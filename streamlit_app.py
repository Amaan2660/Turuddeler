import os
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

# =========================
# Robust DB URL resolution
# =========================
db_url = None
if "DATABASE_URL" in st.secrets:
    db_url = st.secrets["DATABASE_URL"]
elif "database" in st.secrets and "url" in st.secrets["database"]:
    db_url = st.secrets["database"]["url"]

db_url = os.getenv("DATABASE_URL", db_url)

if not db_url:
    st.warning("No DATABASE_URL found in secrets or env; using local SQLite (app.db).")
    db_url = "sqlite:///app.db"

# add psycopg2 and SSL for Supabase if missing
if db_url.startswith("postgresql://") and "psycopg2" not in db_url:
    db_url = db_url.replace("postgresql://", "postgresql+psycopg2://")
if db_url.startswith("postgresql") and "sslmode=" not in db_url:
    sep = "&" if "?" in db_url else "?"
    db_url = f"{db_url}{sep}sslmode=require"

engine = create_engine(db_url, pool_pre_ping=True)

# =========================
# Config & constants
# =========================
CAR_POOL = [
    "209","768","251","179","180","874","875","091","281","280",
    "S-klasse","525","979","516","225"
]
AIRPORT_KW = {"cph", "cph airport", "copenhagen airport", "kastrup", "københavn lufthavn", "kastrup lufthavn"}

DRIVER_COOLDOWN = timedelta(hours=1, minutes=15)  # min gap between a driver’s rides
CAR_HANDOVER_BUFFER = timedelta(minutes=75)       # charge/wash before next driver takes same car

# =========================
# Data access
# =========================
def load_drivers():
    try:
        with engine.connect() as conn:
            df = pd.read_sql("SELECT id, name FROM drivers ORDER BY name", conn)
        if "name" not in df.columns:
            st.error("The 'drivers' table must have a 'name' column.")
        return df
    except Exception as e:
        st.error(f"DB read failed. Check DATABASE_URL. Details: {e}")
        return pd.DataFrame(columns=["id","name"])

# =========================
# Helpers for busy time
# =========================
def is_airport_pickup(pickup_text: str) -> bool:
    s = (pickup_text or "").strip().lower()
    return any(k in s for k in AIRPORT_KW)

def parse_minutes(value):
    try:
        if pd.isna(value) or str(value).strip()=="":
            return None
        return int(float(value))
    except:
        return None

def busy_block_for_row(row: pd.Series) -> timedelta:
    """
    Compute extra busy block (in addition to the ride itself) based on rules:
      - Far-away: Distance_km > 45 => block = trip_minutes + (30m if pickup is airport)
      - Roadshow / Site Inspection: use Duration (Hours/Minutes) if provided; else trip + 60m
    If no trip_minutes known, treat trip ~30m.
    Columns used if present:
      Distance_km, Pickup, Customer Name, Type, Duration Minutes, Duration Hours, Trip Minutes
    """
    trip_min = parse_minutes(row.get("Trip Minutes"))
    if trip_min is None:
        trip_min = 30  # fallback

    distance_km = None
    try:
        if "Distance_km" in row and str(row["Distance_km"]).strip() != "":
            distance_km = float(row["Distance_km"])
    except:
        distance_km = None

    pickup_txt = str(row.get("Pickup", "") or "")
    cust_txt = (str(row.get("Customer Name", "") or "") + " " + str(row.get("Type", "") or "")).lower()

    # Roadshow / Site Inspection — highest priority
    if "roadshow" in cust_txt or "site inspection" in cust_txt:
        dur_min = parse_minutes(row.get("Duration Minutes"))
        if dur_min is None:
            # allow duration in hours too
            dur_hours = row.get("Duration Hours")
            try:
                if dur_hours not in (None, ""):
                    dur_min = int(float(dur_hours) * 60)
            except:
                pass
        if dur_min and dur_min > 0:
            return timedelta(minutes=dur_min)  # block for full stated duration
        return timedelta(minutes=trip_min + 60)  # fallback: trip + 60 on-site

    # Far-away distance rule
    if distance_km is not None and distance_km > 45:
        extra = 30 if is_airport_pickup(pickup_txt) else 0
        return timedelta(minutes=trip_min + extra)

    # Default: just the trip time
    return timedelta(minutes=trip_min)

# =========================
# Assignment logic
# =========================
def assign_plan(rides_df: pd.DataFrame, driver_names: list[str], active_cars: list[str]) -> pd.DataFrame:
    """
    Assign rides to drivers and cars with constraints:
      - Driver cooldown >= 1h15m after each ride
      - Car handover buffer 75m before next driver uses same car
      - Busy block per ride from far-away/roadshow/site inspection rules
    Notes include 'Busy until HH:MM'.
    """
    if not driver_names:
        return rides_df.assign(**{"Assigned Driver":"", "Car":"", "Notes":"No drivers selected"})

    # Parse/normalize times
    rides = rides_df.copy()
    if "Pickup Time" not in rides.columns:
        raise ValueError("CSV must include 'Pickup Time' column.")
    rides["Pickup Time"] = pd.to_datetime(rides["Pickup Time"], errors="coerce")
    if rides["Pickup Time"].isna().any():
        raise ValueError("Some 'Pickup Time' values are invalid. Use a parseable format like 'YYYY-MM-DD HH:MM' or 'HH:MM' (with a separate Date column).")
    # Optional Date column to anchor time-of-day to a specific date
    if "Date" in rides.columns:
        base_date = pd.to_datetime(rides["Date"], errors="coerce").dt.date
        rides["Pickup Time"] = pd.to_datetime(base_date.astype(str) + " " + rides["Pickup Time"].dt.strftime("%H:%M"))

    # Sort by pickup
    rides = rides.sort_values("Pickup Time").reset_index(drop=True)

    # Prepare outputs
    rides["Assigned Driver"] = ""
    rides["Car"] = ""
    rides["Notes"] = ""

    # State trackers
    driver_next_free = {name: datetime.min for name in driver_names}   # when each driver can take the next job
    car_next_free = {car: datetime.min for car in active_cars}         # when each car is free for next (after handover buffer)
    # Optional: remember last driver who used a car (cosmetic, not required here)

    # Round-robins
    driver_idx = 0
    car_idx = 0

    for i, row in rides.iterrows():
        pickup_dt = row["Pickup Time"]

        # Compute busy block for this ride
        busy_td = busy_block_for_row(row)

        # Find a driver who is free by pickup_dt
        assigned_driver = None
        for _ in range(len(driver_names)):
            name = driver_names[driver_idx]
            if pickup_dt >= driver_next_free[name]:
                assigned_driver = name
                driver_idx = (driver_idx + 1) % len(driver_names)
                break
            driver_idx = (driver_idx + 1) % len(driver_names)

        if assigned_driver is None:
            # No driver free — leave unassigned with reason
            rides.at[i, "Notes"] = "No driver free"
            continue

        # Find a car free by pickup_dt
        assigned_car = None
        for _ in range(len(active_cars)):
            car = active_cars[car_idx]
            if pickup_dt >= car_next_free[car]:
                assigned_car = car
                car_idx = (car_idx + 1) % len(active_cars)
                break
            car_idx = (car_idx + 1) % len(active_cars)

        if assigned_car is None:
            rides.at[i, "Notes"] = "No car free"
            # roll back the driver_idx advance so next ride can try the same driver order again
            continue

        # All good — assign
        next_free_driver = pickup_dt + DRIVER_COOLDOWN + busy_td
        next_free_car = pickup_dt + busy_td + CAR_HANDOVER_BUFFER

        rides.at[i, "Assigned Driver"] = assigned_driver
        rides.at[i, "Car"] = assigned_car
        rides.at[i, "Notes"] = f"Busy until {next_free_driver.strftime('%H:%M')}"

        # Update states
        driver_next_free[assigned_driver] = next_free_driver
        car_next_free[assigned_car] = next_free_car

    return rides

# =========================
# UI
# =========================
st.set_page_config(page_title="🚖 Daily Car Plan", page_icon="🚗", layout="wide")
st.title("🚖 Daily Car Plan")

# Load drivers
drivers_df = load_drivers()
if drivers_df.empty:
    st.stop()

# Upload rides CSV
rides_file = st.file_uploader("Upload today's rides CSV", type=["csv"])

if rides_file:
    try:
        rides_df = pd.read_csv(rides_file).fillna("")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # Choose active cars
    default_active = [c for c in CAR_POOL if c.lower() != "s-klasse"]
    active_cars = st.multiselect("Select active cars", CAR_POOL, default=default_active)
    if not active_cars:
        st.warning("Activate at least one car.")
        st.stop()

    # Choose active drivers (limit by number of cars)
    all_driver_names = drivers_df["name"].tolist()
    default_drivers = all_driver_names[: min(len(all_driver_names), len(active_cars))]
    selected_drivers = st.multiselect(
        f"Select drivers for today (max {len(active_cars)} at a time)",
        all_driver_names,
        default=default_drivers
    )

    # Enforce capacity: only as many drivers as cars
    if len(selected_drivers) > len(active_cars):
        st.warning(f"You selected {len(selected_drivers)} drivers but only {len(active_cars)} active cars — only the first {len(active_cars)} will be used.")
        selected_drivers = selected_drivers[: len(active_cars)]

    # Optional knobs
    with st.expander("Advanced timing (optional)"):
        global DRIVER_COOLDOWN, CAR_HANDOVER_BUFFER
        dc_min = st.number_input("Driver cooldown minutes (default 75)", 0, 240, 75, 5)
        hb_min = st.number_input("Car handover buffer minutes (default 75)", 0, 240, 75, 5)
        DRIVER_COOLDOWN = timedelta(minutes=int(dc_min))
        CAR_HANDOVER_BUFFER = timedelta(minutes=int(hb_min))

    # Generate plan
    if st.button("Generate Plan"):
        try:
            plan_df = assign_plan(rides_df, selected_drivers, active_cars)
        except Exception as e:
            st.error(f"Planning failed: {e}")
            st.stop()

        cols = ["Pickup Time", "Ride ID", "Pickup", "Dropoff", "Assigned Driver", "Car", "Notes"]
        for c in cols:
            if c not in plan_df.columns:
                plan_df[c] = ""
        plan_df = plan_df[cols].copy()

        st.subheader("Full Day Plan")
        st.dataframe(plan_df, use_container_width=True)

        st.download_button(
            "Download Plan CSV",
            data=plan_df.to_csv(index=False).encode("utf-8"),
            file_name="daily_car_plan.csv",
            mime="text/csv"
        )

        st.success("✅ Plan generated!")
else:
    st.info("Upload the day's rides CSV to start planning.")
