import os
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
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

# Add psycopg2 driver + SSL for Supabase pooler automatically
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

DEFAULT_DRIVER_COOLDOWN = timedelta(minutes=75)   # gap between a driver's rides
DEFAULT_CAR_HANDOVER = timedelta(minutes=75)      # buffer before a car can switch drivers

# =========================
# Data access
# =========================
def load_drivers():
    with engine.connect() as conn:
        # alias driver_id -> id so the rest of the code keeps working
        df = pd.read_sql(
            "SELECT driver_id AS id, name, number FROM drivers ORDER BY name",
            conn
        )
    return df

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
    Extra busy time based on rules:
      - Roadshow / Site Inspection: use Duration (Minutes/Hours) if present; else Trip + 60.
      - Far-away: Distance_km > 45 => Trip + 30 if pickup is airport else Trip.
      - Else: Trip (or 30 if not provided).
    Uses optional CSV columns:
      Trip Minutes, Distance_km, Pickup, Customer Name, Type, Duration Minutes, Duration Hours
    """
    trip_min = parse_minutes(row.get("Trip Minutes"))
    if trip_min is None:
        trip_min = 30  # fallback trip estimate

    distance_km = None
    try:
        if "Distance_km" in row and str(row["Distance_km"]).strip() != "":
            distance_km = float(row["Distance_km"])
    except:
        distance_km = None

    pickup_txt = str(row.get("Pickup", "") or "")
    cust_txt = (str(row.get("Customer Name", "") or "") + " " + str(row.get("Type", "") or "")).lower()

    # Roadshow / Site Inspection first
    if "roadshow" in cust_txt or "site inspection" in cust_txt:
        dur_min = parse_minutes(row.get("Duration Minutes"))
        if dur_min is None:
            dur_hours = row.get("Duration Hours")
            try:
                if dur_hours not in (None, ""):
                    dur_min = int(float(dur_hours) * 60)
            except:
                dur_min = None
        if dur_min and dur_min > 0:
            return timedelta(minutes=dur_min)
        return timedelta(minutes=trip_min + 60)

    # Far-away rule
    if distance_km is not None and distance_km > 45:
        extra = 30 if is_airport_pickup(pickup_txt) else 0
        return timedelta(minutes=trip_min + extra)

    # Default = trip time
    return timedelta(minutes=trip_min)

# =========================
# Assignment logic
# =========================
def assign_plan(
    rides_df: pd.DataFrame,
    driver_names: list,
    active_cars: list,
    driver_cooldown: timedelta,
    car_handover: timedelta
) -> pd.DataFrame:
    """
    Assign rides with constraints:
      - Driver cooldown after each assigned ride
      - Car handover buffer before car can be used by next driver
      - Busy block per ride per rules above
    """
    if not driver_names:
        return rides_df.assign(**{"Assigned Driver":"", "Car":"", "Notes":"No drivers selected"})

    rides = rides_df.copy()

    # Require 'Pickup Time'
    if "Pickup Time" not in rides.columns:
        raise ValueError("CSV must include 'Pickup Time' column.")
    rides["Pickup Time"] = pd.to_datetime(rides["Pickup Time"], errors="coerce")
    if rides["Pickup Time"].isna().any():
        raise ValueError("Some 'Pickup Time' values are invalid. Use a parseable format like 'YYYY-MM-DD HH:MM' or 'HH:MM' (with a separate Date column).")

    # Optional 'Date' column to anchor HH:MM to a day
    if "Date" in rides.columns and rides["Date"].notna().any():
        base_date = pd.to_datetime(rides["Date"], errors="coerce").dt.date
        rides["Pickup Time"] = pd.to_datetime(base_date.astype(str) + " " + rides["Pickup Time"].dt.strftime("%H:%M"))

    rides = rides.sort_values("Pickup Time").reset_index(drop=True)

    # Prepare outputs
    rides["Assigned Driver"] = ""
    rides["Car"] = ""
    rides["Notes"] = ""

    # State trackers
    driver_next_free = {name: datetime.min for name in driver_names}
    car_next_free = {car: datetime.min for car in active_cars}

    # Round-robins
    driver_idx = 0
    car_idx = 0

    for i, row in rides.iterrows():
        pickup_dt = row["Pickup Time"]
        busy_td = busy_block_for_row(row)

        # Find free driver
        assigned_driver = None
        for _ in range(len(driver_names)):
            name = driver_names[driver_idx]
            if pickup_dt >= driver_next_free[name]:
                assigned_driver = name
                driver_idx = (driver_idx + 1) % len(driver_names)
                break
            driver_idx = (driver_idx + 1) % len(driver_names)

        if assigned_driver is None:
            rides.at[i, "Notes"] = "No driver free"
            continue

        # Find free car
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
            # (we keep the driver assignment attempt order; next ride tries the next car)
            continue

        # Assign + update states
        next_free_driver = pickup_dt + driver_cooldown + busy_td
        next_free_car = pickup_dt + busy_td + car_handover

        rides.at[i, "Assigned Driver"] = assigned_driver
        rides.at[i, "Car"] = assigned_car
        rides.at[i, "Notes"] = f"Busy until {next_free_driver.strftime('%H:%M')}"

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

    # Choose active drivers (cap by active cars)
    all_driver_names = drivers_df["name"].tolist()
    default_drivers = all_driver_names[: min(len(all_driver_names), len(active_cars))]
    selected_drivers = st.multiselect(
        f"Select drivers for today (max {len(active_cars)} at a time)",
        all_driver_names,
        default=default_drivers
    )
    if len(selected_drivers) > len(active_cars):
        st.warning(f"You selected {len(selected_drivers)} drivers but only {len(active_cars)} active cars — only the first {len(active_cars)} will be used.")
        selected_drivers = selected_drivers[: len(active_cars)]

    # Advanced timing knobs (no globals)
    with st.expander("Advanced timing (optional)"):
        dc_min = st.number_input("Driver cooldown minutes (default 75)", 0, 240, 75, 5)
        hb_min = st.number_input("Car handover buffer minutes (default 75)", 0, 240, 75, 5)
        driver_cooldown = timedelta(minutes=int(dc_min))
        car_handover = timedelta(minutes=int(hb_min))

    # Generate plan
    if st.button("Generate Plan"):
        try:
            plan_df = assign_plan(
                rides_df,
                selected_drivers,
                active_cars,
                driver_cooldown if 'driver_cooldown' in locals() else DEFAULT_DRIVER_COOLDOWN,
                car_handover if 'car_handover' in locals() else DEFAULT_CAR_HANDOVER
            )
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
