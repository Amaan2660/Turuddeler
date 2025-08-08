import os
import re
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

# Ensure psycopg2 + SSL for Supabase pooler
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

# =========================
# Data access
# =========================
def load_drivers():
    with engine.connect() as conn:
        df = pd.read_sql(
            "SELECT driver_id AS id, name, number FROM drivers ORDER BY name",
            conn
        )
    return df

# =========================
# Helpers
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

def parse_time(hhmm: str):
    if not hhmm: return None
    hhmm = str(hhmm).strip()
    for fmt in ("%H:%M", "%H.%M", "%H%M"):
        try:
            return datetime.strptime(hhmm, fmt).time()
        except:
            pass
    return None

def combine(service_date: datetime.date, t: datetime.time) -> datetime:
    return datetime.combine(service_date, t)

def in_flex_window(pickup_dt: datetime, start_dt: datetime, end_dt: datetime,
                   soft_early: int, hard_early: int, late_allow: int):
    """
    Return "soft"/"hard"/None if pickup fits shift with flex.
    Overnight handled by rolling end to next day if <= start.
    """
    if end_dt <= start_dt:
        end_dt = end_dt + timedelta(days=1)
    soft_start = start_dt - timedelta(minutes=soft_early)
    soft_end   = end_dt + timedelta(minutes=late_allow)
    if soft_start <= pickup_dt <= soft_end:
        return "soft"
    hard_start = start_dt - timedelta(minutes=hard_early)
    hard_end   = end_dt + timedelta(minutes=late_allow)
    if hard_start <= pickup_dt <= hard_end:
        return "hard"
    return None

def busy_block_for_row(row: pd.Series) -> timedelta:
    """
    Extra busy time:
      - Roadshow / Site Inspection: use Duration Minutes/Hours if given; else Trip + 60.
      - Far-away: Distance_km > 45 => Trip + (30 if pickup is airport else 0).
      - Else: Trip (or 30 fallback).
    CSV columns used if present:
      Trip Minutes, Distance_km, Pickup, Customer Name, Type, Duration Minutes, Duration Hours
    """
    trip_min = parse_minutes(row.get("Trip Minutes")) or 30
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

    # Far-away
    if distance_km is not None and distance_km > 45:
        extra = 30 if is_airport_pickup(pickup_txt) else 0
        return timedelta(minutes=trip_min + extra)

    return timedelta(minutes=trip_min)

# =========================
# Assignment logic (prefs & shifts)
# =========================
def assign_plan_with_prefs(
    rides_df: pd.DataFrame,
    driver_rows: pd.DataFrame,
    active_cars: list,
    service_date: datetime.date,
    driver_cooldown: timedelta,
    car_handover: timedelta,
    soft_early: int,
    hard_early: int,
    late_allow: int
) -> pd.DataFrame:
    """
    driver_rows columns: name, start, end, preferred_car (optional "")
    - Per-driver shift window with soft/hard early + late allowance
    - Optional preferred car; else round-robin across active cars
    - Enforces car handover & driver cooldown & busy rules
    - Notes include Busy-until and Hard Early Start flag
    """
    rides = rides_df.copy()

    # Require 'Pickup Time'
    if "Pickup Time" not in rides.columns:
        raise ValueError("CSV must include 'Pickup Time' column.")
    rides["Pickup Time"] = pd.to_datetime(rides["Pickup Time"], errors="coerce")
    if rides["Pickup Time"].isna().any():
        raise ValueError("Some 'Pickup Time' values are invalid. Use 'YYYY-MM-DD HH:MM' or 'HH:MM' (+ Date column).")

    # Optional 'Date'
    if "Date" in rides.columns and rides["Date"].notna().any():
        base_date = pd.to_datetime(rides["Date"], errors="coerce").dt.date
        rides["Pickup Time"] = pd.to_datetime(base_date.astype(str) + " " + rides["Pickup Time"].dt.strftime("%H:%M"))

    rides = rides.sort_values("Pickup Time").reset_index(drop=True)

    # Prepare outputs
    rides["Assigned Driver"] = ""
    rides["Car"] = ""
    rides["Notes"] = ""

    # Build driver state
    states = {}
    for _, r in driver_rows.iterrows():
        name = r["name"]
        stt = parse_time(r.get("start",""))
        endt = parse_time(r.get("end",""))
        if stt is None or endt is None:
            stt = parse_time("00:00")
            endt = parse_time("23:59")
        states[name] = {
            "start_dt": combine(service_date, stt),
            "end_dt":   combine(service_date, endt),
            "preferred_car": (r.get("preferred_car") or "").strip(),
            "next_free": datetime.min
        }

    # Car state
    car_next_free = {car: datetime.min for car in active_cars}

    # Helper to pick a car for a driver
    rr_idx = 0
    def pick_car(for_driver: str, pickup_dt: datetime):
        pref = states[for_driver]["preferred_car"]
        if pref and pref in active_cars:
            if pickup_dt >= car_next_free[pref]:
                return pref
        nonlocal rr_idx
        for _ in range(len(active_cars)):
            car = active_cars[rr_idx]
            rr_idx = (rr_idx + 1) % len(active_cars)
            if pickup_dt >= car_next_free[car]:
                return car
        return None

    driver_names = list(states.keys())
    d_rr = 0

    for i, row in rides.iterrows():
        pickup_dt = row["Pickup Time"]
        busy_td = busy_block_for_row(row)

        # find feasible drivers by shift window + next_free
        feasible = []
        hard_needed = {}
        for _ in range(len(driver_names)):
            name = driver_names[d_rr]
            d_rr = (d_rr + 1) % len(driver_names)

            st_dt = states[name]["start_dt"]
            en_dt = states[name]["end_dt"]
            flex = in_flex_window(pickup_dt, st_dt, en_dt, soft_early, hard_early, late_allow)
            if flex is None:
                continue
            if pickup_dt < states[name]["next_free"]:
                continue
            feasible.append(name)
            hard_needed[name] = (flex == "hard")

        if not feasible:
            rides.at[i, "Notes"] = "No driver free (window)"
            continue

        chosen = feasible[0]
        chosen_hard = hard_needed[chosen]

        car = pick_car(chosen, pickup_dt)
        if car is None:
            rides.at[i, "Notes"] = "No car free"
            continue

        next_free_driver = pickup_dt + driver_cooldown + busy_td
        next_free_car = pickup_dt + busy_td + car_handover

        rides.at[i, "Assigned Driver"] = chosen
        rides.at[i, "Car"] = car
        note = f"Busy until {next_free_driver.strftime('%H:%M')}"
        if chosen_hard:
            note += " · Hard Early Start"
        rides.at[i, "Notes"] = note

        states[chosen]["next_free"] = next_free_driver
        car_next_free[car] = next_free_car

    return rides

# ============== BILPLAN (car plan) export =================
def extract_busy_until(note: str):
    if not note: return None
    m = re.search(r"Busy until (\d{2}:\d{2})", note)
    if not m: return None
    return m.group(1)

def build_bilplan(plan_df: pd.DataFrame, service_date: datetime.date) -> str:
    """
    Produce lines like:
    BILPLAN dd/mm/yy
    Lina - 209 fra 07:00-15:30
    Barik - 209 fra 16:00-20:00 (after handover)
    """
    if plan_df.empty:
        return f"BILPLAN {service_date.strftime('%d/%m/%y')}\n(No assignments)"
    # Compute per (driver, car) windows: min pickup, max busy-until
    rows = []
    for _, r in plan_df.iterrows():
        name = r.get("Assigned Driver","")
        car = r.get("Car","")
        if not name or not car: 
            continue
        start = pd.to_datetime(r["Pickup Time"])
        bu = extract_busy_until(str(r.get("Notes","")))
        if bu:
            end = datetime.combine(service_date, datetime.strptime(bu, "%H:%M").time())
            if end <= start:
                end += timedelta(days=1)
        else:
            end = start + timedelta(hours=1)  # fallback
        rows.append({"name": name, "car": car, "start": start, "end": end})

    if not rows:
        return f"BILPLAN {service_date.strftime('%d/%m/%y')}\n(No assignments)"

    tmp = pd.DataFrame(rows)
    agg = tmp.groupby(["name","car"]).agg(start=("start","min"), end=("end","max")).reset_index()
    agg = agg.sort_values(["car","start"]).reset_index(drop=True)

    lines = [f"BILPLAN {service_date.strftime('%d/%m/%y')}"]
    for _, r in agg.iterrows():
        span = f"{r['start'].strftime('%H:%M')}-{r['end'].strftime('%H:%M')}"
        lines.append(f"{r['name']} - {r['car']} fra {span}")
    return "\n".join(lines)

# =========================
# UI
# =========================
st.set_page_config(page_title="🚖 Daily Car Plan", page_icon="🚗", layout="wide")
st.title("🚖 Daily Car Plan")

drivers_df = load_drivers()
if drivers_df.empty:
    st.stop()

rides_file = st.file_uploader("Upload today's rides CSV", type=["csv"])

if rides_file:
    try:
        rides_df = pd.read_csv(rides_file).fillna("")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    service_date = st.date_input("Service date", value=datetime.now().date())

    # Active cars
    default_active = [c for c in CAR_POOL if c.lower() != "s-klasse"]
    active_cars = st.multiselect("Active cars (only these can be used)", CAR_POOL, default=default_active)
    if not active_cars:
        st.warning("Activate at least one car.")
        st.stop()

    # Drivers for today + per-driver options
    st.subheader("Drivers for today")
    all_names = drivers_df["name"].tolist()
    default_names = all_names[: min(len(all_names), len(active_cars))]
    selected_names = st.multiselect(
        f"Select drivers (capacity limited by active cars: {len(active_cars)})",
        all_names,
        default=default_names
    )
    if len(selected_names) > len(active_cars):
        st.warning(f"You selected {len(selected_names)} drivers but only {len(active_cars)} cars — only the first {len(active_cars)} will be used.")
        selected_names = selected_names[: len(active_cars)]

    st.caption("Optional per-driver settings (leave blank to let allocator decide):")
    rows = []
    cols = st.columns(3)
    for i, name in enumerate(selected_names):
        with cols[i % 3]:
            st.markdown(f"**{name}**")
            start = st.text_input(f"{name} start (HH:MM)", key=f"start_{name}", placeholder="08:00")
            end   = st.text_input(f"{name} end (HH:MM)", key=f"end_{name}", placeholder="16:00")
            pref  = st.selectbox(f"{name} preferred car (optional)", [""] + active_cars, index=0, key=f"car_{name}")
            rows.append({"name": name, "start": start, "end": end, "preferred_car": pref})
    driver_rows = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["name","start","end","preferred_car"])

    # Timing knobs (no globals)
    with st.expander("Advanced timing (optional)"):
        dc_min = st.number_input("Driver cooldown minutes (default 75)", 0, 240, 75, 5)
        hb_min = st.number_input("Car handover buffer minutes (default 75)", 0, 240, 75, 5)
        soft_min = st.number_input("Soft early start allowed (min)", 0, 180, 30, 5)
        hard_min = st.number_input("Hard early start allowed (min)", 0, 240, 60, 5)
        late_min = st.number_input("Late end allowed (min)", 0, 120, 20, 5)

        driver_cooldown = timedelta(minutes=int(dc_min))
        car_handover = timedelta(minutes=int(hb_min))
        soft_early = int(soft_min)
        hard_early = int(hard_min)
        late_allow = int(late_min)

    if st.button("Generate Plan"):
        try:
            plan_df = assign_plan_with_prefs(
                rides_df=rides_df,
                driver_rows=driver_rows,
                active_cars=active_cars,
                service_date=service_date,
                driver_cooldown=driver_cooldown,
                car_handover=car_handover,
                soft_early=soft_early,
                hard_early=hard_early,
                late_allow=late_allow
            )
        except Exception as e:
            st.error(f"Planning failed: {e}")
            st.stop()

        # Ensure columns exist and show the WHOLE plan
        cols = ["Pickup Time", "Ride ID", "Pickup", "Dropoff", "Assigned Driver", "Car", "Notes"]
        for c in cols:
            if c not in plan_df.columns:
                plan_df[c] = ""
        plan_df = plan_df[cols].copy()
        plan_df = plan_df.sort_values("Pickup Time").reset_index(drop=True)

        st.subheader("Full Day Plan")
        st.dataframe(plan_df, use_container_width=True, height=600)

        # Downloads
        st.download_button(
            "Download Plan CSV",
            data=plan_df.to_csv(index=False).encode("utf-8"),
            file_name="daily_car_plan.csv",
            mime="text/csv"
        )

        # BILPLAN export
        bilplan_text = build_bilplan(plan_df, service_date)
        st.subheader("BILPLAN")
        st.text_area("BILPLAN text", bilplan_text, height=300)
        st.download_button(
            "Download BILPLAN (.txt)",
            data=bilplan_text.encode("utf-8"),
            file_name=f"bilplan_{service_date.isoformat()}.txt",
            mime="text/plain"
        )

        st.success("✅ Plan generated!")
else:
    st.info("Upload the day's rides CSV to start planning.")
