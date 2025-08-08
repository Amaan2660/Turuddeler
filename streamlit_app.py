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
# Config & keywords
# =========================
CAR_POOL = [
    "209","768","251","179","180","874","875","091","281","280",
    "S-klasse","525","979","516","225"
]

# Keywords to detect airport & Copenhagen in free text
AIRPORT_KW = {"cph", "cph airport", "copenhagen airport", "kastrup", "københavn lufthavn", "kastrup lufthavn"}
CPH_KW = {"copenhagen", "københavn", "cph", "kastrup", "frederiksberg"}

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
def _norm(s: str) -> str:
    return (s or "").strip().lower()

def is_airport_pickup(pickup_text: str) -> bool:
    s = _norm(pickup_text)
    return any(k in s for k in AIRPORT_KW)

def is_copenhagen(text_a: str, text_b: str) -> bool:
    s1, s2 = _norm(text_a), _norm(text_b)
    return any(k in s1 for k in CPH_KW) or any(k in s2 for k in CPH_KW)

def parse_minutes(value):
    try:
        if pd.isna(value) or str(value).strip()=="":
            return None
        return int(float(value))
    except:
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

    # Roadshow / Site Inspection
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

    return timedelta(minutes=trip_min)

def rule_gap_after(prev_row: pd.Series, next_row: pd.Series) -> timedelta:
    """
    Extra gap *between* bookings on top of the previous job's busy time.
    Copenhagen-only:
      - If previous pickup was AIRPORT -> +50 min before the next job
      - If previous pickup was CITY and next pickup is AIRPORT -> +0 min (no extra)
      - Else -> +0 min
    """
    prev_pick = str(prev_row.get("Pickup", "") or "")
    prev_drop = str(prev_row.get("Dropoff", "") or "")
    next_pick = str(next_row.get("Pickup", "") or "")
    next_drop = str(next_row.get("Dropoff", "") or "")

    if not is_copenhagen(prev_pick, prev_drop) or not is_copenhagen(next_pick, next_drop):
        return timedelta(0)

    prev_is_airport = is_airport_pickup(prev_pick)
    next_is_airport = is_airport_pickup(next_pick)

    if prev_is_airport:
        return timedelta(minutes=50)
    if (not prev_is_airport) and next_is_airport:
        return timedelta(0)
    return timedelta(0)

def extract_busy_until(note: str):
    if not note: return None
    m = re.search(r"Busy until (\d{2}:\d{2})", note)
    if not m: return None
    return m.group(1)

def in_flex_window(pickup_dt: datetime, start_dt: datetime, end_dt: datetime,
                   soft_early: int, hard_early: int, late_allow: int):
    """
    Return 'soft'/'hard'/None if pickup fits shift+flex. Overnight handled.
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

def build_bilplan(plan_df: pd.DataFrame, service_date: datetime.date) -> str:
    """
    BILPLAN dd/mm/yy
    Name - Car fra HH:MM-HH:MM
    (aggregates contiguous usage per (driver, car))
    """
    if plan_df.empty:
        return f"BILPLAN {service_date.strftime('%d/%m/%y')}\n(No assignments)"

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
            end = start + timedelta(hours=1)
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
# Assignment (prefs, shifts, handovers, rules)
# =========================
def assign_plan_with_prefs(
    rides_df: pd.DataFrame,
    driver_rows: pd.DataFrame,
    active_cars: list,
    service_date: datetime.date,
    car_handover: timedelta,
    soft_early: int,
    hard_early: int,
    late_allow: int,
    auto_relax_to_soft: bool
) -> pd.DataFrame:
    """
    - Per-driver shift window with soft/hard early & late allow (via time_input)
    - Optional per-driver preferred car; else round-robin across active cars
    - Enforce car handover buffer
    - Busy time from busy_block_for_row(row)
    - EXTRA GAP RULES between jobs (CPH airport/city)
    """
    rides = rides_df.copy()

    # Require 'Pickup Time'
    if "Pickup Time" not in rides.columns:
        raise ValueError("CSV must include 'Pickup Time' column.")
    rides["Pickup Time"] = pd.to_datetime(rides["Pickup Time"], errors="coerce")
    if rides["Pickup Time"].isna().any():
        raise ValueError("Some 'Pickup Time' values are invalid. Use 'YYYY-MM-DD HH:MM' or 'HH:MM' (+Date column).")

    # Optional 'Date'
    if "Date" in rides.columns and rides["Date"].notna().any():
        base_date = pd.to_datetime(rides["Date"], errors="coerce").dt.date
        rides["Pickup Time"] = pd.to_datetime(base_date.astype(str) + " " + rides["Pickup Time"].dt.strftime("%H:%M"))

    rides = rides.sort_values("Pickup Time").reset_index(drop=True)

    # Prepare outputs
    rides["Assigned Driver"] = ""
    rides["Car"] = ""
    rides["Notes"] = ""

    # Driver state
    states = {}
    for _, r in driver_rows.iterrows():
        name = r["name"]
        st_dt = r.get("start")
        en_dt = r.get("end")
        if st_dt is None or en_dt is None:
            st_dt = datetime.combine(service_date, datetime.strptime("00:00", "%H:%M").time())
            en_dt = datetime.combine(service_date, datetime.strptime("23:59", "%H:%M").time())
        if en_dt <= st_dt:
            en_dt = en_dt + timedelta(days=1)
        states[name] = {
            "start_dt": st_dt,
            "end_dt":   en_dt,
            "preferred_car": (r.get("preferred_car") or "").strip(),
            "last_row": None,
        }

    # Car state
    car_next_free = {car: datetime.min for car in active_cars}

    # Helper to pick a car for a driver
    rr_idx = 0
    def pick_car(for_driver: str, pickup_dt: datetime):
        nonlocal rr_idx
        pref = states[for_driver]["preferred_car"]
        if pref and pref in active_cars and pickup_dt >= car_next_free[pref]:
            return pref
        for _ in range(len(active_cars)):
            car = active_cars[rr_idx]
            rr_idx = (rr_idx + 1) % len(active_cars)
            if pickup_dt >= car_next_free[car]:
                return car
        return None

    # Driver round-robin order
    driver_names = list(states.keys())
    d_rr = 0

    for i, row in rides.iterrows():
        pickup_dt = row["Pickup Time"]
        busy_td = busy_block_for_row(row)

        feasible = []
        hard_needed = {}
        for _ in range(len(driver_names)):
            name = driver_names[d_rr]
            d_rr = (d_rr + 1) % len(driver_names)

            st_dt = states[name]["start_dt"]
            en_dt = states[name]["end_dt"]
            flex = in_flex_window(pickup_dt, st_dt, en_dt, soft_early, hard_early, late_allow)
            if flex is None:
                if auto_relax_to_soft:
                    soft_start = st_dt - timedelta(minutes=soft_early)
                    soft_end   = en_dt + timedelta(minutes=late_allow)
                    if not (soft_start <= pickup_dt <= soft_end):
                        continue
                    flex = "soft"
                else:
                    continue

            # rule-based availability vs last job
            last_row = states[name]["last_row"]
            if last_row is not None:
                prev_pickup = pd.to_datetime(last_row["Pickup Time"])
                prev_busy = busy_block_for_row(last_row)
                prev_finish = prev_pickup + prev_busy
                extra_gap = rule_gap_after(last_row, row)
                earliest_next = prev_finish + extra_gap
                if pickup_dt < earliest_next:
                    continue

            feasible.append(name)
            hard_needed[name] = (flex == "hard")

        if not feasible:
            rides.at[i, "Notes"] = "No driver free (window/gap)"
            continue

        chosen = feasible[0]
        chosen_hard = hard_needed[chosen]

        # car choice (handover buffer enforced per-car)
        car = pick_car(chosen, pickup_dt)
        if car is None:
            rides.at[i, "Notes"] = "No car free"
            continue

        # Update states
        states[chosen]["last_row"] = row.copy()

        # car blocked until busy done + handover
        next_free_car = pickup_dt + busy_td + car_handover
        car_next_free[car] = next_free_car

        # Notes & write
        busy_until = pickup_dt + busy_td
        note = f"Busy until {busy_until.strftime('%H:%M')}"
        if chosen_hard:
            note += " · Hard Early Start"
        rides.at[i, "Assigned Driver"] = chosen
        rides.at[i, "Car"] = car
        rides.at[i, "Notes"] = note

    return rides

# =========================
# UI (Generate-only updates)
# =========================
st.set_page_config(page_title="🚖 Daily Car Plan", page_icon="🚗", layout="wide")
st.title("🚖 Daily Car Plan")

drivers_df = load_drivers()
if drivers_df.empty:
    st.stop()

# Session state
if "run_plan" not in st.session_state:
    st.session_state.run_plan = False
if "snapshot" not in st.session_state:
    st.session_state.snapshot = {}

# Upload rides CSV
rides_file = st.file_uploader("Upload today's rides CSV", type=["csv"])

if rides_file:
    try:
        rides_df_live = pd.read_csv(rides_file).fillna("")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    service_date = st.date_input("Service date", value=datetime.now().date())

    # Active cars
    default_active = [c for c in CAR_POOL if c.lower() != "s-klasse"]
    active_cars_live = st.multiselect("Active cars (handovers allowed via buffer)", CAR_POOL, default=default_active)
    if not active_cars_live:
        st.warning("Activate at least one car.")
        st.stop()

    # Drivers for today + per-driver options
    st.subheader("Drivers for today")
    all_names = drivers_df["name"].tolist()
    selected_names_live = st.multiselect(
        "Select drivers (you can select more than active cars)",
        all_names,
        default=all_names[:min(len(all_names), 10)]
    )

    st.caption("Optional per-driver settings (leave blank to let allocator decide):")
    rows = []
    cols = st.columns(2)
    for i, name in enumerate(selected_names_live):
        with cols[i % 2]:
            st.markdown(f"**{name}**")
            start_t = st.time_input(f"{name} start", value=None, key=f"start_{name}")
            end_t   = st.time_input(f"{name} end", value=None, key=f"end_{name}")
            pref    = st.selectbox(f"{name} preferred car (optional)", [""] + active_cars_live, index=0, key=f"car_{name}")
            # Convert to datetimes anchored to service_date (None => all-day)
            start_dt = datetime.combine(service_date, start_t) if start_t else None
            end_dt   = datetime.combine(service_date, end_t) if end_t else None
            rows.append({"name": name, "start": start_dt, "end": end_dt, "preferred_car": pref})
    driver_rows_live = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["name","start","end","preferred_car"])

    # Timing sliders
    with st.expander("Advanced timing (optional)"):
        hb_min   = st.slider("Car handover buffer (minutes)", 0, 240, 75, 5)
        soft_min = st.slider("Soft early start allowed (min)", 0, 180, 30, 5)
        hard_min = st.slider("Hard early start allowed (min)", 0, 240, 60, 5)
        late_min = st.slider("Late end allowed (min)", 0, 120, 20, 5)
        auto_relax = st.checkbox("If a job is unassigned for window only, auto-relax to SOFT", value=True)

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Generate"):
            # snapshot inputs so UI edits don't auto-rerun result
            st.session_state.snapshot = {
                "rides_df": rides_df_live.copy(),
                "service_date": service_date,
                "active_cars": list(active_cars_live),
                "driver_rows": driver_rows_live.copy(),
                "car_handover": timedelta(minutes=int(hb_min)),
                "soft_early": int(soft_min),
                "hard_early": int(hard_min),
                "late_allow": int(late_min),
                "auto_relax": bool(auto_relax),
            }
            st.session_state.run_plan = True
    with c2:
        if st.button("Clear"):
            st.session_state.run_plan = False
            st.session_state.snapshot = {}

    if st.session_state.run_plan:
        s = st.session_state.snapshot
        try:
            plan_df = assign_plan_with_prefs(
                rides_df=s["rides_df"],
                driver_rows=s["driver_rows"],
                active_cars=s["active_cars"],
                service_date=s["service_date"],
                car_handover=s["car_handover"],
                soft_early=s["soft_early"],
                hard_early=s["hard_early"],
                late_allow=s["late_allow"],
                auto_relax_to_soft=s["auto_relax"]
            )
        except Exception as e:
            st.error(f"Planning failed: {e}")
            st.stop()

        # Ensure columns & show the WHOLE plan
        cols = ["Pickup Time", "Ride ID", "Pickup", "Dropoff", "Allocated Driver", "Allocated Car", "Notes"]
        # Rename for nicer headers
        plan_df.rename(columns={"Assigned Driver":"Allocated Driver", "Car":"Allocated Car"}, inplace=True)
        for c in cols:
            if c not in plan_df.columns:
                plan_df[c] = ""
        plan_df = plan_df[cols].copy().sort_values("Pickup Time").reset_index(drop=True)

        st.subheader("Full Day Plan")
        st.dataframe(plan_df, use_container_width=True, height=620)

        st.download_button(
            "Download Plan CSV",
            data=plan_df.to_csv(index=False).encode("utf-8"),
            file_name="daily_car_plan.csv",
            mime="text/csv"
        )

        bilplan_text = build_bilplan(plan_df.rename(columns={"Allocated Driver":"Assigned Driver", "Allocated Car":"Car"}), s["service_date"])
        st.subheader("BILPLAN")
        st.text_area("BILPLAN text", bilplan_text, height=320)
        st.download_button(
            "Download BILPLAN (.txt)",
            data=bilplan_text.encode("utf-8"),
            file_name=f"bilplan_{s['service_date'].isoformat()}.txt",
            mime="text/plain"
        )
else:
    st.info("Upload the day's rides CSV to start planning.")
