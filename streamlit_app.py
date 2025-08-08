import os
import re
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, timedelta, date

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

AIRPORT_KW = {"cph", "cph airport", "copenhagen airport", "kastrup", "københavn lufthavn", "kastrup lufthavn"}
CPH_KW = {"copenhagen", "københavn", "cph", "kastrup", "frederiksberg"}

# Far-away settings (force to 209)
FAR_DISTANCE_KM = 45.0
FAR_CAR_ID = "209"

# =========================
# Data access
# =========================
def load_drivers():
    """drivers table with columns: driver_id, name, number"""
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

def parse_float(value):
    try:
        if pd.isna(value) or str(value).strip()=="":
            return None
        return float(str(value).replace(",", "."))
    except:
        return None

def is_far_away_row(row: pd.Series) -> bool:
    """True if Distance_km >= FAR_DISTANCE_KM (treat as far-away)."""
    dk = parse_float(row.get("Distance_km"))
    return (dk is not None) and (dk >= FAR_DISTANCE_KM)

def busy_block_for_row(row: pd.Series) -> timedelta:
    """
    Busy time for the ride (duration driver is occupied from pickup):
      - Roadshow / Site Inspection: Duration Minutes/Hours if given; else Trip + 60.
      - Far-away: Distance_km > FAR_DISTANCE_KM => Trip + (30 if pickup is airport else 0).
      - Else: Trip (or 30 fallback).
    CSV columns used if present:
      Trip Minutes, Distance_km, Pickup, Customer Name, Type, Duration Minutes, Duration Hours
    """
    trip_min = parse_minutes(row.get("Trip Minutes")) or 30

    distance_km = parse_float(row.get("Distance_km"))
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
    if distance_km is not None and distance_km >= FAR_DISTANCE_KM:
        extra = 30 if is_airport_pickup(pickup_txt) else 0
        return timedelta(minutes=trip_min + extra)

    return timedelta(minutes=trip_min)

def rule_gap_after(prev_row: pd.Series, next_row: pd.Series) -> timedelta:
    """
    Extra gap *between* bookings on top of the previous job's busy time.
    Copenhagen-only:
      - If previous pickup was AIRPORT -> +50 min before the next job
      - If previous pickup was CITY and next pickup is AIRPORT -> +0 min
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

# =========================
# Car roster (one car per driver per shift)
# =========================
def build_car_roster(driver_rows: pd.DataFrame, active_cars: list, handover: timedelta):
    """
    Assign exactly ONE car per driver for the entire shift.
    - Honors preferred_car if possible
    - If car is still busy, next driver's effective start is pushed to (car_free + handover)
    Returns:
      roster: dict name -> car_id
      effective_windows: dict name -> (effective_start, end)
    """
    car_free_at = {car: datetime.min for car in active_cars}
    roster = {}
    windows = {}

    def _start_key(r):
        st = r.get("start")
        return st if isinstance(st, datetime) else datetime.min

    rows = driver_rows.to_dict("records")
    rows.sort(key=_start_key)

    for r in rows:
        name = r["name"]
        req_start = r.get("start")
        req_end = r.get("end")
        if req_start is None or req_end is None:
            req_start = datetime.combine(date.today(), datetime.strptime("00:00","%H:%M").time())
            req_end   = datetime.combine(date.today(), datetime.strptime("23:59","%H:%M").time())
        if req_end <= req_start:
            req_end = req_end + timedelta(days=1)

        pref = (r.get("preferred_car") or "").strip()

        # Candidate order: preferred first (if any), otherwise choose earliest-free car
        if pref and pref in active_cars:
            candidates = [pref] + [c for c in active_cars if c != pref]
        else:
            candidates = list(active_cars)

        # pick the car that minimizes delay from requested start
        best_car = None
        best_earliest_start = None
        for car in candidates:
            earliest_start = max(req_start, car_free_at[car] + handover)  # must respect handover before next shift
            if best_car is None or earliest_start < best_earliest_start:
                best_car = car
                best_earliest_start = earliest_start

        roster[name] = best_car
        eff_start = best_earliest_start
        windows[name] = (eff_start, req_end)
        # After this driver, the car becomes free at their end (handover applied when next driver is chosen)
        car_free_at[best_car] = req_end

    return roster, windows

def build_bilplan(plan_df: pd.DataFrame, service_date: date) -> str:
    """
    BILPLAN dd/mm/yy
    Name - Car fra HH:MM-HH:MM
    (aggregates contiguous usage per (driver, car))
    """
    if plan_df.empty:
        return f"BILPLAN {service_date.strftime('%d/%m/%y')}\n(No assignments)"

    rows = []
    for _, r in plan_df.iterrows():
        name = r.get("Allocated Driver","") or r.get("Assigned Driver","")
        car = r.get("Allocated Car","") or r.get("Car","")
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
# Assignment (uses roster + far-away->209 rule)
# =========================
def assign_plan_with_rostered_cars(
    rides_df: pd.DataFrame,
    driver_rows: pd.DataFrame,
    active_cars: list,
    service_date: date,
    car_handover: timedelta,
    soft_early: int,
    hard_early: int,
    late_allow: int,
    auto_relax_to_soft: bool,
    respect_dates: bool
) -> pd.DataFrame:
    """
    - Build a car roster (one car per driver for whole shift; shifts may be pushed by car availability + handover)
    - Assign rides respecting:
        * driver effective shift window (with soft/hard early & late allow)
        * driver's rostered car only
        * far-away rides require FAR_CAR_ID (e.g., 209)
        * ride busy time + CPH airport/city gap rules (between jobs for the same driver)
    """
    rides = rides_df.copy()

    # parse / anchor pickup times
    if "Pickup Time" not in rides.columns:
        raise ValueError("CSV must include 'Pickup Time' column.")
    rides["Pickup Time"] = pd.to_datetime(rides["Pickup Time"], errors="coerce")
    if rides["Pickup Time"].isna().any():
        raise ValueError("Some 'Pickup Time' values are invalid. Use 'HH:MM' or 'YYYY-MM-DD HH:MM'.")

    def _anchor(dt: pd.Timestamp) -> datetime:
        if pd.isna(dt):
            return dt
        if respect_dates and (dt.date() != datetime(1970,1,1).date()):
            return dt.to_pydatetime()
        return datetime.combine(service_date, dt.time())

    if respect_dates and "Date" in rides.columns and rides["Date"].notna().any():
        try:
            dcol = pd.to_datetime(rides["Date"], errors="coerce").dt.date
            rides["Pickup Time"] = pd.to_datetime(dcol.astype(str) + " " + rides["Pickup Time"].dt.strftime("%H:%M"))
        except Exception:
            rides["Pickup Time"] = rides["Pickup Time"].apply(_anchor)
    else:
        rides["Pickup Time"] = rides["Pickup Time"].apply(_anchor)

    rides = rides.sort_values("Pickup Time").reset_index(drop=True)

    # outputs
    rides["Assigned Driver"] = ""
    rides["Car"] = ""
    rides["Notes"] = ""

    # Build roster
    roster, effective_windows = build_car_roster(driver_rows, active_cars, handover=car_handover)

    # Per-driver state for job chaining
    states = {}
    for _, r in driver_rows.iterrows():
        name = r["name"]
        car = roster.get(name, (r.get("preferred_car") or "").strip())
        if not car:
            # Shouldn't happen, but guard
            continue
        eff_start, eff_end = effective_windows[name]
        if eff_end <= eff_start:
            eff_end = eff_end + timedelta(days=1)
        states[name] = {
            "car": car,
            "eff_start": eff_start,
            "eff_end": eff_end,
            "last_row": None,
        }

    # Quick hint if far-away rides exist but FAR_CAR_ID isn't active
    has_far = any(is_far_away_row(r) for _, r in rides.iterrows())
    if has_far and FAR_CAR_ID not in active_cars:
        st.warning(f"Far-away bookings detected but car {FAR_CAR_ID} is not active. Those jobs will not be assigned.")

    # assign jobs
    driver_names = list(states.keys())
    d_rr = 0
    for i, row in rides.iterrows():
        pickup_dt = row["Pickup Time"]
        busy_td = busy_block_for_row(row)
        far_req = is_far_away_row(row)

        feasible = []
        hard_needed = {}
        for _ in range(len(driver_names)):
            name = driver_names[d_rr]
            d_rr = (d_rr + 1) % len(driver_names)

            car = states[name]["car"]
            # Enforce far-away->FAR_CAR_ID
            if far_req:
                if car != FAR_CAR_ID:
                    continue
                if FAR_CAR_ID not in active_cars:
                    continue  # cannot assign far job at all if 209 inactive

            # shift window with flex (use effective window from roster)
            st_dt = states[name]["eff_start"]
            en_dt = states[name]["eff_end"]
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

            # chain against last job (gap rules)
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
            if far_req:
                msg = f"No {FAR_CAR_ID} driver free (window/gap)"
            else:
                msg = "No driver free (window/gap)"
            rides.at[i, "Notes"] = msg
            continue

        chosen = feasible[0]
        chosen_hard = hard_needed[chosen]
        chosen_car = states[chosen]["car"]

        # lock assignment
        states[chosen]["last_row"] = row.copy()
        busy_until = pickup_dt + busy_td

        note = f"Busy until {busy_until.strftime('%H:%M')}"
        if chosen_hard:
            note += " · Hard Early Start"
        if far_req and chosen_car != FAR_CAR_ID:
            # Safety (shouldn't happen due to filters)
            note += f" · WARNING: far-away not on {FAR_CAR_ID}"
        rides.at[i, "Assigned Driver"] = chosen
        rides.at[i, "Car"] = chosen_car
        rides.at[i, "Notes"] = note

    return rides, roster, effective_windows

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

def _infer_service_date(df: pd.DataFrame):
    # If Date column exists, use its mode
    if "Date" in df.columns and df["Date"].notna().any():
        try:
            d = pd.to_datetime(df["Date"], errors="coerce").dt.date.dropna()
            if not d.empty:
                return d.mode().iloc[0]
        except Exception:
            pass
    # Else, if Pickup Time includes real dates, use mode
    if "Pickup Time" in df.columns:
        try:
            pt = pd.to_datetime(df["Pickup Time"], errors="coerce")
            dates = pt.dt.date.dropna()
            if not dates.empty:
                return dates.mode().iloc[0]
        except Exception:
            pass
    return datetime.now().date()

if rides_file:
    try:
        rides_df_live = pd.read_csv(rides_file).fillna("")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    inferred_date = _infer_service_date(rides_df_live)
    service_date = st.date_input("Service date", value=inferred_date)
    respect_dates = st.checkbox("Respect dates in file (keep them if present)", value=True)

    # Active cars
    default_active = [c for c in CAR_POOL if c.lower() != "s-klasse"]
    active_cars_live = st.multiselect("Active cars (one driver per car at a time; handovers allowed)", CAR_POOL, default=default_active)
    if not active_cars_live:
        st.warning("Activate at least one car.")
        st.stop()

    # Drivers for today + per-driver options
    st.subheader("Drivers for today")
    all_names = drivers_df["name"].tolist()
    selected_names_live = st.multiselect(
        "Select drivers (you can select more than active cars; roster will queue them on cars)",
        all_names,
        default=all_names[:min(len(all_names), 12)]
    )

    st.caption("Per-driver settings (leave times empty to assume all-day):")
    rows = []
    cols = st.columns(2)
    for i, name in enumerate(selected_names_live):
        with cols[i % 2]:
            st.markdown(f"**{name}**")
            start_t = st.time_input(f"{name} start", value=None, key=f"start_{name}")
            end_t   = st.time_input(f"{name} end", value=None, key=f"end_{name}")
            pref    = st.selectbox(f"{name} preferred car (optional)", [""] + active_cars_live, index=0, key=f"car_{name}")
            start_dt = datetime.combine(service_date, start_t) if start_t else None
            end_dt   = datetime.combine(service_date, end_t) if end_t else None
            rows.append({"name": name, "start": start_dt, "end": end_dt, "preferred_car": pref})
    driver_rows_live = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["name","start","end","preferred_car"])

    # Timing sliders
    with st.expander("Advanced timing (optional)"):
        hb_min   = st.slider("Car handover buffer between drivers (minutes)", 0, 240, 75, 5)
        soft_min = st.slider("Soft early start allowed (min)", 0, 180, 30, 5)
        hard_min = st.slider("Hard early start allowed (min)", 0, 240, 60, 5)
        late_min = st.slider("Late end allowed (min)", 0, 120, 20, 5)
        auto_relax = st.checkbox("If a job is unassigned for window only, auto-relax to SOFT", value=True)

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Generate"):
            st.session_state.snapshot = {
                "rides_df": rides_df_live.copy(),
                "service_date": service_date,
                "respect_dates": bool(respect_dates),
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
            plan_df, roster, eff_windows = assign_plan_with_rostered_cars(
                rides_df=s
