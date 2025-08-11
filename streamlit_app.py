import os
import re
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, timedelta, date, time

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

# Airport detection: DO NOT use "terminal" or "kastrup" as airport keywords.
# Any text containing "lufthavn" counts; plus explicit CPH names.
AIRPORT_KW = {"cph", "cph airport", "copenhagen airport", "københavns lufthavn", "lufthavn"}
CPH_KW = {"copenhagen", "københavn", "cph", "kastrup", "frederiksberg"}  # "kastrup" kept as city indicator (not airport)

FAR_DISTANCE_KM = 45.0
FAR_CAR_ID = "209"

# =========================
# Data access
# =========================
def load_drivers():
    """drivers table with columns: driver_id, name, number"""
    with engine.connect() as conn:
        df = pd.read_sql("SELECT driver_id AS id, name, number FROM drivers ORDER BY name", conn)
    return df

# =========================
# Helpers
# =========================
def _norm(s: str) -> str:
    return (s or "").strip().lower()

def is_airport_pickup(pickup_text: str) -> bool:
    s = _norm(pickup_text)
    # airport if it contains 'lufthavn' or matches a known airport token
    if "lufthavn" in s:
        return True
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

def parse_float_maybe(value):
    try:
        if value is None: return None
        s = str(value).strip()
        if s == "": return None
        return float(s.replace(",", "."))
    except:
        return None

# --- distance detection (scan common headers) ---
DISTANCE_CANDIDATES = [
    "Distance_km","Distance (km)","KM","Km","km","distance","distance km","Total Km","Total_km"
]
def get_distance_km(row: pd.Series):
    for k in DISTANCE_CANDIDATES:
        if k in row:
            v = parse_float_maybe(row[k])
            if v is not None:
                return v
    return None

def is_far_away_row(row: pd.Series) -> bool:
    dk = get_distance_km(row)
    return (dk is not None) and (dk >= FAR_DISTANCE_KM)

# ---------- Date / time parsing (EU-first, robust) ----------
_DATE_RE = re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")  # detects a DD/MM or MM/DD like token

def _parse_date_eu_first(val) -> date | None:
    """Parse a 'Date' cell to a date using dayfirst=True."""
    try:
        d = pd.to_datetime(val, errors="coerce", dayfirst=True)
        return d.date() if pd.notna(d) else None
    except Exception:
        return None

def _parse_time_only(val) -> time | None:
    """Parse 'HH:MM' or 'H:MM' or 'HHMM' as time (no date)."""
    s = str(val).strip()
    if not s:
        return None
    # accept 4-digit like 0915
    if re.fullmatch(r"\d{3,4}", s):
        s = s.zfill(4)
        s = f"{s[:2]}:{s[2:]}"
    for fmt in ("%H:%M", "%H.%M", "%H%M"):
        try:
            return datetime.strptime(s, fmt).time()
        except Exception:
            continue
    # fallback to pandas (may attach today; we only use .time())
    try:
        t = pd.to_datetime(s, errors="coerce", format="%H:%M")
        if pd.notna(t):
            return t.time()
    except Exception:
        pass
    return None

def _parse_dt_maybe(val, service_date: date) -> datetime | None:
    """
    Parse a Pickup Time cell:
      - If it contains a date -> to_datetime(dayfirst=True)
      - If it's only a time   -> combine with service_date
    """
    s = str(val).strip()
    if not s:
        return None
    if _DATE_RE.search(s):
        # has date: respect EU day-first
        d = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if pd.isna(d):
            return None
        return d.to_pydatetime()
    # time-only -> anchor to service_date
    t = _parse_time_only(s)
    if t is None:
        return None
    return datetime.combine(service_date, t)

# -------- Route type (restricted to 3 types) --------
def booking_type(row: pd.Series) -> str:
    """
    Exactly three types:
      C2C = City -> City
      A2C = Airport -> City
      C2A = City -> Airport
    Airport->Airport (if ever) is normalized to A2C.
    """
    pick = str(row.get("Pickup", "") or "")
    drop = str(row.get("Dropoff", "") or "")
    pick_air = is_airport_pickup(pick)
    drop_air = is_airport_pickup(drop)
    if pick_air and drop_air:
        return "A2C"
    if pick_air and not drop_air:
        return "A2C"
    if (not pick_air) and drop_air:
        return "C2A"
    return "C2C"

def busy_block_for_row(row: pd.Series) -> timedelta:
    """
    Busy time for the ride (duration driver is occupied from pickup):
      - Roadshow / Site Inspection: Duration Minutes/Hours if given; else Trip + 60.
      - Far-away: km >= FAR_DISTANCE_KM => Trip + (30 if pickup is airport else 0).
      - Else: Trip (or 30 fallback).
    """
    trip_min = parse_minutes(row.get("Trip Minutes")) or 30

    distance_km = get_distance_km(row)
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

def rule_gap_after(prev_row: pd.Series, next_row: pd.Series, gaps: dict) -> timedelta:
    """
    Extra gap *between* bookings on top of the previous job's busy time.
    We apply this only when both bookings are in greater CPH (by pickup).
    gaps keys: 'C2C','A2C','C2A' as prev, each maps to dict of next {'C2C':m, 'A2C':m, 'C2A':m}
    """
    prev_pick = str(prev_row.get("Pickup", "") or "")
    next_pick = str(next_row.get("Pickup", "") or "")
    if not is_copenhagen(prev_pick, prev_pick) or not is_copenhagen(next_pick, next_pick):
        return timedelta(0)

    prev_t = booking_type(prev_row)
    next_t = booking_type(next_row)
    mins = int(gaps.get(prev_t, {}).get(next_t, 0))
    return timedelta(minutes=mins)

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

# ---- Cancelled bookings helper ----
CANCEL_MATCHES = ("cancel", "cancelled", "canceled", "no show", "no-show", "noshow")
def row_is_cancelled(row: pd.Series) -> bool:
    for col in row.index:
        col_l = str(col).lower()
        if col_l in ("status", "booking status", "state", "cancelled", "canceled"):
            val = str(row[col]).strip().lower()
            if val in ("1", "true", "yes", "y"):
                return True
            if any(tok in val for tok in CANCEL_MATCHES):
                return True
    return False

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
            # treat as all-day if missing (anchor to today purely for ordering; allocator uses real service_date later)
            today = date.today()
            req_start = datetime.combine(today, time(0,0))
            req_end   = datetime.combine(today, time(23,59))
        if req_end <= req_start:
            req_end = req_end + timedelta(days=1)

        pref = (r.get("preferred_car") or "").strip()

        # Candidate order: preferred first (if any), otherwise earliest-free car
        if pref and pref in active_cars:
            candidates = [pref] + [c for c in active_cars if c != pref]
        else:
            candidates = list(active_cars)

        # pick the car that minimizes delay from requested start
        best_car = None
        best_earliest_start = None
        for car in candidates:
            earliest_start = max(req_start, car_free_at[car] + handover)  # must respect handover before next driver
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

    tmp = pd.DataFrame(rows, columns=["name","car","start","end"])
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
    respect_dates: bool,
    gap_matrix: dict
):
    """
    - Build a car roster (one car per driver for whole shift; shifts may be pushed by car availability + handover)
    - Assign rides respecting:
        * driver effective shift window (with soft/hard early & late allow)
        * driver's rostered car only
        * far-away rides require FAR_CAR_ID (e.g., 209)
        * ride busy time + CPH airport/city **custom gap** rules (between jobs for the same driver)
    """
    # Safety: drop cancelled / no-show rows if the caller forgot to filter
    rides = rides_df.copy()
    try:
        rides = rides[~rides.apply(row_is_cancelled, axis=1)].copy()
    except Exception:
        pass

    # ---------- Parse / anchor pickup times (EU day-first safe) ----------
    if "Pickup Time" not in rides.columns:
        raise ValueError("CSV must include 'Pickup Time' column.")

    # If a Date column exists and we respect it, parse it day-first and combine
    if respect_dates and "Date" in rides.columns and rides["Date"].astype(str).str.strip().ne("").any():
        date_parsed = rides["Date"].apply(_parse_date_eu_first)
        # parse pickup both ways: if it already has a date, honor it; else use time-only with parsed Date
        def _combine_row(row):
            d = row["_parsed_date"]
            raw_pt = row["Pickup Time"]
            if d is None:
                # fall back to parse Pickup Time with dayfirst, or as time+service_date
                return _parse_dt_maybe(raw_pt, service_date)
            # parse time-only from Pickup Time
            t = _parse_time_only(raw_pt)
            if t is not None:
                return datetime.combine(d, t)
            # if Pickup Time itself has a date, parse day-first
            return _parse_dt_maybe(raw_pt, service_date)

        rides["_parsed_date"] = date_parsed
        rides["Pickup Time"] = rides.apply(_combine_row, axis=1)
        rides.drop(columns=["_parsed_date"], inplace=True)
    else:
        # No Date col respected -> parse each Pickup Time; if it's time-only, anchor to service_date
        rides["Pickup Time"] = rides["Pickup Time"].apply(lambda v: _parse_dt_maybe(v, service_date))

    if rides["Pickup Time"].isna().any():
        bad = rides[runs := rides["Pickup Time"].isna()]
        raise ValueError("Some 'Pickup Time' values are invalid or unparsable with EU day-first/time-only logic.")

    rides = rides.sort_values("Pickup Time").reset_index(drop=True)

    # outputs
    rides["Assigned Driver"] = ""
    rides["Car"] = ""
    rides["Notes"] = ""

    # Build roster
    roster, effective_windows = build_car_roster(driver_rows, active_cars, handover=car_handover)

    # Per-driver state for job chaining
    states = {}
    ordered_names = []
    for _, r in driver_rows.iterrows():  # respect UI order
        name = r["name"]
        car = roster.get(name, (r.get("preferred_car") or "").strip())
        if not car:
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
        ordered_names.append(name)

    if not states:
        return rides, roster, effective_windows

    # Far-away hint
    has_far = any(is_far_away_row(r) for _, r in rides.iterrows())
    if has_far and FAR_CAR_ID not in active_cars:
        st.warning(f"Far-away bookings detected but car {FAR_CAR_ID} is not active. Those jobs will not be assigned.")

    # assign jobs — strict shift window check
    driver_names = ordered_names[:]  # stable order from UI
    d_rr = 0  # round-robin cursor
    for i, row in rides.iterrows():
        pickup_dt = row["Pickup Time"]
        busy_td = busy_block_for_row(row)
        far_req = is_far_away_row(row)

        feasible = []
        hard_needed = {}
        scanned = 0
        # scan all drivers starting at cursor
        while scanned < len(driver_names):
            name = driver_names[d_rr]
            d_rr = (d_rr + 1) % len(driver_names)
            scanned += 1

            # Driver must exist in roster
            if name not in states:
                continue
            car = states[name]["car"]

            # Enforce far-away->FAR_CAR_ID
            if far_req:
                if car != FAR_CAR_ID:
                    continue
                if FAR_CAR_ID not in active_cars:
                    continue

            # strict window with flex
            st_dt = states[name]["eff_start"]
            en_dt = states[name]["eff_end"]
            flex = in_flex_window(pickup_dt, st_dt, en_dt, soft_early, hard_early, late_allow)
            if flex is None:
                # optional relax to soft window, but still must fit
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
                extra_gap = rule_gap_after(last_row, row, gap_matrix)
                earliest_next = prev_pickup + prev_busy + extra_gap
                if pickup_dt < earliest_next:
                    continue

            feasible.append(name)
            hard_needed[name] = (flex == "hard")

        if not feasible:
            rides.at[i, "Notes"] = ("No " + FAR_CAR_ID + " driver free (window/gap)") if far_req else "No driver free (window/gap)"
            continue

        # choose the first feasible in this round-robin pass (respects shifts)
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
            note += f" · WARNING: far-away not on {FAR_CAR_ID}"
        rides.at[i, "Assigned Driver"] = chosen
        rides.at[i, "Car"] = chosen_car
        rides.at[i, "Notes"] = note

    return rides, roster, effective_windows

# =========================
# Booking ID helper (use Ref No)
# =========================
ID_CANDIDATES = ("Ref No","Ride ID","Booking ID","BookingID","ID","Id","id")
def get_booking_id(row: pd.Series) -> str:
    for k in ID_CANDIDATES:
        if k in row and str(row[k]).strip() != "":
            return str(row[k]).strip()
    return ""

# normalize/mirror columns to the names we use in the UI
def ensure_column(df: pd.DataFrame, target: str, cands: list):
    if target in df.columns:
        return
    for c in cands:
        if c in df.columns:
            df[target] = df[c]
            return
    df[target] = ""  # fallback

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
    # Prefer explicit Date column (EU day-first)
    if "Date" in df.columns and df["Date"].astype(str).str.strip().ne("").any():
        try:
            d = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True).dt.date.dropna()
            if not d.empty:
                return d.mode().iloc[0]
        except Exception:
            pass
    # Else, peek into Pickup Time trying to extract a date (EU day-first)
    if "Pickup Time" in df.columns:
        s = df["Pickup Time"].astype(str)
        mask_has_date = s.str.contains(_DATE_RE)
        if mask_has_date.any():
            try:
                d = pd.to_datetime(s[mask_has_date], errors="coerce", dayfirst=True).dt.date.dropna()
                if not d.empty:
                    return d.mode().iloc[0]
            except Exception:
                pass
    return datetime.now().date()

if rides_file:
    try:
        rides_df_live = pd.read_csv(rides_file).fillna("")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # Drop cancelled / no-show bookings early
    _pre = len(rides_df_live)
    try:
        rides_df_live = rides_df_live[~rides_df_live.apply(row_is_cancelled, axis=1)].copy()
    except Exception:
        pass
    _post = len(rides_df_live)
    if _pre != _post:
        st.info(f"Filtered {_pre - _post} cancelled/no-show booking(s).")

    inferred_date = _infer_service_date(rides_df_live)
    service_date = st.date_input("Service date", value=inferred_date)
    respect_dates = st.checkbox("Respect dates in file (keep them if present)", value=True)

    # Active cars
    default_active = [c for c in CAR_POOL if c.lower() != "s-klasse"]
    active_cars_live = st.multiselect(
        "Active cars (one driver per car at a time; handovers allowed)",
        CAR_POOL,
        default=default_active
    )
    if not active_cars_live:
        st.warning("Activate at least one car.")
        st.stop()

    # Drivers for today + per-driver options (with per-driver dates)
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
            # Per-driver dates default to service_date but can be changed
            start_date = st.date_input(f"{name} start date", value=service_date, key=f"sd_{name}")
            start_time = st.time_input(f"{name} start time", value=None, key=f"st_{name}")
            end_date   = st.date_input(f"{name} end date", value=service_date, key=f"ed_{name}")
            end_time   = st.time_input(f"{name} end time", value=None, key=f"et_{name}")
            pref       = st.selectbox(f"{name} preferred car (optional)", [""] + active_cars_live, index=0, key=f"car_{name}")

            start_dt = datetime.combine(start_date, start_time) if start_time else None
            end_dt   = datetime.combine(end_date, end_time) if end_time else None
            rows.append({"name": name, "start": start_dt, "end": end_dt, "preferred_car": pref})
    driver_rows_live = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["name","start","end","preferred_car"])

    # Timing sliders
    with st.expander("Advanced timing (optional)"):
        hb_min   = st.slider("Car handover buffer between drivers (minutes)", 0, 240, 75, 5)
        soft_min = st.slider("Soft early start allowed (min)", 0, 180, 30, 5)
        hard_min = st.slider("Hard early start allowed (min)", 0, 240, 60, 5)
        late_min = st.slider("Late end allowed (min)", 0, 120, 20, 5)
        auto_relax = st.checkbox("If a job is unassigned for window only, auto-relax to SOFT", value=True)

    # TYPE➜TYPE gap matrix (C2C, A2C, C2A)
    with st.expander("CPH gap rules between bookings (minutes)"):
        st.caption("Types: C2C (City→City), A2C (Airport→City), C2A (City→Airport). Applied only when both pickups are in CPH.")
        c_cols = st.columns(3)
        types = ["C2C","A2C","C2A"]
        gap_matrix = {}
        default_gaps = {
            "C2C": {"C2C": 25, "A2C": 30, "C2A": 25},
            "A2C": {"C2C": 50, "A2C": 50, "C2A": 50},
            "C2A": {"C2C": 50, "A2C": 0,  "C2A": 40},
        }
        for r in types:
            gap_matrix[r] = {}
            cols_row = st.columns(3)
            for j, c in enumerate(types):
                gap_matrix[r][c] = cols_row[j].number_input(f"{r} ➜ {c}", 0, 240, default_gaps[r][c], 5, key=f"gap_{r}_{c}")

    col_run, col_clear = st.columns(2)
    with col_run:
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
                "gap_matrix": gap_matrix,
            }
            st.session_state.run_plan = True
    with col_clear:
        if st.button("Clear"):
            st.session_state.run_plan = False
            st.session_state.snapshot = {}

    if st.session_state.run_plan:
        s = st.session_state.snapshot
        try:
            plan_df, roster, eff_windows = assign_plan_with_rostered_cars(
                rides_df=s["rides_df"],
                driver_rows=s["driver_rows"],
                active_cars=s["active_cars"],
                service_date=s["service_date"],
                car_handover=s["car_handover"],
                soft_early=s["soft_early"],
                hard_early=s["hard_early"],
                late_allow=s["late_allow"],
                auto_relax_to_soft=s["auto_relax"],
                respect_dates=s["respect_dates"],
                gap_matrix=s["gap_matrix"]
            )
        except Exception as e:
            st.error(f"Planning failed: {e}")
            st.stop()

        # Map columns EXACTLY as you asked:
        # Pickup := "Pick Up"
        # Dropoff := prefer exact "Drop Off"
        ensure_column(plan_df, "Pickup", ["Pick Up","Pick up","Pickup","Pickup Address","From","Start","Start Address","PickupLocation","Pickup Loc"])
        ensure_column(plan_df, "Dropoff", ["Drop Off","Drop off","Dropoff","Dropoff Address","To","End","End Address","DropoffLocation","Dropoff Loc"])

        # Name := "Pax Name"
        ensure_column(plan_df, "Name", ["Pax Name","Passenger Name","PAX","Pax","Client Name","Customer Name"])

        # Pretty table (whole day) with Booking ID from Ref No, Name, Pickup & Dropoff
        plan_df.rename(columns={"Assigned Driver":"Allocated Driver", "Car":"Allocated Car"}, inplace=True)

        # Ensure columns used below exist
        for c in ["Pickup Time","Pickup","Dropoff","Allocated Driver","Allocated Car","Notes","Ref No","Ride ID","Booking ID","Name"]:
            if c not in plan_df.columns:
                plan_df[c] = ""

        # Derive Booking ID from Ref No (fallbacks included)
        plan_df["Booking ID"] = plan_df.apply(get_booking_id, axis=1)

        display_cols = [
            "Pickup Time",
            "Booking ID",   # from Ref No
            "Name",         # from Pax Name
            "Pickup",
            "Dropoff",
            "Allocated Driver",
            "Allocated Car",
            "Notes",
        ]
        display_cols = [c for c in display_cols if c in plan_df.columns]

        plan_df_display = plan_df[display_cols].copy().sort_values("Pickup Time").reset_index(drop=True)

        st.subheader("Full Day Plan")
        st.dataframe(plan_df_display, use_container_width=True, height=900)

        st.download_button(
            "Download Plan CSV",
            data=plan_df_display.to_csv(index=False).encode("utf-8"),
            file_name="daily_car_plan.csv",
            mime="text/csv"
        )

        # Show roster summary (safe formatting)
        st.subheader("Car Roster (one car per driver)")

        def _fmt_dt(dt):
            try:
                if isinstance(dt, pd.Timestamp):
                    dt = dt.to_pydatetime()
                return dt.strftime("%Y-%m-%d %H:%M") if isinstance(dt, datetime) else ""
            except Exception:
                return ""

        roster_rows = []
        for name, car in roster.items():
            st_dt, en_dt = eff_windows.get(name, (None, None))
            roster_rows.append({
                "Driver": name,
                "Car": car or "",
                "From": _fmt_dt(st_dt),
                "To": _fmt_dt(en_dt),
            })

        roster_df = pd.DataFrame(roster_rows)
        if not roster_df.empty:
            roster_df["Car"] = roster_df["Car"].fillna("")
            roster_df["From"] = roster_df["From"].fillna("")
            roster_df = roster_df.sort_values(["Car","From"])
        st.dataframe(roster_df, use_container_width=True)

        # BILPLAN export
        bilplan_text = build_bilplan(plan_df, s["service_date"])
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
