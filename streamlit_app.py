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

# Airport detection (NO "terminal", NO "kastrup")
# Match any 'lufthavn', plus specific CPH names
AIRPORT_KW = {
    "cph", "cph airport", "copenhagen airport",
    "københavns lufthavn", "lufthavn"
}
# CPH area keywords (ok to keep "kastrup" here as a district/city indicator)
CPH_KW = {"copenhagen", "københavn", "cph", "kastrup", "frederiksberg"}

FAR_DISTANCE_KM = 45.0
FAR_CAR_ID = "209"

# Column candidates (NOTE: "Drop Off" is first for destination)
PICKUP_CANDIDATES  = [
    "Pickup","Pick Up","Pick up","Pickup Address","From","Start",
    "Start Address","PickupLocation","Pickup Loc","Pick Up Address"
]
DROPOFF_CANDIDATES = [
    "Drop Off",  # exact column you pointed to
    "Dropoff","Drop off","Dropoff Address","To","End",
    "End Address","DropoffLocation","Dropoff Loc","Drop Off Address"
]

def _first_present(row: pd.Series, candidates) -> str:
    for c in candidates:
        if c in row and str(row[c]).strip():
            return str(row[c])
    return ""

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

def is_airport_place(text: str) -> bool:
    """
    Airport if:
      - contains 'lufthavn' anywhere, OR
      - matches any AIRPORT_KW token
    """
    s = _norm(text)
    if "lufthavn" in s:
        return True
    return any(k in s for k in AIRPORT_KW)

def is_copenhagen(a: str, b: str) -> bool:
    s1, s2 = _norm(a), _norm(b)
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

def get_pickup_text(row: pd.Series) -> str:
    return _first_present(row, PICKUP_CANDIDATES)

def get_dropoff_text(row: pd.Series) -> str:
    return _first_present(row, DROPOFF_CANDIDATES)

# -------- Route type (restricted to 3 types) --------
def booking_type(row: pd.Series) -> str:
    """
    Exactly three types:
      C2C = City -> City
      A2C = Airport -> City
      C2A = City -> Airport
    Any Airport->Airport (if it appears) is normalized to A2C for your ops.
    """
    pick = get_pickup_text(row)
    drop = get_dropoff_text(row)
    pick_air = is_airport_place(pick)
    drop_air = is_airport_place(drop)
    if pick_air and drop_air:
        return "A2C"  # normalize A->A into A2C bucket
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
    pickup_txt = get_pickup_text(row)
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
        extra = 30 if is_airport_place(pickup_txt) else 0
        return timedelta(minutes=trip_min + extra)

    return timedelta(minutes=trip_min)

# ---- Cancelled bookings helper (robust) ----
CANCEL_MATCHES = ("cancel", "cancelled", "canceled", "no show", "no-show", "noshow")
STATUS_EXACT = {
    "status", "booking status", "ride status", "trip status", "job status",
    "state", "is cancelled", "is canceled", "cancelled", "canceled"
}
TRUTHY = {"1", "true", "t", "yes", "y"}

def row_is_cancelled(row: pd.Series) -> bool:
    for col in row.index:
        name = str(col).strip().lower()
        val  = str(row[col]).strip().lower()
        is_status_col = (
            name in STATUS_EXACT
            or "cancel" in name
            or "no show" in name
            or "no-show" in name
            or "noshow" in name
        )
        if not is_status_col:
            continue
        if val in TRUTHY:
            return True
        if any(tok in val for tok in CANCEL_MATCHES):
            return True
    return False

def rule_gap_after(prev_row: pd.Series, next_row: pd.Series, gap_matrix: dict, cph_only: bool) -> timedelta:
    """
    Extra gap between bookings based on prev_type ➜ next_type (C2C, A2C, C2A).
    If cph_only=True, apply only when both pickups are in the CPH area.
    """
    if cph_only:
        prev_pick = get_pickup_text(prev_row)
        next_pick = get_pickup_text(next_row)
        if not (is_copenhagen(prev_pick, prev_pick) and is_copenhagen(next_pick, next_pick)):
            return timedelta(0)

    t_prev = booking_type(prev_row)
    t_next = booking_type(next_row)
    mins = int(gap_matrix.get(t_prev, {}).get(t_next, 0))
    return timedelta(minutes=mins)

def extract_busy_until(note: str):
    if not note: return None
    m = re.search(r"Busy until (\d{2}:\d{2})", note)
    if not m: return None
    return m.group(1)

def in_flex_window(pickup_dt: datetime, start_dt: datetime, end_dt: datetime,
                   soft_early: int, hard_early: int, late_allow: int):
    """Return 'soft'/'hard'/None if pickup fits shift+flex. Overnight handled."""
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
    Honors preferred car if possible; next driver's start may be pushed by handover.
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
            today = date.today()
            req_start = datetime.combine(today, datetime.strptime("00:00","%H:%M").time())
            req_end   = datetime.combine(today, datetime.strptime("23:59","%H:%M").time())
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
            earliest_start = max(req_start, car_free_at[car] + handover)
            if best_car is None or earliest_start < best_earliest_start:
                best_car = car
                best_earliest_start = earliest_start

        roster[name] = best_car
        eff_start = best_earliest_start
        windows[name] = (eff_start, req_end)
        car_free_at[best_car] = req_end

    return roster, windows

def build_bilplan(plan_df: pd.DataFrame, service_date: date) -> str:
    """Text export aggregating contiguous usage per (driver, car)."""
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
# Assignment (roster + far-away->209) + balancing + gaps
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
    gap_matrix: dict,
    gap_cph_only: bool,
    use_gap_only: bool
):
    """
    - Roster cars (one per driver for full shift).
    - Assign rides respecting shift windows, rostered car, far-away rules.
    - Chain with **type→type gap**; if use_gap_only=True, the GAP ALONE controls spacing (ignoring Trip Minutes).
    - Balance: fewest jobs, then smallest idle time.
    """
    rides = rides_df.copy()
    try:
        rides = rides[~rides.apply(row_is_cancelled, axis=1)].copy()
    except Exception:
        pass

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

    # Per-driver state
    states = {}
    for _, r in driver_rows.iterrows():
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
            "count": 0,
        }

    if not states:
        return rides, roster, effective_windows

    # Hint if far-away rides exist but FAR_CAR_ID isn't active
    has_far = any(is_far_away_row(r) for _, r in rides.iterrows())
    if has_far and FAR_CAR_ID not in active_cars:
        st.warning(f"Far-away bookings detected but car {FAR_CAR_ID} is not active. Those jobs will not be assigned.")

    # assign jobs
    driver_names = list(states.keys())
    d_rr = 0  # rotating start index
    for i, row in rides.iterrows():
        pickup_dt = row["Pickup Time"]
        busy_td = busy_block_for_row(row)
        far_req = is_far_away_row(row)

        feasible = []
        hard_needed = {}
        available_at_map = {}
        applied_gap_map = {}
        prev_type_map = {}

        # rotate start for fairness
        for _ in range(len(driver_names)):
            name = driver_names[d_rr]
            d_rr = (d_rr + 1) % len(driver_names)

            car = states[name]["car"]

            # Enforce far-away->FAR_CAR_ID
            if far_req:
                if car != FAR_CAR_ID:
                    continue
                if FAR_CAR_ID not in active_cars:
                    continue

            # shift window with flex
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

            # chain against last job (type-to-type gap)
            last_row = states[name]["last_row"]
            if last_row is not None:
                extra_gap = rule_gap_after(last_row, row, gap_matrix, gap_cph_only)
                prev_pickup = pd.to_datetime(last_row["Pickup Time"])
                if use_gap_only:
                    earliest_next = prev_pickup + extra_gap
                else:
                    prev_busy = busy_block_for_row(last_row)
                    earliest_next = prev_pickup + prev_busy + extra_gap
                if pickup_dt < earliest_next:
                    continue
                available_at_map[name] = earliest_next
                applied_gap_map[name] = int(extra_gap.total_seconds() // 60)
                prev_type_map[name] = booking_type(last_row)
            else:
                available_at_map[name] = st_dt
                applied_gap_map[name] = 0
                prev_type_map[name] = None

            feasible.append(name)
            hard_needed[name] = (flex == "hard")

        if not feasible:
            rides.at[i, "Notes"] = ("No " + FAR_CAR_ID + " driver free (window/gap)") if far_req else "No driver free (window/gap)"
            continue

        # Balanced selection: fewest jobs, then smallest idle time
        def _key(nm):
            idle = (pickup_dt - available_at_map[nm]).total_seconds()
            if idle < 0:
                idle = float('inf')
            return (states[nm]["count"], idle)

        chosen = min(feasible, key=_key)
        chosen_hard = hard_needed[chosen]
        chosen_car = states[chosen]["car"]

        # lock assignment
        last_row_prev = states[chosen]["last_row"]
        applied_gap_min = applied_gap_map.get(chosen, 0)
        prev_type = prev_type_map.get(chosen, None)
        this_type = booking_type(row)

        states[chosen]["last_row"] = row.copy()
        states[chosen]["count"] += 1

        # Note formatting
        busy_until = pickup_dt + busy_td
        note_parts = [f"Busy until {busy_until.strftime('%H:%M')}"]
        if chosen_hard:
            note_parts.append("Hard Early Start")
        if applied_gap_min and prev_type:
            note_parts.append(f"Gap {prev_type}➜{this_type} {applied_gap_min}m")
        if far_req and chosen_car != FAR_CAR_ID:
            note_parts.append(f"WARNING: far-away not on {FAR_CAR_ID}")
        note = " · ".join(note_parts)

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
        use_gap_only = st.checkbox("Use gaps only for chaining (ignore Trip Minutes)", value=True)

    # 3×3 TYPE➜TYPE gap matrix (C2C, A2C, C2A)
    with st.expander("Gap rules between booking TYPES (prev ➜ next)"):
        st.caption("Types: C2C (City→City), A2C (Airport→City), C2A (City→Airport)")
        gap_cph_only = st.checkbox("Apply gaps only when both pickups are in the CPH area", value=False)

        types = ["C2C","A2C","C2A"]
        gap_matrix = {}
        for r in types:
            gap_matrix[r] = {}
            row_cols = st.columns(len(types))
            for j, c in enumerate(types):
                gap_matrix[r][c] = row_cols[j].number_input(f"{r} ➜ {c}", 0, 240, 0, 5, key=f"gap_{r}_{c}")

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
                "gap_cph_only": bool(gap_cph_only),
                "use_gap_only": bool(use_gap_only),
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
                gap_matrix=s["gap_matrix"],
                gap_cph_only=s["gap_cph_only"],
                use_gap_only=s["use_gap_only"],
            )
        except Exception as e:
            st.error(f"Planning failed: {e}")
            st.stop()

        # Normalize columns for display/exports (include exact "Drop Off")
        ensure_column(plan_df, "Pickup", PICKUP_CANDIDATES)
        ensure_column(plan_df, "Dropoff", DROPOFF_CANDIDATES)  # ensures "Drop Off" is used if present
        ensure_column(plan_df, "Name", ["Pax Name","Passenger Name","PAX","Pax","Client Name","Customer Name"])

        # Optional debug: show detected route type
        show_debug_type = st.checkbox("Show detected Type column (C2C/A2C/C2A)", value=False)
        if show_debug_type:
            plan_df["Type"] = plan_df.apply(booking_type, axis=1)

        plan_df.rename(columns={"Assigned Driver":"Allocated Driver", "Car":"Allocated Car"}, inplace=True)

        for c in ["Pickup Time","Pickup","Dropoff","Allocated Driver","Allocated Car","Notes","Ref No","Ride ID","Booking ID","Name"]:
            if c not in plan_df.columns:
                plan_df[c] = ""

        plan_df["Booking ID"] = plan_df.apply(get_booking_id, axis=1)

        display_cols = [
            "Pickup Time",
            "Booking ID",
            "Name",
            "Pickup",
            "Dropoff",
            "Allocated Driver",
            "Allocated Car",
            "Notes",
        ]
        if show_debug_type and "Type" in plan_df.columns:
            display_cols.insert(5, "Type")  # show Type before driver columns

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

        # Roster summary
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
            roster_rows.append({"Driver": name, "Car": car or "", "From": _fmt_dt(st_dt), "To": _fmt_dt(en_dt)})

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
