import os
import re
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
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
# Schema (save drivers' working times per date + named profiles)
# =========================
def ensure_schema():
    ddl1 = """
    CREATE TABLE IF NOT EXISTS driver_shifts (
        service_date DATE NOT NULL,
        driver_name  TEXT NOT NULL,
        start_ts     TIMESTAMP NULL,
        end_ts       TIMESTAMP NULL,
        preferred_car TEXT NULL,
        PRIMARY KEY (service_date, driver_name)
    );
    """
    ddl2 = """
    CREATE TABLE IF NOT EXISTS shift_profiles (
        profile_name TEXT PRIMARY KEY,
        created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    ddl3 = """
    CREATE TABLE IF NOT EXISTS shift_profile_entries (
        profile_name TEXT NOT NULL,
        driver_name  TEXT NOT NULL,
        start_hhmm   TEXT NULL,
        end_hhmm     TEXT NULL,
        preferred_car TEXT NULL,
        PRIMARY KEY (profile_name, driver_name),
        FOREIGN KEY (profile_name) REFERENCES shift_profiles(profile_name) ON DELETE CASCADE
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl1))
        conn.execute(text(ddl2))
        conn.execute(text(ddl3))

def _dialect():
    return engine.dialect.name

def upsert_shift_row(service_date: date, name: str, start_ts, end_ts, preferred_car: str):
    if _dialect() == "postgresql":
        sql = text("""
            INSERT INTO driver_shifts (service_date, driver_name, start_ts, end_ts, preferred_car)
            VALUES (:d, :n, :s, :e, :p)
            ON CONFLICT (service_date, driver_name)
            DO UPDATE SET start_ts = EXCLUDED.start_ts,
                          end_ts = EXCLUDED.end_ts,
                          preferred_car = EXCLUDED.preferred_car;
        """)
    else:
        sql = text("""
            INSERT INTO driver_shifts (service_date, driver_name, start_ts, end_ts, preferred_car)
            VALUES (:d, :n, :s, :e, :p)
            ON CONFLICT(service_date, driver_name) DO UPDATE SET
                start_ts = excluded.start_ts,
                end_ts = excluded.end_ts,
                preferred_car = excluded.preferred_car;
        """)
    with engine.begin() as conn:
        conn.execute(sql, {"d": service_date, "n": name, "s": start_ts, "e": end_ts, "p": preferred_car})

def load_shifts_for_date(service_date: date) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT driver_name AS name, start_ts AS start, end_ts AS end, preferred_car "
                 "FROM driver_shifts WHERE service_date = :d ORDER BY driver_name"),
            conn,
            params={"d": service_date}
        )
    return df

# ---- Profiles (times only, no dates) ----
def list_profiles() -> list:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT profile_name FROM shift_profiles ORDER BY profile_name")).fetchall()
    return [r[0] for r in rows]

def save_profile(profile_name: str, driver_rows: list):
    """Save/replace a profile from driver_rows (start/end taken as HH:MM strings)."""
    if not profile_name:
        raise ValueError("Profile name is required.")
    entries = []
    for r in driver_rows:
        nm = r["name"]
        st_ts = r.get("start")
        en_ts = r.get("end")
        pc = (r.get("preferred_car") or "")
        st_hhmm = datetime.strftime(st_ts, "%H:%M") if isinstance(st_ts, datetime) else None
        en_hhmm = datetime.strftime(en_ts, "%H:%M") if isinstance(en_ts, datetime) else None
        entries.append((nm, st_hhmm, en_hhmm, pc))

    with engine.begin() as conn:
        if _dialect() == "postgresql":
            conn.execute(text("""
                INSERT INTO shift_profiles (profile_name) VALUES (:p)
                ON CONFLICT (profile_name) DO NOTHING;
            """), {"p": profile_name})
        else:
            conn.execute(text("""
                INSERT OR IGNORE INTO shift_profiles (profile_name) VALUES (:p);
            """), {"p": profile_name})

        conn.execute(text("DELETE FROM shift_profile_entries WHERE profile_name = :p"), {"p": profile_name})

        ins = text("""
            INSERT INTO shift_profile_entries (profile_name, driver_name, start_hhmm, end_hhmm, preferred_car)
            VALUES (:p, :n, :s, :e, :c)
        """)
        for nm, s, e, pc in entries:
            conn.execute(ins, {"p": profile_name, "n": nm, "s": s, "e": e, "c": pc})

def load_profile(profile_name: str, service_date: date) -> pd.DataFrame:
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT driver_name, start_hhmm, end_hhmm, preferred_car
            FROM shift_profile_entries
            WHERE profile_name = :p
            ORDER BY driver_name
        """), {"p": profile_name}).fetchall()
    if not rows:
        return pd.DataFrame(columns=["name","start","end","preferred_car"])
    recs = []
    for driver_name, start_hhmm, end_hhmm, pc in rows:
        st_dt = datetime.combine(service_date, datetime.strptime(start_hhmm, "%H:%M").time()) if start_hhmm else None
        en_dt = datetime.combine(service_date, datetime.strptime(end_hhmm, "%H:%M").time()) if end_hhmm else None
        recs.append({"name": driver_name, "start": st_dt, "end": en_dt, "preferred_car": pc or ""})
    return pd.DataFrame(recs)

ensure_schema()

# =========================
# Config & keywords
# =========================
CAR_POOL = [
    "209","768","251","179","180","874","875","091","281","280",
    "S-klasse","525","979","516","225"
]

# Airport detection (NO "terminal", NO airport "kastrup")
AIRPORT_KW = {
    "cph", "cph airport", "copenhagen airport",
    "københavns lufthavn", "lufthavn"
}
# CPH area keywords (keep "kastrup" as district indicator)
CPH_KW = {"copenhagen", "københavn", "cph", "kastrup", "frederiksberg"}

FAR_DISTANCE_KM = 45.0
FAR_CAR_ID = "209"

# Default type→type gap matrix (minutes)
DEFAULT_GAP_MATRIX = {
    "C2C": {"C2C": 25, "A2C": 30, "C2A": 25},
    "A2C": {"C2C": 50, "A2C": 50, "C2A": 50},
    "C2A": {"C2C": 50, "A2C": 0,  "C2A": 40},
}

# Column candidates (NOTE: "Drop Off" first)
PICKUP_CANDIDATES  = [
    "Pickup","Pick Up","Pick up","Pickup Address","From","Start",
    "Start Address","PickupLocation","Pickup Loc","Pick Up Address"
]
DROPOFF_CANDIDATES = [
    "Drop Off",
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

# --- distance detection ---
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
    pick = get_pickup_text(row)
    drop = get_dropoff_text(row)
    pick_air = is_airport_place(pick)
    drop_air = is_airport_place(drop)
    if pick_air and drop_air:
        return "A2C"
    if pick_air and not drop_air:
        return "A2C"
    if (not pick_air) and drop_air:
        return "C2A"
    return "C2C"

def busy_block_for_row(row: pd.Series) -> timedelta:
    trip_min = parse_minutes(row.get("Trip Minutes")) or 30
    distance_km = get_distance_km(row)
    pickup_txt = get_pickup_text(row)
    cust_txt = (str(row.get("Customer Name", "") or "") + " " + str(row.get("Type", "") or "")).lower()

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
# Car roster (one car per driver per shift) WITH early-finish flex
# =========================
def build_car_roster(driver_rows: pd.DataFrame, active_cars: list, handover: timedelta, early_end_flex: timedelta):
    """
    Assign exactly ONE car per driver for the entire shift.
    Honors preferred car if possible; sequences drivers per car with handover at start.
    NEW: previous driver's end is treated as (end - early_end_flex) for the next driver's earliest start on that car.
    NOTE: This only affects car-to-driver mapping and effective start; real car overlap is prevented by per-booking lock.
    """
    car_prev_end = {car: datetime.min for car in active_cars}  # last scheduled END on that car (by shifts)
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

        # Candidate order: preferred first (if any), otherwise earliest "available"
        if pref and pref in active_cars:
            candidates = [pref] + [c for c in active_cars if c != pref]
        else:
            candidates = list(active_cars)

        best_car = None
        best_earliest_start = None
        for car in candidates:
            prev_end = car_prev_end[car]
            # allow early end flex for the *previous* driver on this car
            if prev_end == datetime.min:
                avail = req_start  # no prior driver on this car
            else:
                avail = (prev_end - early_end_flex) + handover
            earliest_start = max(req_start, avail)
            if best_car is None or earliest_start < best_earliest_start:
                best_car = car
                best_earliest_start = earliest_start

        roster[name] = best_car
        eff_start = best_earliest_start
        windows[name] = (eff_start, req_end)
        # update this car's scheduled end to this driver's requested end
        car_prev_end[best_car] = req_end

    return roster, windows

def build_bilplan(plan_df: pd.DataFrame, service_date: date) -> str:
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
    tmp = pd.DataFrame(rows)
    agg = tmp.groupby(["name","car"]).agg(start=("start","min"), end=("end","max")).reset_index()
    agg = agg.sort_values(["car","start"]).reset_index(drop=True)
    lines = [f"BILPLAN {service_date.strftime('%d/%m/%y')}"]
    for _, r in agg.iterrows():
        span = f"{r['start'].strftime('%H:%M')}-{r['end'].strftime('%H:%M')}"
        lines.append(f"{r['name']} - {r['car']} fra {span}")
    return "\n".join(lines)

# =========================
# Assignment (round-robin + fairness + per-booking vehicle lock)
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
    use_gap_only: bool,
    roster_early_end_flex: timedelta
):
    # 1) Preprocess rides
    rides = rides_df.copy()
    try:
        rides = rides[~rides.apply(row_is_cancelled, axis=1)].copy()
    except Exception:
        pass

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

    # 2) Outputs
    rides["Assigned Driver"] = ""
    rides["Car"] = ""
    rides["Notes"] = ""

    # 3) Build roster + driver states (WITH early-finish flex)
    roster, effective_windows = build_car_roster(driver_rows, active_cars, handover=car_handover, early_end_flex=roster_early_end_flex)

    states = {}
    base_order = []
    for _, r in driver_rows.iterrows():  # preserve UI order
        name = r["name"]
        car = roster.get(name, (r.get("preferred_car") or "").strip())
        if not car:
            continue
        st_dt, en_dt = effective_windows[name]
        if en_dt <= st_dt:
            en_dt = en_dt + timedelta(days=1)
        shift_hours = max((en_dt - st_dt).total_seconds() / 3600.0, 0.01)
        states[name] = {
            "car": car,
            "eff_start": st_dt,
            "eff_end": en_dt,
            "last_row": None,
            "count": 0,
            "shift_hours": shift_hours,
        }
        base_order.append(name)

    if not states:
        return rides, roster, effective_windows

    # 4) Active list + cursor (we analyze ALL active drivers every time and then pick best)
    active = []
    cursor = 0
    in_active = set()

    def refresh_active(now_dt: datetime):
        nonlocal active, cursor, in_active
        for nm in base_order:
            if nm in in_active:
                continue
            st_dt = states[nm]["eff_start"]
            en_dt = states[nm]["eff_end"]
            if st_dt <= now_dt < en_dt:
                active.append(nm)
                in_active.add(nm)
        i = 0
        while i < len(active):
            nm = active[i]
            if now_dt >= states[nm]["eff_end"]:
                if i < cursor:
                    cursor -= 1
                in_active.remove(nm)
                del active[i]
                if len(active) == 0:
                    cursor = 0
                else:
                    cursor %= len(active)
                continue
            i += 1
        if active:
            cursor %= len(active)
        else:
            cursor = 0

    # 5) Per-car lock state
    car_free_at = {}      # car -> datetime (end of last assigned booking)
    car_last_driver = {}  # car -> name

    # Far-away hint
    has_far = any(is_far_away_row(r) for _, r in rides.iterrows())
    if has_far and FAR_CAR_ID not in active_cars:
        st.warning(f"Far-away bookings detected but car {FAR_CAR_ID} is not active. Those jobs will not be assigned.")

    # 6) Assignment loop
    gm = gap_matrix or DEFAULT_GAP_MATRIX

    for i, row in rides.iterrows():
        pickup_dt = row["Pickup Time"]
        busy_td = busy_block_for_row(row)
        far_req = is_far_away_row(row)

        refresh_active(pickup_dt)

        if not active:
            rides.at[i, "Notes"] = "No driver free (no active shifts)"
            continue

        # Analyze the WHOLE active list, then choose best
        n = len(active)
        order = [(cursor + k) % n for k in range(n)]
        feasible = []
        applied_gap_min_map = {}
        prev_type_map = {}

        for idx in order:
            nm = active[idx]
            car = states[nm]["car"]

            if far_req and car != FAR_CAR_ID:
                continue

            st_dt = states[nm]["eff_start"]
            en_dt = states[nm]["eff_end"]
            flex = in_flex_window(pickup_dt, st_dt, en_dt, soft_early, hard_early, late_allow)
            if flex is None:
                if auto_relax_to_soft:
                    soft_start = st_dt - timedelta(minutes=soft_early)
                    soft_end   = en_dt + timedelta(minutes=late_allow)
                    if not (soft_start <= pickup_dt <= soft_end):
                        continue
                else:
                    continue

            # driver chaining gap
            last_row = states[nm]["last_row"]
            if last_row is not None:
                extra_gap = rule_gap_after(last_row, row, gm, gap_cph_only)
                prev_pick = pd.to_datetime(last_row["Pickup Time"])
                if use_gap_only:
                    earliest_next_driver = prev_pick + extra_gap
                else:
                    prev_busy = busy_block_for_row(last_row)
                    earliest_next_driver = prev_pick + prev_busy + extra_gap
                if pickup_dt < earliest_next_driver:
                    continue
                applied_gap_min_map[nm] = int(extra_gap.total_seconds() // 60)
                prev_type_map[nm] = booking_type(last_row)
            else:
                applied_gap_min_map[nm] = 0
                prev_type_map[nm] = None

            # vehicle lock (+ handover if driver switches)
            car_free = car_free_at.get(car, datetime.min)
            needs_handover = (car_last_driver.get(car) is not None and car_last_driver.get(car) != nm)
            car_ready = car_free + (car_handover if needs_handover else timedelta(0))
            if pickup_dt < car_ready:
                continue

            # feasible; fairness
            load_ratio = states[nm]["count"] / states[nm]["shift_hours"]
            ring_distance = (idx - cursor) % n
            feasible.append((nm, load_ratio, ring_distance))

        if not feasible:
            rides.at[i, "Notes"] = "No driver free (window/gap/car)"
            continue

        # Choose: lowest load ratio; tie -> closest ahead of cursor
        feasible.sort(key=lambda t: (t[1], t[2]))
        chosen, _, chosen_ring_dist = feasible[0]
        chosen_idx = (cursor + chosen_ring_dist) % len(active)
        chosen_car = states[chosen]["car"]

        # lock
        busy_until = pickup_dt + busy_td
        states[chosen]["last_row"] = row.copy()
        states[chosen]["count"] += 1

        # car state
        needs_handover_note = ""
        if car_last_driver.get(chosen_car) is not None and car_last_driver[chosen_car] != chosen:
            needs_handover_note = f"Handover {int(car_handover.total_seconds()//60)}m"
        car_free_at[chosen_car] = busy_until
        car_last_driver[chosen_car] = chosen

        # notes
        this_type = booking_type(row)
        prev_type = prev_type_map.get(chosen)
        gap_mins  = applied_gap_min_map.get(chosen, 0)

        note_parts = [f"Busy until {busy_until.strftime('%H:%M')}"]
        if gap_mins and prev_type:
            note_parts.append(f"Gap {prev_type}➜{this_type} {gap_mins}m")
        if needs_handover_note:
            note_parts.append(needs_handover_note)
        if far_req and chosen_car != FAR_CAR_ID:
            note_parts.append(f"WARNING: far-away not on {FAR_CAR_ID}")

        rides.at[i, "Assigned Driver"] = chosen
        rides.at[i, "Car"] = chosen_car
        rides.at[i, "Notes"] = " · ".join(note_parts)

        if active:
            cursor = (chosen_idx + 1) % len(active)

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

# normalize/mirror columns used in UI
def ensure_column(df: pd.DataFrame, target: str, cands: list):
    if target in df.columns:
        return
    for c in cands:
        if c in df.columns:
            df[target] = df[c]
            return
    df[target] = ""

# =========================
# App state & UI
# =========================
st.set_page_config(page_title="🚖 Daily Car Plan", page_icon="🚗", layout="wide")
st.title("🚖 Daily Car Plan")

# session buckets
if "rides_df" not in st.session_state:
    st.session_state.rides_df = None
if "driver_rows" not in st.session_state:
    st.session_state.driver_rows = None
if "active_cars" not in st.session_state:
    st.session_state.active_cars = []
if "settings" not in st.session_state:
    st.session_state.settings = {}
if "plan" not in st.session_state:
    st.session_state.plan = None
if "service_date" not in st.session_state:
    st.session_state.service_date = datetime.now().date()

drivers_df = load_drivers()
if drivers_df.empty:
    st.stop()

tab_rides, tab_setup, tab_plan = st.tabs(["📄 Rides", "👥 Drivers & Settings", "🧮 Plan"])

# ===== Rides tab =====
with tab_rides:
    st.subheader("Load rides once")
    up = st.file_uploader("Upload today's rides CSV", type=["csv"], key="uploader")
    if up:
        tmp = pd.read_csv(up).fillna("")
        st.dataframe(tmp.head(30), use_container_width=True, height=260)
        if st.button("Use this file"):
            st.session_state.rides_df = tmp
            st.success("Rides loaded into session. Switch tabs freely.")
    if st.session_state.rides_df is not None:
        st.info(f"Rides in memory: {len(st.session_state.rides_df)} row(s).")
        if st.button("Replace rides file"):
            st.session_state.rides_df = None
            st.session_state.plan = None
            st.rerun()

# ===== Drivers & Settings tab =====
with tab_setup:
    st.subheader("Service date & DB shifts")
    sd = st.date_input("Service date", value=st.session_state.service_date, key="svc_date_picker")
    st.session_state.service_date = sd

    colA, colB = st.columns(2)
    with colA:
        if st.button("Load saved shifts for this date"):
            df_load = load_shifts_for_date(st.session_state.service_date)
            if df_load.empty:
                st.warning("No saved shifts for this date.")
            else:
                st.session_state.driver_rows = df_load.to_dict("records")
                st.success(f"Loaded {len(df_load)} shift(s) for {st.session_state.service_date}.")
    with colB:
        st.caption("You can save after editing below.")

    st.markdown("---")
    st.subheader("Drivers & Timing (saved only on submit)")
    with st.form("driver_setup_form", clear_on_submit=False):
        # Active cars
        default_active = [c for c in CAR_POOL if c.lower() != "s-klasse"]
        active_cars = st.multiselect(
            "Active cars (one driver per car at a time; handovers allowed)",
            options=CAR_POOL,
            default=st.session_state.active_cars or default_active
        )

        # Respect dates in file
        respect_dates = st.checkbox("Respect dates in file (keep them if present)", value=st.session_state.settings.get("respect_dates", True))

        # Driver multi-select and per-driver inputs
        all_names = drivers_df["name"].tolist()
        if st.session_state.driver_rows:
            default_names = [r["name"] for r in st.session_state.driver_rows]
        else:
            default_names = all_names[:min(len(all_names), 12)]
        selected = st.multiselect("Drivers today", options=all_names, default=default_names)

        rows = []
        cols = st.columns(2)
        saved_rows_df = pd.DataFrame(st.session_state.driver_rows) if st.session_state.driver_rows else pd.DataFrame(columns=["name","start","end","preferred_car"])
        for i, name in enumerate(selected):
            with cols[i % 2]:
                st.markdown(f"**{name}**")
                rec = {}
                if not saved_rows_df.empty and name in saved_rows_df["name"].values:
                    rec = saved_rows_df.loc[saved_rows_df["name"] == name].iloc[0].to_dict()
                start_time = st.time_input(
                    f"{name} start time",
                    value=(pd.to_datetime(rec["start"]).time() if rec.get("start") not in (None, "", pd.NaT) else None),
                    key=f"st_{name}"
                )
                end_time   = st.time_input(
                    f"{name} end time",
                    value=(pd.to_datetime(rec["end"]).time() if rec.get("end") not in (None, "", pd.NaT) else None),
                    key=f"et_{name}"
                )
                pref = st.selectbox(
                    f"{name} preferred car",
                    [""] + active_cars,
                    index=([""] + active_cars).index(rec.get("preferred_car","")) if rec.get("preferred_car","") in ([""] + active_cars) else 0,
                    key=f"pref_{name}"
                )
                start_dt = datetime.combine(st.session_state.service_date, start_time) if start_time else None
                end_dt   = datetime.combine(st.session_state.service_date, end_time) if end_time else None
                rows.append({"name": name, "start": start_dt, "end": end_dt, "preferred_car": pref})

        driver_rows_live = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["name","start","end","preferred_car"])

        # Timing sliders
        st.markdown("### Timing & rules")
        hb_min   = st.slider("Car handover (min) — when a car switches drivers", 0, 240, st.session_state.settings.get("car_handover", 75), 5)
        soft_min = st.slider("Soft early start allowed (min)", 0, 180, st.session_state.settings.get("soft_early", 30), 5)
        hard_min = st.slider("Hard early start allowed (min)", 0, 240, st.session_state.settings.get("hard_early", 60), 5)
        late_min = st.slider("Late end allowed (min)", 0, 120, st.session_state.settings.get("late_allow", 20), 5)
        auto_relax = st.checkbox("If a job is unassigned for window only, auto-relax to SOFT", value=st.session_state.settings.get("auto_relax", True))
        use_gap_only = st.checkbox("Use TYPE gaps only for chaining (ignore Trip Minutes)", value=st.session_state.settings.get("use_gap_only", True))

        # NEW: early finish flex in roster
        roster_early_end_min = st.slider("Allow previous driver to finish up to (min) EARLY (roster only)", 0, 120, st.session_state.settings.get("roster_early_end_min", 60), 5)

        st.markdown("#### Gap rules between booking TYPES (prev ➜ next)")
        st.caption("Types: C2C (City→City), A2C (Airport→City), C2A (City→Airport)")
        gap_cph_only = st.checkbox("Apply TYPE gaps only when both pickups are in CPH area", value=st.session_state.settings.get("gap_cph_only", False))

        types = ["C2C","A2C","C2A"]
        gap_matrix = {}
        for r in types:
            gap_matrix[r] = {}
            row_cols = st.columns(len(types))
            for j, c in enumerate(types):
                default_val = int(st.session_state.settings.get("gap_matrix", DEFAULT_GAP_MATRIX).get(r, {}).get(c, DEFAULT_GAP_MATRIX[r][c]))
                gap_matrix[r][c] = row_cols[j].number_input(f"{r} ➜ {c}", 0, 240, default_val, 5, key=f"gap_{r}_{c}")

        submitted = st.form_submit_button("Save driver setup (kept in session)")
        if submitted:
            st.session_state.active_cars = list(active_cars)
            st.session_state.driver_rows = driver_rows_live.to_dict("records")
            st.session_state.settings = {
                "service_date": st.session_state.service_date,
                "respect_dates": bool(respect_dates),
                "car_handover": int(hb_min),
                "soft_early": int(soft_min),
                "hard_early": int(hard_min),
                "late_allow": int(late_min),
                "auto_relax": bool(auto_relax),
                "gap_matrix": gap_matrix,
                "gap_cph_only": bool(gap_cph_only),
                "use_gap_only": bool(use_gap_only),
                "roster_early_end_min": int(roster_early_end_min),
            }
            st.success("Setup saved in session.")

    # Save current setup (driver working times) to DB for this date
    if st.session_state.driver_rows:
        if st.button("💾 Save these drivers' working times to DB (for this date)"):
            for r in st.session_state.driver_rows:
                upsert_shift_row(
                    service_date=st.session_state.service_date,
                    name=r["name"],
                    start_ts=r.get("start"),
                    end_ts=r.get("end"),
                    preferred_car=(r.get("preferred_car") or "")
                )
            st.success(f"Saved {len(st.session_state.driver_rows)} shift(s) for {st.session_state.service_date}.")

    # ---- Profiles UI ----
    st.markdown("---")
    st.subheader("Shift Profiles (times only, no dates)")

    existing = list_profiles()
    colp1, colp2 = st.columns([2,1])
    with colp1:
        profile_name = st.text_input("Profile name (create or overwrite)")
    with colp2:
        pick_existing = st.selectbox("Existing profiles", options=[""] + existing, index=0)

    cpa, cpb, cpc = st.columns(3)
    with cpa:
        if st.button("Save current drivers as profile"):
            if not st.session_state.driver_rows:
                st.warning("Save driver setup first (top of this tab), then save as profile.")
            else:
                try:
                    save_profile(profile_name.strip(), st.session_state.driver_rows)
                    st.success(f"Saved profile '{profile_name}'.")
                except Exception as e:
                    st.error(f"Failed to save profile: {e}")
    with cpb:
        if st.button("Load selected profile into today"):
            name_to_load = pick_existing or profile_name.strip()
            if not name_to_load:
                st.warning("Pick or type a profile name to load.")
            else:
                dfp = load_profile(name_to_load, st.session_state.service_date)
                if dfp.empty:
                    st.warning(f"No entries found for profile '{name_to_load}'.")
                else:
                    st.session_state.driver_rows = dfp.to_dict("records")
                    st.success(f"Loaded profile '{name_to_load}' for {st.session_state.service_date}.")
    with cpc:
        if st.button("Delete selected profile"):
            name_to_delete = pick_existing or profile_name.strip()
            if not name_to_delete:
                st.warning("Pick or type a profile name to delete.")
            else:
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM shift_profile_entries WHERE profile_name = :p"), {"p": name_to_delete})
                    conn.execute(text("DELETE FROM shift_profiles WHERE profile_name = :p"), {"p": name_to_delete})
                st.success(f"Deleted profile '{name_to_delete}'.")
                st.rerun()

# ===== Plan tab =====
with tab_plan:
    st.subheader("Generate / Update Plan")
    rides_df = st.session_state.rides_df
    driver_rows = st.session_state.driver_rows
    active_cars = st.session_state.active_cars
    settings = st.session_state.settings

    if rides_df is None:
        st.info("Load rides in the **Rides** tab first.")
        st.stop()
    if not driver_rows or not active_cars or not settings:
        st.info("Save your **Drivers & Settings** in the previous tab.")
        st.stop()

    if st.button("Generate / Update Plan"):
        rides_df_live = pd.DataFrame(rides_df).fillna("")
        driver_rows_live = pd.DataFrame(driver_rows)
        svc_date = settings.get("service_date", st.session_state.service_date)

        try:
            plan_df, roster, eff_windows = assign_plan_with_rostered_cars(
                rides_df=rides_df_live,
                driver_rows=driver_rows_live,
                active_cars=active_cars,
                service_date=svc_date,
                car_handover=timedelta(minutes=settings["car_handover"]),
                soft_early=settings["soft_early"],
                hard_early=settings["hard_early"],
                late_allow=settings["late_allow"],
                auto_relax_to_soft=settings["auto_relax"],
                respect_dates=settings.get("respect_dates", True),
                gap_matrix=settings["gap_matrix"],
                gap_cph_only=settings["gap_cph_only"],
                use_gap_only=settings["use_gap_only"],
                roster_early_end_flex=timedelta(minutes=settings.get("roster_early_end_min", 60)),
            )
        except Exception as e:
            st.error(f"Planning failed: {e}")
            st.stop()

        # Normalize columns for display (use exact Drop Off when present)
        ensure_column(plan_df, "Pickup", PICKUP_CANDIDATES)
        ensure_column(plan_df, "Dropoff", DROPOFF_CANDIDATES)
        ensure_column(plan_df, "Name", ["Pax Name","Passenger Name","PAX","Pax","Client Name","Customer Name"])

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
            display_cols.insert(5, "Type")

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
        bilplan_text = build_bilplan(plan_df, svc_date)
        st.subheader("BILPLAN")
        st.text_area("BILPLAN text", bilplan_text, height=320)
        st.download_button(
            "Download BILPLAN (.txt)",
            data=bilplan_text.encode("utf-8"),
            file_name=f"bilplan_{svc_date.isoformat()}.txt",
            mime="text/plain"
        )
