
import pandas as pd
from datetime import datetime, timedelta, date
import re
import math

DEFAULT_AIRPORT_KEYWORDS = [
    "cph", "cph airport", "copenhagen airport", "kastrup", "københavn lufthavn", "kastrup lufthavn"
]
DEFAULT_CPH_KEYWORDS = ["copenhagen", "københavn", "kobenhavn", "cph"]
DEFAULT_FAR_AWAY_KEYWORDS = ["roskilde", "helsingør", "hilleroed", "hillerød", "odense", "aarhus", "aalborg"]

# ----------------- helpers -----------------
def _col(df, *candidates):
    lowermap = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowermap:
            return lowermap[cand.lower()]
    for c in df.columns:
        lc = c.lower()
        for cand in candidates:
            if cand.lower() in lc:
                return c
    return None

def parse_time(t):
    if pd.isna(t) or str(t).strip() == "":
        return None
    s = str(t).strip()
    for fmt in ["%H:%M", "%H.%M"]:
        try:
            return datetime.strptime(s, fmt).time()
        except ValueError:
            pass
    if re.fullmatch(r"\d{3,4}", s):
        s = s.zfill(4)
        return datetime.strptime(s, "%H%M").time()
    try:
        return pd.to_datetime(s).time()
    except Exception:
        return None

def parse_date(d):
    if pd.isna(d) or str(d).strip() == "":
        return None
    s = str(d).strip()
    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%m/%d/%Y"]:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None

def normalize_str(x):
    if isinstance(x, str):
        return x.strip().lower()
    return ""

def _split_csv_list(val):
    if pd.isna(val) or str(val).strip() == "":
        return []
    return [v.strip().lower() for v in str(val).split(",") if v.strip()]

def text_has_any(text, keywords):
    lt = normalize_str(text)
    return any(k in lt for k in keywords)

def is_airport(text, airport_keywords):
    return text_has_any(text, airport_keywords)

def classify_area(row, cph_keywords, far_keywords, pickup_city_col=None, pickup_addr_col=None):
    city_txt = normalize_str(row.get(pickup_city_col)) if pickup_city_col else ""
    addr_txt = normalize_str(row.get(pickup_addr_col)) if pickup_addr_col else ""
    if any(k in city_txt for k in cph_keywords) or any(k in addr_txt for k in cph_keywords):
        return "copenhagen"
    if any(k in city_txt for k in far_keywords) or any(k in addr_txt for k in far_keywords):
        return "far"
    return "other"

# ----------------- travel time provider wrappers -----------------
class TravelTimeProvider:
    def trip_minutes(self, o_lat, o_lng, d_lat, d_lng):
        """Return estimated driving minutes; can return None on failure."""
        return None

class OSRMProvider(TravelTimeProvider):
    def __init__(self, base_url="https://router.project-osrm.org"):
        self.base_url = base_url.rstrip("/")

    def trip_minutes(self, o_lat, o_lng, d_lat, d_lng):
        try:
            import requests
            url = f"{self.base_url}/route/v1/driving/{o_lng},{o_lat};{d_lng},{d_lat}?overview=false"
            r = requests.get(url, timeout=8)
            r.raise_for_status()
            data = r.json()
            secs = data["routes"][0]["duration"]
            return max(1, int(math.ceil(secs / 60)))
        except Exception:
            return None

class GoogleDMProvider(TravelTimeProvider):
    def __init__(self, api_key):
        self.api_key = api_key

    def trip_minutes(self, o_lat, o_lng, d_lat, d_lng):
        try:
            import requests
            params = {
                "origins": f"{o_lat},{o_lng}",
                "destinations": f"{d_lat},{d_lng}",
                "mode": "driving",
                "key": self.api_key,
            }
            r = requests.get("https://maps.googleapis.com/maps/api/distancematrix/json", params=params, timeout=8)
            r.raise_for_status()
            data = r.json()
            secs = data["rows"][0]["elements"][0]["duration"]["value"]
            return max(1, int(math.ceil(secs / 60)))
        except Exception:
            return None

def geocode_google(address, api_key):
    try:
        import requests
        params = {"address": address, "key": api_key, "region": "dk"}
        r = requests.get("https://maps.googleapis.com/maps/api/geocode/json", params=params, timeout=8)
        r.raise_for_status()
        js = r.json()
        loc = js["results"][0]["geometry"]["location"]
        return float(loc["lat"]), float(loc["lng"])
    except Exception:
        return None, None

# ----------------- allocation -----------------
def allocate(bookings: pd.DataFrame,
             drivers: pd.DataFrame,
             # area rules
             min_gap_after_airport_cph_min: int = 50,
             min_gap_after_airport_far_min: int = 60,
             min_general_gap_far_min: int = 15,
             # vehicle + vip
             require_vehicle_match: bool = True,
             vip_driver_preferred_for_levels=("super vip",),
             # breaks
             enable_breaks: bool = True,
             default_break_after_hours: float = 6.0,
             default_break_minutes: int = 30,
             # keywords
             airport_keywords=None,
             cph_keywords=None,
             far_keywords=None,
             # dates
             force_service_date: date | None = None,
             # travel-time
             use_travel_time: bool = False,
             travel_provider: TravelTimeProvider | None = None,
             geocode_with_google_api_key: str | None = None,
             # column candidates
             vip_level_col_candidates=("VIP Level", "Vip Level", "VIP", "Booking Type"),
             vehicle_col_candidates=("Vehicle Type", "Vehicle", "Requested Vehicle"),
             pickup_lat_candidates=("Pick Up Lat", "Pickup Lat", "From Lat", "From Latitude"),
             pickup_lng_candidates=("Pick Up Lng", "Pickup Lng", "From Lng", "From Longitude"),
             drop_lat_candidates=("Drop Off Lat", "Drop Lat", "To Lat", "To Latitude"),
             drop_lng_candidates=("Drop Off Lng", "Drop Lng", "To Lng", "To Longitude"),
             ) -> pd.DataFrame:

    airport_keywords = airport_keywords or DEFAULT_AIRPORT_KEYWORDS
    cph_keywords = cph_keywords or DEFAULT_CPH_KEYWORDS
    far_keywords = far_keywords or DEFAULT_FAR_AWAY_KEYWORDS

    # Identify columns
    pickup_time_col = _col(bookings, "Pickup Time", "Pick Up Time", "Pick Up Time ", "PickupTime")
    pickup_date_col = _col(bookings, "Pickup Date", "Pick Up Date", "PickupDate", "Date")
    pickup_addr_col = _col(bookings, "Pick Up", "Pick Up Address", "Pickup Address", "From")
    pickup_city_col = _col(bookings, "Pick Up City", "Pickup City")
    drop_addr_col = _col(bookings, "Drop Off", "Dropoff", "Drop Off Address", "To")

    vip_col = None
    for c in vip_level_col_candidates:
        vip_col = _col(bookings, c) or vip_col
    vehicle_col = None
    for c in vehicle_col_candidates:
        vehicle_col = _col(bookings, c) or vehicle_col

    p_lat_col = None
    for c in pickup_lat_candidates:
        p_lat_col = _col(bookings, c) or p_lat_col
    p_lng_col = None
    for c in pickup_lng_candidates:
        p_lng_col = _col(bookings, c) or p_lng_col
    d_lat_col = None
    for c in drop_lat_candidates:
        d_lat_col = _col(bookings, c) or d_lat_col
    d_lng_col = None
    for c in drop_lng_candidates:
        d_lng_col = _col(bookings, c) or d_lng_col

    if pickup_time_col is None or pickup_addr_col is None:
        raise ValueError("Couldn't find required columns: 'Pickup Time' and 'Pick Up'/'Pick Up Address'.")

    # Build normalized pickup datetime
    if force_service_date is None and pickup_date_col is None:
        force_service_date = date.today()

    dt_list = []
    for _, r in bookings.iterrows():
        d = force_service_date if force_service_date else parse_date(r.get(pickup_date_col))
        t = parse_time(r.get(pickup_time_col))
        dt_list.append(datetime.combine(d, t) if d and t else None)

    out = bookings.copy()
    out["__pickup_dt"] = dt_list
    out["__pickup_is_airport"] = out[pickup_addr_col].apply(lambda x: is_airport(x, airport_keywords))
    out["__area"] = out.apply(lambda r: classify_area(r, cph_keywords, far_keywords, pickup_city_col, pickup_addr_col), axis=1)

    # Attach coordinates (if available or via geocoding when asked)
    def to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    if p_lat_col: out["__p_lat"] = out[p_lat_col].apply(to_float)
    else: out["__p_lat"] = None
    if p_lng_col: out["__p_lng"] = out[p_lng_col].apply(to_float)
    else: out["__p_lng"] = None
    if d_lat_col: out["__d_lat"] = out[d_lat_col].apply(to_float)
    else: out["__d_lat"] = None
    if d_lng_col: out["__d_lng"] = out[d_lng_col].apply(to_float)
    else: out["__d_lng"] = None

    if use_travel_time and geocode_with_google_api_key:
        # Best-effort geocoding for missing coords
        for idx, r in out.iterrows():
            if (r["__p_lat"] is None or r["__p_lng"] is None) and isinstance(r.get(pickup_addr_col), str):
                lat, lng = geocode_google(r[pickup_addr_col], geocode_with_google_api_key)
                out.loc[idx, "__p_lat"] = lat
                out.loc[idx, "__p_lng"] = lng
            if (r["__d_lat"] is None or r["__d_lng"] is None) and isinstance(r.get(drop_addr_col), str):
                lat, lng = geocode_google(r[drop_addr_col], geocode_with_google_api_key)
                out.loc[idx, "__d_lat"] = lat
                out.loc[idx, "__d_lng"] = lng

    # Estimate trip minutes (pickup->drop)
    def estimate_trip_minutes(row):
        if not use_travel_time or travel_provider is None:
            return None
        o_lat, o_lng, d_lat, d_lng = row["__p_lat"], row["__p_lng"], row["__d_lat"], row["__d_lng"]
        if None in (o_lat, o_lng, d_lat, d_lng):
            return None
        return travel_provider.trip_minutes(o_lat, o_lng, d_lat, d_lng)

    out["__est_trip_min"] = out.apply(estimate_trip_minutes, axis=1)

    # Prepare driver states & capabilities
    def _pt(t): return parse_time(t) if pd.notna(t) else None
    driver_state = {}
    for _, drow in drivers.iterrows():
        sid = str(drow.get("driver_id") or "").strip()
        sname = str(drow.get("name") or "").strip()
        if not sid: 
            continue
        vehicle_types = set(_split_csv_list(drow.get("vehicle_types")))
        is_vip = str(drow.get("is_vip") or "").strip().lower() in ("1", "true", "yes", "y")
        b_after = float(drow.get("break_after_hours")) if pd.notna(drow.get("break_after_hours")) and str(drow.get("break_after_hours")).strip() != "" else default_break_after_hours
        b_min = int(float(drow.get("break_minutes"))) if pd.notna(drow.get("break_minutes")) and str(drow.get("break_minutes")).strip() != "" else default_break_minutes

        driver_state[sid] = {
            "name": sname,
            "shift_start": _pt(drow.get("shift_start")),
            "shift_end": _pt(drow.get("shift_end")),
            "vehicle_types": vehicle_types,
            "is_vip": is_vip,
            "break_after_hours": b_after,
            "break_minutes": b_min,
            "last_job_end_dt": None,         # datetime end of previous job
            "last_job_pickup_is_airport": False,
            "last_drop_lat": None,
            "last_drop_lng": None,
            "continuous_work_start": None,
        }

    assignments = []
    unassigned = []

    def in_shift(st, dt):
        if not st["shift_start"] or not st["shift_end"]:
            return True
        pt = dt.time()
        if st["shift_start"] <= st["shift_end"]:
            return st["shift_start"] <= pt <= st["shift_end"]
        return pt >= st["shift_start"] or pt <= st["shift_end"]

    def meets_break_rule(st, next_pickup_dt, gap_minutes):
        if not enable_breaks:
            return True
        if st["continuous_work_start"] is None or st["last_job_end_dt"] is None:
            return True
        worked_hours = (st["last_job_end_dt"] - st["continuous_work_start"]).total_seconds() / 3600.0
        if worked_hours >= st["break_after_hours"]:
            return gap_minutes >= st["break_minutes"]
        return True

    def vehicle_ok(st, req_vehicle):
        if not require_vehicle_match or not req_vehicle:
            return True
        if not st["vehicle_types"]:
            return True
        return normalize_str(req_vehicle) in st["vehicle_types"]

    def required_gap_minutes(st, area, travel_minutes_from_prev_to_next):
        # Base required gap is travel time between jobs if known, else 0
        base = travel_minutes_from_prev_to_next or 0
        extra = 0
        if area == "copenhagen":
            if st["last_job_pickup_is_airport"]:
                extra = max(extra, min_gap_after_airport_cph_min)
        elif area == "far":
            if st["last_job_pickup_is_airport"]:
                extra = max(extra, min_gap_after_airport_far_min)
            extra = max(extra, min_general_gap_far_min)
        return max(base, extra)

    # utility to compute travel time from last drop to next pickup
    def travel_from_last_to_next(st, next_row):
        if not use_travel_time or travel_provider is None:
            return None
        if st["last_drop_lat"] is None or st["last_drop_lng"] is None:
            return None
        o_lat, o_lng = st["last_drop_lat"], st["last_drop_lng"]
        d_lat, d_lng = next_row["__p_lat"], next_row["__p_lng"]
        if None in (o_lat, o_lng, d_lat, d_lng):
            return None
        return travel_provider.trip_minutes(o_lat, o_lng, d_lat, d_lng)

    for idx, r in out.iterrows():
        pickup_dt = r["__pickup_dt"]
        if pickup_dt is None:
            unassigned.append((idx, "missing_pickup_datetime"))
            continue

        area = r["__area"]
        req_vehicle = r.get(vehicle_col) if vehicle_col else None
        vip_level_raw = normalize_str(r.get(vip_col)) if vip_col else ""
        is_super_vip = vip_level_raw in [normalize_str(x) for x in vip_driver_preferred_for_levels]
        est_trip_min = r["__est_trip_min"]

        feasible = []
        vip_feasible = []

        for sid, st in driver_state.items():
            if not in_shift(st, pickup_dt):
                continue

            # compute travel time from last job drop to this pickup
            travel_min = travel_from_last_to_next(st, r)

            # If previous job has an end, compute actual gap and required gap
            if st["last_job_end_dt"] is not None:
                gap = (pickup_dt - st["last_job_end_dt"]).total_seconds() / 60.0
                req_gap = required_gap_minutes(st, area, travel_min)
                if gap < req_gap - 1e-6:
                    continue

                # break rule
                if not meets_break_rule(st, pickup_dt, gap):
                    continue

            # Vehicle matching
            if not vehicle_ok(st, req_vehicle):
                continue

            feasible.append(sid)
            if is_super_vip and st["is_vip"]:
                vip_feasible.append(sid)

        if not feasible:
            unassigned.append((idx, "no_feasible_driver"))
            continue

        # Prefer VIP if needed
        pool = vip_feasible if (is_super_vip and vip_feasible) else feasible
        pool.sort(key=lambda sid: driver_state[sid]["last_job_end_dt"] or datetime.min)
        chosen = pool[0]
        st = driver_state[chosen]

        assignments.append({"row_index": idx, "driver_id": chosen, "driver_name": st["name"]})

        # Update state: compute end_dt for this job
        end_dt = pickup_dt
        if use_travel_time and est_trip_min and isinstance(est_trip_min, (int, float)):
            end_dt = pickup_dt + timedelta(minutes=int(est_trip_min))

        if st["last_job_end_dt"] is None:
            st["continuous_work_start"] = pickup_dt
        else:
            gap = (pickup_dt - st["last_job_end_dt"]).total_seconds() / 60.0
            if enable_breaks and gap >= st["break_minutes"]:
                st["continuous_work_start"] = pickup_dt

        st["last_job_end_dt"] = end_dt
        st["last_job_pickup_is_airport"] = bool(r["__pickup_is_airport"])
        st["last_drop_lat"] = r["__d_lat"]
        st["last_drop_lng"] = r["__d_lng"]

    # Output
    res = out.copy()
    res["Allocated Driver ID"] = ""
    res["Allocated Driver Name"] = ""
    for a in assignments:
        res.loc[a["row_index"], "Allocated Driver ID"] = a["driver_id"]
        res.loc[a["row_index"], "Allocated Driver Name"] = a["driver_name"]
    res["Unassigned Reason"] = ""
    for idx, reason in unassigned:
        res.loc[idx, "Unassigned Reason"] = reason

    return res


# ---- shift flexibility & two-phase assignment wrapper ----
def allocate_with_shift_flex(
    *,
    bookings: pd.DataFrame,
    drivers: pd.DataFrame,
    early_start_soft_min: int = 30,
    early_start_hard_min: int = 60,
    late_end_allow_min: int = 20,
    **kwargs
) -> pd.DataFrame:
    """
    Two-phase allocation:
      Phase 1: allow up to early_start_soft_min early start, late_end_allow_min late end
      Phase 2: retry remaining unassigned with up to early_start_hard_min (ONLY IF necessary)
    """
    # We call the inner allocate() while temporarily widening shifts in the driver data frame.
    def widen_shifts(df: pd.DataFrame, early: int, late: int) -> pd.DataFrame:
        from datetime import datetime, timedelta
        import pandas as pd
        out = df.copy()
        def _shift(t, delta_min):
            if pd.isna(t) or str(t).strip() == "":
                return t
            from datetime import datetime, timedelta
            s = str(t).strip()
            fmt = "%H:%M"
            try:
                tm = datetime.strptime(s, fmt)
            except Exception:
                try:
                    tm = datetime.strptime(s, "%H.%M")
                except Exception:
                    return t
            tm = tm + timedelta(minutes=delta_min)
            return tm.strftime(fmt)
        out["shift_start"] = out.get("shift_start", "").apply(lambda v: _shift(v, -early))
        out["shift_end"] = out.get("shift_end", "").apply(lambda v: _shift(v, late))
        return out

    # Phase 1: soft early start
    d1 = widen_shifts(drivers, early_start_soft_min, late_end_allow_min)
    res1 = allocate(
        bookings=bookings,
        drivers=d1,
        **kwargs
    )

    remaining = res1[res1["Allocated Driver ID"] == ""].copy()
    if remaining.empty:
        return res1

    # Phase 2: hard early start (only if absolutely necessary)
    d2 = widen_shifts(drivers, early_start_hard_min, late_end_allow_min)
    res2 = allocate(
        bookings=bookings,
        drivers=d2,
        **kwargs
    )

    # Merge: prefer res1 assignments, fill gaps from res2
    final = res1.copy()
    for idx, row in res2.iterrows():
        if final.loc[idx, "Allocated Driver ID"] == "" and row["Allocated Driver ID"] != "":
            final.loc[idx, "Allocated Driver ID"] = row["Allocated Driver ID"]
            final.loc[idx, "Allocated Driver Name"] = row["Allocated Driver Name"]
            final.loc[idx, "Unassigned Reason"] = ""

    # Tag which bookings used hard early start for transparency
    final["Shift Flex Used"] = ""
    for idx in final.index:
        if res1.loc[idx, "Allocated Driver ID"] == "" and final.loc[idx, "Allocated Driver ID"] != "":
            final.loc[idx, "Shift Flex Used"] = f"early_start_used_{early_start_hard_min}m"
    return final
