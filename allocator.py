
import pandas as pd
from datetime import datetime, timedelta, date
import re, math

DEFAULT_AIRPORT_KEYWORDS = ["cph","cph airport","copenhagen airport","kastrup","københavn lufthavn","kastrup lufthavn"]
DEFAULT_CPH_KEYWORDS = ["copenhagen","københavn","kobenhavn","cph"]
DEFAULT_FAR_AWAY_KEYWORDS = ["roskilde","helsingør","hilleroed","hillerød","odense","aarhus","aalborg"]

def _col(df, *cands):
    lowermap = {c.lower(): c for c in df.columns}
    for cand in cands:
        if cand.lower() in lowermap:
            return lowermap[cand.lower()]
    for c in df.columns:
        lc = c.lower()
        for cand in cands:
            if cand.lower() in lc:
                return c
    return None

def parse_time(t):
    if pd.isna(t) or str(t).strip()=="": return None
    s = str(t).strip()
    for fmt in ["%H:%M","%H.%M"]:
        try: return datetime.strptime(s, fmt).time()
        except ValueError: pass
    if re.fullmatch(r"\\d{3,4}", s):
        s = s.zfill(4); return datetime.strptime(s,"%H%M").time()
    try: return pd.to_datetime(s).time()
    except: return None

def parse_date(d):
    if pd.isna(d) or str(d).strip()=="": return None
    s = str(d).strip()
    for fmt in ["%Y-%m-%d","%d/%m/%Y","%d-%m-%Y","%d.%m.%Y","%m/%d/%Y"]:
        try: return datetime.strptime(s, fmt).date()
        except ValueError: pass
    try: return pd.to_datetime(s).date()
    except: return None

def normalize_str(x): return x.strip().lower() if isinstance(x,str) else ""

def _split_csv_list(val):
    if pd.isna(val) or str(val).strip()=="": return []
    return [v.strip().lower() for v in str(val).split(",") if v.strip()]

def text_has_any(text, kws): return any(k in normalize_str(text) for k in kws)
def is_airport(text, kws): return text_has_any(text, kws)

def classify_area(row, cph_kws, far_kws, pickup_city_col=None, pickup_addr_col=None):
    city = normalize_str(row.get(pickup_city_col)) if pickup_city_col else ""
    addr = normalize_str(row.get(pickup_addr_col)) if pickup_addr_col else ""
    if any(k in city for k in cph_kws) or any(k in addr for k in cph_kws): return "copenhagen"
    if any(k in city for k in far_kws) or any(k in addr for k in far_kws): return "far"
    return "other"

class RoutingProvider:
    def route(self, o_lat,o_lng,d_lat,d_lng): return (None, None)

class OSRMProvider(RoutingProvider):
    def __init__(self, base_url="https://router.project-osrm.org"): self.base_url = base_url.rstrip("/")
    def route(self, o_lat,o_lng,d_lat,d_lng):
        try:
            import requests
            url=f"{self.base_url}/route/v1/driving/{o_lng},{o_lat};{d_lng},{d_lat}?overview=false"
            r=requests.get(url,timeout=8); r.raise_for_status(); js=r.json()
            secs = js["routes"][0]["duration"]; meters = js["routes"][0]["distance"]
            return max(1, int(math.ceil(secs/60))), meters/1000.0
        except: return (None, None)

class GoogleDMProvider(RoutingProvider):
    def __init__(self, api_key): self.api_key=api_key
    def route(self, o_lat,o_lng,d_lat,d_lng):
        try:
            import requests
            params={"origins":f"{o_lat},{o_lng}","destinations":f"{d_lat},{d_lng}","mode":"driving","key":self.api_key}
            r=requests.get("https://maps.googleapis.com/maps/api/distancematrix/json",params=params,timeout=8); r.raise_for_status()
            js=r.json(); secs=js["rows"][0]["elements"][0]["duration"]["value"]; meters=js["rows"][0]["elements"][0]["distance"]["value"]
            return max(1,int(math.ceil(secs/60))), meters/1000.0
        except: return (None, None)

def geocode_google(address, api_key):
    try:
        import requests
        params={"address":address,"key":api_key,"region":"dk"}
        r=requests.get("https://maps.googleapis.com/maps/api/geocode/json",params=params,timeout=8); r.raise_for_status()
        loc=r.json()["results"][0]["geometry"]["location"]; return float(loc["lat"]), float(loc["lng"])
    except: return (None,None)

def allocate(bookings: pd.DataFrame,
             drivers: pd.DataFrame,
             # area rules
             min_gap_after_airport_cph_min: int = 50,
             min_gap_after_airport_far_min: int = 60,
             min_general_gap_far_min: int = 15,
             # vehicle & VIP
             require_vehicle_match: bool = True,
             vip_driver_preferred_for_levels=("super vip",),
             # breaks
             enable_breaks: bool = True,
             default_break_after_hours: float = 6.0,
             default_break_minutes: int = 30,
             # keywords
             airport_keywords=None, cph_keywords=None, far_keywords=None,
             # dates
             force_service_date: date | None = None,
             # routing
             use_routing: bool = False,
             routing_provider: RoutingProvider | None = None,
             geocode_with_google_api_key: str | None = None,
             # special rules
             far_distance_km_threshold: float = 45.0,
             airport_wait_min_estimate: int = 30,
             customer_name_cols=("Customer","Customer Name","Client","Passenger","Company","Name"),
             duration_cols_hours=("Duration Hours","Hours","Duration"),
             duration_cols_minutes=("Duration Minutes","Minutes"),
             # column candidates
             vip_level_col_candidates=("VIP Level","Vip Level","VIP","Booking Type"),
             vehicle_col_candidates=("Vehicle Type","Vehicle","Requested Vehicle"),
             pickup_lat_candidates=("Pick Up Lat","Pickup Lat","From Lat","From Latitude"),
             pickup_lng_candidates=("Pick Up Lng","Pickup Lng","From Lng","From Longitude"),
             drop_lat_candidates=("Drop Off Lat","Drop Lat","To Lat","To Latitude"),
             drop_lng_candidates=("Drop Off Lng","Drop Lng","To Lng","To Longitude"),
             ) -> pd.DataFrame:

    airport_keywords = airport_keywords or DEFAULT_AIRPORT_KEYWORDS
    cph_keywords = cph_keywords or DEFAULT_CPH_KEYWORDS
    far_keywords = far_keywords or DEFAULT_FAR_AWAY_KEYWORDS

    pu_time = _col(bookings,"Pickup Time","Pick Up Time","Pick Up Time ","PickupTime")
    pu_date = _col(bookings,"Pickup Date","Pick Up Date","PickupDate","Date")
    pu_addr = _col(bookings,"Pick Up","Pick Up Address","Pickup Address","From")
    pu_city = _col(bookings,"Pick Up City","Pickup City")
    do_addr = _col(bookings,"Drop Off","Dropoff","Drop Off Address","To")

    vip_col = None
    for c in vip_level_col_candidates: vip_col = _col(bookings, c) or vip_col
    vehicle_col = None
    for c in vehicle_col_candidates: vehicle_col = _col(bookings, c) or vehicle_col

    p_lat = None
    for c in pickup_lat_candidates: p_lat = _col(bookings, c) or p_lat
    p_lng = None
    for c in pickup_lng_candidates: p_lng = _col(bookings, c) or p_lng
    d_lat = None
    for c in drop_lat_candidates: d_lat = _col(bookings, c) or d_lat
    d_lng = None
    for c in drop_lng_candidates: d_lng = _col(bookings, c) or d_lng

    if pu_time is None or pu_addr is None:
        raise ValueError("Missing required columns: 'Pickup Time' and 'Pick Up'/'Pick Up Address'.")

    if force_service_date is None and pu_date is None:
        force_service_date = date.today()

    dts = []
    for _, r in bookings.iterrows():
        d = force_service_date if force_service_date else parse_date(r.get(pu_date))
        t = parse_time(r.get(pu_time))
        dts.append(datetime.combine(d,t) if d and t else None)

    out = bookings.copy()
    out["__pickup_dt"] = dts
    out["__pickup_is_airport"] = out[pu_addr].apply(lambda x: is_airport(x, airport_keywords))
    out["__area"] = out.apply(lambda r: classify_area(r, cph_keywords, far_keywords, pu_city, pu_addr), axis=1)

    def to_float(x):
        try: return float(x)
        except: return None

    out["__p_lat"] = out[p_lat].apply(to_float) if p_lat else None
    out["__p_lng"] = out[p_lng].apply(to_float) if p_lng else None
    out["__d_lat"] = out[d_lat].apply(to_float) if d_lat else None
    out["__d_lng"] = out[d_lng].apply(to_float) if d_lng else None

    # routing: estimate (minutes, km) pickup->drop
    def route_est(row):
        if not use_routing or routing_provider is None: return (None, None)
        o_lat,o_lng,d_lat,d_lng = row["__p_lat"],row["__p_lng"],row["__d_lat"],row["__d_lng"]
        if None in (o_lat,o_lng,d_lat,d_lng): return (None, None)
        return routing_provider.route(o_lat,o_lng,d_lat,d_lng)
    rmins, rkm = [], []
    for _, r in out.iterrows():
        m, k = route_est(r); rmins.append(m); rkm.append(k)
    out["__est_trip_min"] = rmins
    out["__route_km"] = rkm

    # customer text + duration
    def get_customer_text(row):
        parts = []
        for c in customer_name_cols:
            cc = _col(out, c)
            if cc and isinstance(row.get(cc), str): parts.append(row.get(cc))
        return " ".join(parts).lower()

    def get_duration_minutes(row):
        mins = 0
        for c in duration_cols_hours:
            cc = _col(out, c)
            if cc and str(row.get(cc, "")).strip() != "":
                try: mins += int(float(row.get(cc)) * 60)
                except: pass
        for c in duration_cols_minutes:
            cc = _col(out, c)
            if cc and str(row.get(cc, "")).strip() != "":
                try: mins += int(float(row.get(cc)))
                except: pass
        return mins if mins>0 else None

    # busy minutes logic
    def busy_minutes(row):
        area = row["__area"]; trip_min = row["__est_trip_min"]; route_km = row["__route_km"]
        pickup_airport = bool(row["__pickup_is_airport"])
        base = trip_min if isinstance(trip_min,(int,float)) else 30

        # Far-away threshold rule
        if area == "far" and isinstance(route_km,(int,float)) and route_km > far_distance_km_threshold:
            extra = airport_wait_min_estimate if pickup_airport else 0
            return int(base + extra)

        # Roadshow / Site Inspection
        cust = get_customer_text(row)
        if "roadshow" in cust or "site inspection" in cust:
            dur = get_duration_minutes(row)
            if dur: return int(dur)
            return int(base + 60)

        return int(base)

    out["__busy_minutes"] = out.apply(busy_minutes, axis=1)

    # ---- driver states ----
    def _pt(t): return parse_time(t) if pd.notna(t) else None
    driver_state = {}
    for _, drow in drivers.iterrows():
        sid = str(drow.get("driver_id") or "").strip()
        sname = str(drow.get("name") or "").strip()
        if not sid: continue
        vehicle_types = set(_split_csv_list(drow.get("vehicle_types")))
        is_vip = str(drow.get("is_vip") or "").strip().lower() in ("1","true","yes","y")
        b_after = float(drow.get("break_after_hours") or 6.0)
        b_min = int(float(drow.get("break_minutes") or 30))
        driver_state[sid] = {
            "name": sname,
            "shift_start": _pt(drow.get("shift_start")),
            "shift_end": _pt(drow.get("shift_end")),
            "vehicle_types": vehicle_types, "is_vip": is_vip,
            "break_after_hours": b_after, "break_minutes": b_min,
            "last_job_end_dt": None, "last_job_pickup_is_airport": False,
            "last_drop_lat": None, "last_drop_lng": None,
            "continuous_work_start": None,
        }

    assignments, unassigned = [], []

    def in_shift(st, dt):
        if not st["shift_start"] or not st["shift_end"]: return True
        pt = dt.time()
        if st["shift_start"] <= st["shift_end"]: return st["shift_start"] <= pt <= st["shift_end"]
        return pt >= st["shift_start"] or pt <= st["shift_end"]

    def meets_break_rule(st, gap_min):
        if st["continuous_work_start"] is None or st["last_job_end_dt"] is None: return True
        worked = (st["last_job_end_dt"] - st["continuous_work_start"]).total_seconds()/3600.0
        if worked >= st["break_after_hours"]: return gap_min >= st["break_minutes"]
        return True

    def vehicle_ok(st, req_v):
        if not require_vehicle_match or not req_v: return True
        if not st["vehicle_types"]: return True
        return normalize_str(req_v) in st["vehicle_types"]

    def travel_gap_from_last(st, row):
        if not use_routing or routing_provider is None: return None
        if st["last_drop_lat"] is None or st["last_drop_lng"] is None: return None
        o_lat,o_lng = st["last_drop_lat"], st["last_drop_lng"]
        d_lat,d_lng = row["__p_lat"], row["__p_lng"]
        if None in (o_lat,o_lng,d_lat,d_lng): return None
        m, _ = routing_provider.route(o_lat,o_lng,d_lat,d_lng); return m

    req_vehicle_col = vehicle_col

    for idx, r in out.iterrows():
        pickup_dt = r["__pickup_dt"]
        if pickup_dt is None: unassigned.append((idx,"missing_pickup_datetime")); continue
        area = r["__area"]
        req_vehicle = r.get(req_vehicle_col) if req_vehicle_col else None
        vip_raw = normalize_str(r.get(vip_col)) if vip_col else ""
        is_super_vip = vip_raw in [normalize_str(x) for x in vip_driver_preferred_for_levels]
        busy_min = r["__busy_minutes"]

        feasible, vip_feasible = [], []
        for sid, st in driver_state.items():
            if not in_shift(st, pickup_dt): continue

            travel_min = travel_gap_from_last(st, r)
            if st["last_job_end_dt"] is not None:
                gap = (pickup_dt - st["last_job_end_dt"]).total_seconds()/60.0
                req_gap = travel_min if isinstance(travel_min,(int,float)) else 0
                if area == "copenhagen":
                    if st["last_job_pickup_is_airport"]:
                        req_gap = max(req_gap, min_gap_after_airport_cph_min)
                elif area == "far":
                    if st["last_job_pickup_is_airport"]:
                        req_gap = max(req_gap, min_gap_after_airport_far_min)
                    req_gap = max(req_gap, min_general_gap_far_min)
                if gap < req_gap - 1e-6: continue
                if not meets_break_rule(st, gap): continue

            if not vehicle_ok(st, req_vehicle): continue

            feasible.append(sid)
            if is_super_vip and st["is_vip"]: vip_feasible.append(sid)

        if not feasible: unassigned.append((idx,"no_feasible_driver")); continue
        pool = vip_feasible if (is_super_vip and vip_feasible) else feasible
        pool.sort(key=lambda sid: driver_state[sid]["last_job_end_dt"] or datetime.min)
        chosen = pool[0]; st = driver_state[chosen]

        assignments.append({"row_index": idx, "driver_id": chosen, "driver_name": st["name"]})

        end_dt = pickup_dt + timedelta(minutes=int(busy_min))
        if st["last_job_end_dt"] is None:
            st["continuous_work_start"] = pickup_dt
        else:
            gap = (pickup_dt - st["last_job_end_dt"]).total_seconds()/60.0
            if gap >= st["break_minutes"]:
                st["continuous_work_start"] = pickup_dt
        st["last_job_end_dt"] = end_dt
        st["last_job_pickup_is_airport"] = bool(r["__pickup_is_airport"])
        st["last_drop_lat"], st["last_drop_lng"] = r["__d_lat"], r["__d_lng"]

    res = out.copy()
    res["Allocated Driver ID"] = ""; res["Allocated Driver Name"] = ""
    for a in assignments:
        res.loc[a["row_index"], "Allocated Driver ID"] = a["driver_id"]
        res.loc[a["row_index"], "Allocated Driver Name"] = a["driver_name"]
    res["Unassigned Reason"] = ""
    for idx, reason in unassigned: res.loc[idx,"Unassigned Reason"]=reason
    return res

def allocate_with_shift_flex(*, bookings: pd.DataFrame, drivers: pd.DataFrame,
                             early_start_soft_min: int = 30,
                             early_start_hard_min: int = 60,
                             late_end_allow_min: int = 20,
                             **kwargs) -> pd.DataFrame:
    def widen(df, early, late):
        from datetime import datetime, timedelta
        out = df.copy()
        def shift_time(val, delta):
            import pandas as pd
            if pd.isna(val) or str(val).strip()=="": return val
            s=str(val).strip()
            for fmt in ["%H:%M","%H.%M"]:
                try:
                    tm=datetime.strptime(s,fmt); tm=tm+timedelta(minutes=delta); return tm.strftime("%H:%M")
                except: pass
            return val
        out["shift_start"]=out.get("shift_start","").apply(lambda v: shift_time(v, -early))
        out["shift_end"]=out.get("shift_end","").apply(lambda v: shift_time(v, late))
        return out

    d1 = widen(drivers, early_start_soft_min, late_end_allow_min)
    res1 = allocate(bookings=bookings, drivers=d1, **kwargs)
    if (res1["Allocated Driver ID"] == "").sum() == 0:
        res1["Shift Flex Used"] = ""; res1["Hard Early Start"]="No"; return res1

    d2 = widen(drivers, early_start_hard_min, late_end_allow_min)
    res2 = allocate(bookings=bookings, drivers=d2, **kwargs)

    final = res1.copy()
    final["Shift Flex Used"] = ""; final["Hard Early Start"] = "No"
    for idx in final.index:
        if final.loc[idx,"Allocated Driver ID"] == "" and res2.loc[idx,"Allocated Driver ID"] != "":
            final.loc[idx,"Allocated Driver ID"] = res2.loc[idx,"Allocated Driver ID"]
            final.loc[idx,"Allocated Driver Name"] = res2.loc[idx,"Allocated Driver Name"]
            final.loc[idx,"Unassigned Reason"] = ""
            final.loc[idx,"Shift Flex Used"] = f"early_start_used_{early_start_hard_min}m"
            final.loc[idx,"Hard Early Start"] = "Yes"
    return final
