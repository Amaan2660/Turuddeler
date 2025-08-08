
import pandas as pd
import streamlit as st
from datetime import date, datetime, timedelta
import plotly.express as px

from db import init_db, session_scope, Driver, Vehicle
from allocator import (
    allocate_with_shift_flex,
    DEFAULT_AIRPORT_KEYWORDS, DEFAULT_CPH_KEYWORDS, DEFAULT_FAR_AWAY_KEYWORDS,
    OSRMProvider, GoogleDMProvider
)

st.set_page_config(page_title="Allocator + Fleet + Car Plan", page_icon="🚗", layout="wide")
st.title("🚗 Allocator + Fleet + Car Plan")

init_db()

tab_alloc, tab_drivers, tab_fleet, tab_plan = st.tabs(["🧮 Allocation", "👤 Drivers", "🚘 Fleet", "🗓️ Car Plan"])

# ---------------- Drivers tab ----------------
with tab_drivers:
    st.subheader("Drivers (DB)")
    with session_scope() as s:
        with st.form("driver_form", clear_on_submit=True):
            edit_mode = st.checkbox("Edit existing", value=False)
            existing_ids = [d.driver_id for d in s.query(Driver).order_by(Driver.driver_id).all()]
            sel_id = st.selectbox("Driver ID", [""] + existing_ids)
            name = st.text_input("Name")
            vehicle_types = st.text_input("Vehicle types (comma-separated)", value="standard")
            is_vip = st.checkbox("VIP driver", value=False)
            break_after = st.number_input("Break after hours", min_value=1.0, value=6.0, step=0.5)
            break_minutes = st.number_input("Break length (minutes)", min_value=5, value=30, step=5)
            shift_start = st.text_input("Default shift start (HH:MM)", value="08:00")
            shift_end = st.text_input("Default shift end (HH:MM)", value="16:00")
            active = st.checkbox("Active", value=True)
            submitted = st.form_submit_button("Save")

        if submitted:
            with session_scope() as s2:
                if edit_mode and sel_id:
                    d = s2.query(Driver).filter(Driver.driver_id == sel_id).first()
                    if d:
                        d.name = name or d.name
                        d.vehicle_types = vehicle_types or d.vehicle_types
                        d.is_vip = is_vip
                        d.break_after_hours = float(break_after)
                        d.break_minutes = int(break_minutes)
                        d.shift_start = shift_start or d.shift_start
                        d.shift_end = shift_end or d.shift_end
                        d.active = active
                        st.success(f"Updated driver {sel_id}.")
                    else:
                        st.error("Driver not found.")
                else:
                    if not sel_id or not name:
                        st.error("Driver ID and Name required.")
                    else:
                        exists = s2.query(Driver).filter(Driver.driver_id == sel_id).first()
                        if exists:
                            st.error("Driver ID exists. Use edit mode.")
                        else:
                            d = Driver(driver_id=sel_id, name=name, vehicle_types=vehicle_types, is_vip=is_vip,
                                       break_after_hours=float(break_after), break_minutes=int(break_minutes),
                                       shift_start=shift_start, shift_end=shift_end, active=active)
                            s2.add(d)
                            st.success(f"Added driver {sel_id}.")

    with session_scope() as s:
        st.markdown("**Current drivers**")
        data = [{
            "driver_id": d.driver_id, "name": d.name, "is_vip": d.is_vip,
            "shift_start": d.shift_start, "shift_end": d.shift_end, "active": d.active,
        } for d in s.query(Driver).order_by(Driver.driver_id).all()]
        st.dataframe(pd.DataFrame(data), use_container_width=True)

    st.markdown("---")
    st.markdown("**Import drivers from CSV**")
    up = st.file_uploader("CSV with drivers", type=["csv"], key="driver_import")
    if up is not None and st.button("Import drivers"):
        try:
            imp = pd.read_csv(up, dtype=str).fillna("")
            def col(df, *cands):
                lower = {c.lower(): c for c in df.columns}
                for cand in cands:
                    if cand.lower() in lower: return lower[cand.lower()]
                for c in df.columns:
                    for cand in cands:
                        if cand.lower() in c.lower(): return c
                return None
            c_id = col(imp, "driver_id","id","code")
            c_name = col(imp, "name")
            c_vs = col(imp, "vehicle_types","vehicles","vehicle type")
            c_vip = col(imp, "is_vip","vip")
            c_bh = col(imp, "break_after_hours")
            c_bm = col(imp, "break_minutes")
            c_ss = col(imp, "shift_start")
            c_se = col(imp, "shift_end")
            c_act = col(imp, "active","is_active")
            if not c_id or not c_name:
                st.error("CSV must include at least 'driver_id' and 'name'.")
            else:
                added, updated = 0, 0
                with session_scope() as s2:
                    for _, r in imp.iterrows():
                        rid = str(r.get(c_id,"")).strip()
                        if not rid: continue
                        d = s2.query(Driver).filter(Driver.driver_id == rid).first()
                        creating = d is None
                        if creating:
                            d = Driver(driver_id=rid, name=str(r.get(c_name,"")).strip())
                            s2.add(d); added += 1
                        else: updated += 1
                        d.vehicle_types = str(r.get(c_vs, d.vehicle_types or ""))
                        vip_raw = str(r.get(c_vip, d.is_vip)).strip().lower()
                        d.is_vip = vip_raw in ("1","true","yes","y")
                        try: d.break_after_hours = float(r.get(c_bh, d.break_after_hours or 6.0) or 6.0)
                        except: d.break_after_hours = 6.0
                        try: d.break_minutes = int(float(r.get(c_bm, d.break_minutes or 30) or 30))
                        except: d.break_minutes = 30
                        d.shift_start = str(r.get(c_ss, d.shift_start or "08:00") or "08:00")
                        d.shift_end = str(r.get(c_se, d.shift_end or "16:00") or "16:00")
                        act_raw = str(r.get(c_act, d.active)).strip().lower()
                        d.active = act_raw in ("", "1", "true", "yes", "y")
                st.success(f"Imported drivers. Added: {added}, Updated: {updated}.")
        except Exception as e:
            st.exception(e)

# ---------------- Fleet tab ----------------
with tab_fleet:
    st.subheader("Vehicles (DB)")
    DEFAULT_VEHICLE_IDS = ["209","768","251","179","180","874","875","091","281","280","S-klasse","525","979","516","225"]
    with session_scope() as s:
        existing = {v.car_id for v in s.query(Vehicle).all()}
        to_seed = [cid for cid in DEFAULT_VEHICLE_IDS if cid not in existing]
        if to_seed:
            for cid in to_seed:
                s.add(Vehicle(car_id=cid, is_sklasse=(cid.lower()=="s-klasse"), active=(cid.lower()!="s-klasse")))
            st.info(f"Seeded vehicles: {', '.join(to_seed)} (S-klasse default inactive)")

    with session_scope() as s:
        df = pd.DataFrame([{"car_id": v.car_id, "is_sklasse": v.is_sklasse, "active": v.active} for v in s.query(Vehicle).order_by(Vehicle.car_id).all()])
    st.dataframe(df, use_container_width=True)

    with session_scope() as s:
        col1, col2, col3 = st.columns(3)
        with col1:
            toggle = st.text_input("Toggle Active (car_id)")
            if st.button("Toggle Active"):
                v = s.query(Vehicle).filter(Vehicle.car_id == toggle).first()
                if v: v.active = not v.active; st.success(f"{v.car_id} active={v.active}")
                else: st.error("car_id not found.")
        with col2:
            mk = st.text_input("Toggle S-klasse (car_id)")
            if st.button("Toggle S-klasse"):
                v = s.query(Vehicle).filter(Vehicle.car_id == mk).first()
                if v: v.is_sklasse = not v.is_sklasse; st.success(f"{v.car_id} is_sklasse={v.is_sklasse}")
                else: st.error("car_id not found.")
        with col3:
            st.caption("Only active cars count toward concurrent limit.")

# ---------------- Allocation tab ----------------
with tab_alloc:
    st.subheader("1) Upload bookings")
    bookings_file = st.file_uploader("Bookings CSV", type=["csv"])

    st.subheader("2) Select drivers for this schedule")
    with session_scope() as s:
        all_drivers = s.query(Driver).filter(Driver.active==True).order_by(Driver.driver_id).all()
    if not all_drivers:
        st.info("No active drivers in DB. Add them on the Drivers tab.")
    else:
        selected_ids = st.multiselect("Pick drivers", [d.driver_id for d in all_drivers],
                                      default=[d.driver_id for d in all_drivers])
        st.markdown("**Optional per-run shift overrides**")
        overrides = {}
        cols = st.columns(4)
        for i, d in enumerate([d for d in all_drivers if d.driver_id in selected_ids]):
            with cols[i % 4]:
                st.caption(f"{d.driver_id} · {d.name}")
                os = st.text_input(f"{d.driver_id} start", key=f"a_{d.driver_id}_start", placeholder=d.shift_start)
                oe = st.text_input(f"{d.driver_id} end", key=f"a_{d.driver_id}_end", placeholder=d.shift_end)
                overrides[d.driver_id] = {"start": os or d.shift_start, "end": oe or d.shift_end}

    st.subheader("3) Rules")
    c1, c2, c3 = st.columns(3)
    with c1:
        airport_kw = st.text_input("Airport keywords", ", ".join(DEFAULT_AIRPORT_KEYWORDS))
        cph_kw = st.text_input("Copenhagen keywords", ", ".join(DEFAULT_CPH_KEYWORDS))
        far_kw = st.text_input("Far-away keywords", ", ".join(DEFAULT_FAR_AWAY_KEYWORDS))
    with c2:
        min_gap_cph = st.number_input("CPH: min gap after airport pickup (min)", 0, 240, 50, 5)
        min_gap_far_air = st.number_input("Far: min gap after airport pickup (min)", 0, 240, 60, 5)
        min_gap_far = st.number_input("Far: general min gap (min)", 0, 240, 15, 5)
    with c3:
        require_vehicle = st.checkbox("Require vehicle match", value=True)
        vip_levels = st.text_input("VIP levels preferring VIP drivers", "Super VIP")

    st.subheader("4) Shift flexibility")
    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        early_soft = st.number_input("Start up to (soft) earlier, min", 0, 180, 30, 5)
    with colf2:
        early_hard = st.number_input("If needed: start up to (hard) earlier, min", 0, 180, 60, 5)
    with colf3:
        late_allow = st.number_input("Allow end overrun (minutes)", 0, 120, 20, 5)

    st.subheader("5) Routing (minutes & km)")
    use_rt = st.checkbox("Use routing", value=True)
    provider = st.selectbox("Provider", ["OSRM (free)", "Google Distance Matrix"])
    osrm_url = st.text_input("OSRM URL", "https://router.project-osrm.org")
    gmaps_key = st.text_input("Google API Key", type="password")
    geocode_missing = st.checkbox("Geocode missing coords (Google)", value=False)
    far_km = st.number_input("Far-away distance threshold (km)", 1, 300, 45, 1)
    airport_wait = st.number_input("Airport wait add (min) when far-away", 0, 120, 30, 5)

    if st.button("Run allocation"):
        if bookings_file is None or not selected_ids:
            st.error("Upload bookings and select at least one driver.")
        else:
            rows = []
            for d in all_drivers:
                if d.driver_id not in selected_ids: continue
                o = overrides.get(d.driver_id, {})
                rows.append({
                    "driver_id": d.driver_id, "name": d.name, "vehicle_types": d.vehicle_types, "is_vip": d.is_vip,
                    "break_after_hours": d.break_after_hours, "break_minutes": d.break_minutes,
                    "shift_start": o.get("start", d.shift_start), "shift_end": o.get("end", d.shift_end),
                })
            drivers_df = pd.DataFrame(rows)
            bookings = pd.read_csv(bookings_file, dtype=str)

            provider_obj = None
            if use_rt:
                if provider == "OSRM (free)":
                    provider_obj = OSRMProvider(osrm_url)
                else:
                    provider_obj = GoogleDMProvider(gmaps_key) if gmaps_key else None

            out = allocate_with_shift_flex(
                bookings=bookings, drivers=drivers_df,
                early_start_soft_min=int(early_soft), early_start_hard_min=int(early_hard), late_end_allow_min=int(late_allow),
                min_gap_after_airport_cph_min=int(min_gap_cph), min_gap_after_airport_far_min=int(min_gap_far_air),
                min_general_gap_far_min=int(min_gap_far), require_vehicle_match=bool(require_vehicle),
                vip_driver_preferred_for_levels=tuple([v.strip().lower() for v in vip_levels.split(",") if v.strip()]),
                enable_breaks=True,
                airport_keywords=[k.strip().lower() for k in airport_kw.split(",") if k.strip()],
                cph_keywords=[k.strip().lower() for k in cph_kw.split(",") if k.strip()],
                far_keywords=[k.strip().lower() for k in far_kw.split(",") if k.strip()],
                use_routing=bool(use_rt), routing_provider=provider_obj,
                geocode_with_google_api_key=(gmaps_key if (use_rt and geocode_missing and gmaps_key) else None),
                far_distance_km_threshold=float(far_km), airport_wait_min_estimate=int(airport_wait),
            )

            st.success("Allocation done.")
            def _hi(r):
                return ['background-color: #fff3cd' if (r.get('Hard Early Start','')=='Yes') else '' for _ in r]
            try:
                st.dataframe(out.head(300).style.apply(_hi, axis=1), use_container_width=True)
            except Exception:
                st.dataframe(out.head(300), use_container_width=True)

            only_hard = st.checkbox("Show only HARD early-start rows", value=False)
            if only_hard:
                st.dataframe(out[out["Hard Early Start"]=="Yes"], use_container_width=True)

            st.download_button("Download assignments CSV",
                               data=out.to_csv(index=False).encode("utf-8"),
                               file_name="assignments.csv", mime="text/csv")

            st.subheader("Gantt")
            if "__pickup_dt" in out.columns:
                g = out[["__pickup_dt","__busy_minutes","Allocated Driver Name"]].copy()
                g = g[g["Allocated Driver Name"] != ""]
                if not g.empty:
                    g["start"] = pd.to_datetime(g["__pickup_dt"])
                    g["finish"] = g["start"] + pd.to_timedelta(g["__busy_minutes"].astype(int), unit="m")
                    g.rename(columns={"Allocated Driver Name": "Driver"}, inplace=True)
                    fig = px.timeline(g, x_start="start", x_end="finish", y="Driver", color="Driver")
                    fig.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No assigned jobs to chart.")
            else:
                st.info("No pickup datetimes to chart.")

# ---------------- Car Plan tab ----------------
with tab_plan:
    st.subheader("Build Car Plan (BILPLAN)")
    service_date = st.date_input("Service date", value=date.today())

    with session_scope() as s:
        drivers = [d for d in s.query(Driver).filter(Driver.active==True).order_by(Driver.driver_id).all()]
        vehicles = [v for v in s.query(Vehicle).filter(Vehicle.active==True).order_by(Vehicle.car_id).all()]

    if not drivers or not vehicles:
        st.info("Need at least one active driver and one ACTIVE vehicle.")
    else:
        sel = st.multiselect("Select drivers for today", [f"{d.driver_id} · {d.name}" for d in drivers],
                             default=[f"{d.driver_id} · {d.name}" for d in drivers])
        st.markdown("**Optional per-driver shift overrides**")
        overrides = {}
        cols = st.columns(4)
        bykey = {f"{d.driver_id} · {d.name}": d for d in drivers}
        selected = [bykey[k] for k in sel if k in bykey]
        for i, d in enumerate(selected):
            with cols[i % 4]:
                st.caption(f"{d.driver_id} · {d.name}")
                os = st.text_input(f"{d.driver_id} start", key=f"cp_{d.driver_id}_start", placeholder=d.shift_start)
                oe = st.text_input(f"{d.driver_id} end", key=f"cp_{d.driver_id}_end", placeholder=d.shift_end)
                overrides[d.driver_id] = {"start": os or d.shift_start, "end": oe or d.shift_end}

        avoid_sklasse = st.checkbox("Avoid 'S-klasse' unless needed", value=True)
        handover_minutes = st.number_input("Handover buffer (charge/wash) minutes", 30, 240, 75, 5)

        if st.button("Generate plan"):
            def parse_hhmm(s):
                for fmt in ("%H:%M","%H.%M","%H%M"):
                    try: return datetime.strptime(s, fmt).time()
                    except: pass
                return None

            windows = []
            for d in selected:
                t1 = parse_hhmm(overrides[d.driver_id]["start"])
                t2 = parse_hhmm(overrides[d.driver_id]["end"])
                if t1 and t2:
                    start_dt = datetime.combine(service_date, t1)
                    end_dt   = datetime.combine(service_date, t2)
                    if end_dt <= start_dt: end_dt += timedelta(days=1)
                    windows.append({"driver_id": d.driver_id, "name": d.name, "start": start_dt, "end": end_dt})

            regular_cars = [v.car_id for v in vehicles if not v.is_sklasse]
            s_cars = [v.car_id for v in vehicles if v.is_sklasse]
            cars = regular_cars + (s_cars if not avoid_sklasse else [])

            cars_in_use = {}  # car_id -> free_at
            assignments = []

            def find_free_car(at_time):
                free = []
                for cid in cars:
                    free_at = cars_in_use.get(cid, None)
                    if free_at is None or free_at <= at_time:
                        free.append(cid)
                return free[0] if free else None

            def choose_donor(at_time):
                # car that frees the earliest after at_time
                busy = [(cid, free_at) for cid, free_at in cars_in_use.items() if free_at and free_at > at_time]
                if not busy: return None, None
                busy.sort(key=lambda x: x[1])
                return busy[0]

            windows.sort(key=lambda w: w["start"])

            for w in windows:
                start, end = w["start"], w["end"]
                cid = find_free_car(start)
                if cid is not None:
                    assignments.append((w["name"], cid, start, end, ""))
                    cars_in_use[cid] = end + timedelta(minutes=handover_minutes)
                else:
                    donor_car, free_by = choose_donor(start)
                    if donor_car is None:
                        assignments.append((w["name"], "(no car)", start, end, ""))
                        continue
                    new_start = free_by + timedelta(minutes=handover_minutes)
                    assignments.append((w["name"], donor_car, new_start, end, f"after handover from {donor_car}"))
                    cars_in_use[donor_car] = end + timedelta(minutes=handover_minutes)

            lines = [f"BILPLAN {service_date.strftime('%d/%m/%y')}"]
            assignments.sort(key=lambda x: x[2])
            for name, cid, sdt, edt, note in assignments:
                span = f"{sdt.strftime('%H:%M')}-{edt.strftime('%H:%M')}"
                suffix = f" ({note})" if note else ""
                lines.append(f"{name} - {cid} fra {span}{suffix}")

            bilplan_text = "\n".join(lines)
            st.text_area("BILPLAN", bilplan_text, height=400)
            st.download_button("Download BILPLAN (.txt)", data=bilplan_text.encode("utf-8"),
                               file_name=f"bilplan_{service_date.isoformat()}.txt", mime="text/plain")

            df_out = pd.DataFrame([{
                "name": name, "car_id": cid,
                "start": sdt.strftime("%Y-%m-%d %H:%M"),
                "end": edt.strftime("%Y-%m-%d %H:%M"),
                "note": note
            } for name, cid, sdt, edt, note in assignments])
            st.dataframe(df_out, use_container_width=True)
            st.download_button("Download Car Plan CSV", data=df_out.to_csv(index=False).encode("utf-8"),
                               file_name=f"car_plan_{service_date.isoformat()}.csv", mime="text/csv")
