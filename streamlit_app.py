
import os
import pandas as pd
import streamlit as st
from datetime import date
import plotly.express as px

from db import init_db, session_scope, Driver
from allocator import (
    allocate, allocate_with_shift_flex,
    DEFAULT_AIRPORT_KEYWORDS, DEFAULT_CPH_KEYWORDS, DEFAULT_FAR_AWAY_KEYWORDS,
    OSRMProvider, GoogleDMProvider
)

st.set_page_config(page_title="Driver Allocator", page_icon="🚗", layout="wide")
st.title("🚗 Driver Allocator – DB-backed Drivers & Schedule")

# Init DB (SQLite by default, or set DATABASE_URL env var for Postgres/Supabase)
init_db()

tab_alloc, tab_drivers = st.tabs(["🧮 Allocation", "👤 Drivers & Schedule"])

with tab_drivers:
    st.subheader("Drivers (database)")
    with session_scope() as s:
        if "refresh_drivers" not in st.session_state:
            st.session_state.refresh_drivers = 0

        # Add / edit form
        st.markdown("**Add or edit a driver**")
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
            if edit_mode and sel_id:
                d = s.query(Driver).filter(Driver.driver_id == sel_id).first()
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
                    st.error("Selected driver not found.")
            else:
                if not sel_id or not name:
                    st.error("Driver ID and Name are required.")
                else:
                    exists = s.query(Driver).filter(Driver.driver_id == sel_id).first()
                    if exists:
                        st.error("Driver ID already exists. Use edit mode to modify.")
                    else:
                        d = Driver(
                            driver_id=sel_id, name=name, vehicle_types=vehicle_types, is_vip=is_vip,
                            break_after_hours=float(break_after), break_minutes=int(break_minutes),
                            shift_start=shift_start, shift_end=shift_end, active=active
                        )
                        s.add(d)
                        st.success(f"Added driver {sel_id}.")

        st.markdown("---")
        st.markdown("**Current drivers**")
        data = [
            {
                "driver_id": d.driver_id,
                "name": d.name,
                "vehicle_types": d.vehicle_types,
                "is_vip": d.is_vip,
                "break_after_hours": d.break_after_hours,
                "break_minutes": d.break_minutes,
                "shift_start": d.shift_start,
                "shift_end": d.shift_end,
                "active": d.active,
            }
            for d in s.query(Driver).order_by(Driver.driver_id).all()
        ]
        st.dataframe(pd.DataFrame(data))

        # Delete
        del_id = st.text_input("Delete driver ID")
        if st.button("Delete driver"):
            d = s.query(Driver).filter(Driver.driver_id == del_id).first()
            if d:
                s.delete(d)
                st.success(f"Deleted driver {del_id}.")
            else:
                st.error("Driver not found.")

with tab_alloc:
    st.subheader("1) Upload bookings")
    bookings_file = st.file_uploader("Bookings CSV", type=["csv"])

    st.subheader("2) Select drivers for this schedule")
    with session_scope() as s:
        all_drivers = s.query(Driver).filter(Driver.active == True).order_by(Driver.driver_id).all()
    if not all_drivers:
        st.info("No active drivers in database. Add them on the 'Drivers & Schedule' tab.")
    else:
        selected_ids = st.multiselect(
            "Pick drivers for this run",
            [d.driver_id for d in all_drivers],
            default=[d.driver_id for d in all_drivers]
        )

        # Per-run shift overrides
        st.markdown("**Optional per-run shift overrides** (blank = use default)")
        overrides = {}
        cols = st.columns(4)
        for i, d in enumerate([d for d in all_drivers if d.driver_id in selected_ids]):
            with cols[i % 4]:
                st.caption(f"{d.driver_id} · {d.name}")
                os = st.text_input(f"{d.driver_id} start", key=f"{d.driver_id}_start", placeholder=d.shift_start)
                oe = st.text_input(f"{d.driver_id} end", key=f"{d.driver_id}_end", placeholder=d.shift_end)
                overrides[d.driver_id] = {"start": os or d.shift_start, "end": oe or d.shift_end}

    st.subheader("3) Rules")
    c1, c2, c3 = st.columns(3)
    with c1:
        cph_only = st.checkbox("Only allocate Copenhagen pickups", value=False)
        airport_kw = st.text_input("Airport keywords", ", ".join(DEFAULT_AIRPORT_KEYWORDS))
    with c2:
        cph_kw = st.text_input("Copenhagen keywords", ", ".join(DEFAULT_CPH_KEYWORDS))
        far_kw = st.text_input("Far-away keywords", ", ".join(DEFAULT_FAR_AWAY_KEYWORDS))
    with c3:
        require_vehicle = st.checkbox("Require vehicle type match", value=True)
        vip_levels = st.text_input("VIP levels preferring VIP drivers", "Super VIP")

    c4, c5, c6 = st.columns(3)
    with c4:
        min_gap_cph = st.number_input("CPH: min gap after airport pickup (min)", 0, 240, 50, 5)
    with c5:
        min_gap_far_air = st.number_input("Far-away: min gap after airport pickup (min)", 0, 240, 60, 5)
    with c6:
        min_gap_far = st.number_input("Far-away: general min gap (min)", 0, 240, 15, 5)

    st.subheader("4) Shift flexibility")
    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        early_soft = st.number_input("Start up to (soft) earlier, min", 0, 180, 30, 5)
    with colf2:
        early_hard = st.number_input("If needed: start up to (hard) earlier, min", 0, 180, 60, 5,
                                     help="Only used if a job can't be assigned otherwise.")
    with colf3:
        late_allow = st.number_input("Allow end overrun (minutes)", 0, 120, 20, 5)

    st.subheader("5) Travel time")
    use_tt = st.checkbox("Use travel-time spacing", value=True)
    provider = st.selectbox("Provider", ["OSRM (free)", "Google Distance Matrix"])
    osrm_url = st.text_input("OSRM URL", value="https://router.project-osrm.org")
    gmaps_key = st.text_input("Google API Key", type="password")
    geocode_missing = st.checkbox("Geocode missing coords (Google)", value=False)

    if st.button("Run allocation"):
        if bookings_file is None or not selected_ids:
            st.error("Upload bookings and select at least one driver.")
        else:
            # Build a drivers DataFrame from DB + overrides + selection
            rows = []
            for d in all_drivers:
                if d.driver_id not in selected_ids:
                    continue
                o = overrides.get(d.driver_id, {})
                rows.append({
                    "driver_id": d.driver_id,
                    "name": d.name,
                    "vehicle_types": d.vehicle_types,
                    "is_vip": d.is_vip,
                    "break_after_hours": d.break_after_hours,
                    "break_minutes": d.break_minutes,
                    "shift_start": o.get("start", d.shift_start),
                    "shift_end": o.get("end", d.shift_end),
                })
            drivers_df = pd.DataFrame(rows)

            bookings = pd.read_csv(bookings_file, dtype=str)

            # Providers
            provider_obj = None
            if use_tt:
                if provider == "OSRM (free)":
                    provider_obj = OSRMProvider(osrm_url)
                else:
                    provider_obj = GoogleDMProvider(gmaps_key) if gmaps_key else None

            out = allocate_with_shift_flex(
                bookings=bookings,
                drivers=drivers_df,
                early_start_soft_min=int(early_soft),
                early_start_hard_min=int(early_hard),
                late_end_allow_min=int(late_allow),
                min_gap_after_airport_cph_min=int(min_gap_cph),
                min_gap_after_airport_far_min=int(min_gap_far_air),
                min_general_gap_far_min=int(min_gap_far),
                require_vehicle_match=bool(require_vehicle),
                vip_driver_preferred_for_levels=tuple([v.strip().lower() for v in vip_levels.split(",") if v.strip()]),
                enable_breaks=True,
                airport_keywords=[k.strip().lower() for k in airport_kw.split(",") if k.strip()],
                cph_keywords=[k.strip().lower() for k in cph_kw.split(",") if k.strip()],
                far_keywords=[k.strip().lower() for k in far_kw.split(",") if k.strip()],
                use_travel_time=bool(use_tt),
                travel_provider=provider_obj,
                geocode_with_google_api_key=(gmaps_key if (use_tt and geocode_missing and gmaps_key) else None),
            )

            st.success("Allocation done.")
            st.dataframe(out.head(200), use_container_width=True)

            total = len(out)
            unassigned = int((out["Unassigned Reason"] != "").sum())
            st.metric("Total bookings", total)
            st.metric("Unassigned", unassigned)

            st.download_button(
                "Download assignments CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="assignments.csv",
                mime="text/csv",
            )

            # Gantt
            st.subheader("Gantt view")
            if "__pickup_dt" in out.columns:
                g = out[["__pickup_dt", "__est_trip_min", "Allocated Driver Name"]].copy()
                g = g[g["Allocated Driver Name"] != ""]
                if not g.empty:
                    g["start"] = pd.to_datetime(g["__pickup_dt"])
                    g["finish"] = g["start"]
                    mask = g["__est_trip_min"].notna()
                    g.loc[mask, "finish"] = g.loc[mask, "start"] + pd.to_timedelta(g.loc[mask, "__est_trip_min"].astype(int), unit="m")
                    g.loc[~mask, "finish"] = g.loc[~mask, "start"] + pd.Timedelta(minutes=30)
                    g.rename(columns={"Allocated Driver Name": "Driver"}, inplace=True)
                    fig = px.timeline(g, x_start="start", x_end="finish", y="Driver", color="Driver")
                    fig.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No assigned jobs to show in Gantt.")
            else:
                st.info("No timing columns for Gantt.")

st.markdown("---")
st.caption("DB default is SQLite (file). Set DATABASE_URL env var for Postgres/Supabase.")
