import streamlit as st
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from datetime import datetime, timedelta

# -----------------------------
# DB connection (Supabase - port 6543)
# -----------------------------
DATABASE_URL = st.secrets["DATABASE_URL"]
engine = create_engine(DATABASE_URL)

def load_drivers():
    query = "SELECT id, name, car_model FROM drivers ORDER BY name"
    return pd.read_sql(query, engine)

# -----------------------------
# Ride assignment logic
# -----------------------------
def assign_rides(rides_df, drivers_df, active_cars):
    """
    Assign rides to active drivers with a 1h15m gap between rides per driver.
    Special cases for far bookings, roadshows, site inspections.
    Adds blocked time window to Notes column.
    """
    active_drivers = drivers_df[drivers_df["car_model"].isin(active_cars)].copy()

    rides_df["Pickup Time"] = pd.to_datetime(rides_df["Pickup Time"])
    rides_df = rides_df.sort_values("Pickup Time").reset_index(drop=True)

    rides_df["Assigned Driver"] = ""
    rides_df["Car"] = ""
    rides_df["Notes"] = ""

    driver_next_time = {row["name"]: datetime.min for _, row in active_drivers.iterrows()}

    driver_cycle = active_drivers["name"].tolist()
    driver_idx = 0

    for i, ride in rides_df.iterrows():
        assigned = False
        attempts = 0

        while not assigned and attempts < len(driver_cycle):
            driver_name = driver_cycle[driver_idx]
            car_model = active_drivers.loc[active_drivers["name"] == driver_name, "car_model"].iloc[0]
            pickup_time = ride["Pickup Time"]

            notes = ""
            extra_block = timedelta()

            ride_type = str(ride.get("Type", "")).lower()
            if "far booking" in ride_type:
                extra_block = timedelta(hours=1)
                notes = "Far booking"
            elif "roadshow" in ride_type:
                extra_block = timedelta(hours=2)
                notes = "Roadshow"
            elif "site inspection" in ride_type:
                extra_block = timedelta(hours=1, minutes=30)
                notes = "Site inspection"

            if pickup_time >= driver_next_time[driver_name]:
                next_free = pickup_time + timedelta(hours=1, minutes=15) + extra_block
                rides_df.at[i, "Assigned Driver"] = driver_name
                rides_df.at[i, "Car"] = car_model
                rides_df.at[i, "Notes"] = f"{notes} | Busy until {next_free.strftime('%H:%M')}" if notes else f"Busy until {next_free.strftime('%H:%M')}"

                driver_next_time[driver_name] = next_free
                assigned = True

            driver_idx = (driver_idx + 1) % len(driver_cycle)
            attempts += 1

    return rides_df

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🚖 Daily Car Plan")

drivers_df = load_drivers()

rides_file = st.file_uploader("Upload today's rides CSV", type=["csv"])

if rides_file:
    rides_df = pd.read_csv(rides_file)

    all_cars = drivers_df["car_model"].unique().tolist()
    active_cars = st.multiselect("Select active cars", all_cars, default=[c for c in all_cars if "S-Klasse" not in c])

    s_klasse_drivers = drivers_df[drivers_df["car_model"].str.contains("S-Klasse", case=False)]
    if not s_klasse_drivers.empty:
        s_klasse_driver = st.selectbox("S-Klasse driver", ["None"] + s_klasse_drivers["name"].tolist())
        if s_klasse_driver != "None" and "S-Klasse" not in active_cars:
            active_cars.append("S-Klasse")

    if st.button("Generate Plan"):
        plan_df = assign_rides(rides_df, drivers_df, active_cars)

        st.subheader("Full Day Plan")
        st.dataframe(
            plan_df[["Pickup Time", "Ride ID", "Pickup", "Dropoff", "Assigned Driver", "Car", "Notes"]],
            use_container_width=True
        )

        st.success("✅ Plan generated for the day!")
else:
    st.info("Please upload the day's rides CSV to start planning.")
