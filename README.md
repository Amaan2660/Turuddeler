
# Allocator + Fleet + Car Plan (All-in-one)

Includes:
- Allocation with far-away distance rule (>45km + 30m airport wait) and Roadshow/Site Inspection duration blocking
- Routing via OSRM/Google (optional geocoding)
- Shift flexibility (soft 30m, hard 60m) with "Hard Early Start" highlight
- DB-backed Drivers (CSV import) and Vehicles
- Fleet tab with ACTIVE cars control (S-klasse default inactive)
- Car Plan that uses ONLY active cars with handover buffer (default 75m) and avoids S-klasse unless toggled

Run locally:
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Deploy to Streamlit Cloud: push to GitHub → New app → main file `streamlit_app.py`.
Set `DATABASE_URL` in app secrets for Postgres/Supabase.
