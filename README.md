
# Driver Allocator – DB-backed drivers, shift flexibility, travel-time, Gantt

## Database
- Default: SQLite file (`drivers.db` in app working dir).
- Production: set `DATABASE_URL` (e.g. Supabase Postgres). Example:
  `postgresql://USER:PASS@HOST:PORT/DBNAME`

## Run locally
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy (Streamlit Cloud)
Push this folder to a GitHub repo → New app → main file `streamlit_app.py`.
Optionally set `DATABASE_URL` in app secrets to use Postgres.
