from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Supabase connection pooling URL (Port 6543)
# Replace with your actual Supabase details if they change
DATABASE_URL = "postgresql://postgres.eafxeoejcedwmotqpvxe:HALABEL12!1@aws-0-eu-north-1.pooler.supabase.com:6543/postgres"

# Allow override from environment variable if set
DATABASE_URL = os.getenv("DATABASE_URL", DATABASE_URL)

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base model for ORM classes
Base = declarative_base()

def init_db():
    # Import models here so they are registered before creating tables
    import models  # Make sure models.py exists
    Base.metadata.create_all(bind=engine)
