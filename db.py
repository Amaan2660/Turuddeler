
import os
from contextlib import contextmanager
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///app.db")
engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

class Driver(Base):
    __tablename__ = "drivers"
    id = Column(Integer, primary_key=True)
    driver_id = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    vehicle_types = Column(String, default="")
    is_vip = Column(Boolean, default=False)
    break_after_hours = Column(Float, default=6.0)
    break_minutes = Column(Integer, default=30)
    shift_start = Column(String, default="08:00")
    shift_end = Column(String, default="16:00")
    active = Column(Boolean, default=True)

class Vehicle(Base):
    __tablename__ = "vehicles"
    id = Column(Integer, primary_key=True)
    car_id = Column(String, unique=True, nullable=False)  # e.g., '209', 'S-klasse'
    is_sklasse = Column(Boolean, default=False)
    active = Column(Boolean, default=True)

def init_db():
    Base.metadata.create_all(engine)

@contextmanager
def session_scope():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
