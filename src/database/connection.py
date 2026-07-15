import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import Base from models so the schemas are registered for metadata creation
from src.database.models import Base

# Resolve base directory (parent of src/) and load dotenv
BASE_DIR = Path(__file__).resolve().parent.parent.parent
dotenv_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=dotenv_path)

# Retrieve DATABASE_URL from .env
DATABASE_URL = os.getenv(
    "DATABASE_URL", "sqlite:///./cortexray.db"
)

# Create engine for connecting to the PostgreSQL server
engine = create_engine(DATABASE_URL)

# Configure the session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Auto-create tables if they do not exist
try:
    Base.metadata.create_all(bind=engine)
    print("Database tables initialized/verified successfully.")
except Exception as e:
    print("=" * 80)
    print("WARNING: Could not connect to database or auto-create tables.")
    print("Please ensure you have:")
    print("  1. Started your local PostgreSQL server.")
    print("  2. Run 'CREATE DATABASE fracture_db;' inside pgAdmin 4.")
    print("  3. Configured the correct credentials in your '.env' file.")
    print(f"Error details: {e}")
    print("=" * 80)


# Dependency generator to obtain a DB session and clean it up after the request finishes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
