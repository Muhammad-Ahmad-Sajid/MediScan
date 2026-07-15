import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = "postgresql+pg8000://postgres:pgadmin4@localhost:5432/fracture_db"

engine = create_engine(DATABASE_URL)

sql = """
CREATE TABLE IF NOT EXISTS arthritis_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id),
    scan_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    original_file_path VARCHAR(500),
    grade INTEGER,
    grade_name VARCHAR(50),
    confidence FLOAT,
    confidence_flag VARCHAR(50),
    heatmap_path VARCHAR(500),
    clinical_recommendation TEXT,
    model_version VARCHAR(50),
    report_path VARCHAR(500),
    clinician_override VARCHAR(50),
    override_notes TEXT,
    override_timestamp TIMESTAMP
);
"""

try:
    with engine.begin() as conn:
        conn.execute(text(sql))
    print("SUCCESS: arthritis_predictions table created successfully in pgAdmin 4!")
except Exception as e:
    print(f"FAILED: {e}")
