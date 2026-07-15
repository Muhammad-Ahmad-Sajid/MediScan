from database import engine
from sqlalchemy import text

with engine.connect() as conn:
    conn.execute(text("ALTER TABLE xray_scans ALTER COLUMN image_quality_flag TYPE VARCHAR(50) USING image_quality_flag::VARCHAR;"))
    conn.commit()
    print("Successfully altered table")
