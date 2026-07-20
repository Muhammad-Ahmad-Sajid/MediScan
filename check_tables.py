from sqlalchemy import create_engine, text
from config import DATABASE_URL

engine = create_engine(DATABASE_URL)
with engine.connect() as conn:
    result = conn.execute(
        text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
        )
    )
    tables = [row[0] for row in result]
    print(f"Found {len(tables)} tables:")
    for t in sorted(tables):
        print(f" - {t}")
