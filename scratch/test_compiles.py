import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))

from sqlalchemy.types import TypeDecorator, TEXT
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class SQLiteARRAY(TypeDecorator):
    impl = TEXT
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return []
        return json.loads(value)


# Import models
from src.database.models import Base, Patient

# Swap type dynamically
Patient.comorbidities.property.columns[0].type = SQLiteARRAY()

engine = create_engine("sqlite:///:memory:")
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

db = TestingSessionLocal()
try:
    p = Patient(
        full_name="Bruce Wayne", age=35, gender="Male", comorbidities=["Osteoporosis", "Diabetes"]
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    print("Insertion Success!")
    print(f"Patient comorbidities: {p.comorbidities} (Type: {type(p.comorbidities)})")
except Exception as e:
    print(f"Error during insertion: {e}")
    db.rollback()
    sys.exit(1)
finally:
    db.close()
