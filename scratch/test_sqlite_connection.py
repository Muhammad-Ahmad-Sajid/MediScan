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

Patient.comorbidities.property.columns[0].type = SQLiteARRAY()

# Create engine and keep connection open
engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
connection = engine.connect()

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=connection)

# Create tables using connection
Base.metadata.create_all(bind=connection)

# Use one session to insert
db1 = TestingSessionLocal()
p = Patient(full_name="Alice Wayne", age=32, gender="Female", comorbidities=[])
db1.add(p)
db1.commit()
p_id = p.id
db1.close()

# Use a separate session to query
db2 = TestingSessionLocal()
queried_p = db2.query(Patient).filter(Patient.id == p_id).first()
if queried_p:
    print(f"Success! Found patient: {queried_p.full_name}")
else:
    print("Failed: Patient not found!")
db2.close()

# Clean up connection
connection.close()
