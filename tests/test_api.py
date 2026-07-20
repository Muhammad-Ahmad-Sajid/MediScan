import pytest
import uuid
import json
from httpx import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.types import TypeDecorator, TEXT

# Add project root to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from main import app
from src.database.connection import get_db
from src.database import models as db_models
from src.database.models import Base


# 1. Custom TypeDecorator to serialize lists to JSON strings in SQLite
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
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []


# Swap Patient.comorbidities column type to our SQLiteARRAY type at runtime
db_models.Patient.comorbidities.property.columns[0].type = SQLiteARRAY()

# SQLite in-memory engine
engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})


@pytest.fixture
def anyio_backend():
    """Returns the name of the backend to run tests with (anyio plugin)."""
    return "asyncio"


@pytest.fixture(scope="function")
def db_connection():
    """Keeps a single SQLite connection open and creates all tables on it for the test duration."""
    connection = engine.connect()
    Base.metadata.create_all(bind=connection)
    yield connection
    Base.metadata.drop_all(bind=connection)
    connection.close()


@pytest.fixture(scope="function")
def db(db_connection):
    """Provides a database session bound to the shared SQLite connection."""
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=db_connection
    )
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(autouse=True)
def override_db_dependency(db):
    """Overrides get_db dependency in main FastAPI app to point to the shared test session."""

    def _get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = _get_db
    yield
    app.dependency_overrides.pop(get_db, None)


# 2. Test Cases


@pytest.mark.anyio
async def test_create_patient():
    """Tests that POST /patients/ registers a patient and returns a valid UUID id."""
    patient_payload = {
        "full_name": "Bruce Wayne",
        "age": 35,
        "gender": "Male",
        "comorbidities": ["Osteoporosis"],
    }

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/patients/", json=patient_payload)

        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert uuid.UUID(data["id"])
        assert data["full_name"] == "Bruce Wayne"
        assert data["age"] == 35
        assert data["comorbidities"] == ["Osteoporosis"]


@pytest.mark.anyio
async def test_create_patient_invalid_age():
    """Tests that POST /patients/ with age 150 yields a 422 validation error."""
    patient_payload = {
        "full_name": "Old Wayne",
        "age": 150,
        "gender": "Male",
        "comorbidities": [],
    }

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/patients/", json=patient_payload)

        assert response.status_code == 422
        assert "Age must be between 1 and 120" in response.text


@pytest.mark.anyio
async def test_get_history_nonexistent_patient():
    """Tests that GET /patients/{id}/history returns 404 for a nonexistent patient."""
    nonexistent_uuid = str(uuid.uuid4())

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get(f"/patients/{nonexistent_uuid}/history")

        assert response.status_code == 404
        assert "Patient not found" in response.json()["detail"]


@pytest.mark.anyio
async def test_patch_prognosis_override(db):
    """Tests that PATCH /prognosis/{id}/override updates prognosis details with clinician overrides."""
    # Seed patient
    patient = db_models.Patient(
        full_name="Ellen Ripley", age=32, gender="Female", comorbidities=[]
    )
    db.add(patient)
    db.flush()

    # Seed scan
    scan = db_models.XrayScan(
        patient_id=patient.id,
        original_file_path="uploads/test_scan.png",
        bone_affected="ankle",
        image_quality_flag="Good",
        dataset_source=db_models.DatasetSource.uploaded,
    )
    db.add(scan)
    db.flush()

    # Seed prediction
    prediction = db_models.FracturePrediction(
        scan_id=scan.id,
        fracture_detected=True,
        severity=db_models.Severity.simple,
        confidence_score=0.92,
        model_version="v2.0.0",
    )
    db.add(prediction)
    db.flush()

    # Seed prognosis
    prognosis = db_models.PrognosisResult(
        prediction_id=prediction.id,
        rest_weeks_min=6,
        rest_weeks_max=8,
        cast_type="Short Leg Plaster Cast",
        plaster_required=True,
        weight_bearing_status="Partial weight-bearing",
        referral_flag=db_models.ReferralFlag.conservative,
    )
    db.add(prognosis)
    db.commit()

    prognosis_id = str(prognosis.id)
    override_payload = {
        "clinician_override": "Dr. Sarah Connor",
        "override_notes": "Slight healing acceleration observed. Moving to partial weight-bearing.",
    }

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.patch(
            f"/prognosis/{prognosis_id}/override", json=override_payload
        )

        assert response.status_code == 200
        data = response.json()
        assert data["clinician_override"] is True
        assert "Dr. Sarah Connor" in data["override_notes"]
        assert "healing acceleration" in data["override_notes"]
        assert data["override_timestamp"] is not None
