from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from uuid import UUID
from typing import List
from pydantic import BaseModel, Field
from datetime import datetime

from src.database.connection import get_db
from src.database import models as db_models

router = APIRouter(prefix="/api/records", tags=["Patient Records"])


# ------------------------------------------------------------------------------
# Pydantic Schemas for Input Validation and Serialization
# ------------------------------------------------------------------------------
class PatientCreate(BaseModel):
    full_name: str = Field(..., example="Jane Miller")
    age: int = Field(..., ge=0, le=120, example=34)
    gender: str = Field(..., example="Female")
    comorbidities: List[str] = Field(default_factory=list, example=["None"])


class PatientOut(BaseModel):
    id: UUID
    full_name: str
    age: int
    gender: str
    comorbidities: List[str]
    created_at: datetime

    class Config:
        from_attributes = True


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------


@router.get("/patients", response_model=List[PatientOut])
def get_patients(db: Session = Depends(get_db)):
    """Retrieves all patients registered in the system."""
    patients = db.query(db_models.Patient).all()
    return patients


@router.get("/patients/{patient_id}")
def get_patient_detail(patient_id: UUID, db: Session = Depends(get_db)):
    """
    Retrieves detailed clinical profile of a patient including their
    historical X-ray scans, predictions, and prognosis plans.
    """
    patient = (
        db.query(db_models.Patient).filter(db_models.Patient.id == patient_id).first()
    )
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found"
        )

    # Gather related records
    scans_list = []
    for scan in patient.scans:
        scan_data = {
            "scan_id": scan.id,
            "upload_timestamp": scan.upload_timestamp,
            "original_file_path": scan.original_file_path,
            "bone_affected": scan.bone_affected,
            "image_quality_flag": scan.image_quality_flag,
            "dataset_source": scan.dataset_source.value,
            "prediction": None,
        }

        # If prediction exists
        if scan.prediction:
            pred = scan.prediction
            scan_data["prediction"] = {
                "prediction_id": pred.id,
                "fracture_detected": pred.fracture_detected,
                "severity": pred.severity.value if pred.severity else None,
                "confidence_score": pred.confidence_score,
                "heatmap_path": pred.heatmap_path,
                "model_version": pred.model_version,
                "prognosis": None,
            }

            # If prognosis exists
            if pred.prognosis:
                prog = pred.prognosis
                scan_data["prediction"]["prognosis"] = {
                    "prognosis_id": prog.id,
                    "rest_weeks_min": prog.rest_weeks_min,
                    "rest_weeks_max": prog.rest_weeks_max,
                    "cast_type": prog.cast_type,
                    "plaster_required": prog.plaster_required,
                    "weight_bearing_status": prog.weight_bearing_status,
                    "referral_flag": prog.referral_flag.value,
                    "clinician_override": prog.clinician_override,
                    "override_notes": prog.override_notes,
                    "override_timestamp": prog.override_timestamp,
                }

        scans_list.append(scan_data)

    return {
        "patient_id": patient.id,
        "full_name": patient.full_name,
        "age": patient.age,
        "gender": patient.gender,
        "comorbidities": patient.comorbidities,
        "created_at": patient.created_at,
        "scans": scans_list,
    }


@router.post(
    "/patients", response_model=PatientOut, status_code=status.HTTP_201_CREATED
)
def create_patient(patient_in: PatientCreate, db: Session = Depends(get_db)):
    """Registers a new patient clinical record in the database."""
    new_patient = db_models.Patient(
        full_name=patient_in.full_name,
        age=patient_in.age,
        gender=patient_in.gender,
        comorbidities=patient_in.comorbidities,
    )
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)
    return new_patient
