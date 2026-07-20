import os
import uuid
from pathlib import Path
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    UploadFile,
    File,
    Form,
    Request,
)
from sqlalchemy.orm import Session

from src.config import UPLOAD_FOLDER
from src.database.connection import get_db
from src.database import models as db_models
from src.prognosis.rule_engine import calculate_prognosis
from inference import run_inference

router = APIRouter(prefix="/api/detection", tags=["Fracture Detection"])


@router.post("/scan")
async def detect_fracture(
    request: Request,
    patient_id: str = Form(..., description="UUID string of the patient"),
    bone_affected: str = Form(
        ...,
        description="Clinician-designated bone affected (e.g., Radius, Tibia, Femur)",
    ),
    dataset_source: str = Form(
        "uploaded", description="Source: MURA, FracAtlas, or uploaded"
    ),
    file: UploadFile = File(..., description="Grayscale X-ray image (PNG or JPG)"),
    db: Session = Depends(get_db),
):
    """
    Uploads an X-ray scan, delegates prediction and heatmap generation to the
    inference module, executes the prognosis rules engine, and logs everything to PostgreSQL.
    """
    # 1. Verify patient exists in database
    try:
        patient_uuid = uuid.UUID(patient_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid patient_id format. Must be a UUID.",
        )

    patient = (
        db.query(db_models.Patient).filter(db_models.Patient.id == patient_uuid).first()
    )
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with ID {patient_id} not found.",
        )

    # 2. Save uploaded file to the UPLOAD_FOLDER
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    file_extension = Path(file.filename).suffix
    unique_filename = f"scan_{uuid.uuid4()}{file_extension}"
    saved_file_path = UPLOAD_FOLDER / unique_filename

    try:
        with open(saved_file_path, "wb") as buffer:
            import shutil

            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save file to disk: {e}",
        )

    # 3. Create database scan log record
    rel_file_path = f"uploads/{unique_filename}"

    # Map dataset source string to enum value
    try:
        ds_source_enum = db_models.DatasetSource(dataset_source)
    except ValueError:
        ds_source_enum = db_models.DatasetSource.uploaded

    new_scan = db_models.XrayScan(
        patient_id=patient.id,
        original_file_path=rel_file_path,
        bone_affected=bone_affected,
        image_quality_flag="Good",
        dataset_source=ds_source_enum,
    )
    db.add(new_scan)
    db.flush()  # Populates new_scan.id

    # 4. Delegate to the inference module (run_inference handles model execution and Grad-CAM)
    try:
        inference_result = run_inference(str(saved_file_path))
    except Exception as e:
        db.rollback()
        if saved_file_path.exists():
            os.remove(saved_file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fracture Model inference execution failed: {e}",
        )

    # Adjust paths returned by generate_heatmap to be relative web URLs
    rel_heatmap_path = ""
    if inference_result.heatmap_path:
        rel_heatmap_path = f"heatmaps/{Path(inference_result.heatmap_path).name}"

    # 5. Log ML prediction record in DB
    # Severity is mapped to None if no fracture is detected
    db_severity = None
    if inference_result.fracture_detected:
        try:
            db_severity = db_models.Severity(inference_result.severity)
        except ValueError:
            pass

    new_prediction = db_models.FracturePrediction(
        scan_id=new_scan.id,
        fracture_detected=inference_result.fracture_detected,
        severity=db_severity,
        confidence_score=inference_result.severity_confidence,
        heatmap_path=rel_heatmap_path if rel_heatmap_path else None,
        model_version="v2.0.0",  # v2 corresponds to the multi-task FractureModel
    )
    db.add(new_prediction)
    db.flush()  # Populates new_prediction.id

    # 6. Run Prognosis Rule Engine
    try:
        # Use severity class for prognosis (default to hairline if normal to estimate base metrics,
        # but in a real system we only calculate prognosis if fracture is detected)
        severity_for_prognosis = (
            inference_result.severity
            if inference_result.fracture_detected
            else "hairline"
        )

        prog_data = calculate_prognosis(
            severity=severity_for_prognosis,
            confidence=inference_result.severity_confidence,
            age=patient.age,
            comorbidities=patient.comorbidities,
            bone_affected=bone_affected,  # Use clinician-designated bone affected
        )

        # If no fracture is detected, reset rest weeks and cast type to reflect normal scan
        if not inference_result.fracture_detected:
            prog_data["rest_weeks_min"] = 0
            prog_data["rest_weeks_max"] = 0
            prog_data["cast_type"] = "None"
            prog_data["plaster_required"] = False
            prog_data["weight_bearing_status"] = "Full weight bearing"
            prog_data["referral_flag"] = "conservative"

        new_prognosis = db_models.PrognosisResult(
            prediction_id=new_prediction.id,
            rest_weeks_min=prog_data["rest_weeks_min"],
            rest_weeks_max=prog_data["rest_weeks_max"],
            cast_type=prog_data["cast_type"],
            plaster_required=prog_data["plaster_required"],
            weight_bearing_status=prog_data["weight_bearing_status"],
            referral_flag=db_models.ReferralFlag(prog_data["referral_flag"]),
            clinician_override=False,
        )
        db.add(new_prognosis)
    except Exception as e:
        db.rollback()
        if saved_file_path.exists():
            os.remove(saved_file_path)
        # Clean up generated heatmap on fail
        if rel_heatmap_path:
            abs_heatmap = Path("d:/X-ray ML Model") / rel_heatmap_path
            if abs_heatmap.exists():
                os.remove(abs_heatmap)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prognosis calculation execution failed: {e}",
        )

    # Commit all database records
    db.commit()

    return {
        "status": "success",
        "scan_id": new_scan.id,
        "patient": {
            "name": patient.full_name,
            "age": patient.age,
            "comorbidities": patient.comorbidities,
        },
        "scan": {
            "original_file_path": rel_file_path,
            "bone_affected": bone_affected,
            "dataset_source": dataset_source,
        },
        "prediction": {
            "prediction_id": new_prediction.id,
            "fracture_detected": inference_result.fracture_detected,
            "severity": inference_result.severity,
            "bone_predicted": inference_result.bone_affected,
            "confidence_score": inference_result.severity_confidence,
            "heatmap_path": rel_heatmap_path,
            "model_version": "v2.0.0",
        },
        "prognosis": {
            "rest_weeks_min": new_prognosis.rest_weeks_min,
            "rest_weeks_max": new_prognosis.rest_weeks_max,
            "cast_type": new_prognosis.cast_type,
            "plaster_required": new_prognosis.plaster_required,
            "weight_bearing_status": new_prognosis.weight_bearing_status,
            "referral_flag": new_prognosis.referral_flag.value,
        },
    }
