from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from uuid import UUID
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime

from src.database.connection import get_db
from src.database import models as db_models

router = APIRouter(prefix="/api/prognosis", tags=["Prognosis Management"])


# ------------------------------------------------------------------------------
# Pydantic Schema for Override Input
# ------------------------------------------------------------------------------
class PrognosisOverride(BaseModel):
    rest_weeks_min: Optional[int] = Field(None, ge=0)
    rest_weeks_max: Optional[int] = Field(None, ge=0)
    cast_type: Optional[str] = Field(None, max_length=100)
    plaster_required: Optional[bool] = None
    weight_bearing_status: Optional[str] = Field(None, max_length=100)
    referral_flag: Optional[str] = Field(None, description="'conservative' or 'surgical'")
    override_notes: str = Field(
        ..., min_length=5, description="Justification for clinical override"
    )


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------


@router.put("/{prognosis_id}/override")
def override_prognosis(
    prognosis_id: UUID, override_in: PrognosisOverride, db: Session = Depends(get_db)
):
    """
    Allows a clinician to manually override recovery prognosis recommendations.
    Logs clinician details, override notes, and updates the database record.
    """
    prognosis = (
        db.query(db_models.PrognosisResult)
        .filter(db_models.PrognosisResult.id == prognosis_id)
        .first()
    )

    if not prognosis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Prognosis record not found."
        )

    # Validate referral flag if provided
    if override_in.referral_flag:
        try:
            ref_enum = db_models.ReferralFlag(override_in.referral_flag.lower())
            prognosis.referral_flag = ref_enum
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid referral_flag. Must be 'conservative' or 'surgical'.",
            )

    # Apply clinical overrides
    if override_in.rest_weeks_min is not None:
        prognosis.rest_weeks_min = override_in.rest_weeks_min
    if override_in.rest_weeks_max is not None:
        prognosis.rest_weeks_max = override_in.rest_weeks_max
    if override_in.cast_type is not None:
        prognosis.cast_type = override_in.cast_type
    if override_in.plaster_required is not None:
        prognosis.plaster_required = override_in.plaster_required
    if override_in.weight_bearing_status is not None:
        prognosis.weight_bearing_status = override_in.weight_bearing_status

    # Set clinician audit fields
    prognosis.clinician_override = True
    prognosis.override_notes = override_in.override_notes
    prognosis.override_timestamp = datetime.utcnow()

    # Commit transactions
    db.commit()
    db.refresh(prognosis)

    return {
        "status": "success",
        "message": "Clinician override applied successfully",
        "prognosis": {
            "prognosis_id": prognosis.id,
            "rest_weeks_min": prognosis.rest_weeks_min,
            "rest_weeks_max": prognosis.rest_weeks_max,
            "cast_type": prognosis.cast_type,
            "plaster_required": prognosis.plaster_required,
            "weight_bearing_status": prognosis.weight_bearing_status,
            "referral_flag": prognosis.referral_flag.value,
            "clinician_override": prognosis.clinician_override,
            "override_notes": prognosis.override_notes,
            "override_timestamp": prognosis.override_timestamp,
        },
    }
