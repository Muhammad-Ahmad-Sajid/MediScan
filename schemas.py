from pydantic import BaseModel, Field, field_validator, ConfigDict
from uuid import UUID
from datetime import datetime
from typing import List, Optional


# ------------------------------------------------------------------------------
# Patient Schemas
# ------------------------------------------------------------------------------
class PatientCreate(BaseModel):
    full_name: str = Field(..., description="Full legal name of the patient")
    age: int = Field(..., description="Age in years")
    gender: str = Field(..., description="Gender of the patient (e.g. Male, Female, Other)")
    comorbidities: List[str] = Field(
        default_factory=list, description="Array of health comorbidities"
    )

    @field_validator("age")
    @classmethod
    def validate_age(cls, v: int) -> int:
        if not (1 <= v <= 120):
            raise ValueError("Age must be between 1 and 120")
        return v


class PatientResponse(BaseModel):
    id: UUID
    full_name: str
    age: int
    gender: str
    comorbidities: List[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------------------
# Scan Upload & Inference Schemas
# ------------------------------------------------------------------------------
class ScanUploadResponse(BaseModel):
    scan_id: UUID
    fracture_detected: bool
    bone_region: str
    fracture_confidence: float
    confidence_flag: str
    message: str
    cast_type: Optional[str] = None
    rest_weeks_min: Optional[int] = None
    rest_weeks_max: Optional[int] = None
    plaster_required: Optional[bool] = None
    weight_bearing_status: Optional[str] = None
    referral_flag: Optional[str] = None
    heatmap_url: Optional[str] = None
    report_url: Optional[str] = None
    file_path: Optional[str] = None
    model_version: str


# ------------------------------------------------------------------------------
# Prognosis Override Schemas
# ------------------------------------------------------------------------------
class PrognosisOverrideRequest(BaseModel):
    clinician_override: str = Field(
        ..., description="Name or identifier of the clinician authorizing the override"
    )
    override_notes: str = Field(..., description="Clinical justification notes for the override")


class PrognosisOverrideResponse(BaseModel):
    id: UUID
    prediction_id: UUID
    rest_weeks_min: int
    rest_weeks_max: int
    cast_type: str
    plaster_required: bool
    weight_bearing_status: str
    referral_flag: str
    clinician_override: bool
    override_notes: Optional[str] = None
    override_timestamp: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------------------
# Patient History Detail Schemas
# ------------------------------------------------------------------------------
class PrognosisDetail(BaseModel):
    id: UUID
    rest_weeks_min: int
    rest_weeks_max: int
    cast_type: str
    plaster_required: bool
    weight_bearing_status: str
    referral_flag: str
    clinician_override: bool
    override_notes: Optional[str] = None
    override_timestamp: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class PredictionDetail(BaseModel):
    id: UUID
    fracture_detected: bool
    severity: Optional[str] = None
    confidence_score: float
    heatmap_path: Optional[str] = None
    model_version: str
    prognosis: Optional[PrognosisDetail] = None

    model_config = ConfigDict(from_attributes=True)


class ScanDetail(BaseModel):
    id: UUID
    upload_timestamp: datetime
    original_file_path: str
    bone_affected: str
    image_quality_flag: str
    dataset_source: str
    prediction: Optional[PredictionDetail] = None

    model_config = ConfigDict(from_attributes=True)


class PatientHistoryResponse(BaseModel):
    id: UUID
    full_name: str
    age: int
    gender: str
    comorbidities: List[str]
    created_at: datetime
    scans: List[ScanDetail]

    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------------------
# Arthritis Schemas
# ------------------------------------------------------------------------------
class ArthritisScanResponse(BaseModel):
    scan_id: UUID
    original_file_path: Optional[str] = None
    grade: int
    grade_name: str
    grade_description: str
    confidence: float
    confidence_flag: str
    clinical_recommendation: str
    heatmap_url: Optional[str] = None
    report_url: Optional[str] = None
    model_version: str
    message: str

    model_config = ConfigDict(from_attributes=True)


class ArthritisOverrideRequest(BaseModel):
    clinician_override: str = Field(
        ..., description="Name or identifier of the clinician authorizing the override"
    )
    override_notes: str = Field(..., description="Clinical justification notes for the override")

