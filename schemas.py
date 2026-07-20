import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field

# ====================
# SHARED SCHEMAS
# ====================
class UserCreate(BaseModel):
    full_name: str
    email: str
    password: str
    role: str = "doctor"

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str

class PatientCreate(BaseModel):
    full_name: str
    age: int = Field(..., ge=1, le=120)
    gender: str
    comorbidities: List[str] = []

class PatientResponse(BaseModel):
    id: uuid.UUID
    full_name: str
    age: int
    gender: str
    comorbidities: List[str]
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

# ====================
# OVERRIDE SCHEMAS
# ====================
class BaseOverrideRequest(BaseModel):
    clinician_override: str
    override_notes: Optional[str] = None

class FractureOverrideRequest(BaseOverrideRequest): pass
class ArthritisOverrideRequest(BaseOverrideRequest): pass
class OsteoporosisOverrideRequest(BaseOverrideRequest): pass
class TBOverrideRequest(BaseOverrideRequest): pass
class LungNoduleOverrideRequest(BaseOverrideRequest): pass
class BrainTumorOverrideRequest(BaseOverrideRequest): pass
class BrainHemorrhageOverrideRequest(BaseOverrideRequest): pass
class RetinopathyOverrideRequest(BaseOverrideRequest): pass

# ====================
# MODULE 1: FRACTURE
# ====================
class FracturePrognosisResponse(BaseModel):
    rest_weeks_min: int
    rest_weeks_max: int
    cast_type: str
    plaster_required: bool
    weight_bearing_status: str
    referral_flag: str
    model_config = ConfigDict(from_attributes=True)

class FractureResponse(BaseModel):
    id: uuid.UUID
    patient_id: uuid.UUID
    upload_timestamp: datetime
    original_file_path: str
    fracture_detected: bool
    body_region: str
    confidence: float
    confidence_flag: str
    heatmap_path: Optional[str]
    model_version: str
    report_path: Optional[str]
    clinician_override: Optional[str]
    override_notes: Optional[str]
    override_timestamp: Optional[datetime]
    prognosis: Optional[FracturePrognosisResponse] = None
    model_config = ConfigDict(from_attributes=True)

# ====================
# MODULE 2: ARTHRITIS
# ====================
class ArthritisResponse(BaseModel):
    id: uuid.UUID
    patient_id: uuid.UUID
    upload_timestamp: datetime
    original_file_path: str
    grade: int
    grade_name: str
    confidence: float
    confidence_flag: str
    heatmap_path: Optional[str]
    clinical_recommendation: str
    model_version: str
    report_path: Optional[str]
    clinician_override: Optional[str]
    override_notes: Optional[str]
    override_timestamp: Optional[datetime]
    model_config = ConfigDict(from_attributes=True)

# ====================
# MODULE 3: OSTEOPOROSIS
# ====================
class OsteoporosisResponse(BaseModel):
    id: uuid.UUID
    patient_id: uuid.UUID
    upload_timestamp: datetime
    original_file_path: str
    has_osteoporosis: bool
    confidence: float
    confidence_flag: str
    heatmap_path: Optional[str]
    clinical_recommendation: str
    model_version: str
    report_path: Optional[str]
    clinician_override: Optional[str]
    override_notes: Optional[str]
    override_timestamp: Optional[datetime]
    model_config = ConfigDict(from_attributes=True)

# ====================
# MODULE 4: TB
# ====================
class TBResponse(BaseModel):
    id: uuid.UUID
    patient_id: uuid.UUID
    upload_timestamp: datetime
    original_file_path: str
    has_tb: bool
    tb_probability: float
    confidence: float
    confidence_flag: str
    urgency: str
    heatmap_path: Optional[str]
    clinical_recommendation: str
    model_version: str
    report_path: Optional[str]
    clinician_override: Optional[str]
    override_notes: Optional[str]
    override_timestamp: Optional[datetime]
    model_config = ConfigDict(from_attributes=True)

# ====================
# MODULE 5: LUNG NODULE
# ====================
class LungNoduleResponse(BaseModel):
    id: uuid.UUID
    patient_id: uuid.UUID
    upload_timestamp: datetime
    original_file_path: str
    has_nodule: bool
    nodule_probability: float
    confidence: float
    confidence_flag: str
    urgency: str
    heatmap_path: Optional[str]
    clinical_recommendation: str
    model_version: str
    report_path: Optional[str]
    clinician_override: Optional[str]
    override_notes: Optional[str]
    override_timestamp: Optional[datetime]
    model_config = ConfigDict(from_attributes=True)

# ====================
# MODULE 6: BRAIN TUMOR
# ====================
class BrainTumorResponse(BaseModel):
    id: uuid.UUID
    patient_id: uuid.UUID
    upload_timestamp: datetime
    original_file_path: str
    tumor_detected: bool
    tumor_type: str
    confidence: float
    confidence_flag: str
    glioma_risk_flag: bool
    all_probabilities: Dict[str, float]
    urgency: str
    heatmap_path: Optional[str]
    clinical_recommendation: str
    model_version: str
    report_path: Optional[str]
    clinician_override: Optional[str]
    override_notes: Optional[str]
    override_timestamp: Optional[datetime]
    model_config = ConfigDict(from_attributes=True)

# ====================
# MODULE 7: BRAIN HEMORRHAGE
# ====================
class BrainHemorrhageResponse(BaseModel):
    id: uuid.UUID
    patient_id: uuid.UUID
    upload_timestamp: datetime
    original_file_path: str
    has_hemorrhage: bool
    hemorrhage_probability: float
    confidence: float
    confidence_flag: str
    urgency: str
    small_dataset_warning: bool
    heatmap_path: Optional[str]
    clinical_recommendation: str
    model_version: str
    report_path: Optional[str]
    clinician_override: Optional[str]
    override_notes: Optional[str]
    override_timestamp: Optional[datetime]
    model_config = ConfigDict(from_attributes=True)

# ====================
# MODULE 8: BONE AGE
# ====================
class BoneAgeResponse(BaseModel):
    id: uuid.UUID
    patient_id: uuid.UUID
    upload_timestamp: datetime
    original_file_path: str
    predicted_months: float
    predicted_display: str
    uncertainty_months: float
    confidence_flag: str
    chronological_age_months: Optional[int]
    deviation_months: Optional[float]
    skeletal_age_flag: str
    heatmap_path: Optional[str]
    clinical_recommendation: str
    model_version: str
    report_path: Optional[str]
    model_config = ConfigDict(from_attributes=True)

# ====================
# MODULE 9: RETINOPATHY
# ====================
class RetinopathyResponse(BaseModel):
    id: uuid.UUID
    patient_id: uuid.UUID
    upload_timestamp: datetime
    original_file_path: str
    grade: int
    grade_name: str
    confidence: float
    confidence_flag: str
    referable_dr: bool
    referable_risk_flag: bool
    all_probabilities: Any
    urgency: str
    follow_up_months: int
    heatmap_path: Optional[str]
    clinical_recommendation: str
    model_version: str
    report_path: Optional[str]
    clinician_override: Optional[str]
    override_notes: Optional[str]
    override_timestamp: Optional[datetime]
    model_config = ConfigDict(from_attributes=True)
