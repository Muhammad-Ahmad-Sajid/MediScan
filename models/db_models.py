import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSON
from sqlalchemy.orm import relationship
from database import Base, engine

class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    full_name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="doctor")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Patient(Base):
    __tablename__ = "patients"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    full_name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(10), nullable=False)
    comorbidities = Column(ARRAY(String), default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships for backref convenience
    fracture_scans = relationship("FractureScan", back_populates="patient", cascade="all, delete-orphan")
    arthritis_scans = relationship("ArthritisScan", back_populates="patient", cascade="all, delete-orphan")
    osteoporosis_scans = relationship("OsteoporosisScan", back_populates="patient", cascade="all, delete-orphan")
    tb_scans = relationship("TBScan", back_populates="patient", cascade="all, delete-orphan")
    lung_nodule_scans = relationship("LungNoduleScan", back_populates="patient", cascade="all, delete-orphan")
    brain_tumor_scans = relationship("BrainTumorScan", back_populates="patient", cascade="all, delete-orphan")
    brain_hemorrhage_scans = relationship("BrainHemorrhageScan", back_populates="patient", cascade="all, delete-orphan")
    bone_age_scans = relationship("BoneAgeScan", back_populates="patient", cascade="all, delete-orphan")
    retinopathy_scans = relationship("RetinopathyScan", back_populates="patient", cascade="all, delete-orphan")

class FractureScan(Base):
    __tablename__ = "fracture_scans"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"))
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    original_file_path = Column(String(500))
    fracture_detected = Column(Boolean)
    body_region = Column(String(50))
    confidence = Column(Float)
    confidence_flag = Column(String(20))
    heatmap_path = Column(String(500), nullable=True)
    model_version = Column(String(50))
    report_path = Column(String(500), nullable=True)
    clinician_override = Column(String(50), nullable=True)
    override_notes = Column(Text, nullable=True)
    override_timestamp = Column(DateTime, nullable=True)
    
    patient = relationship("Patient", back_populates="fracture_scans")
    prognosis = relationship("FracturePrognosis", back_populates="scan", uselist=False, cascade="all, delete-orphan")

class FracturePrognosis(Base):
    __tablename__ = "fracture_prognosis"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_id = Column(UUID(as_uuid=True), ForeignKey("fracture_scans.id"))
    rest_weeks_min = Column(Integer)
    rest_weeks_max = Column(Integer)
    cast_type = Column(String(100))
    plaster_required = Column(Boolean)
    weight_bearing_status = Column(String(50))
    referral_flag = Column(String(20))
    
    scan = relationship("FractureScan", back_populates="prognosis")

class ArthritisScan(Base):
    __tablename__ = "arthritis_scans"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"))
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    original_file_path = Column(String(500))
    grade = Column(Integer)
    grade_name = Column(String(50))
    confidence = Column(Float)
    confidence_flag = Column(String(20))
    heatmap_path = Column(String(500), nullable=True)
    clinical_recommendation = Column(Text)
    model_version = Column(String(50))
    report_path = Column(String(500), nullable=True)
    clinician_override = Column(String(50), nullable=True)
    override_notes = Column(Text, nullable=True)
    override_timestamp = Column(DateTime, nullable=True)
    
    patient = relationship("Patient", back_populates="arthritis_scans")

class OsteoporosisScan(Base):
    __tablename__ = "osteoporosis_scans"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"))
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    original_file_path = Column(String(500))
    has_osteoporosis = Column(Boolean)
    confidence = Column(Float)
    confidence_flag = Column(String(20))
    heatmap_path = Column(String(500), nullable=True)
    clinical_recommendation = Column(Text)
    model_version = Column(String(50))
    report_path = Column(String(500), nullable=True)
    clinician_override = Column(String(50), nullable=True)
    override_notes = Column(Text, nullable=True)
    override_timestamp = Column(DateTime, nullable=True)
    
    patient = relationship("Patient", back_populates="osteoporosis_scans")

class TBScan(Base):
    __tablename__ = "tb_scans"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"))
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    original_file_path = Column(String(500))
    has_tb = Column(Boolean)
    tb_probability = Column(Float)
    confidence = Column(Float)
    confidence_flag = Column(String(20))
    urgency = Column(String(20))
    heatmap_path = Column(String(500), nullable=True)
    clinical_recommendation = Column(Text)
    model_version = Column(String(50))
    report_path = Column(String(500), nullable=True)
    clinician_override = Column(String(50), nullable=True)
    override_notes = Column(Text, nullable=True)
    override_timestamp = Column(DateTime, nullable=True)
    
    patient = relationship("Patient", back_populates="tb_scans")

class LungNoduleScan(Base):
    __tablename__ = "lung_nodule_scans"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"))
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    original_file_path = Column(String(500))
    has_nodule = Column(Boolean)
    nodule_probability = Column(Float)
    confidence = Column(Float)
    confidence_flag = Column(String(20))
    urgency = Column(String(20))
    heatmap_path = Column(String(500), nullable=True)
    clinical_recommendation = Column(Text)
    model_version = Column(String(50))
    report_path = Column(String(500), nullable=True)
    clinician_override = Column(String(50), nullable=True)
    override_notes = Column(Text, nullable=True)
    override_timestamp = Column(DateTime, nullable=True)
    
    patient = relationship("Patient", back_populates="lung_nodule_scans")

class BrainTumorScan(Base):
    __tablename__ = "brain_tumor_scans"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"))
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    original_file_path = Column(String(500))
    tumor_detected = Column(Boolean)
    tumor_type = Column(String(50))
    confidence = Column(Float)
    confidence_flag = Column(String(20))
    glioma_risk_flag = Column(Boolean, default=False)
    all_probabilities = Column(JSON)
    urgency = Column(String(20))
    heatmap_path = Column(String(500), nullable=True)
    clinical_recommendation = Column(Text)
    model_version = Column(String(50))
    report_path = Column(String(500), nullable=True)
    clinician_override = Column(String(50), nullable=True)
    override_notes = Column(Text, nullable=True)
    override_timestamp = Column(DateTime, nullable=True)
    
    patient = relationship("Patient", back_populates="brain_tumor_scans")

class BrainHemorrhageScan(Base):
    __tablename__ = "brain_hemorrhage_scans"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"))
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    original_file_path = Column(String(500))
    has_hemorrhage = Column(Boolean)
    hemorrhage_probability = Column(Float)
    confidence = Column(Float)
    confidence_flag = Column(String(20))
    urgency = Column(String(20))
    small_dataset_warning = Column(Boolean, default=True)
    heatmap_path = Column(String(500), nullable=True)
    clinical_recommendation = Column(Text)
    model_version = Column(String(50))
    report_path = Column(String(500), nullable=True)
    clinician_override = Column(String(50), nullable=True)
    override_notes = Column(Text, nullable=True)
    override_timestamp = Column(DateTime, nullable=True)
    
    patient = relationship("Patient", back_populates="brain_hemorrhage_scans")

class BoneAgeScan(Base):
    __tablename__ = "bone_age_scans"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"))
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    original_file_path = Column(String(500))
    predicted_months = Column(Float)
    predicted_display = Column(String(50))
    uncertainty_months = Column(Float)
    confidence_flag = Column(String(20))
    chronological_age_months = Column(Integer, nullable=True)
    deviation_months = Column(Float, nullable=True)
    skeletal_age_flag = Column(String(30))
    heatmap_path = Column(String(500), nullable=True)
    clinical_recommendation = Column(Text)
    model_version = Column(String(50))
    report_path = Column(String(500), nullable=True)
    
    patient = relationship("Patient", back_populates="bone_age_scans")

class RetinopathyScan(Base):
    __tablename__ = "retinopathy_scans"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"))
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    original_file_path = Column(String(500))
    grade = Column(Integer)
    grade_name = Column(String(30))
    confidence = Column(Float)
    confidence_flag = Column(String(20))
    referable_dr = Column(Boolean)
    referable_risk_flag = Column(Boolean, default=False)
    all_probabilities = Column(JSON)
    urgency = Column(String(20))
    follow_up_months = Column(Integer)
    heatmap_path = Column(String(500), nullable=True)
    clinical_recommendation = Column(Text)
    model_version = Column(String(50))
    report_path = Column(String(500), nullable=True)
    clinician_override = Column(String(50), nullable=True)
    override_notes = Column(Text, nullable=True)
    override_timestamp = Column(DateTime, nullable=True)
    
    patient = relationship("Patient", back_populates="retinopathy_scans")

# Create all tables in the database
Base.metadata.create_all(bind=engine)
