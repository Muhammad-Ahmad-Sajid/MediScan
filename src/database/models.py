import uuid
import enum
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Enum, Text
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship, declarative_base

# Create the declarative base
Base = declarative_base()


# ------------------------------------------------------------------------------
# Enums matching database types
# ------------------------------------------------------------------------------
class DatasetSource(str, enum.Enum):
    MURA = "MURA"
    FracAtlas = "FracAtlas"
    uploaded = "uploaded"


class Severity(str, enum.Enum):
    hairline = "hairline"
    simple = "simple"
    displaced = "displaced"
    comminuted = "comminuted"


class ReferralFlag(str, enum.Enum):
    conservative = "conservative"
    surgical = "surgical"


# ------------------------------------------------------------------------------
# SQLAlchemy Models
# ------------------------------------------------------------------------------


class Patient(Base):
    __tablename__ = "patients"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="UUID primary key identifying the patient",
    )
    full_name = Column(String(255), nullable=False, comment="Full legal name of the patient")
    age = Column(Integer, nullable=False, comment="Age of the patient in years")
    gender = Column(
        String(50), nullable=False, comment="Gender of the patient (e.g. Male, Female, Other)"
    )
    comorbidities = Column(
        ARRAY(String),
        default=list,
        nullable=False,
        comment="Array of text items representing patient health comorbidities (e.g., Osteoporosis, Diabetes)",
    )
    created_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
        comment="Timestamp when the patient record was created",
    )

    # Bidirectional One-to-Many Relationship: Patient -> XrayScans
    scans = relationship("XrayScan", back_populates="patient", cascade="all, delete-orphan")


class XrayScan(Base):
    __tablename__ = "xray_scans"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="UUID primary key identifying the X-ray scan",
    )
    patient_id = Column(
        UUID(as_uuid=True),
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
        comment="Foreign key referencing patients(id)",
    )
    upload_timestamp = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
        comment="Timestamp when the image scan was uploaded",
    )
    original_file_path = Column(
        String(512),
        nullable=False,
        comment="Absolute path to the raw X-ray scan file stored on disk",
    )
    bone_affected = Column(
        String(100),
        nullable=False,
        comment="Name of the bone shown in the scan (e.g., Radius, Femur, Tibia)",
    )
    image_quality_flag = Column(
        String(50),
        default="Good",
        nullable=False,
        comment="Status of scan quality (e.g., Good, Blurry, Low Exposure)",
    )
    dataset_source = Column(
        Enum(DatasetSource, name="dataset_source_enum"),
        nullable=False,
        comment="Origin of the image (MURA, FracAtlas, or uploaded by user)",
    )

    # Bidirectional relationships
    patient = relationship("Patient", back_populates="scans")

    # Bidirectional One-to-One: XrayScan -> FracturePrediction
    prediction = relationship(
        "FracturePrediction", back_populates="scan", uselist=False, cascade="all, delete-orphan"
    )


class FracturePrediction(Base):
    __tablename__ = "fracture_predictions"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="UUID primary key identifying the ML prediction",
    )
    scan_id = Column(
        UUID(as_uuid=True),
        ForeignKey("xray_scans.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        comment="Unique foreign key referencing xray_scans(id) (one-to-one)",
    )
    fracture_detected = Column(
        Boolean, nullable=False, comment="True if a bone fracture is detected, False otherwise"
    )
    severity = Column(
        Enum(Severity, name="severity_enum"),
        nullable=True,
        comment="Severity rating of the fracture if detected (NULL if no fracture)",
    )
    confidence_score = Column(
        Float, nullable=False, comment="Prediction confidence score between 0.0 and 1.0"
    )
    heatmap_path = Column(
        String(512),
        nullable=True,
        comment="Path to the generated Grad-CAM heatmap visualization file",
    )
    model_version = Column(
        String(50),
        nullable=False,
        comment="Version tag of the ML model used to generate this prediction",
    )

    # Bidirectional relationships
    scan = relationship("XrayScan", back_populates="prediction")

    # Bidirectional One-to-One: FracturePrediction -> PrognosisResult
    prognosis = relationship(
        "PrognosisResult", back_populates="prediction", uselist=False, cascade="all, delete-orphan"
    )


class PrognosisResult(Base):
    __tablename__ = "prognosis_results"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="UUID primary key identifying the prognosis report",
    )
    prediction_id = Column(
        UUID(as_uuid=True),
        ForeignKey("fracture_predictions.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        comment="Unique foreign key referencing fracture_predictions(id) (one-to-one)",
    )
    rest_weeks_min = Column(
        Integer, nullable=False, comment="Minimum recommended recovery/rest duration in weeks"
    )
    rest_weeks_max = Column(
        Integer, nullable=False, comment="Maximum recommended recovery/rest duration in weeks"
    )
    cast_type = Column(
        String(100),
        nullable=False,
        comment="Type of cast or support suggested for the injury (e.g., Short Arm Cast, Boot)",
    )
    plaster_required = Column(
        Boolean, nullable=False, comment="Boolean flag indicating if a plaster cast is required"
    )
    weight_bearing_status = Column(
        String(100),
        nullable=False,
        comment="Restricted weight bearing instructions (e.g., Non-weight bearing, Partial)",
    )
    referral_flag = Column(
        Enum(ReferralFlag, name="referral_flag_enum"),
        nullable=False,
        comment="Required clinical action: conservative management or surgical referral",
    )
    clinician_override = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Flag indicating if a clinician manually updated these values",
    )
    override_notes = Column(
        Text, nullable=True, comment="Reason or notes for the clinician override"
    )
    override_timestamp = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when the clinician override occurred",
    )

    # Bidirectional relationships
    prediction = relationship("FracturePrediction", back_populates="prognosis")


class ArthritisPrediction(Base):
    __tablename__ = "arthritis_predictions"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="UUID primary key identifying the arthritis prediction",
    )
    patient_id = Column(
        UUID(as_uuid=True),
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
        comment="Foreign key referencing patients(id)",
    )
    scan_timestamp = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
        comment="Timestamp when the scan was run",
    )
    original_file_path = Column(String(512), nullable=True)
    grade = Column(Integer, nullable=True)
    grade_name = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    confidence_flag = Column(String(50), nullable=True)
    heatmap_path = Column(String(512), nullable=True)
    clinical_recommendation = Column(Text, nullable=True)
    model_version = Column(String(50), nullable=True)
    report_path = Column(String(512), nullable=True)
    clinician_override = Column(String(50), nullable=True)
    override_notes = Column(Text, nullable=True)
    override_timestamp = Column(DateTime(timezone=True), nullable=True)
