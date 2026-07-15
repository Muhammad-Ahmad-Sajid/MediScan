"""
Root-level database gateway, exposing SQLAlchemy engine, session maker,
base class, utility functions, and ORM models.
"""

from src.database.connection import engine, SessionLocal, get_db
from src.database.models import (
    Base,
    DatasetSource,
    Severity,
    ReferralFlag,
    Patient,
    XrayScan,
    FracturePrediction,
    PrognosisResult,
)

__all__ = [
    "engine",
    "SessionLocal",
    "get_db",
    "Base",
    "DatasetSource",
    "Severity",
    "ReferralFlag",
    "Patient",
    "XrayScan",
    "FracturePrediction",
    "PrognosisResult",
]
