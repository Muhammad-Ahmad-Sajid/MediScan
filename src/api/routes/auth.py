from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/auth", tags=["Clinician Authentication"])


class ClinicianLogin(BaseModel):
    username: str = Field(..., example="doctor_sajid")
    password: str = Field(..., example="securepass123")


class ClinicianRegister(BaseModel):
    username: str = Field(..., example="doctor_sajid")
    full_name: str = Field(..., example="Dr. Ahmad Sajid")
    license_number: str = Field(..., example="MC-987654")
    password: str = Field(..., example="securepass123")


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register_clinician(clinician: ClinicianRegister):
    """Registers a clinician account for system access."""
    # Simplified authentication mock
    return {
        "status": "success",
        "message": f"Clinician account for {clinician.full_name} registered successfully.",
        "username": clinician.username,
    }


@router.post("/login")
def login_clinician(credentials: ClinicianLogin):
    """Authenticates a clinician and issues an access token."""
    # Simplified authentication mock checking default credentials
    if (
        credentials.username == "doctor_sajid"
        and credentials.password == "securepass123"
    ):
        return {
            "status": "success",
            "access_token": "mock_jwt_session_token_xyz123",
            "token_type": "bearer",
            "clinician": {
                "username": "doctor_sajid",
                "name": "Dr. Ahmad Sajid",
                "role": "Clinician",
            },
        }
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials. Try doctor_sajid / securepass123.",
    )
