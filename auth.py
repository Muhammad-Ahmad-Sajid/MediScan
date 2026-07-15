import os
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.security.utils import get_authorization_scheme_param
from sqlalchemy import Column, String, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, ConfigDict
from passlib.context import CryptContext
from jose import JWTError, jwt
from dotenv import load_dotenv

from database import Base, get_db

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here-make-it-long-and-random")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
try:
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))
except ValueError:
    ACCESS_TOKEN_EXPIRE_MINUTES = 480

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ==============================================================================
# DATABASE MODEL
# ==============================================================================
class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    full_name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(200), nullable=False)
    role = Column(String(20), default="doctor")  # 'doctor' or 'admin'
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.current_timestamp())


# ==============================================================================
# PYDANTIC SCHEMAS
# ==============================================================================
class UserCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    role: str = "doctor"


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: uuid.UUID
    full_name: str
    email: EmailStr
    role: str
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    email: Optional[str] = None


# ==============================================================================
# AUTH FUNCTIONS
# ==============================================================================
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            logger.warning("Token validation failed: missing 'sub' (email) in payload.")
            raise credentials_exception
        token_data = TokenData(email=email)
        logger.info(f"Token validation successful for email: {email}")
    except JWTError as e:
        logger.warning(f"Token validation failed: JWTError: {str(e)}")
        raise credentials_exception

    user = db.query(User).filter(User.email == token_data.email).first()
    if user is None:
        logger.warning(f"Token validation failed: User {token_data.email} not found in database.")
        raise credentials_exception
    if not user.is_active:
        logger.warning(f"Token validation failed: User {token_data.email} is inactive.")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")
    return user


def require_doctor(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role not in ["doctor", "admin"]:
        logger.warning(
            f"Access denied: User {current_user.email} (role: {current_user.role}) attempted to access a doctor route."
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions. Doctor role required.",
        )
    return current_user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "admin":
        logger.warning(
            f"Access denied: User {current_user.email} (role: {current_user.role}) attempted to access an admin route."
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions. Admin role required.",
        )
    return current_user


# ==============================================================================
# ROUTER & ROUTES
# ==============================================================================
auth_router = APIRouter(prefix="/auth", tags=["auth"])


@auth_router.post("/register", response_model=UserResponse)
def register(request: Request, user_in: UserCreate, db: Session = Depends(get_db)):
    # Exception: if users table is empty, allow registration without auth
    user_count = db.query(User).count()
    if user_count > 0:
        # Require admin
        authorization = request.headers.get("Authorization")
        if not authorization:
            logger.warning("Registration failed: Missing token and table is not empty.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
            )

        scheme, param = get_authorization_scheme_param(authorization)
        if scheme.lower() != "bearer":
            logger.warning("Registration failed: Invalid authentication scheme.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication scheme"
            )

        current_user = get_current_user(token=param, db=db)
        require_admin(current_user)

    # Check if email exists
    existing_user = db.query(User).filter(User.email == user_in.email).first()
    if existing_user:
        logger.warning(f"Registration failed: Email {user_in.email} already exists.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
        )

    # Create new user
    db_user = User(
        full_name=user_in.full_name,
        email=user_in.email,
        hashed_password=hash_password(user_in.password),
        role=user_in.role,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    logger.info(f"Successfully registered new user: {db_user.email} with role: {db_user.role}")
    return db_user


@auth_router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        logger.warning(f"Login failed for email: {form_data.username}. Invalid credentials.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        logger.warning(f"Login failed for email: {form_data.username}. Account is inactive.")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")

    # Token payload MUST include: sub (email), role, exp (expiry)
    token_data = {"sub": user.email, "role": user.role}
    access_token = create_access_token(data=token_data)
    logger.info(f"Login successful for email: {user.email}")
    return {"access_token": access_token, "token_type": "bearer"}


@auth_router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user


# ==============================================================================
# TEST BLOCK
# ==============================================================================
if __name__ == "__main__":
    # Do not connect to database in the test block
    logger.info("=== Starting auth.py tests ===")

    # 1. Test hash_password and verify_password
    test_password = "SuperSecretPassword123!"
    hashed = hash_password(test_password)

    verify_success = verify_password(test_password, hashed)
    verify_fail = not verify_password("WrongPassword!", hashed)

    if verify_success and verify_fail:
        logger.info("PASS: hash_password and verify_password")
    else:
        logger.error("FAIL: hash_password and verify_password")

    # 2. Test create_access_token and decode it back manually
    test_data = {"sub": "doctor@hospital.com", "role": "doctor"}
    token = create_access_token(test_data)

    try:
        decoded_payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if (
            decoded_payload.get("sub") == "doctor@hospital.com"
            and decoded_payload.get("role") == "doctor"
            and "exp" in decoded_payload
        ):
            logger.info("PASS: create_access_token and decode")
        else:
            logger.error("FAIL: create_access_token and decode (payload mismatch)")
    except Exception as e:
        logger.error(f"FAIL: create_access_token and decode (exception: {e})")

    logger.info("=== Tests finished ===")
