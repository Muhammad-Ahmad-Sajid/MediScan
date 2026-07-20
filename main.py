import os
import uuid
import time
import shutil
import json
import logging
import traceback
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request, APIRouter
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from database import engine, Base, get_db
from models import db_models
import schemas

from auth import auth_router, get_current_doctor, get_current_admin
from auth import UserCreate # using db_models.User in endpoints

# Import existing inference engines
from inference import run_inference as run_fracture_inference
from prognosis_engine import get_prognosis as get_fracture_prognosis
from arthritis_inference import analyze_arthritis as run_arthritis_inference
from osteoporosis_inference import run_inference as run_osteoporosis_inference
from report_generator import generate_report, generate_tb_report

models_loaded = ["fracture", "arthritis", "osteoporosis"]
try:
    from tb_inference import run_tb_inference
    models_loaded.append("tb")
except ImportError as e:
    logger.warning(f"Failed to load TB inference module: {e}")

try:
    from lung_nodule_inference import run_lung_nodule_inference
    models_loaded.append("lung_nodule")
except ImportError as e:
    logger.warning(f"Failed to load Lung Nodule inference module: {e}")

try:
    from brain_tumor_inference import run_brain_tumor_inference
    models_loaded.append("brain_tumor")
except ImportError as e:
    logger.warning(f"Failed to load Brain Tumor inference module: {e}")

try:
    from brain_hemorrhage_inference import run_brain_hemorrhage_inference
    models_loaded.append("brain_hemorrhage")
except ImportError as e:
    logger.warning(f"Failed to load Brain Hemorrhage inference module: {e}")



try:
    from bone_age_inference import run_bone_age_inference
    models_loaded.append("bone_age")
except ImportError as e:
    logger.warning(f"Failed to load Bone Age inference module: {e}")

try:
    from retinopathy_inference import run_retinopathy_inference
    models_loaded.append("retinopathy")
except ImportError as e:
    logger.warning(f"Failed to load Retinopathy inference module: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# FastAPI App
app = FastAPI(title="MediScan AI", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/heatmaps", StaticFiles(directory="heatmaps"), name="heatmaps")
app.mount("/reports", StaticFiles(directory="reports"), name="reports")
app.mount("/static", StaticFiles(directory="templates"), name="templates")

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    logger.error(f"Unhandled server error: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# Health
@app.get("/health", tags=["System"])
def health_check():
    try:
        with engine.connect() as conn: pass
        db_status = "connected"
    except Exception:
        db_status = "error"
        
    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "database": db_status
    }

# UI routes
@app.get("/", tags=["UI"], include_in_schema=False)
def serve_dashboard():
    return FileResponse("templates/index.html")

@app.get("/history", tags=["UI"], include_in_schema=False)
def serve_history():
    return FileResponse("templates/history.html")

# Overrides
@app.get("/overrides", tags=["Admin"])
def get_all_overrides(db: Session = Depends(get_db), current_user = Depends(get_current_admin)):
    # Simple aggregation of overrides
    overrides = []
    return overrides # To be expanded later if needed across all modules

app.include_router(auth_router, prefix="/auth", tags=["Auth"])

# ====================
# PATIENT ROUTER
# ====================
patient_router = APIRouter(prefix="/patients", tags=["Patients"])

@patient_router.post("/", response_model=schemas.PatientResponse)
def create_patient(patient: schemas.PatientCreate, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    db_patient = db_models.Patient(**patient.model_dump())
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

@patient_router.get("/", response_model=List[schemas.PatientResponse])
def get_all_patients(db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    return db.query(db_models.Patient).all()

@patient_router.get("/{patient_id}", response_model=schemas.PatientResponse)
def get_patient(patient_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    p = db.query(db_models.Patient).filter(db_models.Patient.id == patient_id).first()
    if not p: raise HTTPException(404, "Patient not found")
    return p

@patient_router.delete("/{patient_id}")
def delete_patient(patient_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_admin)):
    p = db.query(db_models.Patient).filter(db_models.Patient.id == patient_id).first()
    if not p: raise HTTPException(404, "Patient not found")
    db.delete(p) # Hard delete for now, as is_active not on patient
    db.commit()
    return {"detail": "Deleted"}

app.include_router(patient_router)

# ====================
# MODULE 1: FRACTURE
# ====================
fracture_router = APIRouter(prefix="/fracture", tags=["Fracture"])

@fracture_router.post("/scan/upload", response_model=schemas.FractureResponse)
async def upload_fracture(patient_id: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    p = db.query(db_models.Patient).filter(db_models.Patient.id == patient_id).first()
    if not p: raise HTTPException(404, "Patient not found")
    
    uid = uuid.uuid4().hex[:8]
    filepath = f"uploads/frac_{uid}_{file.filename}".replace("\\", "/")
    with open(filepath, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
        
    res = run_fracture_inference(filepath)
    prog = None
    if res.confidence_flag != "inconclusive":
        prog = get_fracture_prognosis(res.bone_region, "simple" if res.fracture_detected else "hairline", p.age, p.comorbidities)
        
    db_scan = db_models.FractureScan(
        patient_id=p.id, original_file_path=filepath, fracture_detected=res.fracture_detected,
        body_region=res.bone_region, confidence=res.fracture_confidence, confidence_flag=res.confidence_flag,
        heatmap_path=res.heatmap_path, model_version=res.model_version
    )
    db.add(db_scan)
    db.flush()
    
    if prog:
        db_prog = db_models.FracturePrognosis(
            scan_id=db_scan.id, rest_weeks_min=prog.rest_weeks_min, rest_weeks_max=prog.rest_weeks_max,
            cast_type=prog.cast_type, plaster_required=prog.plaster_required, weight_bearing_status=prog.weight_bearing_status,
            referral_flag=prog.referral_flag
        )
        db.add(db_prog)
        
    db.commit()
    db.refresh(db_scan)
    return db_scan

@fracture_router.get("/scan/{scan_id}", response_model=schemas.FractureResponse)
def get_fracture(scan_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    scan = db.query(db_models.FractureScan).filter(db_models.FractureScan.id == scan_id).first()
    if not scan: raise HTTPException(404, "Scan not found")
    return scan

@fracture_router.patch("/{scan_id}/override", response_model=schemas.FractureResponse)
def override_fracture(scan_id: str, req: schemas.FractureOverrideRequest, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    scan = db.query(db_models.FractureScan).filter(db_models.FractureScan.id == scan_id).first()
    if not scan: raise HTTPException(404, "Scan not found")
    scan.clinician_override = req.clinician_override
    scan.override_notes = req.override_notes
    scan.override_timestamp = datetime.utcnow()
    db.commit()
    db.refresh(scan)
    return scan

@app.get("/patients/{patient_id}/fracture-history", tags=["Fracture"])
def fracture_history(patient_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    return db.query(db_models.FractureScan).filter(db_models.FractureScan.patient_id == patient_id).order_by(db_models.FractureScan.upload_timestamp.desc()).all()

app.include_router(fracture_router)

# ====================
# MODULE 2: ARTHRITIS
# ====================
arthritis_router = APIRouter(prefix="/arthritis", tags=["Arthritis"])

@arthritis_router.post("/scan/upload", response_model=schemas.ArthritisResponse)
async def upload_arthritis(patient_id: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    p = db.query(db_models.Patient).filter(db_models.Patient.id == patient_id).first()
    if not p: raise HTTPException(404, "Patient not found")
    
    uid = uuid.uuid4().hex[:8]
    filepath = f"uploads/arth_{uid}_{file.filename}".replace("\\", "/")
    with open(filepath, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
        
    res = run_arthritis_inference(filepath)
    
    db_scan = db_models.ArthritisScan(
        patient_id=p.id, original_file_path=filepath, grade=res.grade, grade_name=res.grade_name,
        confidence=res.confidence, confidence_flag=res.confidence_flag, heatmap_path=res.heatmap_path,
        clinical_recommendation=res.clinical_recommendation, model_version=res.model_version
    )
    db.add(db_scan)
    db.commit()
    db.refresh(db_scan)
    return db_scan

@arthritis_router.get("/scan/{scan_id}", response_model=schemas.ArthritisResponse)
def get_arthritis(scan_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    scan = db.query(db_models.ArthritisScan).filter(db_models.ArthritisScan.id == scan_id).first()
    if not scan: raise HTTPException(404, "Scan not found")
    return scan

@arthritis_router.patch("/{scan_id}/override", response_model=schemas.ArthritisResponse)
def override_arthritis(scan_id: str, req: schemas.ArthritisOverrideRequest, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    scan = db.query(db_models.ArthritisScan).filter(db_models.ArthritisScan.id == scan_id).first()
    if not scan: raise HTTPException(404, "Scan not found")
    scan.clinician_override = req.clinician_override
    scan.override_notes = req.override_notes
    scan.override_timestamp = datetime.utcnow()
    db.commit()
    db.refresh(scan)
    return scan

@app.get("/patients/{patient_id}/arthritis-history", tags=["Arthritis"])
def arthritis_history(patient_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    return db.query(db_models.ArthritisScan).filter(db_models.ArthritisScan.patient_id == patient_id).order_by(db_models.ArthritisScan.upload_timestamp.desc()).all()

app.include_router(arthritis_router)

# ====================
# MODULE 3: OSTEOPOROSIS
# ====================
osteo_router = APIRouter(prefix="/osteoporosis", tags=["Osteoporosis"])

@osteo_router.post("/scan/upload", response_model=schemas.OsteoporosisResponse)
async def upload_osteo(patient_id: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    p = db.query(db_models.Patient).filter(db_models.Patient.id == patient_id).first()
    if not p: raise HTTPException(404, "Patient not found")
    
    uid = uuid.uuid4().hex[:8]
    filepath = f"uploads/osteo_{uid}_{file.filename}".replace("\\", "/")
    with open(filepath, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
        
    res = run_osteoporosis_inference(filepath)
    
    db_scan = db_models.OsteoporosisScan(
        patient_id=p.id, original_file_path=filepath, has_osteoporosis=res.has_osteoporosis,
        confidence=res.confidence, confidence_flag=res.confidence_flag, heatmap_path=res.heatmap_path,
        clinical_recommendation=res.clinical_recommendation, model_version=res.model_version
    )
    db.add(db_scan)
    db.commit()
    db.refresh(db_scan)
    return db_scan

@osteo_router.get("/scan/{scan_id}", response_model=schemas.OsteoporosisResponse)
def get_osteo(scan_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    scan = db.query(db_models.OsteoporosisScan).filter(db_models.OsteoporosisScan.id == scan_id).first()
    if not scan: raise HTTPException(404, "Scan not found")
    return scan

@osteo_router.patch("/{scan_id}/override", response_model=schemas.OsteoporosisResponse)
def override_osteo(scan_id: str, req: schemas.OsteoporosisOverrideRequest, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    scan = db.query(db_models.OsteoporosisScan).filter(db_models.OsteoporosisScan.id == scan_id).first()
    if not scan: raise HTTPException(404, "Scan not found")
    scan.clinician_override = req.clinician_override
    scan.override_notes = req.override_notes
    scan.override_timestamp = datetime.utcnow()
    db.commit()
    db.refresh(scan)
    return scan

@app.get("/patients/{patient_id}/osteoporosis-history", tags=["Osteoporosis"])
def osteo_history(patient_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    return db.query(db_models.OsteoporosisScan).filter(db_models.OsteoporosisScan.patient_id == patient_id).order_by(db_models.OsteoporosisScan.upload_timestamp.desc()).all()

app.include_router(osteo_router)


# ====================
# PLACEHOLDER MODULES 4-9
# ====================

tb_router = APIRouter(prefix="/tb", tags=["TB"])

@tb_router.post("/scan/upload", response_model=schemas.TBResponse)
async def upload_tb(patient_id: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    p = db.query(db_models.Patient).filter(db_models.Patient.id == patient_id).first()
    if not p: raise HTTPException(404, "Patient not found")
    if not file.content_type.startswith("image/"): raise HTTPException(400, "File must be an image")
    
    uid = uuid.uuid4().hex[:8]
    filepath = f"uploads/tb_{uid}_{file.filename}".replace("\\", "/")
    with open(filepath, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
        
    try:
        res = run_tb_inference(filepath)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Inference failed: {str(e)}")
        
    urgency = "routine"
    if res.has_tb:
        urgency = "emergency" if res.confidence_flag == "clear" else "urgent"
        
    db_scan = db_models.TBScan(
        patient_id=p.id, original_file_path=filepath, has_tb=res.has_tb, tb_probability=res.tb_probability,
        confidence=res.confidence, confidence_flag=res.confidence_flag, urgency=urgency, heatmap_path=res.heatmap_path,
        clinical_recommendation=res.clinical_recommendation, model_version=res.model_version
    )
    db.add(db_scan)
    db.commit()
    db.refresh(db_scan)
    return db_scan

@tb_router.get("/scan/{scan_id}", response_model=schemas.TBResponse)
def get_tb(scan_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    scan = db.query(db_models.TBScan).filter(db_models.TBScan.id == scan_id).first()
    if not scan: raise HTTPException(404, "Scan not found")
    return scan

@tb_router.get("/scan/{scan_id}/report")
def get_tb_report(scan_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    scan = db.query(db_models.TBScan).filter(db_models.TBScan.id == scan_id).first()
    if not scan: raise HTTPException(404, "Scan not found")
    if scan.report_path and os.path.exists(scan.report_path):
        return FileResponse(scan.report_path, media_type="application/pdf", filename=f"TB_Report_{scan_id}.pdf")
        
    p = db.query(db_models.Patient).filter(db_models.Patient.id == scan.patient_id).first()
    
    report_path = generate_tb_report(
        patient={"full_name": p.full_name, "age": p.age, "gender": p.gender, "comorbidities": p.comorbidities},
        scan={"scan_id": scan.id, "upload_timestamp": scan.upload_timestamp.strftime("%Y-%m-%d %H:%M:%S") if scan.upload_timestamp else "", "original_file_path": scan.original_file_path},
        inference_result=scan
    )
    
    if not report_path: raise HTTPException(500, "Failed to generate report")
    scan.report_path = report_path
    db.commit()
    return FileResponse(report_path, media_type="application/pdf", filename=f"TB_Report_{scan_id}.pdf")

@tb_router.patch("/{scan_id}/override", response_model=schemas.TBResponse)
def override_tb(scan_id: str, req: schemas.TBOverrideRequest, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    scan = db.query(db_models.TBScan).filter(db_models.TBScan.id == scan_id).first()
    if not scan: raise HTTPException(404, "Scan not found")
    scan.clinician_override = req.clinician_override
    scan.override_notes = req.override_notes
    scan.override_timestamp = datetime.utcnow()
    db.commit()
    db.refresh(scan)
    return scan

@app.get("/patients/{patient_id}/tb-history", tags=["TB"], response_model=List[schemas.TBResponse])
def tb_history(patient_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    return db.query(db_models.TBScan).filter(db_models.TBScan.patient_id == patient_id).order_by(db_models.TBScan.upload_timestamp.desc()).all()

app.include_router(tb_router)

lung_nodule_router = APIRouter(prefix="/lung-nodule", tags=["Lung Nodule"])

@lung_nodule_router.post("/scan/upload", response_model=schemas.LungNoduleResponse)
async def upload_lung_nodule(patient_id: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    p = db.query(db_models.Patient).filter(db_models.Patient.id == patient_id).first()
    if not p: raise HTTPException(404, "Patient not found")
    if not file.content_type.startswith("image/"): raise HTTPException(400, "File must be an image")
    
    uid = uuid.uuid4().hex[:8]
    filepath = f"uploads/lung_nodule_{uid}_{file.filename}".replace("\\", "/")
    with open(filepath, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
        
    try:
        res = run_lung_nodule_inference(filepath)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Inference failed: {str(e)}")
        
    db_scan = db_models.LungNoduleScan(
        patient_id=p.id, original_file_path=filepath, has_nodule=res.has_nodule, nodule_probability=res.nodule_probability,
        confidence=res.confidence, confidence_flag=res.confidence_flag, urgency=res.urgency, heatmap_path=res.heatmap_path,
        clinical_recommendation=res.clinical_recommendation, model_version=res.model_version
    )
    db.add(db_scan)
    db.commit()
    db.refresh(db_scan)
    return db_scan

@lung_nodule_router.get("/scan/{scan_id}", response_model=schemas.LungNoduleResponse)
def get_lung_nodule(scan_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    scan = db.query(db_models.LungNoduleScan).filter(db_models.LungNoduleScan.id == scan_id).first()
    if not scan: raise HTTPException(404, "Scan not found")
    return scan

@lung_nodule_router.get("/scan/{scan_id}/report")
def get_lung_nodule_report(scan_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    from report_generator import generate_lung_nodule_report
    scan = db.query(db_models.LungNoduleScan).filter(db_models.LungNoduleScan.id == scan_id).first()
    if not scan: raise HTTPException(404, "Scan not found")
    if scan.report_path and os.path.exists(scan.report_path):
        return FileResponse(scan.report_path, media_type="application/pdf", filename=f"LungNodule_Report_{scan_id}.pdf")
        
    p = db.query(db_models.Patient).filter(db_models.Patient.id == scan.patient_id).first()
    
    report_path = generate_lung_nodule_report(
        patient={"full_name": p.full_name, "age": p.age, "gender": p.gender, "comorbidities": p.comorbidities},
        scan={"scan_id": scan.id, "upload_timestamp": scan.upload_timestamp.strftime("%Y-%m-%d %H:%M:%S") if scan.upload_timestamp else "", "original_file_path": scan.original_file_path},
        inference_result=scan
    )
    
    if not report_path: raise HTTPException(500, "Failed to generate report")
    scan.report_path = report_path
    db.commit()
    return FileResponse(report_path, media_type="application/pdf", filename=f"LungNodule_Report_{scan_id}.pdf")

@lung_nodule_router.patch("/{scan_id}/override", response_model=schemas.LungNoduleResponse)
def override_lung_nodule(scan_id: str, req: schemas.LungNoduleOverrideRequest, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    scan = db.query(db_models.LungNoduleScan).filter(db_models.LungNoduleScan.id == scan_id).first()
    if not scan: raise HTTPException(404, "Scan not found")
    scan.clinician_override = req.clinician_override
    scan.override_notes = req.override_notes
    scan.override_timestamp = datetime.utcnow()
    db.commit()
    db.refresh(scan)
    return scan

@app.get("/patients/{patient_id}/lung-nodule-history", tags=["Lung Nodule"], response_model=List[schemas.LungNoduleResponse])
def lung_nodule_history(patient_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    return db.query(db_models.LungNoduleScan).filter(db_models.LungNoduleScan.patient_id == patient_id).order_by(db_models.LungNoduleScan.upload_timestamp.desc()).all()

app.include_router(lung_nodule_router)

brain_tumor_router = APIRouter(prefix="/brain-tumor", tags=["Brain Tumor"])

@brain_tumor_router.post("/scan/upload", response_model=schemas.BrainTumorResponse)
async def upload_brain_tumor(patient_id: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    p = db.query(db_models.Patient).filter(db_models.Patient.id == patient_id).first()
    if not p: raise HTTPException(404, "Patient not found")
    if not file.content_type.startswith("image/"): raise HTTPException(400, "File must be an image")
    
    uid = uuid.uuid4().hex[:8]
    filepath = f"uploads/brain_tumor_{uid}_{file.filename}".replace("\\", "/")
    with open(filepath, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
        
    try:
        res = run_brain_tumor_inference(filepath)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Inference failed: {str(e)}")
        
    db_scan = db_models.BrainTumorScan(
        patient_id=p.id, original_file_path=filepath, tumor_detected=res.tumor_detected, tumor_type=res.tumor_type,
        confidence=res.confidence, confidence_flag=res.confidence_flag, glioma_risk_flag=res.glioma_risk_flag, 
        all_probabilities=res.all_probabilities, urgency=res.urgency, heatmap_path=res.heatmap_path,
        clinical_recommendation=res.clinical_recommendation, model_version=res.model_version
    )
    db.add(db_scan)
    db.commit()
    db.refresh(db_scan)
    return db_scan

@brain_tumor_router.get("/scan/{scan_id}", response_model=schemas.BrainTumorResponse)
def get_brain_tumor(scan_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    scan = db.query(db_models.BrainTumorScan).filter(db_models.BrainTumorScan.id == scan_id).first()
    if not scan: raise HTTPException(404, "Scan not found")
    return scan

@brain_tumor_router.get("/scan/{scan_id}/report")
def get_brain_tumor_report(scan_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    from report_generator import generate_brain_tumor_report
    scan = db.query(db_models.BrainTumorScan).filter(db_models.BrainTumorScan.id == scan_id).first()
    if not scan: raise HTTPException(404, "Scan not found")
    if scan.report_path and os.path.exists(scan.report_path):
        return FileResponse(scan.report_path, media_type="application/pdf", filename=f"BrainTumor_Report_{scan_id}.pdf")
        
    p = db.query(db_models.Patient).filter(db_models.Patient.id == scan.patient_id).first()
    
    report_path = generate_brain_tumor_report(
        patient={"full_name": p.full_name, "age": p.age, "gender": p.gender, "comorbidities": p.comorbidities},
        scan={"scan_id": scan.id, "upload_timestamp": scan.upload_timestamp.strftime("%Y-%m-%d %H:%M:%S") if scan.upload_timestamp else "", "original_file_path": scan.original_file_path},
        inference_result=scan
    )
    
    if not report_path: raise HTTPException(500, "Failed to generate report")
    scan.report_path = report_path
    db.commit()
    return FileResponse(report_path, media_type="application/pdf", filename=f"BrainTumor_Report_{scan_id}.pdf")

@brain_tumor_router.patch("/{scan_id}/override", response_model=schemas.BrainTumorResponse)
def override_brain_tumor(scan_id: str, req: schemas.BrainTumorOverrideRequest, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    scan = db.query(db_models.BrainTumorScan).filter(db_models.BrainTumorScan.id == scan_id).first()
    if not scan: raise HTTPException(404, "Scan not found")
    scan.clinician_override = req.clinician_override
    scan.override_notes = req.override_notes
    scan.override_timestamp = datetime.utcnow()
    db.commit()
    db.refresh(scan)
    return scan

@app.get("/patients/{patient_id}/brain-tumor-history", tags=["Brain Tumor"], response_model=List[schemas.BrainTumorResponse])
def brain_tumor_history(patient_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    return db.query(db_models.BrainTumorScan).filter(db_models.BrainTumorScan.patient_id == patient_id).order_by(db_models.BrainTumorScan.upload_timestamp.desc()).all()

app.include_router(brain_tumor_router)

brain_hemorrhage_router = APIRouter(prefix="/brain-hemorrhage", tags=["Brain Hemorrhage"])

@brain_hemorrhage_router.post("/scan/upload", response_model=schemas.BrainHemorrhageResponse)
async def upload_brain_hemorrhage(patient_id: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    p = db.query(db_models.Patient).filter(db_models.Patient.id == patient_id).first()
    if not p: raise HTTPException(404, "Patient not found")
    if not file.content_type.startswith("image/"): raise HTTPException(400, "File must be an image")
    
    uid = uuid.uuid4().hex[:8]
    filepath = f"uploads/brain_hemorrhage_{uid}_{file.filename}".replace("\\", "/")
    with open(filepath, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
        
    try:
        res = run_brain_hemorrhage_inference(filepath)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Inference failed: {str(e)}")
        
    db_scan = db_models.BrainHemorrhageScan(
        patient_id=p.id, original_file_path=filepath, has_hemorrhage=res.has_hemorrhage, hemorrhage_probability=res.hemorrhage_probability,
        confidence=res.confidence, confidence_flag=res.confidence_flag, urgency=res.urgency, small_dataset_warning=True, 
        heatmap_path=res.heatmap_path, clinical_recommendation=res.clinical_recommendation, model_version=res.model_version
    )
    db.add(db_scan)
    db.commit()
    db.refresh(db_scan)
    
    if res.has_hemorrhage:
        logger.warning(f"HEMORRHAGE DETECTED for patient {patient_id}, scan {db_scan.id}. Urgency: EMERGENCY.")
        
    return db_scan

@brain_hemorrhage_router.get("/scan/{scan_id}", response_model=schemas.BrainHemorrhageResponse)
def get_brain_hemorrhage(scan_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    scan = db.query(db_models.BrainHemorrhageScan).filter(db_models.BrainHemorrhageScan.id == scan_id).first()
    if not scan: raise HTTPException(404, "Scan not found")
    return scan

@brain_hemorrhage_router.get("/scan/{scan_id}/report")
def get_brain_hemorrhage_report(scan_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    from report_generator import generate_brain_hemorrhage_report
    scan = db.query(db_models.BrainHemorrhageScan).filter(db_models.BrainHemorrhageScan.id == scan_id).first()
    if not scan: raise HTTPException(404, "Scan not found")
    if scan.report_path and os.path.exists(scan.report_path):
        return FileResponse(scan.report_path, media_type="application/pdf", filename=f"BrainHemorrhage_Report_{scan_id}.pdf")
        
    p = db.query(db_models.Patient).filter(db_models.Patient.id == scan.patient_id).first()
    
    report_path = generate_brain_hemorrhage_report(
        patient={"full_name": p.full_name, "age": p.age, "gender": p.gender, "comorbidities": p.comorbidities},
        scan={"scan_id": scan.id, "upload_timestamp": scan.upload_timestamp.strftime("%Y-%m-%d %H:%M:%S") if scan.upload_timestamp else "", "original_file_path": scan.original_file_path},
        inference_result=scan
    )
    
    if not report_path: raise HTTPException(500, "Failed to generate report")
    scan.report_path = report_path
    db.commit()
    return FileResponse(report_path, media_type="application/pdf", filename=f"BrainHemorrhage_Report_{scan_id}.pdf")

@brain_hemorrhage_router.patch("/{scan_id}/override", response_model=schemas.BrainHemorrhageResponse)
def override_brain_hemorrhage(scan_id: str, req: schemas.BrainHemorrhageOverrideRequest, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    scan = db.query(db_models.BrainHemorrhageScan).filter(db_models.BrainHemorrhageScan.id == scan_id).first()
    if not scan: raise HTTPException(404, "Scan not found")
    scan.clinician_override = req.clinician_override
    scan.override_notes = req.override_notes
    scan.override_timestamp = datetime.utcnow()
    db.commit()
    db.refresh(scan)
    return scan

@app.get("/patients/{patient_id}/brain-hemorrhage-history", tags=["Brain Hemorrhage"], response_model=List[schemas.BrainHemorrhageResponse])
def brain_hemorrhage_history(patient_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    return db.query(db_models.BrainHemorrhageScan).filter(db_models.BrainHemorrhageScan.patient_id == patient_id).order_by(db_models.BrainHemorrhageScan.upload_timestamp.desc()).all()

app.include_router(brain_hemorrhage_router)

bone_age_router = APIRouter(prefix="/bone-age", tags=["Bone Age"])

@bone_age_router.post("/scan/upload", response_model=schemas.BoneAgeResponse)
async def upload_bone_age(patient_id: str = Form(...), chronological_age_months: Optional[int] = Form(None), file: UploadFile = File(...), db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    p = db.query(db_models.Patient).filter(db_models.Patient.id == patient_id).first()
    if not p: raise HTTPException(404, "Patient not found")
    if not file.content_type.startswith("image/"): raise HTTPException(400, "File must be an image")
    
    uid = uuid.uuid4().hex[:8]
    filepath = f"uploads/bone_age_{uid}_{file.filename}".replace("\\", "/")
    with open(filepath, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
        
    try:
        res = run_bone_age_inference(filepath, chronological_age_months=chronological_age_months)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Inference failed: {str(e)}")
        
    db_scan = db_models.BoneAgeScan(
        patient_id=p.id, original_file_path=filepath, predicted_months=res.predicted_months, 
        predicted_display=res.predicted_display, uncertainty_months=res.uncertainty_months,
        confidence_flag=res.confidence_flag, chronological_age_months=chronological_age_months,
        deviation_months=res.deviation_months, skeletal_age_flag=res.skeletal_age_flag,
        heatmap_path=res.heatmap_path, clinical_recommendation=res.clinical_recommendation, 
        model_version=res.model_version
    )
    db.add(db_scan)
    db.commit()
    db.refresh(db_scan)
    return db_scan

@bone_age_router.get("/scan/{scan_id}", response_model=schemas.BoneAgeResponse)
def get_bone_age(scan_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    scan = db.query(db_models.BoneAgeScan).filter(db_models.BoneAgeScan.id == scan_id).first()
    if not scan: raise HTTPException(404, "Scan not found")
    return scan

@bone_age_router.get("/scan/{scan_id}/report")
def get_bone_age_report(scan_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    from report_generator import generate_bone_age_report
    scan = db.query(db_models.BoneAgeScan).filter(db_models.BoneAgeScan.id == scan_id).first()
    if not scan: raise HTTPException(404, "Scan not found")
    if scan.report_path and os.path.exists(scan.report_path):
        return FileResponse(scan.report_path, media_type="application/pdf", filename=f"BoneAge_Report_{scan_id}.pdf")
        
    p = db.query(db_models.Patient).filter(db_models.Patient.id == scan.patient_id).first()
    
    report_path = generate_bone_age_report(
        patient={"full_name": p.full_name, "age": p.age, "gender": p.gender, "comorbidities": p.comorbidities},
        scan={"scan_id": scan.id, "upload_timestamp": scan.upload_timestamp.strftime("%Y-%m-%d %H:%M:%S") if scan.upload_timestamp else "", "original_file_path": scan.original_file_path, "patient_id": p.id},
        inference_result=scan
    )
    
    if not report_path: raise HTTPException(500, "Failed to generate report")
    scan.report_path = report_path
    db.commit()
    return FileResponse(report_path, media_type="application/pdf", filename=f"BoneAge_Report_{scan_id}.pdf")

@app.get("/patients/{patient_id}/bone-age-history", tags=["Bone Age"], response_model=List[schemas.BoneAgeResponse])
def bone_age_history(patient_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    return db.query(db_models.BoneAgeScan).filter(db_models.BoneAgeScan.patient_id == patient_id).order_by(db_models.BoneAgeScan.upload_timestamp.desc()).all()

app.include_router(bone_age_router)

retinopathy_router = APIRouter(prefix="/retinopathy", tags=["Retinopathy"])
@retinopathy_router.post("/scan/upload", response_model=schemas.RetinopathyResponse)
async def upload_retinopathy(patient_id: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    patient = db.query(db_models.Patient).filter(db_models.Patient.id == patient_id).first()
    if not patient: raise HTTPException(404, "Patient not found")
    
    os.makedirs("uploads", exist_ok=True)
    ext = file.filename.split(".")[-1]
    filename = f"retinopathy_{uuid.uuid4().hex[:8]}_{int(time.time())}.{ext}"
    filepath = os.path.join("uploads", filename)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    try:
        from retinopathy_inference import run_retinopathy_inference
        result = run_retinopathy_inference(filepath)
    except Exception as e:
        logger.error(f"Retinopathy inference failed: {e}", exc_info=True)
        raise HTTPException(500, "Inference engine failed")
        
    db_scan = db_models.RetinopathyScan(
        id=str(uuid.uuid4()),
        patient_id=patient_id,
        original_file_path=filepath,
        grade=result.grade,
        grade_name=result.grade_name,
        confidence=result.confidence,
        confidence_flag=result.confidence_flag,
        referable_dr=result.referable_dr,
        referable_risk_flag=result.referable_risk_flag,
        all_probabilities=result.all_probabilities,
        urgency=result.urgency,
        follow_up_months=result.follow_up_months,
        heatmap_path=result.heatmap_path,
        clinical_recommendation=result.clinical_recommendation,
        model_version=result.model_version
    )
    db.add(db_scan)
    db.commit()
    db.refresh(db_scan)
    return db_scan

@retinopathy_router.get("/scan/{scan_id}", response_model=schemas.RetinopathyResponse)
def get_retinopathy(scan_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    scan = db.query(db_models.RetinopathyScan).filter(db_models.RetinopathyScan.id == scan_id).first()
    if not scan: raise HTTPException(404, "Scan not found")
    return scan

@retinopathy_router.get("/scan/{scan_id}/report")
def get_retinopathy_report(scan_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    from report_generator import generate_retinopathy_report
    scan = db.query(db_models.RetinopathyScan).filter(db_models.RetinopathyScan.id == scan_id).first()
    if not scan: raise HTTPException(404, "Scan not found")
    if scan.report_path and os.path.exists(scan.report_path):
        return FileResponse(scan.report_path, media_type="application/pdf", filename=f"Retinopathy_Report_{scan_id}.pdf")
        
    p = db.query(db_models.Patient).filter(db_models.Patient.id == scan.patient_id).first()
    
    report_path = generate_retinopathy_report(
        patient={"full_name": p.full_name, "age": p.age, "gender": p.gender, "comorbidities": p.comorbidities, "patient_id": p.id},
        scan={"scan_id": scan.id, "upload_timestamp": scan.upload_timestamp.strftime("%Y-%m-%d %H:%M:%S") if scan.upload_timestamp else "", "original_file_path": scan.original_file_path, "patient_id": p.id},
        inference_result=scan
    )
    
    if not report_path: raise HTTPException(500, "Failed to generate report")
    scan.report_path = report_path
    db.commit()
    return FileResponse(report_path, media_type="application/pdf", filename=f"Retinopathy_Report_{scan_id}.pdf")

@retinopathy_router.patch("/scan/{scan_id}/override", response_model=schemas.RetinopathyResponse)
def override_retinopathy(scan_id: str, payload: schemas.RetinopathyOverrideRequest, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    scan = db.query(db_models.RetinopathyScan).filter(db_models.RetinopathyScan.id == scan_id).first()
    if not scan: raise HTTPException(404, "Scan not found")
    scan.clinician_override = payload.clinician_override
    scan.override_notes = payload.override_notes
    scan.override_timestamp = datetime.utcnow()
    db.commit()
    db.refresh(scan)
    return scan

@app.get("/patients/{patient_id}/retinopathy-history", tags=["Retinopathy"], response_model=List[schemas.RetinopathyResponse])
def retinopathy_history(patient_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    return db.query(db_models.RetinopathyScan).filter(db_models.RetinopathyScan.patient_id == patient_id).order_by(db_models.RetinopathyScan.upload_timestamp.desc()).all()

app.include_router(retinopathy_router)
