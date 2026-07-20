import re

with open("d:/X-ray ML Model/main.py", "r", encoding="utf-8") as f:
    content = f.read()

endpoint_code = """
@patient_router.get("/{patient_id}/scans", tags=["Patients"])
def get_patient_scans(patient_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_doctor)):
    scans = []
    models_to_check = [
        (db_models.FractureScan, "fracture"),
        (db_models.ArthritisScan, "arthritis"),
        (db_models.OsteoporosisScan, "osteoporosis"),
        (db_models.TBScan, "tb"),
        (db_models.LungNoduleScan, "lung-nodule"),
        (db_models.BrainTumorScan, "brain-tumor"),
        (db_models.BrainHemorrhageScan, "brain-hemorrhage"),
        (db_models.BoneAgeScan, "bone-age"),
        (db_models.RetinopathyScan, "retinopathy"),
    ]
    for model_class, mod_name in models_to_check:
        try:
            records = db.query(model_class).filter(model_class.patient_id == patient_id).all()
            for r in records:
                scans.append({
                    "id": r.id,
                    "module": mod_name,
                    "upload_timestamp": r.upload_timestamp.isoformat() if r.upload_timestamp else None,
                    "file_path": r.file_path,
                    "heatmap_path": getattr(r, "heatmap_path", None),
                    "diagnosis": getattr(r, "diagnosis", getattr(r, "result_class", getattr(r, "predicted_class", getattr(r, "bone_age_months", "Unknown")))),
                    "confidence": getattr(r, "confidence", getattr(r, "probability", getattr(r, "confidence_score", 1.0))),
                    "recommendation": getattr(r, "recommendation", "")
                })
        except Exception:
            pass
    scans.sort(key=lambda x: x["upload_timestamp"] or "", reverse=True)
    return scans
"""

if "@patient_router.get(\"/{patient_id}\"" in content:
    # Insert right before the GET /patients/{patient_id} route
    content = content.replace(
        '@patient_router.get("/{patient_id}", response_model=schemas.PatientResponse)',
        endpoint_code + '\n@patient_router.get("/{patient_id}", response_model=schemas.PatientResponse)'
    )
    with open("d:/X-ray ML Model/main.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Success: Added GET /patients/{patient_id}/scans to main.py")
else:
    print("Error: Could not find @patient_router.get(/{patient_id}) in main.py")
