import re

with open("d:/X-ray ML Model/main.py", "r", encoding="utf-8") as f:
    content = f.read()

endpoint_code = """
@app.get("/admin/dashboard_stats", tags=["Admin"])
def get_dashboard_stats(db: Session = Depends(get_db)):
    # Calculate real stats across all modules
    models = [
        (db_models.FractureScan, "fracture"),
        (db_models.ArthritisScan, "arthritis"),
        (db_models.OsteoporosisScan, "osteoporosis"),
        (db_models.TBScan, "tb"),
        (db_models.LungNoduleScan, "lung_nodule"),
        (db_models.BrainTumorScan, "brain_tumor"),
        (db_models.BrainHemorrhageScan, "brain_hemorrhage"),
        (db_models.BoneAgeScan, "bone_age"),
        (db_models.RetinopathyScan, "retinopathy"),
    ]
    
    total_scans = 0
    module_counts = {}
    critical_alerts = 0
    reports_generated = 0
    
    for model_class, mod_key in models:
        try:
            records = db.query(model_class).all()
            total_scans += len(records)
            module_counts[mod_key] = len(records)
            
            for r in records:
                if getattr(r, "report_path", None):
                    reports_generated += 1
                
                # Check for critical alerts based on available attributes
                is_critical = False
                if getattr(r, "glioma_risk_flag", False): is_critical = True
                elif getattr(r, "referable_risk_flag", False): is_critical = True
                elif getattr(r, "surgery_referral", False): is_critical = True
                elif str(getattr(r, "result_class", "")).lower() in ["grade 4", "grade 3"]: is_critical = True
                elif "hemorrhage detected" in str(getattr(r, "diagnosis", "")).lower(): is_critical = True
                
                if is_critical:
                    critical_alerts += 1
        except Exception:
            module_counts[mod_key] = 0
            
    patients_count = db.query(db_models.Patient).count()
    
    return {
        "total_scans": total_scans,
        "patients": patients_count,
        "alerts": critical_alerts,
        "reports": reports_generated,
        "module_counts": module_counts
    }
"""

if '@app.get("/admin/dashboard_stats' not in content:
    content = content.replace("# Overrides", endpoint_code + "\n# Overrides")
    with open("d:/X-ray ML Model/main.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Success: Added GET /admin/dashboard_stats to main.py")
else:
    print("Already exists")
