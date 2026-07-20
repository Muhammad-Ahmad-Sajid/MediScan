with open("d:/X-ray ML Model/main.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add Jinja2Templates import
content = content.replace(
    "from fastapi.staticfiles import StaticFiles",
    "from fastapi.staticfiles import StaticFiles\nfrom fastapi.templating import Jinja2Templates",
)

# 2. Update app.mount
content = content.replace(
    'app.mount("/static", StaticFiles(directory="templates"), name="templates")',
    'app.mount("/static", StaticFiles(directory="static"), name="static")\ntemplates = Jinja2Templates(directory="templates")',
)

# 3. Replace the UI routes and /overrides endpoint
old_ui_routes = """# UI routes
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
    return overrides # To be expanded later if needed across all modules"""

new_ui_routes = """# UI routes
@app.get("/", response_class=HTMLResponse, tags=["UI"], include_in_schema=False)
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/dashboard")

@app.get("/login", response_class=HTMLResponse, tags=["UI"], include_in_schema=False)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse, tags=["UI"], include_in_schema=False)
async def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/scan/{module}", response_class=HTMLResponse, tags=["UI"], include_in_schema=False)
async def scan_page(request: Request, module: str):
    return templates.TemplateResponse("scan.html", {"request": request, "module": module})

@app.get("/results/{module}/{scan_id}", response_class=HTMLResponse, tags=["UI"], include_in_schema=False)
async def results_page(request: Request, module: str, scan_id: str):
    return templates.TemplateResponse("results.html", {"request": request, "module": module, "scan_id": scan_id})

@app.get("/history", response_class=HTMLResponse, tags=["UI"], include_in_schema=False)
async def history_page(request: Request):
    return templates.TemplateResponse("history.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse, tags=["UI"], include_in_schema=False)
async def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

# Overrides
@app.get("/admin/overrides", tags=["Admin"])
def get_all_overrides(db: Session = Depends(get_db), current_user = Depends(get_current_admin)):
    overrides = []
    
    # Aggregating overrides from all scan tables
    models_to_check = [
        (db_models.FractureScan, "Fracture"),
        (db_models.ArthritisScan, "Arthritis"),
        (db_models.OsteoporosisScan, "Osteoporosis"),
        (db_models.TBScan, "TB"),
        (db_models.LungNoduleScan, "Lung Nodule"),
        (db_models.BrainTumorScan, "Brain Tumor"),
        (db_models.BrainHemorrhageScan, "Brain Hemorrhage"),
        (db_models.BoneAgeScan, "Bone Age"),
        (db_models.RetinopathyScan, "Retinopathy"),
    ]
    
    for model_class, mod_name in models_to_check:
        try:
            if hasattr(model_class, "clinician_override"):
                records = db.query(model_class, db_models.Patient).join(db_models.Patient, model_class.patient_id == db_models.Patient.id).filter(model_class.clinician_override.isnot(None)).all()
                for scan, patient in records:
                    overrides.append({
                        "date": scan.upload_timestamp.strftime("%Y-%m-%d %H:%M:%S") if scan.upload_timestamp else "Unknown",
                        "doctor": scan.overridden_by or "Unknown",
                        "module": mod_name,
                        "patient": f"{patient.full_name} ({patient.id})",
                        "original": "N/A",  # Ideally we'd log the original before override, but we don't have it strictly stored.
                        "override": scan.clinician_override,
                        "notes": getattr(scan, "override_notes", "")
                    })
        except Exception as e:
            pass # Ignore if table doesn't have it

    # Sort by date descending
    overrides.sort(key=lambda x: x["date"], reverse=True)
    return overrides"""

if old_ui_routes in content:
    content = content.replace(old_ui_routes, new_ui_routes)

    # Add HTMLResponse to fastapi imports
    if "HTMLResponse" not in content:
        content = content.replace(
            "from fastapi.responses import JSONResponse, FileResponse",
            "from fastapi.responses import JSONResponse, FileResponse, HTMLResponse",
        )

    with open("d:/X-ray ML Model/main.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Success: Updated UI routes and /admin/overrides in main.py")
else:
    print("Error: Could not find old UI routes in main.py")
