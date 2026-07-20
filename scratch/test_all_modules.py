import requests
import json
import uuid
import sys
import os
from datetime import datetime

BASE_URL = "http://localhost:8000"
DUMMY_IMAGE = "d:/X-ray ML Model/scratch/dummy_test.png"

# Ensure dummy image exists
if not os.path.exists(DUMMY_IMAGE):
    with open(DUMMY_IMAGE, "wb") as f:
        f.write(b"dummy image data")

def login():
    data = {
        "username": "doctor_test@test.com",
        "password": "password123"
    }
    res = requests.post(f"{BASE_URL}/auth/login", data=data)
    if res.status_code == 200:
        return res.json()["access_token"]
        
    # Try registering if login fails
    req = {"full_name": "Test Doctor", "email": "doctor_test@test.com", "password": "password123", "role": "doctor"}
    res_reg = requests.post(f"{BASE_URL}/auth/register", json=req)
    res = requests.post(f"{BASE_URL}/auth/login", data=data)
    if res.status_code == 200:
        return res.json()["access_token"]
        
    print("Login failed!")
    sys.exit(1)

def get_first_patient(token):
    headers = {"Authorization": f"Bearer {token}"}
    res = requests.get(f"{BASE_URL}/patients/", headers=headers)
    if res.status_code == 200:
        patients = res.json()
        return patients[0]["id"]
    else:
        print("Failed to fetch patients")
        sys.exit(1)

def test_module(module_name, endpoint, token, patient_id):
    headers = {"Authorization": f"Bearer {token}"}
    try:
        with open(DUMMY_IMAGE, "rb") as f:
            files = {"file": ("dummy.png", f, "image/png")}
            data = {"patient_id": patient_id}
            
            # Bone age uses chronological_age
            if "bone-age" in endpoint:
                data["chronological_age"] = "150"
                
            res = requests.post(f"{BASE_URL}{endpoint}", headers=headers, files=files, data=data)
            
            if res.status_code == 200:
                result = res.json()
                scan_id = result.get("id", "Unknown")
                return {"status": "PASS", "scan_id": scan_id, "details": "HTTP 200 OK"}
            else:
                return {"status": "FAIL", "scan_id": None, "details": f"HTTP {res.status_code} - {res.text}"}
    except Exception as e:
        return {"status": "ERROR", "scan_id": None, "details": str(e)}

def test_report(endpoint_prefix, scan_id, token):
    if not scan_id or scan_id == "Unknown":
        return "SKIP"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        res = requests.get(f"{BASE_URL}{endpoint_prefix}/{scan_id}/report", headers=headers)
        if res.status_code == 200:
            return "PASS"
        else:
            return f"FAIL (HTTP {res.status_code})"
    except Exception as e:
        return f"ERROR ({str(e)})"

def test_history(endpoint, token, patient_id):
    headers = {"Authorization": f"Bearer {token}"}
    try:
        res = requests.get(f"{BASE_URL}{endpoint}", headers=headers)
        if res.status_code == 200:
            return "PASS"
        else:
            return f"FAIL (HTTP {res.status_code})"
    except Exception as e:
        return f"ERROR ({str(e)})"

def test_override(endpoint_prefix, scan_id, token):
    if not scan_id or scan_id == "Unknown":
        return "SKIP"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        # We try to send generic data, the exact payload depends on the module, but most accept `clinician_override`
        data = {
            "clinician_override": "Test Override",
            "override_notes": "Test"
        }
        res = requests.patch(f"{BASE_URL}{endpoint_prefix}/{scan_id}/override", json=data, headers=headers)
        if res.status_code == 200:
            return "PASS"
        else:
            return f"FAIL (HTTP {res.status_code})"
    except Exception as e:
        return f"ERROR ({str(e)})"

MODULES = [
    {"name": "Module 1 & 2 (Fracture)", "upload": "/fracture/scan/upload", "report": None, "history": "/patients/{}/fracture-history", "override": "/fracture"},
    {"name": "Module 3 (Osteoporosis)", "upload": "/osteoporosis/scan/upload", "report": None, "history": "/patients/{}/osteoporosis-history", "override": "/osteoporosis"},
    {"name": "Module 4 (Arthritis)", "upload": "/arthritis/scan/upload", "report": None, "history": "/patients/{}/arthritis-history", "override": "/arthritis"},
    {"name": "Module 5 (TB)", "upload": "/tb/scan/upload", "report": "/tb/scan", "history": "/patients/{}/tb-history", "override": "/tb"},
    {"name": "Module 5b (Lung Nodule)", "upload": "/lung-nodule/scan/upload", "report": "/lung-nodule/scan", "history": "/patients/{}/lung-nodule-history", "override": "/lung-nodule"},
    {"name": "Module 6 (Brain Tumor)", "upload": "/brain-tumor/scan/upload", "report": "/brain-tumor/scan", "history": "/patients/{}/brain-tumor-history", "override": "/brain-tumor"},
    {"name": "Module 7 (Brain Hemorrhage)", "upload": "/brain-hemorrhage/scan/upload", "report": "/brain-hemorrhage/scan", "history": "/patients/{}/brain-hemorrhage-history", "override": "/brain-hemorrhage"},
    {"name": "Module 8 (Bone Age)", "upload": "/bone-age/scan/upload", "report": "/bone-age/scan", "history": "/patients/{}/bone-age-history", "override": None},
    {"name": "Module 9 (Retinopathy)", "upload": "/retinopathy/scan/upload", "report": "/retinopathy/scan", "history": "/patients/{}/retinopathy-history", "override": "/retinopathy/scan"}
]

def main():
    print("Logging in...")
    token = login()
    print("Fetching patient...")
    patient_id = get_first_patient(token)
    
    print(f"Using Patient ID: {patient_id}\n")
    
    results = []
    
    for mod in MODULES:
        print(f"Testing {mod['name']}...")
        upload_res = test_module(mod['name'], mod['upload'], token, patient_id)
        
        scan_id = upload_res['scan_id']
        
        report_res = test_report(mod['report'], scan_id, token)
        history_url = mod['history'].format(patient_id)
        history_res = test_history(history_url, token, patient_id)
        
        override_res = "N/A"
        if mod['override']:
            override_res = test_override(mod['override'], scan_id, token)
            
        results.append({
            "Module": mod['name'],
            "Upload (Inference)": upload_res['status'],
            "Report (PDF)": report_res,
            "History": history_res,
            "Override": override_res,
            "Details": upload_res['details'] if upload_res['status'] != 'PASS' else 'OK'
        })
        
    # Generate Markdown Table
    print("\n\n=== 9-MODULE PLATFORM CHECK STATUS ===\n")
    print("| Module | Upload (Inference) | Report (PDF) | History | Clinician Override |")
    print("|---|---|---|---|---|")
    for r in results:
        print(f"| {r['Module']} | {r['Upload (Inference)']} | {r['Report (PDF)']} | {r['History']} | {r['Override']} |")
        
    print("\nDetailed Errors:")
    for r in results:
        if r['Upload (Inference)'] != 'PASS' or 'FAIL' in r['Report (PDF)'] or 'FAIL' in r['History'] or 'FAIL' in r['Override']:
            print(f"- {r['Module']}: {r['Details']}")

if __name__ == "__main__":
    main()
