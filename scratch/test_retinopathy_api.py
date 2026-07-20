import requests
import json

BASE_URL = "http://localhost:8000"


def login():
    data = {"username": "doctor_test@test.com", "password": "password123"}
    res = requests.post(f"{BASE_URL}/auth/login", data=data)
    if res.status_code == 200:
        return res.json()["access_token"]

    # Try registering
    req = {
        "full_name": "Test Doctor",
        "email": "doctor_test@test.com",
        "password": "password123",
        "role": "doctor",
    }
    res_reg = requests.post(f"{BASE_URL}/auth/register", json=req)
    res = requests.post(f"{BASE_URL}/auth/login", data=data)
    if res.status_code != 200:
        print(f"Login failed: {res.text}")
        return None
    return res.json()["access_token"]


def get_patient(token):
    headers = {"Authorization": f"Bearer {token}"}
    res = requests.get(f"{BASE_URL}/patients/", headers=headers)
    if res.status_code == 200 and res.json():
        return res.json()[0]["id"]
    else:
        data = {
            "full_name": "Retinopathy Test Patient",
            "age": 45,
            "gender": "Male",
            "comorbidities": ["Diabetes"],
        }
        res = requests.post(f"{BASE_URL}/patients/", json=data, headers=headers)
        return res.json()["id"]


def test_upload(patient_id, file_path, token):
    print(f"\n--- Testing upload for {file_path} ---")
    headers = {"Authorization": f"Bearer {token}"}
    with open(file_path, "rb") as f:
        files = {"file": (file_path.split("/")[-1], f, "image/png")}
        data = {"patient_id": patient_id}
        res = requests.post(
            f"{BASE_URL}/retinopathy/scan/upload",
            data=data,
            files=files,
            headers=headers,
        )

    if res.status_code == 200:
        data = res.json()
        print("Upload successful!")
        print(f"Scan ID: {data['id']}")
        print(f"Grade: {data['grade']} - {data['grade_name']}")
        print(f"Referable DR: {data['referable_dr']}")
        print(f"Referable Risk Flag: {data.get('referable_risk_flag', False)}")

        probs = data.get("all_probabilities")
        if probs:
            if isinstance(probs, str):
                probs = json.loads(probs)
            print("Probabilities:")
            for k, v in probs.items():
                print(f"  Grade {k}: {v:.4f}")
        return data["id"]
    else:
        print(f"Upload failed: {res.status_code}")
        print(res.text)
        return None


def test_report(scan_id, token):
    print(f"\n--- Testing report generation for {scan_id} ---")
    headers = {"Authorization": f"Bearer {token}"}
    res = requests.get(f"{BASE_URL}/retinopathy/scan/{scan_id}/report", headers=headers)
    if res.status_code == 200:
        print(f"Report generated successfully. Content type: {res.headers.get('content-type')}")
    else:
        print(f"Report generation failed: {res.status_code}")
        print(res.text)


def test_override(scan_id, token):
    print(f"\n--- Testing clinician override for {scan_id} ---")
    headers = {"Authorization": f"Bearer {token}"}
    data = {
        "clinician_override": "Moderate Nonproliferative DR",
        "override_notes": "Tumor looks more like a moderate DR on secondary review.",
    }
    res = requests.patch(f"{BASE_URL}/retinopathy/scan/{scan_id}/override", json=data, headers=headers)
    if res.status_code == 200:
        print("Override successful!")
    else:
        print(f"Override failed: {res.status_code}")
        print(res.text)


def test_history(patient_id, token):
    print(f"\n--- Testing patient history ---")
    headers = {"Authorization": f"Bearer {token}"}
    res = requests.get(f"{BASE_URL}/patients/{patient_id}/retinopathy-history", headers=headers)
    if res.status_code == 200:
        print(f"History retrieved successfully. Found {len(res.json())} records.")
    else:
        print(f"History failed: {res.status_code}")
        print(res.text)


if __name__ == "__main__":
    token = login()
    if not token:
        exit(1)
    patient_id = get_patient(token)
    print(f"Using Patient ID: {patient_id}")

    images = [
        "d:/X-ray ML Model/Mediscan/retinopathy/colored_images/No_DR/002c21358ce6.png",
        "d:/X-ray ML Model/Mediscan/retinopathy/colored_images/Moderate/000c1434d8d7.png",
        "d:/X-ray ML Model/Mediscan/retinopathy/colored_images/Proliferate_DR/001639a390f0.png",
    ]

    scan_ids = []
    for img in images:
        sid = test_upload(patient_id, img, token)
        if sid:
            scan_ids.append(sid)
            test_report(sid, token)

    if scan_ids:
        test_override(scan_ids[0], token)
        test_history(patient_id, token)
