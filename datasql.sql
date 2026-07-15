CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    hashed_password VARCHAR(200) NOT NULL,
    role VARCHAR(20) DEFAULT 'doctor',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS patients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    full_name VARCHAR(100) NOT NULL,
    age INTEGER NOT NULL CHECK (age BETWEEN 1 AND 120),
    gender VARCHAR(20),
    comorbidities TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS xray_scans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id),
    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    original_file_path VARCHAR(500),
    bone_affected VARCHAR(50),
    image_quality_flag BOOLEAN DEFAULT TRUE,
    dataset_source VARCHAR(50),
    report_path VARCHAR(500)
);

CREATE TABLE IF NOT EXISTS fracture_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id UUID REFERENCES xray_scans(id),
    fracture_detected BOOLEAN,
    severity VARCHAR(50),
    confidence_score FLOAT,
    confidence_flag VARCHAR(50) DEFAULT 'clear',
    heatmap_path VARCHAR(500),
    model_version VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS prognosis_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id UUID REFERENCES fracture_predictions(id),
    rest_weeks_min INTEGER,
    rest_weeks_max INTEGER,
    cast_type VARCHAR(100),
    plaster_required BOOLEAN,
    weight_bearing_status VARCHAR(100),
    referral_flag VARCHAR(20),
    clinician_override VARCHAR(200),
    override_notes TEXT,
    override_timestamp TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scan_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id UUID REFERENCES xray_scans(id),
    report_path VARCHAR(500) NOT NULL,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50)
);