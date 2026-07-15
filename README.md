[![CI Pipeline](https://github.com/Muhammad-Ahmad-Sajid/MediScan/actions/workflows/ci.yml/badge.svg)](https://github.com/Muhammad-Ahmad-Sajid/MediScan/actions/workflows/ci.yml)<br>
![Python](https://img.shields.io/badge/python-3.10-blue.svg)<br>
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)<br>
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)<br>
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)<br>
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

<br>

# MediScan AI Platform
**Comprehensive Multi-Disease Medical AI Diagnostic System**

MediScan is a highly advanced, end-to-end medical AI diagnostic platform designed to assist radiologists and physicians. Originally evolving from the CortexRay bone fracture architecture, MediScan scales the deep learning infrastructure to detect, diagnose, and formulate clinical prognoses for **eight distinct medical conditions** across multiple physiological regions using X-ray, MRI, and CT imagery.

---

## 🌟 Clinical Diagnostic Capabilities

- 🦴 **Bone Age Estimation:** Pediatric skeletal maturity regression (Target MAE < 12 months)
- 💥 **Bone Fractures:** Multi-region skeletal fracture detection (98.15% accuracy)
- 🦴 **Osteoporosis:** Bone density degradation and risk classification
- 🦵 **Arthritis:** Joint space narrowing and rheumatoid classification
- 🫁 **Tuberculosis (TB):** Pulmonary infection detection from chest X-rays
- 🫁 **Lung Nodules:** Pulmonary nodule/malignancy screening
- 🧠 **Brain Tumors:** MRI-based glioma, meningioma, and pituitary tumor classification
- 🩸 **Brain Hemorrhage:** CT-based intracranial hemorrhage detection

---

## 🏗️ Core AI Features

- **Massive ResNet-50 Ensembles:** Dedicated PyTorch deep learning architectures dynamically trained and fine-tuned for each specific disease using hundreds of thousands of medical images.
- **Dynamic Checkpoint Resuming:** Robust model training architecture with dynamic pause, snapshot, and resume capabilities to preserve optimization momentum over multi-day training loops.
- **Grad-CAM Explainability Heatmaps:** Visual, interpretable AI mapping highlighting exact clinical zones of interest for doctors to review.
- **Clinical Prognosis Engine:** Rule-based prognosis engine that formulates recommended rest duration, cast type, and specialist referral flags.
- **Automated PDF Reports:** Instant compilation of patient data, AI findings, and clinical recommendations into standardized medical reports.

---

## 💻 Tech Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **ML Framework** | PyTorch 2.x | Model training and inference |
| **Model Architectures** | ResNet-50 | Pretrained CNN backbones |
| **Backend** | FastAPI | REST API and file serving |
| **Database** | PostgreSQL 16 | Patient and scan storage |
| **ML Tracking** | MLflow | Experiment and inference logging |
| **Containerization**| Docker + Compose | Deployment and portability |

---

## 🚀 Quick Start (Local Development)

1. Clone the repo: `git clone https://github.com/Muhammad-Ahmad-Sajid/MediScan.git`
2. Create virtual environment: `python -m venv myenv` and activate it
3. Install dependencies: `pip install -r requirements.txt`
4. Set up `.env` file (Database, JWT secrets, Model paths)
5. Start the FastAPI server: `uvicorn main:app --reload --port 8000`

---

## ⚠️ Clinical Disclaimer

> ⚠️ **MediScan is an AI-assisted clinical decision support tool intended for use by qualified medical professionals only.** It does not replace radiologist diagnosis or clinical judgment. All AI predictions must be reviewed and confirmed by a licensed clinician before any treatment decision is made. The authors accept no liability for clinical decisions made based on this system's output.

---

## 👤 Author

**Muhammad Ahmad Sajid**  
BSAI Student — Ghulam Ishaq Khan Institute (GIKI)  
GitHub: [Muhammad-Ahmad-Sajid](https://github.com/Muhammad-Ahmad-Sajid)  
Project: [MediScan](https://github.com/Muhammad-Ahmad-Sajid/MediScan)
