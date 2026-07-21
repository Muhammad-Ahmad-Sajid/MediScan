# MediScan AI

MediScan AI is a state-of-the-art, multi-specialty medical imaging intelligence platform designed to assist healthcare professionals in diagnosing a wide range of conditions from medical scans. It integrates 9 distinct AI diagnostic modules into a single, unified clinical dashboard.

## Project Overview

Modern healthcare relies heavily on medical imaging, but the sheer volume of scans can overwhelm radiologists and specialists. MediScan AI serves as an automated "second opinion" system. By leveraging advanced deep learning architectures (primarily customized ResNet-50 models) paired with Grad-CAM Explainable AI (XAI) and clinical prognosis engines, MediScan provides rapid, highly accurate, and interpretable diagnoses.

The platform is designed with a professional, cinematic dark-mode UI for clinicians, offering secure authentication, patient scan history tracking, and a comprehensive admin dashboard for system monitoring.

## The 9 Diagnostic Modules

MediScan is divided into 9 specialized AI modules, each trained on distinct datasets to handle specific clinical tasks:

| # | Module | Clinical Use-Case | ML Task | Key Metric |
|---|--------|-------------------|---------|------------|
| 1 | **Bone Fracture Detection** | Detects fractures in musculoskeletal X-rays (hand, leg, hip, shoulder) and generates severity prognosis. | Binary classification + Body Region | 98.15% Accuracy |
| 2 | **Arthritis Grading** | Evaluates knee X-rays to assign a Kellgren-Lawrence (KL) grade for osteoarthritis severity (0-4). | 5-class KL scale grading | Multi-class Grading |
| 3 | **Osteoporosis Screening** | Screens for reduced bone density in X-rays to flag potential osteoporosis risks. | Binary classification | 91% Sensitivity |
| 4 | **Tuberculosis (TB) Screening** | Analyzes Chest X-Rays (CXRs) to detect signs of pulmonary tuberculosis infections. | Binary classification | 97% Sensitivity |
| 5 | **Lung Nodule Detection** | Identifies the presence of potentially malignant pulmonary nodules in chest imaging. | Binary classification | 100% Sensitivity |
| 6 | **Brain Tumor Classification** | Classifies Brain MRIs into 4 categories: Glioma, Meningioma, Pituitary tumor, or No Tumor. | 4-class categorization | 95% Accuracy |
| 7 | **Brain Hemorrhage Detection** | Rapidly screens head CT scans for intracranial hemorrhages, prioritizing emergency cases. | Binary classification | Emergency Protocol |
| 8 | **Bone Age Estimation** | Analyzes pediatric hand X-rays to estimate skeletal age, aiding in growth disorder diagnosis. | Regression | 7.34-month MAE |
| 9 | **Diabetic Retinopathy Grading** | Examines retinal fundus images to grade diabetic retinopathy on a 0-4 severity scale. | 5-class Ordinal | QWK 0.8708 |

## Datasets Used

This platform leverages several massive, publicly available medical imaging datasets for training the 9 diagnostic modules. You can download the datasets directly from their original sources to retrain or fine-tune the models:

1. **Bone Fracture (Modules 1 & 2):** [Stanford MURA Dataset](https://stanfordmlgroup.github.io/competitions/mura/) & [FracAtlas Dataset](https://figshare.com/articles/dataset/The_FracAtlas_Dataset/22363063)
2. **Arthritis Grading (Module 2):** [Knee Osteoarthritis Dataset with KL Grading (Kaggle)](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity)
3. **Osteoporosis Screening (Module 3):** [Osteoporosis Knee X-Ray Dataset (Kaggle)](https://www.kaggle.com/datasets/smritisingh1997/osteoporosis-knee-xray-dataset)
4. **TB Screening (Module 4):** [Tuberculosis (TB) Chest X-ray Database (Kaggle)](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
5. **Lung Nodule Detection (Module 5):** Chest X-Ray / CT Lung Nodule Datasets (e.g., LUNA16 / Kaggle equivalents)
6. **Brain Tumor Classification (Module 6):** [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
7. **Brain Hemorrhage Detection (Module 7):** [RSNA Intracranial Hemorrhage Detection (Kaggle)](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection)
8. **Bone Age Estimation (Module 8):** [RSNA Pediatric Bone Age Challenge (Kaggle)](https://www.kaggle.com/datasets/kmader/rsna-bone-age)
9. **Diabetic Retinopathy Grading (Module 9):** [APTOS 2019 Blindness Detection (Kaggle)](https://www.kaggle.com/c/aptos2019-blindness-detection)

## Technology Stack

- **Deep Learning / AI:** PyTorch, TorchVision, ResNet-50 architecture, Grad-CAM (for Explainable AI heatmaps), CLAHE (Contrast Limited Adaptive Histogram Equalization for medical image enhancement).
- **Backend API:** FastAPI (Python), providing asynchronous, high-performance REST endpoints for all 9 inference models.
- **Database:** PostgreSQL for persistent storage of patient records and scan histories, managed via SQLAlchemy ORM.
- **Authentication:** Secure JWT-based (JSON Web Token) authentication system.
- **Frontend UI:** Vanilla JavaScript, HTML5, and CSS3, featuring a responsive, dynamic sidebar, glassmorphism elements, and smooth micro-animations.
- **DevOps & MLOps:** Docker, Docker Compose, MLflow (for experiment tracking), and GitHub Actions (for CI/CD and linting).

## Key Features

- **Explainable AI (Grad-CAM):** Every scan analyzed by MediScan AI produces a Grad-CAM heatmap. This visualizes exactly which pixels the neural network focused on to make its diagnosis, providing crucial transparency for doctors.
- **Automated Prognosis Engine:** Beyond just "detecting" a disease, the system uses confidence thresholds and specialized algorithms to generate a readable clinical prognosis report (e.g., suggesting a CT scan follow-up or classifying severity).
- **Secure Patient History:** Doctors can log in securely and view a historical timeline of all patient scans, complete with timestamps, diagnostic confidence scores, and visual heatmap records.
- **Admin Dashboard:** A real-time system monitoring dashboard to track API latency, model load times, database health, and overall system metrics.

## Quick Start (Docker)

To run the entire 9-module platform locally:

```bash
docker-compose up --build
```

- **Frontend Application:** `http://localhost:8000`
- **Interactive API Docs:** `http://localhost:8000/docs`
