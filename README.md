# MediScan AI

Multi-specialty medical imaging intelligence platform with 9 AI diagnostic modules.

## Modules

| # | Module | Task | Key Metric |
|---|--------|------|------------|
| 1 | Bone Fracture Detection | Binary classification + body region | 98.15% accuracy |
| 2 | Arthritis Grading | 5-class KL scale | Multi-class grading |
| 3 | Osteoporosis Screening | Binary classification | 91% sensitivity |
| 4 | TB Screening | Binary classification | 97% sensitivity |
| 5 | Lung Nodule Detection | Binary classification | 100% sensitivity |
| 6 | Brain Tumor Classification | 4-class | 95% accuracy |
| 7 | Brain Hemorrhage Detection | Binary classification | Emergency protocol |
| 8 | Bone Age Estimation | Regression | 7.34-month MAE |
| 9 | Diabetic Retinopathy Grading | 5-class ordinal | QWK 0.8708 |

## Tech Stack

- **ML**: PyTorch, ResNet-50, Transfer Learning, Grad-CAM, CLAHE
- **Backend**: FastAPI, PostgreSQL, SQLAlchemy, JWT Auth
- **DevOps**: Docker, Docker Compose, GitHub Actions, MLflow

## Quick Start

```bash
docker-compose up --build
```

API available at `http://localhost:8000` | Docs at `http://localhost:8000/docs`
