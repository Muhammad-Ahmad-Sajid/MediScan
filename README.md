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

## Datasets Used

This platform leverages several publicly available, high-quality medical imaging datasets for training the 9 diagnostic modules. Since the datasets are massive, they are not included in this repository. You can download them directly from their original sources:

1. **Bone Fracture (Modules 1 & 2):** [Stanford MURA Dataset](https://stanfordmlgroup.github.io/competitions/mura/) & [FracAtlas Dataset](https://figshare.com/articles/dataset/The_FracAtlas_Dataset/22363063)
2. **Arthritis Grading (Module 2):** [Knee Osteoarthritis Dataset with KL Grading (Kaggle)](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity)
3. **Osteoporosis Screening (Module 3):** [Osteoporosis Knee X-Ray Dataset (Kaggle)](https://www.kaggle.com/datasets/smritisingh1997/osteoporosis-knee-xray-dataset)
4. **TB Screening (Module 4):** [Tuberculosis (TB) Chest X-ray Database (Kaggle)](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
5. **Lung Nodule Detection (Module 5):** Chest X-Ray / CT Lung Nodule Datasets (e.g., LUNA16 / Kaggle equivalents)
6. **Brain Tumor Classification (Module 6):** [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
7. **Brain Hemorrhage Detection (Module 7):** [RSNA Intracranial Hemorrhage Detection (Kaggle)](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection)
8. **Bone Age Estimation (Module 8):** [RSNA Pediatric Bone Age Challenge (Kaggle)](https://www.kaggle.com/datasets/kmader/rsna-bone-age)
9. **Diabetic Retinopathy Grading (Module 9):** [APTOS 2019 Blindness Detection (Kaggle)](https://www.kaggle.com/c/aptos2019-blindness-detection)

## Tech Stack

- **ML**: PyTorch, ResNet-50, Transfer Learning, Grad-CAM, CLAHE
- **Backend**: FastAPI, PostgreSQL, SQLAlchemy, JWT Auth
- **DevOps**: Docker, Docker Compose, GitHub Actions, MLflow

## Quick Start

```bash
docker-compose up --build
```

API available at `http://localhost:8000` | Docs at `http://localhost:8000/docs`
