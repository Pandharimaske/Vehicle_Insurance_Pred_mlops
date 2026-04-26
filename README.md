# 🚗 Vehicle Insurance Claim Prediction

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.10+-orange.svg)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/DVC-3.40+-purple.svg)](https://dvc.org)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://docker.com)

A production-grade MLOps pipeline for binary classification — predicting whether a customer is likely to make a vehicle insurance claim. Built with a modular `src`-layout, automated hyperparameter optimisation, MLflow experiment tracking on DagsHub, DVC data versioning, and a containerised FastAPI inference endpoint.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Pipeline Architecture](#-pipeline-architecture)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [MLflow & DagsHub Tracking](#-mlflow--dagshub-tracking)
- [API Reference](#-api-reference)
- [DVC Pipeline](#-dvc-pipeline)
- [Docker](#-docker)
- [CI/CD](#️-cicd)

---

## 🔍 Overview

The project ingests raw vehicle insurance data from **MongoDB Atlas**, runs it through a multi-stage transformation pipeline, and trains a binary classifier using **Optuna-powered hyperparameter optimisation** across XGBoost and LightGBM. A **champion/challenger evaluation** gate compares every new model against the current Production model in the **MLflow Model Registry** — only promoting models that improve AUC-ROC by a defined threshold. **SHAP** explainability charts are logged as MLflow artifacts, and the final model is served via a **Dockerised FastAPI** endpoint with Pydantic input validation.

---

## 🏗️ Pipeline Architecture

```
MongoDB Atlas
     │
     ▼
┌───────────────────────┐
│    Data Ingestion     │  Fetches collection → splits train/test CSVs
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│   Data Validation     │  Schema & column checks against config/schema.yaml
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Data Transformation  │  Gender mapping, OHE, StandardScaler,
│                       │  MinMaxScaler, SMOTEENN resampling
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│    Model Training     │  Optuna TPE sampler — 50 trials
│                       │  Search space: XGBoost × LightGBM
│                       │  Objective: maximise AUC-ROC
│                       │  Best model → metrics + SHAP → MLflow
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│   Model Evaluation    │  Champion AUC ← MLflow Production registry
│  (Champion/Challenger)│  Challenger AUC ← current run
│                       │  Gate: ΔAUC > 0.005 → promote to Production
└──────────┬────────────┘
           │  (accepted)
           ▼
┌───────────────────────┐
│    Model Pusher       │  Uploads accepted model to AWS S3
└───────────────────────┘
```

---

## 📁 Project Structure

```
Vehicle_Insurance_Pred_mlops/
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py        # MongoDB → feature store CSV
│   │   ├── data_validation.py       # Schema & column validation
│   │   ├── data_transformation.py   # Feature engineering + SMOTEENN
│   │   ├── model_trainer.py         # Optuna HPO (XGBoost + LightGBM) + SHAP
│   │   ├── model_evaluation.py      # Champion/challenger AUC gate + MLflow registry
│   │   └── model_pusher.py          # AWS S3 model upload
│   │
│   ├── pipline/
│   │   ├── training_pipeline.py     # Full pipeline orchestration + DagsHub/MLflow setup
│   │   └── prediction_pipeline.py   # Inference data class
│   │
│   ├── entity/
│   │   ├── config_entity.py         # Dataclass configs for each stage
│   │   ├── artifact_entity.py       # Dataclass artifacts (incl. roc_auc)
│   │   ├── estimator.py             # MyModel wrapper (predict + predict_proba)
│   │   └── s3_estimator.py          # S3 model load/save/predict
│   │
│   ├── constants/__init__.py        # All project constants
│   ├── configuration/               # MongoDB + AWS connection managers
│   ├── data_access/                 # MongoDB data access layer
│   ├── cloud_storage/               # S3 storage utilities
│   ├── exception/                   # Custom exception class
│   ├── logger/                      # Logging setup
│   └── utils/main_utils.py          # File I/O helpers
│
├── config/
│   ├── schema.yaml                  # Column schema, feature lists
│   └── model.yaml
│
├── app.py                           # FastAPI app — /predict JSON endpoint
├── dvc.yaml                         # DVC pipeline stage definitions
├── params.yaml                      # Versioned hyperparameters
├── Dockerfile
├── requirements.txt
├── setup.py
├── pyproject.toml
└── notebooks/
    ├── exp-notebook.ipynb           # EDA & feature engineering
    └── mongoDB_demo.ipynb           # Data upload to MongoDB
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.10 |
| ML Models | XGBoost, LightGBM |
| Hyperparameter Optimisation | Optuna (TPE sampler, 50 trials) |
| Experiment Tracking | MLflow + DagsHub |
| Data Versioning | DVC |
| Explainability | SHAP (TreeExplainer) |
| API Serving | FastAPI + Pydantic + Uvicorn |
| Containerisation | Docker |
| Data Store | MongoDB Atlas |
| Model Store | AWS S3 |
| Imbalance Handling | SMOTEENN (imblearn) |
| CI/CD | GitHub Actions + AWS ECR + EC2 |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10
- MongoDB Atlas cluster with data loaded (see `notebooks/mongoDB_demo.ipynb`)
- AWS account with an S3 bucket
- (Optional) DagsHub account for remote MLflow tracking

### 1 — Clone & Install

```bash
git clone https://github.com/<your-username>/Vehicle_Insurance_Pred_mlops.git
cd Vehicle_Insurance_Pred_mlops

conda create -n vehicle python=3.10 -y
conda activate vehicle
pip install -r requirements.txt
```

### 2 — Set Environment Variables

```bash
# MongoDB (required)
export MONGODB_URL="mongodb+srv://<user>:<password>@<cluster>.mongodb.net/"

# AWS S3 (required for model store)
export AWS_ACCESS_KEY_ID="YOUR_KEY_ID"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_KEY"

# DagsHub MLflow tracking (optional — falls back to ./mlruns if not set)
export DAGSHUB_REPO_OWNER="your-dagshub-username"
export DAGSHUB_REPO_NAME="Vehicle_Insurance_Pred_mlops"
```

### 3 — Run Training Pipeline

```bash
python -c "from src.pipline.training_pipeline import TrainPipeline; TrainPipeline().run_pipeline()"
```

### 4 — Start the API Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

---

## 📊 MLflow & DagsHub Tracking

Every training run logs the following to the `vehicle_insurance_claim_prediction` experiment:

| Type | Items |
|---|---|
| **Params** | `model_type`, `optuna_n_trials`, `optuna_sampler`, all best hyperparameters |
| **Metrics** | `auc_roc`, `f1_score`, `precision`, `recall`, `champion_auc`, `auc_improvement` |
| **Artifacts** | Serialised model, SHAP feature importance bar chart |
| **Registry** | Model registered as `vehicle_insurance_model`; auto-promoted to `Production` when `ΔAUC > 0.005` |

### View locally

```bash
mlflow ui --port 5000
# → http://localhost:5000
```

### View on DagsHub

```
https://dagshub.com/<DAGSHUB_REPO_OWNER>/Vehicle_Insurance_Pred_mlops/experiments
```

---

## 🌐 API Reference

### `GET /train`

Triggers the full training pipeline end-to-end.

```bash
curl http://localhost:8000/train
```

---

### `POST /predict`

Returns a binary insurance claim prediction with a confidence score.

**Request**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": 1,
    "Age": 35,
    "Driving_License": 1,
    "Region_Code": 28.0,
    "Previously_Insured": 0,
    "Annual_Premium": 40454.0,
    "Policy_Sales_Channel": 26.0,
    "Vintage": 127,
    "Vehicle_Age_lt_1_Year": 0,
    "Vehicle_Age_gt_2_Years": 1,
    "Vehicle_Damage_Yes": 1
  }'
```

**Response**

```json
{
  "prediction": 1,
  "label": "Response-Yes",
  "confidence": 0.8734
}
```

**Input Fields**

| Field | Type | Constraints | Description |
|---|---|---|---|
| `Gender` | int | 0 or 1 | 0 = Female, 1 = Male |
| `Age` | int | 18–100 | Customer age in years |
| `Driving_License` | int | 0 or 1 | 1 = has valid licence |
| `Region_Code` | float | ≥ 0 | Encoded geographic region |
| `Previously_Insured` | int | 0 or 1 | 1 = had prior insurance |
| `Annual_Premium` | float | > 0 | Annual premium amount (₹) |
| `Policy_Sales_Channel` | float | > 0 | Encoded sales channel |
| `Vintage` | int | ≥ 0 | Days associated with the company |
| `Vehicle_Age_lt_1_Year` | int | 0 or 1 | Vehicle age < 1 year |
| `Vehicle_Age_gt_2_Years` | int | 0 or 1 | Vehicle age > 2 years |
| `Vehicle_Damage_Yes` | int | 0 or 1 | Vehicle previously damaged |

Pydantic validates all fields — invalid requests return `422 Unprocessable Entity` with field-level error details.

---

## 📦 DVC Pipeline

```bash
dvc init          # initialise DVC (first time only)
dvc repro         # reproduce only stale stages
dvc params diff   # compare params vs last run
dvc dag           # visualise the pipeline DAG
```

Stages defined in `dvc.yaml`:

| Stage | Inputs | Outputs |
|---|---|---|
| `data_ingestion` | MongoDB collection | `artifact/data_ingestion/` |
| `data_validation` | ingestion artifacts + `schema.yaml` | `artifact/data_validation/` |
| `data_transformation` | validation artifacts | `artifact/data_transformation/` |
| `model_trainer` | transformation artifacts + `params.yaml` | `artifact/model_trainer/` |
| `model_evaluation` | ingestion + trainer artifacts | MLflow registry update |

Changing `params.yaml` (e.g. `n_trials`, `auc_promotion_gate`) automatically invalidates downstream stages on the next `dvc repro`.

---

## 🐳 Docker

```bash
# Build
docker build -t vehicle-insurance-pred .

# Run
docker run -p 8000:8000 \
  -e MONGODB_URL="..." \
  -e AWS_ACCESS_KEY_ID="..." \
  -e AWS_SECRET_ACCESS_KEY="..." \
  -e DAGSHUB_REPO_OWNER="..." \
  -e DAGSHUB_REPO_NAME="..." \
  vehicle-insurance-pred
```

API available at `http://localhost:8000`.

---

## ⚙️ CI/CD

GitHub Actions workflow (`.github/workflows/aws.yaml`) runs on every push to `main`:

1. **CI** — Builds Docker image and pushes to AWS ECR
2. **CD** — Pulls the new image on a self-hosted EC2 runner and restarts the container

**Required GitHub Secrets:**

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret |
| `AWS_DEFAULT_REGION` | e.g. `us-east-1` |
| `ECR_REPO` | ECR repository name |
| `MONGODB_URL` | MongoDB Atlas connection string |

---

## 📄 License

This project is for educational and portfolio purposes.