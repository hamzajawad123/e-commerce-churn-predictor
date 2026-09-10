# E-Commerce Churn & LTV Predictor

> End-to-end MLOps project that scores e-commerce customers for **churn risk** and **lifetime value (LTV)**, then recommends a retention action — built with Feast, DagsHub MLflow, FastAPI, Streamlit, DVC, Docker, and GitHub Actions.

## Table of Contents

- [Introduction](#introduction)
- [Live app](#live-app)
- [Features](#features)
- [Technologies](#technologies)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Step-by-step setup](#step-by-step-setup)
- [Environment configuration](#environment-configuration)
- [Run the dashboard](#run-the-dashboard)
- [Run the API](#run-the-api)
- [Run the complete local stack](#run-the-complete-local-stack)
- [Docker](#docker)
  - [Option A — Docker Compose (recommended)](#option-a--docker-compose-recommended)
  - [Option B — Build and run the image yourself](#option-b--build-and-run-the-image-yourself)
  - [Option C — Pull from Docker Hub](#option-c--pull-from-docker-hub)
- [Data, training, and notebooks](#data-training-and-notebooks)
- [GitHub Actions](#github-actions)
- [API endpoints](#api-endpoints)
- [Project structure](#project-structure)
- [Folder and file guide](#folder-and-file-guide)
- [Common commands](#common-commands)
- [Tests](#tests)
- [How to stop](#how-to-stop)
- [Troubleshooting](#troubleshooting)
- [Security](#security)
- [Useful links](#useful-links)

---



## Introduction

**E-Commerce Churn & LTV Predictor** serves two supervised models and one rule engine for retention:

| Task | Type | Output |
| ---- | ---- | ------ |
| Churn prediction | Binary classification (`Churned`) | `churn_probability`, `risk_tier` |
| LTV prediction | Regression (`Lifetime_Value`) | `predicted_ltv`, `ltv_tier`, `ltv_confidence` |
| Retention decision | Rule table (not ML) | `retention_strategy` |

Customer features are defined once in **Feast** (offline Parquet + online SQLite). Champion models are loaded from **DagsHub-hosted MLflow** by alias `@champion`. The FastAPI app (`app/main.py`) scores customers and logs predictions. The Streamlit dashboard (`app/Home.py`) calls that API for Customer Lookup and High-Risk Cohort views.

Training used a **50,000-row** Kaggle e-commerce behavior dataset with real `Churned` and `Lifetime_Value` columns. Continuous retraining can inject **synthetic** “new customers” via SDV’s `GaussianCopulaSynthesizer` (clearly labeled `data_source=synthetic`), then promote a new `@champion` only if metrics improve on a fixed **real Q4** holdout.

---



## Live app

Run locally or with Docker (see [Quick Start](#quick-start) and [Docker](#docker)):

```text
Dashboard:  http://localhost:8501
API docs:   http://localhost:8000/docs
```

There is no separate Streamlit Community Cloud deploy wired in this repo by default. Point a host at `app/Home.py` only if the FastAPI service is reachable via `API_BASE_URL`.

---



## Features

- Feast feature store with two services: `churn_ltv_feature_service` (includes `Lifetime_Value`) and `ltv_feature_service` (excludes it to avoid target leakage)
- CatBoost `@champion` models for churn and LTV on DagsHub MLflow (sklearn flavors also supported if a non-CatBoost model wins promotion)
- FastAPI scoring: `/predict_churn`, `/predict_ltv`, `/predict_full`, `/high_risk_customers`
- Streamlit multipage UI: Home, Customer Lookup, High-Risk Cohort
- Rule-based retention recommendations from risk × LTV tiers
- DVC-tracked raw/processed data; Feast online store materialized for serving
- Synthetic batch (5,000 rows) + retrain + conditional `@champion` promotion in CI
- Docker image for API + dashboard (`Dockerfile`, `docker compose up --build`)
- GitHub Actions: lint, tests, Feast setup, retrain cycle, Docker Hub push

---



## Technologies


| Area                                          | What this repo uses                                                                 |
| --------------------------------------------- | ----------------------------------------------------------------------------------- |
| Language                                      | Python **3.11** (GitHub Actions and `Dockerfile`)                                   |
| Dashboard                                     | Streamlit (`app/Home.py`, `app/pages/`)                                             |
| API                                           | FastAPI + Uvicorn (`app/main.py`)                                                   |
| Feature store                                 | Feast **0.66.0** (offline Parquet + online SQLite)                                  |
| Experiment tracking / registry                | MLflow on [DagsHub](https://dagshub.com/)                                           |
| Data versioning                               | DVC + Git (remote on DagsHub)                                                       |
| Training stack (repo root `requirements.txt`) | CatBoost, scikit-learn, XGBoost, LightGBM, SHAP, SDV, imbalanced-learn              |
| Tests                                         | pytest (`tests/`)                                                                   |
| Containers                                    | `Dockerfile`, `docker-compose.yml`                                                  |
| CI                                            | `.github/workflows/mlops_pipeline.yml`                                              |


Raw and processed datasets live under `data/` (DVC). Feast’s own registry and online store live under `feature_repo/data/` (gitignored). Models are **not** stored in Git; serving loads them from DagsHub MLflow.

---



## Architecture

```
Kaggle ecommerce data ──▶ DVC (data/raw, data/processed)
                                    │
                                    ▼
                         Feast feature_repo/ ──▶ offline Parquet + online SQLite
                                    │
         Colab notebooks / CI retrain ──▶ DagsHub MLflow (@candidate / @champion)
                                    │
                                    ▼
                          FastAPI (app/main.py)
                           Feast online + MLflow load
                                    │
                                    ▼
                     Streamlit (app/Home.py + pages)
                     calls API via API_BASE_URL
```

Continuous retrain path (CI publish / weekly cron):

```
dvc pull → synthetic 5000 rows → retrain all models → promote if better on real Q4
        → feast apply/materialize → docker build/push
```

---



## Prerequisites

Install these before you clone.


| Tool                                     | Required?                            | Version in this repo                           | Check                                                              |
| ---------------------------------------- | ------------------------------------ | ---------------------------------------------- | ------------------------------------------------------------------ |
| Git                                      | Yes                                  | Not pinned                                     | `git --version`                                                    |
| Python                                   | Yes                                  | **3.11** (Actions + Docker `python:3.11-slim`) | `python --version`                                                 |
| pip                                      | Yes (comes with Python)              | Not pinned                                     | `python -m pip --version`                                          |
| DagsHub account + token                  | Yes for MLflow + DVC pull            | —                                              | [dagshub.com](https://dagshub.com/)                                |
| Docker Desktop / Docker Engine + Compose | Only if you use Compose              | —                                              | `docker --version` then `docker compose version`                   |
| VS Code                                  | Optional                             | —                                              | —                                                                  |


On Windows, use **Command Prompt**, **PowerShell**, or the **VS Code terminal**. Run clone/install/run commands from the folder where you want the project (or from the repo root after clone).

---



## Quick Start

Shortest path: clone, Python 3.11 venv, dependencies, `.env`, Feast materialize, API + Streamlit.

```bash
git clone https://github.com/hamzajawad123/e-commerce-churn-predictor.git
cd e-commerce-churn-predictor
python -m venv .venv
```

**Windows (Command Prompt / PowerShell / VS Code terminal):**

```bat
.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env
```

**macOS / Linux:**

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` in the **repository root** (see [Environment configuration](#environment-configuration)). Pull data and materialize Feast:

```bash
dvc pull
bash scripts/feast_setup.sh
```

On Windows Git Bash / WSL, use the same `bash scripts/feast_setup.sh`. Then start API and dashboard (two terminals — see below).

```text
API:        http://localhost:8000/docs
Dashboard:  http://localhost:8501
```

You need DagsHub MLflow models registered with alias `@champion`, and a materialized Feast online store. If data or models are missing, follow [Data, training, and notebooks](#data-training-and-notebooks).

To run from Docker instead of a local Python venv, see [Docker](#docker).

---



## Step-by-step setup



### Step 1 — Install required software

1. Install Git.
2. Install **Python 3.11**. Confirm with `python --version`.
3. Create a DagsHub account, open (or create) the project repo, and create an access token.
4. Optional: install Docker Desktop if you will run `docker compose`.
5. Optional: install [VS Code](https://code.visualstudio.com/).



### Step 2 — Clone the repository

In Command Prompt, PowerShell, or a VS Code terminal, `cd` to the parent folder where you want the project, then:

```bash
git clone https://github.com/hamzajawad123/e-commerce-churn-predictor.git
cd e-commerce-churn-predictor
```

GitHub creates a folder named `e-commerce-churn-predictor`. The clone URL is `https://github.com/hamzajawad123/e-commerce-churn-predictor`.

### Step 3 — Open the project in VS Code (optional)

1. Start VS Code.
2. **File → Open Folder…** and select the cloned `e-commerce-churn-predictor` folder.
3. Confirm the explorer root contains `README.md`, `app/`, `src/`, `feature_repo/`, and `.env.example`.
4. Open a terminal: **Terminal → New Terminal** (or `Ctrl+``).
5. The terminal working directory should be the repo root.

If the VS Code `code` CLI is installed:

```bash
cd e-commerce-churn-predictor
code .
```



### Step 4 — Create a virtual environment and install dependencies

From the **repository root**, with Python 3.11:

```bash
python -m venv .venv
```

Activate:

```bat
.venv\Scripts\activate
```

```bash
source .venv/bin/activate
```

**Full project** (API, dashboard, Feast, training scripts, notebooks helpers, pytest):

```bash
python -m pip install -r requirements.txt
```

This repo uses a **single** root `[requirements.txt](requirements.txt)` for local runs, Docker, and CI.

### Step 5 — Configure environment variables

See the next section. Create `.env` in the **repository root**, not inside `app/`.

### Step 6 — Pull data and set up Feast

```bash
dvc pull
bash scripts/feast_setup.sh
```

### Step 7 — Start the API and dashboard

See [Run the API](#run-the-api) and [Run the dashboard](#run-the-dashboard). The Streamlit UI **calls** FastAPI via `API_BASE_URL`.

---



## Environment configuration

1. Copy the example file in the **repository root**:

**Windows:**

```bat
copy .env.example .env
```

**macOS / Linux:**

```bash
cp .env.example .env
```

2. Open `.env` and put **your** values. Never commit `.env` (it is listed in `.gitignore`).

Variables from `[.env.example](.env.example)`:


| Variable                      | Required?                                      | Role                                      |
| ----------------------------- | ---------------------------------------------- | ----------------------------------------- |
| `DAGSHUB_REPO_OWNER`          | Yes for DagsHub / CI                           | DagsHub username                          |
| `DAGSHUB_REPO_NAME`           | Yes for DagsHub / CI                           | Repo name                                 |
| `MLFLOW_TRACKING_URI`         | Yes for model load / retrain                   | `https://dagshub.com/<owner>/<repo>.mlflow` |
| `MLFLOW_TRACKING_USERNAME`    | Yes                                            | Usually same as DagsHub owner             |
| `MLFLOW_TRACKING_PASSWORD`    | Yes                                            | DagsHub token                             |
| `MODEL_NAME_CHURN`            | Optional                                       | Default `churn_model`                     |
| `MODEL_NAME_LTV`              | Optional                                       | Default `ltv_model`                       |
| `MODEL_ALIAS`                 | Optional                                       | Default `champion`                        |
| `FEAST_REPO_PATH`             | Optional                                       | Default `./feature_repo`                  |
| `API_BASE_URL`                | Optional (dashboard)                           | Default `http://localhost:8000`           |
| `PREDICTIONS_DB_PATH`         | Optional                                       | Default `predictions_log.db`              |
| `CHURN_RISK_HIGH_THRESHOLD`   | Optional                                       | Default `0.7`                             |
| `CHURN_RISK_MEDIUM_THRESHOLD` | Optional                                       | Default `0.4`                             |
| `LTV_HIGH_THRESHOLD`          | Optional                                       | Default `1440.63` (EDA mean LTV)          |
| `DVC_REMOTE_URL`              | Optional                                       | DagsHub DVC remote URL                    |


Example shape (use your own secrets):

```env
DAGSHUB_REPO_OWNER=your_dagshub_username
DAGSHUB_REPO_NAME=e-commerce-churn-predictor
MLFLOW_TRACKING_URI=https://dagshub.com/your_dagshub_username/e-commerce-churn-predictor.mlflow
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token_here
API_BASE_URL=http://localhost:8000
FEAST_REPO_PATH=./feature_repo
MODEL_ALIAS=champion
```

Serving fails to load models if `MLFLOW_TRACKING_URI` (and credentials) are missing. The dashboard fails to score customers if `API_BASE_URL` does not reach a running API.

---



## Run the dashboard

**Where:** repository root, venv activated, `.env` filled, `requirements.txt` installed, **API already running**, Feast online store materialized.

**Terminal — Streamlit**

```bash
streamlit run app/Home.py
```

**Browser**

```text
http://localhost:8501
```

Pages:

- **Home** — overview
- **Customer Lookup** — score one customer via `GET /predict_full/{id}`
- **High-Risk Cohort** — list from `GET /high_risk_customers`

Unlike a standalone Hopsworks dashboard, this UI **does** need Uvicorn running (or the Compose `api` service). Leave both terminals open while you use the app.

---



## Run the API

**Where:** repository root, venv activated, `requirements.txt` installed, `.env` with MLflow + Feast settings, `dvc pull` + `bash scripts/feast_setup.sh` already done, `@champion` models available on DagsHub.

**Terminal — FastAPI**

```bash
uvicorn app.main:app --reload --port 8000
```

Uvicorn binds port **8000**. The Docker Compose `api` service also serves on port **8000**.


| Check              | URL                                              |
| ------------------ | ------------------------------------------------ |
| Health             | `http://localhost:8000/health`                   |
| Churn              | `http://localhost:8000/predict_churn/0`          |
| LTV                | `http://localhost:8000/predict_ltv/0`            |
| Full + retention   | `http://localhost:8000/predict_full/0`           |
| High-risk list     | `http://localhost:8000/high_risk_customers`      |
| OpenAPI UI         | `http://localhost:8000/docs`                     |


---



## Run the complete local stack

Two processes: the dashboard calls the API.

### Terminal 1 — API

```bash
uvicorn app.main:app --reload --port 8000
```

```text
http://localhost:8000/health
```



### Terminal 2 — Dashboard

```bash
streamlit run app/Home.py
```

```text
http://localhost:8501
```

Feast must already be materialized and MLflow `@champion` aliases must exist for scores to appear.

---



## Docker

You can run this project from **one Docker image** instead of installing every Python package on your machine. The image is defined in `[Dockerfile](Dockerfile)` (`python:3.11-slim`). Compose starts **two services** from that image: FastAPI on port **8000** and Streamlit on port **8501**.

Published image name (CI): **`DOCKERHUB_USERNAME/ecommerce-churn-ltv`** (for example `hamzajawad/ecommerce-churn-ltv` if that is your Hub user).

You still need Docker Desktop (Windows/macOS) or Docker Engine + Compose, and a root `.env` (DagsHub / MLflow credentials). Secrets are not baked into the image; pass them at run time with Compose `env_file: .env` or `docker run --env-file .env`.

Confirm Docker:

```bash
docker --version
docker compose version
```

Clone (same as [Step 2](#step-2--clone-the-repository)), `cd e-commerce-churn-predictor`, and copy `.env.example` to `.env`.

**Important:** the image expects a **materialized** `feature_repo/` (registry + online store) in the build context. Run `bash scripts/feast_setup.sh` locally (or rely on CI’s publish job) before `docker build` if you need online features inside the container.

### Option A — Docker Compose (recommended)

From the **repository root**, with `.env` present (`docker-compose.yml` uses `env_file: .env`):

```bash
docker compose up --build
```

This builds the image and starts both services. The first run downloads `python:3.11-slim` if it is not already on your machine.


| Service     | Host ports | Command                                              |
| ----------- | ---------- | ---------------------------------------------------- |
| `api`       | **8000**   | `uvicorn app.main:app --host 0.0.0.0 --port 8000`    |
| `dashboard` | **8501**   | `streamlit run app/Home.py` (port 8501)              |


Compose sets `API_BASE_URL=http://api:8000` on the dashboard so it reaches the API service by name.

Open:

```text
Dashboard:  http://localhost:8501
API health: http://localhost:8000/health
API docs:   http://localhost:8000/docs
```

Run in the background:

```bash
docker compose up --build -d
```

Stop and remove the containers:

```bash
docker compose down
```

### Option B — Build and run the image yourself

From the **repository root** (build context is `.` because the Dockerfile copies `app/`, `src/`, and `feature_repo/`):

**1. Build**

```bash
docker build -t hamzajawad/ecommerce-churn-ltv:latest .
```

**2. Confirm the image exists**

```bash
docker images hamzajawad/ecommerce-churn-ltv
```

**3. Run API**

```bash
docker run --rm --name churn-api --env-file .env -p 8000:8000 hamzajawad/ecommerce-churn-ltv:latest
```

**4. Run dashboard** (second container; point at the host API)

```bash
docker run --rm --name churn-dashboard --env-file .env -e API_BASE_URL=http://host.docker.internal:8000 -p 8501:8501 hamzajawad/ecommerce-churn-ltv:latest streamlit run app/Home.py --server.port 8501 --server.address 0.0.0.0
```

Prefer [Option A](#option-a--docker-compose-recommended) so networking between API and dashboard is automatic.

```text
Dashboard:  http://localhost:8501
API health: http://localhost:8000/health
```

**5. Stop**

In the terminal running a container: **Ctrl+C**.

Or from another terminal:

```bash
docker stop churn-api
docker stop churn-dashboard
```

`--env-file .env` passes the same variables Compose uses. `--rm` deletes the container when it stops; the **image** stays until you run `docker rmi hamzajawad/ecommerce-churn-ltv:latest`.

### Option C — Pull from Docker Hub

If the image is already on Docker Hub, you do not need to build from this repo:

```bash
docker pull hamzajawad/ecommerce-churn-ltv:latest
docker run --rm --name churn-api --env-file .env -p 8000:8000 hamzajawad/ecommerce-churn-ltv:latest
```

You still need a local `.env` with DagsHub MLflow credentials. Models are loaded from MLflow at runtime; they are not stored inside the image as pickle files.

To publish a locally built image:

```bash
docker login
docker push hamzajawad/ecommerce-churn-ltv:latest
```

---



## Data, training, and notebooks

Run these from the **repository root** with `[requirements.txt](requirements.txt)` and a complete `.env`.

### Pull versioned data (DVC)

```bash
dvc pull
```

Writes/restores:

- `data/raw/ecommerce_customer_churn_dataset.csv` (via `.dvc` pointer)
- `data/processed/features.parquet` (via `.dvc` pointer)

### Explore and train (Colab notebooks)

| Notebook | Purpose |
| -------- | ------- |
| `[notebooks/01_ecommerce_churn_eda_v1.ipynb](notebooks/01_ecommerce_churn_eda_v1.ipynb)` | EDA |
| `[notebooks/02_e_commerce_churn_feature_engineering_v1.ipynb](notebooks/02_e_commerce_churn_feature_engineering_v1.ipynb)` | Cleaning + feature engineering |
| `[notebooks/03_e_commerce_churn_train_v1.ipynb](notebooks/03_e_commerce_churn_train_v1.ipynb)` | Train, evaluate, register `@candidate` on DagsHub |

Open in Google Colab or locally with Jupyter after dependencies are installed.

### Apply + materialize Feast

```bash
bash scripts/feast_setup.sh
```

### Promote `@candidate` → `@champion` (manual)

After reviewing Phase 5 / notebook metrics:

```bash
python scripts/promote_champion.py
```

Prompts for a DagsHub token, then moves both `churn_model` and `ltv_model` aliases.

### Synthetic data + continuous retrain (local)

```bash
python scripts/fit_synthetic_generator.py
python scripts/generate_synthetic_batch.py --num-rows 5000 --also-write-features
python scripts/run_retrain_cycle.py --num-rows 5000
bash scripts/feast_setup.sh
```

`run_retrain_cycle.py` samples synthetic rows, merges with real data, retrains candidate models, logs to MLflow, and promotes `@champion` only if **real Q4** metrics strictly improve (higher churn ROC-AUC / lower LTV RMSE).

Synthetic rows are labeled `data_source=synthetic` and are **not** presented as live traffic.

---



## GitHub Actions

Add **repository secrets**:

| Secret                 | Used for                          |
| ---------------------- | --------------------------------- |
| `DAGSHUB_REPO_OWNER`   | DVC auth + MLflow URI             |
| `DAGSHUB_REPO_NAME`    | MLflow URI                        |
| `DAGSHUB_TOKEN`        | DVC + MLflow password             |
| `DOCKERHUB_USERNAME`   | Image push                        |
| `DOCKERHUB_TOKEN`      | Image push                        |


| Workflow                                                                       | Trigger                                                         | What it runs |
| ------------------------------------------------------------------------------ | --------------------------------------------------------------- | ------------ |
| `[.github/workflows/mlops_pipeline.yml](.github/workflows/mlops_pipeline.yml)` | PR / push to `main`, weekly cron (`0 6 * * 1`), workflow_dispatch | **ci:** `dvc pull`, flake8, Feast setup, pytest, Docker build. **publish** (main / schedule / manual): synthetic+retrain cycle → Feast → Docker Hub push |


Python **3.11**. Jobs install root `requirements.txt`.

---



## API endpoints

Defined in `[app/main.py](app/main.py)`. No request body. No auth in this file.


| Method | Path                          | Purpose                                                                 |
| ------ | ----------------------------- | ----------------------------------------------------------------------- |
| GET    | `/health`                     | Status plus whether churn/LTV models loaded                             |
| GET    | `/predict_churn/{customer_id}`| Churn probability + risk tier                                           |
| GET    | `/predict_ltv/{customer_id}`  | Predicted LTV + tier + confidence                                       |
| GET    | `/predict_full/{customer_id}` | Churn + LTV + retention strategy; writes SQLite prediction log          |
| GET    | `/high_risk_customers`        | Latest high-risk prediction per customer (deduped), optional `?limit=`  |


Unknown customers return **404** when Feast has no online features for that `Customer_ID`.

Retention tiers use env thresholds (`CHURN_RISK_*`, `LTV_HIGH_THRESHOLD`). Q4 signups get `ltv_confidence=low` because of a known walk-forward distribution shift.

---



## Project structure

```text
e-commerce-churn-predictor/
├── .github/workflows/
│   └── mlops_pipeline.yml
├── app/
│   ├── Home.py
│   ├── main.py
│   ├── feature_client.py
│   ├── model_loader.py
│   ├── schemas.py
│   ├── ui_theme.py
│   ├── __init__.py
│   └── pages/
│       ├── 1_Customer_Lookup.py
│       └── 2_High_Risk_Cohort.py
├── data/
│   ├── raw/                 # DVC-tracked CSV (+ .dvc pointer)
│   └── processed/           # DVC-tracked features.parquet (+ .dvc pointer)
├── feature_repo/
│   ├── feature_store.yaml
│   ├── entities.py
│   ├── data_sources.py
│   ├── features.py
│   ├── feature_services.py
│   └── data/                # Feast registry + online store (gitignored)
├── notebooks/
│   ├── 01_ecommerce_churn_eda_v1.ipynb
│   ├── 02_e_commerce_churn_feature_engineering_v1.ipynb
│   └── 03_e_commerce_churn_train_v1.ipynb
├── scripts/
│   ├── feast_setup.sh
│   ├── fit_synthetic_generator.py
│   ├── generate_synthetic_batch.py
│   ├── promote_champion.py
│   └── run_retrain_cycle.py
├── src/
│   ├── feature_engineering.py
│   ├── retention_policy.py
│   └── retrain_pipeline.py
├── tests/
│   ├── test_api.py
│   ├── test_features.py
│   └── test_retention_policy.py
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── README.md
└── requirements.txt
```

---



## Folder and file guide


| Path                                                                   | Purpose                                                |
| ---------------------------------------------------------------------- | ------------------------------------------------------ |
| `[app/Home.py](app/Home.py)`                                           | Streamlit dashboard entry                              |
| `[app/pages/1_Customer_Lookup.py](app/pages/1_Customer_Lookup.py)`     | Score one customer                                     |
| `[app/pages/2_High_Risk_Cohort.py](app/pages/2_High_Risk_Cohort.py)`   | High-risk cohort table                                 |
| `[app/main.py](app/main.py)`                                           | FastAPI app                                            |
| `[app/feature_client.py](app/feature_client.py)`                       | Feast online feature lookup                            |
| `[app/model_loader.py](app/model_loader.py)`                           | Load `@champion` from MLflow                           |
| `[app/schemas.py](app/schemas.py)`                                     | Pydantic response models                               |
| `[app/ui_theme.py](app/ui_theme.py)`                                   | Shared Streamlit theme + API helpers                   |
| `[src/retention_policy.py](src/retention_policy.py)`                   | Risk/LTV tiers + retention rule table                  |
| `[src/feature_engineering.py](src/feature_engineering.py)`             | Shared clean/engineer for real + synthetic data        |
| `[src/retrain_pipeline.py](src/retrain_pipeline.py)`                   | Retrain, compare on real Q4, conditional promote       |
| `[feature_repo/](feature_repo/)`                                       | Feast definitions                                      |
| `[scripts/feast_setup.sh](scripts/feast_setup.sh)`                     | `feast apply` + materialize                            |
| `[scripts/run_retrain_cycle.py](scripts/run_retrain_cycle.py)`         | Synthetic batch + retrain (CI entrypoint)              |
| `[scripts/promote_champion.py](scripts/promote_champion.py)`           | Manual candidate → champion                            |
| `[.env.example](.env.example)`                                         | Env template                                           |
| `[docker-compose.yml](docker-compose.yml)`                             | API :8000 + dashboard :8501                            |
| `[Dockerfile](Dockerfile)`                                             | Image for API and dashboard                            |
| `[tests/](tests/)`                                                     | pytest                                                 |


---



## Common commands


| Command                                                                 | What it does                                          |
| ----------------------------------------------------------------------- | ----------------------------------------------------- |
| `python -m pip install -r requirements.txt`                             | Install all dependencies                              |
| `dvc pull`                                                              | Restore DVC-tracked data                              |
| `bash scripts/feast_setup.sh`                                           | Apply + materialize Feast                             |
| `uvicorn app.main:app --reload --port 8000`                             | Start API on port 8000                                |
| `streamlit run app/Home.py`                                             | Start dashboard                                       |
| `python scripts/promote_champion.py`                                    | Promote `@candidate` → `@champion`                    |
| `python scripts/fit_synthetic_generator.py`                             | Fit SDV synthesizer                                   |
| `python scripts/generate_synthetic_batch.py --num-rows 5000`            | Sample synthetic rows                                 |
| `python scripts/run_retrain_cycle.py --num-rows 5000`                   | Full synthetic + retrain cycle                        |
| `docker compose up --build`                                             | Build and start API + dashboard                       |
| `docker compose up --build -d`                                          | Same, detached                                        |
| `docker compose down`                                                   | Stop Compose stack                                    |
| `docker build -t hamzajawad/ecommerce-churn-ltv:latest .`               | Build the image                                       |
| `docker pull hamzajawad/ecommerce-churn-ltv:latest`                     | Pull the image from Docker Hub                        |
| `docker run --rm --env-file .env -p 8000:8000 hamzajawad/ecommerce-churn-ltv:latest` | Run API from the image               |
| `pytest tests/`                                                         | Run tests                                             |
| `flake8 app/ src/ tests/ --max-line-length=120`                         | Lint                                                  |


---



## Tests

From the repository root, after installing root `requirements.txt` and materializing Feast (`bash scripts/feast_setup.sh`):

```bash
pytest tests/ -v
```

- `test_retention_policy.py` — rule engine thresholds (no Feast required)
- `test_features.py` — online feature shape / leakage checks (Feast required)
- `test_api.py` — FastAPI routes with stubbed CatBoost models (Feast required)

---



## How to stop

- Streamlit or Uvicorn in a terminal: **Ctrl+C**
- Docker Compose: `docker compose down` from the repo root
- Docker containers started with `docker run --name churn-api` / `churn-dashboard`: `docker stop churn-api` (and `docker stop churn-dashboard`)

---



## Troubleshooting

**Missing** `.env` **/** models fail to load  
Set `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD` (DagsHub token). Copy `.env.example` to `.env` in the repo root.

**Dashboard cannot reach API**  
Confirm Uvicorn is running. Local default is `API_BASE_URL=http://localhost:8000`. Under Compose the dashboard uses `http://api:8000`.

**404 Customer not found**  
That `Customer_ID` is not in the Feast online store. Run `dvc pull` and `bash scripts/feast_setup.sh`. After a synthetic retrain cycle, materialize again so new IDs exist online.

**Empty or wrong features / FileNotFoundError in Feast**  
`feature_repo/data_sources.py` uses a relative path to `data/processed/features.parquet`. Run commands from the repo root. Do not bake absolute Windows paths into the Feast registry.

**DVC pull fails (auth)**  
Configure DagsHub credentials for the DVC remote (same token as MLflow). CI uses `DAGSHUB_REPO_OWNER` + `DAGSHUB_TOKEN`.

**No `@champion` models**  
Register candidates from the training notebook, then run `python scripts/promote_champion.py`, or wait for a successful CI retrain that beats the current champion.

**Port 8501 or 8000 already in use**  
Stop the other process or change the port in the run command. Compose uses 8501 and 8000.

`docker compose` **cannot start**  
Compose requires a root `.env` (`env_file: .env`). Docker Desktop (or Engine + Compose plugin) must be running.

**Python version**  
Use 3.11 to match Actions and `python:3.11-slim`.

**venv not active**  
Windows: `.venv\Scripts\activate`. If `python` is not found, use `py -3.11`.

**Tests fail on Feast**  
Run `bash scripts/feast_setup.sh` before `pytest tests/`.

---



## Security

Never commit DagsHub tokens, Docker Hub passwords, or `.env` to GitHub or into a Docker image. `.gitignore` ignores `.env`. `.dockerignore` keeps `.env` out of the build context. Pass secrets at run time with `--env-file .env`, Compose `env_file`, or GitHub Actions secrets.

There is no `SECURITY.md` in this repository.

---



## Useful links


|                   |                                                                                                                                |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Source repository | [https://github.com/hamzajawad123/e-commerce-churn-predictor](https://github.com/hamzajawad123/e-commerce-churn-predictor)     |
| DagsHub project   | [https://dagshub.com/hamzajawad123/e-commerce-churn-predictor](https://dagshub.com/hamzajawad123/e-commerce-churn-predictor)   |
| Docker Hub        | [https://hub.docker.com/r/hamzajawad/ecommerce-churn-ltv](https://hub.docker.com/r/hamzajawad/ecommerce-churn-ltv) (after push) |
| Issue tracker     | [https://github.com/hamzajawad123/e-commerce-churn-predictor/issues](https://github.com/hamzajawad123/e-commerce-churn-predictor/issues) |
| Feast docs        | [https://docs.feast.dev/](https://docs.feast.dev/)                                                                             |
| MLflow            | [https://mlflow.org/](https://mlflow.org/)                                                                                     |
| DagsHub           | [https://dagshub.com/](https://dagshub.com/)                                                                                   |
| Env template      | `[.env.example](.env.example)`                                                                                                 |


