# Hybrid Recommendation Intelligence Platform

A production-oriented hybrid recommendation system inspired by Netflix and YouTube recommendation architectures. Built as a learning and portfolio project using IBM Bob (AI-assisted development).

## System Overview

The platform combines multiple recommendation strategies:

- **Content-Based Filtering** — recommend items based on item attributes
- **Collaborative Filtering** — recommend items based on user behavior patterns
- **Hybrid Recommendation** — combine multiple signals for better accuracy
- **Candidate Generation** — fast retrieval of candidate items at scale
- **ML Ranking** — re-rank candidates using learned user preferences
- **Feedback Loop** — continuously improve recommendations from user interactions

> **Current Status: Phase 3 — Baseline Recommendation Systems ✅ Complete**
> Phases 1–3 are complete. KuaiRec 2.0 (4.6M interactions) processed, temporal splits created, 5 non-ML baseline recommenders built and evaluated. Best baseline NDCG@10: **0.4461** (PopularityRecommender). Phase 4 (ML-Based Recommendation) is next.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React 19 + TypeScript + Vite |
| Backend | Python 3.14 + FastAPI |
| Database | PostgreSQL 16 |
| ORM | SQLAlchemy |
| Migrations | Alembic |
| Configuration | Pydantic Settings |
| Containerization | Docker + Docker Compose V2 |

---

## Prerequisites

Ensure the following are installed before proceeding:

| Tool | Required Version | Check |
|---|---|---|
| Git | Any recent | `git --version` |
| Python | 3.13+ | `python --version` |
| Node.js | 18+ | `node --version` |
| npm | 9+ | `npm --version` |
| Docker Desktop | Current | `docker --version` |

> **Windows users:** If running `.ps1` scripts is blocked, run this once in PowerShell as your user:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

---

## Quick Start (Windows PowerShell)

### 1. Clone the repository

```powershell
git clone <your-github-repo-url>
cd hybrid-recommendation-intelligence-platform
```

### 2. Configure environment variables

```powershell
Copy-Item .env.example .env
# Open .env and set your values — especially DB_PASSWORD and SECRET_KEY
notepad .env
```

### 3. Start the database

```powershell
docker compose up -d db
# Wait ~10 seconds for PostgreSQL to initialize
docker compose ps   # db should show "healthy"
```

### 4. Set up and start the backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
alembic upgrade head
uvicorn app.main:app --reload
```

Backend available at: `http://localhost:8000`
API docs available at: `http://localhost:8000/docs`

### 5. Start the frontend (new terminal)

```powershell
cd frontend
npm install
npm run dev
```

Frontend available at: `http://localhost:5173`

---

## API Endpoints (Phase 1)

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/health` | Application health status |
| GET | `/api/v1/health/db` | Database connectivity check |
| GET | `/api/v1/system/info` | Safe system information |

---

## Running Tests

### Backend (Phase 1 + Phase 2 + Phase 3 — 134 tests)

```powershell
# From project root
backend\.venv\Scripts\pytest.exe backend/tests/ -v
backend\.venv\Scripts\ruff.exe check backend/app/ scripts/ ml/ backend/tests/ml/
```

### Frontend

```powershell
cd frontend
npm run lint
npx tsc --noEmit
npm run build
```

---

## Phase 2 Data Pipeline

```powershell
# From project root (requires backend venv + dataset downloaded)
backend\.venv\Scripts\python.exe scripts/data/download_dataset.py
backend\.venv\Scripts\python.exe scripts/data/inspect_dataset.py
backend\.venv\Scripts\python.exe scripts/data/validate_dataset.py
backend\.venv\Scripts\python.exe scripts/data/convert_to_parquet.py
backend\.venv\Scripts\python.exe scripts/data/generate_report.py
```

> **Note:** Dataset files (~1.5 GB) are excluded from Git. Run `download_dataset.py` after cloning.

## Phase 3 Baseline Experiments

```powershell
# From project root (requires Phase 2 data files)
backend\.venv\Scripts\python.exe scripts/ml/prepare_splits.py
backend\.venv\Scripts\python.exe scripts/ml/run_baselines.py
```

---

## Project Structure

```
hybrid-recommendation-intelligence-platform/
├── frontend/           # React + TypeScript + Vite
├── backend/            # Python 3.14 + FastAPI
├── data/               # Data files (excluded from Git)
├── ml/                 # ML modules (Phase 2+)
├── notebooks/          # Jupyter notebooks (Phase 2+)
├── docs/               # Technical documentation
├── scripts/            # Utility scripts
├── docker-compose.yml  # Local development services
├── .env.example        # Environment variable template
├── AGENTS.md           # AI development contract
└── phase-1-plan.md     # Phase 1 implementation plan
```

---

## Development Phases

| Phase | Scope | Status |
|---|---|---|
| **Phase 1** | Foundation — dev environment, API skeleton, database | ✅ Complete |
| **Phase 2** | Data Engineering — KuaiRec 2.0 acquisition, Parquet pipeline, quality report | ✅ Complete |
| **Phase 3** | Baseline Systems — EDA, splits, evaluation framework, 5 baseline recommenders | ✅ Complete |
| Phase 4 | ML Recommendation — Content-Based Filtering, Collaborative Filtering (ALS) | 🔜 Next |
| Phase 5 | ML Ranking — candidate generation, LightGBM/XGBoost, MLflow | ⏳ Planned |
| Phase 6 | Feedback Loop — online learning, A/B testing, monitoring | ⏳ Planned |

---

## Documentation

- [`docs/architecture/system-architecture.md`](docs/architecture/system-architecture.md) — Component diagram and data flow
- [`docs/database/database-overview.md`](docs/database/database-overview.md) — Schema rationale and Alembic workflow
- [`docs/api/api-overview.md`](docs/api/api-overview.md) — Endpoint reference
- [`docs/phases/phase-1-setup-guide.md`](docs/phases/phase-1-setup-guide.md) — Step-by-step setup guide for Windows
- [`docs/phases/phase-3-baseline-guide.md`](docs/phases/phase-3-baseline-guide.md) — Phase 3 setup and results
- [`docs/decisions/architecture-decisions.md`](docs/decisions/architecture-decisions.md) — Architecture Decision Records (ADR-001 through ADR-P3-07)
- [`docs/ml/evaluation-methodology.md`](docs/ml/evaluation-methodology.md) — Offline evaluation metrics and split strategy
- [`docs/ml/baseline-algorithms.md`](docs/ml/baseline-algorithms.md) — All 5 baseline recommender algorithms
- [`docs/ml/experiment-guide.md`](docs/ml/experiment-guide.md) — How to run and record experiments
- [`docs/data/dataset-overview.md`](docs/data/dataset-overview.md) — KuaiRec 2.0 dataset summary
- [`docs/data/data-dictionary.md`](docs/data/data-dictionary.md) — Column-level data dictionary
- [`docs/data/data-acquisition-guide.md`](docs/data/data-acquisition-guide.md) — How to re-acquire the dataset
- [`docs/data/data-pipeline.md`](docs/data/data-pipeline.md) — Phase 2 pipeline design and conventions
- [`docs/data/data-quality-report.md`](docs/data/data-quality-report.md) — Data quality findings
- [`AGENTS.md`](AGENTS.md) — AI development contract and project rules

---

## Contributing

This project follows Conventional Commits. See [`AGENTS.md`](AGENTS.md) for coding conventions, branch naming, and security rules.

---

*Developed with IBM Bob (AI-assisted development environment)*
