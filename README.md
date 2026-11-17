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

> **Current Status: Phase 1 — Foundation**
> The recommendation engine is not yet implemented. Phase 1 establishes the complete local development foundation.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + TypeScript + Vite |
| Backend | Python 3.13 + FastAPI |
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

### Backend

```powershell
cd backend
.venv\Scripts\Activate.ps1
pytest tests/ -v
ruff check app/
ruff format --check app/
```

### Frontend

```powershell
cd frontend
npm run lint
npx tsc --noEmit
npm run build
```

---

## Project Structure

```
hybrid-recommendation-intelligence-platform/
├── frontend/           # React + TypeScript + Vite
├── backend/            # Python 3.13 + FastAPI
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
| **Phase 1** | Foundation — dev environment, API skeleton, database | 🔄 In Progress |
| Phase 2 | Data Layer — ingestion, schema, Polars/DuckDB | ⏳ Planned |
| Phase 3 | Recommendation Engine — content-based, collaborative, hybrid | ⏳ Planned |
| Phase 4 | ML Ranking — candidate generation, model serving | ⏳ Planned |
| Phase 5 | Feedback Loop — online learning, A/B testing, monitoring | ⏳ Planned |

---

## Documentation

- [`docs/architecture/system-architecture.md`](docs/architecture/system-architecture.md) — Component diagram and data flow
- [`docs/database/database-overview.md`](docs/database/database-overview.md) — Schema rationale and Alembic workflow
- [`docs/api/api-overview.md`](docs/api/api-overview.md) — Endpoint reference
- [`docs/phases/phase-1-setup-guide.md`](docs/phases/phase-1-setup-guide.md) — Step-by-step setup guide for Windows
- [`docs/decisions/architecture-decisions.md`](docs/decisions/architecture-decisions.md) — Architecture Decision Records
- [`AGENTS.md`](AGENTS.md) — AI development contract and project rules

---

## Contributing

This project follows Conventional Commits. See [`AGENTS.md`](AGENTS.md) for coding conventions, branch naming, and security rules.

---

*Developed with IBM Bob (AI-assisted development environment)*
