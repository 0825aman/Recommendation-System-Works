# AGENTS.md — Hybrid Recommendation Intelligence Platform
## AI Development Contract

This file is the authoritative guide for every AI-assisted development session on this project.
Read this file completely before taking any action. Never deviate from the rules below without
explicit written approval from the project owner.

---

## 1. Project Objective

Build a production-oriented, advanced hybrid recommendation platform inspired by Netflix and
YouTube recommendation systems. The system combines:

- Content-Based Filtering
- Collaborative Filtering
- Hybrid Recommendation
- Candidate Generation
- Machine Learning Ranking
- Feedback Loop

This is a learning and portfolio project developed with IBM Bob (AI-assisted development
environment). IBM Bob is NOT an IBM Cloud hosting service — it is the local AI development tool.

---

## 2. Approved Technology Stack

### Core (All Phases)
| Layer | Technology | Version |
|---|---|---|
| AI-assisted development | IBM Bob | current |
| Version control | Git + GitHub | current |
| Frontend | React + TypeScript + Vite | latest stable |
| Backend | Python + FastAPI | Python 3.14, FastAPI latest stable |
| Database | PostgreSQL | 16 (Docker) |
| ORM | SQLAlchemy | latest stable |
| Migrations | Alembic | latest stable |
| Configuration | Pydantic Settings | latest stable |
| Containerization | Docker + Docker Compose V2 | current |
| Testing (backend) | Pytest | latest stable |
| Linting/formatting | Ruff | latest stable |
| HTTP client (frontend) | Axios | latest stable |

### Phase 2 Active (installed in Phase 2)
| Technology | Purpose |
|---|---|
| Polars 2.x | DataFrame operations — cleaning, type casting, schema inspection |
| DuckDB 2.x | SQL analytics on Parquet files — statistics, distributions |
| PyArrow | Parquet I/O backend for Polars |
| tqdm | Download progress bars |
| rich | Formatted terminal output |

### Future Phases Only — Do NOT Install Before Required
| Technology | Phase |
|---|---|
| NumPy, SciPy | Numerical computing — Phase 4+ |
| scikit-learn | ML — Phase 4+ |
| PyTorch | Deep learning — Phase 4+ |
| LightGBM or XGBoost | Ranking — Phase 4+ |
| TF-IDF, Sentence Transformers | Text processing — Phase 4+ |
| FAISS CPU | Vector retrieval — Phase 4+ |
| MLflow | Experiment tracking — Phase 4+ |
| Redis | Caching — only when justified |

### Development Environment
- OS: Windows 10/11 x64, PowerShell
- RAM: 16 GB
- GPU: AMD Radeon (integrated) — CPU-first strategy
- Avoid GPU-dependent libraries in development

---

## 3. Architecture Principles

1. **Modular monolith** — not microservices. Keep all backend logic in one deployable unit initially.
2. **Logical separation** — frontend, backend, database, and future ML modules are logically separate.
3. **HTTP API boundary** — frontend communicates with backend only through versioned HTTP APIs.
4. **Backend owns data** — all database interactions go through the backend. Frontend never touches the database directly.
5. **Separation of concerns** — database access, business logic, API routes, and configuration are in separate modules.
6. **Environment variables for config** — use Pydantic Settings and `.env`. Never hardcode credentials.
7. **No credential exposure** — never log, return, or commit secrets of any kind.
8. **SQLAlchemy + Alembic** — all schema changes go through Alembic migrations. No manual DDL.
9. **Typed and testable** — all Python code is typed. All TypeScript is strict mode. All public functions have tests.
10. **Minimal dependencies** — do not add a library unless it is required. Prefer standard library equivalents.
11. **No premature infrastructure** — do not introduce services, queues, or caches unless a phase explicitly requires them.
12. **API versioning** — all backend routes are under `/api/v1/`. Future versions use `/api/v2/`.

---

## 4. Repository Structure

```
hybrid-recommendation-intelligence-platform/
├── frontend/               # React + TypeScript + Vite
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/          # Page-level components
│   │   ├── services/       # API service layer (Axios wrappers)
│   │   ├── hooks/          # Custom React hooks
│   │   ├── types/          # TypeScript interfaces and types
│   │   └── App.tsx
│   ├── public/
│   ├── package.json
│   └── vite.config.ts
│
├── backend/                # Python 3.14 + FastAPI
│   ├── app/
│   │   ├── api/v1/         # Versioned route handlers
│   │   ├── core/           # Config, logging, exceptions
│   │   ├── db/             # SQLAlchemy engine, session, base
│   │   ├── models/         # SQLAlchemy ORM models
│   │   ├── schemas/        # Pydantic request/response schemas
│   │   ├── services/       # Business logic (not in routes)
│   │   └── main.py         # FastAPI app factory
│   ├── tests/              # Pytest test suite
│   ├── alembic/            # Migration scripts
│   ├── alembic.ini
│   └── pyproject.toml
│
├── data/                   # Data files (all excluded from Git)
│   ├── raw/
│   ├── processed/
│   └── interim/
│
├── ml/                     # ML modules (Phase 3+, active)
│   ├── baselines/          # Phase 3 non-ML baseline recommenders
│   ├── data/               # Parquet loaders + temporal split logic
│   ├── evaluation/         # Offline metrics + Evaluator class
│   ├── experiments/        # JSON experiment result recorder
│   ├── features/           # Item + user feature extraction
│   ├── content_based/      # Phase 4+
│   ├── collaborative/      # Phase 4+
│   ├── hybrid/             # Phase 4+
│   ├── ranking/            # Phase 4+
│   └── config.py           # ExperimentConfig dataclass
│
├── notebooks/              # Jupyter notebooks (Phase 2+)
├── docs/                   # Technical documentation
│   ├── architecture/
│   ├── database/
│   ├── api/
│   ├── decisions/
│   └── phases/
│
├── scripts/                # Utility scripts
├── .github/workflows/      # CI/CD (Phase 2+)
│
├── .env.example            # Committed — placeholder values only
├── .gitignore
├── .dockerignore
├── docker-compose.yml
├── AGENTS.md               # This file
├── phase-1-plan.md         # Phase 1 implementation plan
└── README.md
```

**Rule:** Create only directories and files needed for the current phase. Do not create empty
placeholder implementations for future phases unless there is a specific, approved reason.

---

## 5. Coding Conventions

### Python
- Python 3.14, fully typed (use `from __future__ import annotations` where needed)
- Follow PEP 8. Enforce with Ruff (`ruff check`, `ruff format`)
- Ruff line length: 88 characters
- Use `snake_case` for functions, variables, modules
- Use `PascalCase` for classes
- Use `UPPER_SNAKE_CASE` for constants
- Docstrings on all public functions and classes
- FastAPI dependency injection via `Depends()`
- Pydantic models for all request and response bodies
- Never use `*` imports
- All database queries go through SQLAlchemy sessions — no raw SQL strings in routes

### TypeScript / React
- TypeScript strict mode (`"strict": true` in `tsconfig.json`)
- Functional components only — no class components
- Use named exports, not default exports for components (exception: page components)
- Use `camelCase` for variables and functions
- Use `PascalCase` for components and types
- Use `UPPER_SNAKE_CASE` for constants
- No `any` types — use `unknown` and narrow where needed
- One component per file
- Axios instance configured centrally in `src/services/api.ts`
- API response types defined in `src/types/`
- Lint with ESLint (`npm run lint`)

### Git
- Branch naming: `feature/<short-description>`, `fix/<short-description>`, `chore/<short-description>`
- Commit messages: Conventional Commits format
  - `feat:` new feature
  - `fix:` bug fix
  - `chore:` maintenance, config, tooling
  - `docs:` documentation only
  - `test:` tests only
  - `refactor:` no behavior change
- Example: `feat: add system info endpoint`
- Never commit directly to `main` for feature work (use feature branches + PR)
- Initial setup commits to `main` are acceptable during Phase 1 bootstrapping

---

## 6. Testing Requirements

### Backend (Pytest)
- Every API endpoint has at least one happy-path test
- Every API endpoint has at least one error/edge-case test
- Database connectivity is tested with the actual database (not mocked) in integration tests
- Use `TestClient` from `httpx` for endpoint tests
- Test configuration overrides database URL to a test database or uses the same DB in a transaction that is rolled back
- Run: `pytest tests/ -v` from `backend/`

### Frontend
- TypeScript type check must pass: `npx tsc --noEmit`
- ESLint must pass: `npm run lint`
- Build must succeed: `npm run build`
- Component tests will be added in a future phase (React Testing Library)

### Quality Gates — All Must Pass Before Committing
```powershell
# Backend (run from project root)
backend\.venv\Scripts\pytest.exe backend/tests/ -v
backend\.venv\Scripts\ruff.exe check backend/app/ scripts/ ml/ backend/tests/ml/
backend\.venv\Scripts\ruff.exe format --check backend/app/ scripts/ ml/

# Frontend
cd frontend
npm run lint
npx tsc --noEmit
npm run build
```

---

## 7. Security Rules

These rules are non-negotiable. Violations must be fixed before any commit.

1. **Never commit `.env`** — it is in `.gitignore`. Always.
2. **Never hardcode credentials** — passwords, API keys, secret keys must come from environment variables.
3. **`.env.example` has placeholders only** — values like `your-secret-key-here`, never real values.
4. **No secrets in logs** — logging configuration must not log environment variable contents.
5. **No secrets in API responses** — health endpoints expose app status, not configuration values.
6. **PostgreSQL not publicly exposed** — bind to `127.0.0.1:5432` only in Docker Compose.
7. **Least-privilege database credentials** — the application user does not have superuser rights.
8. **No destructive commands without approval** — `DROP TABLE`, `DELETE FROM`, `alembic downgrade`, volume deletion require explicit owner approval.
9. **Do not disable security checks** — never weaken type checking, lint rules, or test assertions to make a check pass.
10. **No sensitive data in `data/` committed to Git** — all dataset files are excluded by `.gitignore`.

---

## 8. Phase Boundaries

### Phase 1 — Foundation ✅ Complete
Deliverables: working local dev environment, backend skeleton, frontend skeleton, database, Docker Compose, testing setup, documentation.

### Phase 2 — Data Engineering ✅ Complete
Deliverables: KuaiRec 2.0 acquisition, inspection, validation, cleaning, Parquet conversion, data quality report, pipeline tests, documentation.
Active tools: Polars, DuckDB, PyArrow.

### Phase 3 — Baseline Recommendation Systems ✅ Complete
Deliverables: EDA, target label definition, temporal train/val/test splits, offline evaluation framework (7 metrics), 5 non-ML baseline recommenders, JSON experiment tracking, 72 new tests, full documentation.
ADRs: ADR-P3-01 through ADR-P3-07.
Best NDCG@10: 0.4461 (PopularityRecommender on KuaiRec 2.0 small_matrix).
No new pip packages required — pure Python evaluation layer.

### Phase 4 — ML-Based Recommendation (Current Target)
Deliverables: Content-Based Filtering (TF-IDF / item features), Collaborative Filtering (Matrix Factorization / ALS), comparison against Phase 3 baselines on NDCG@10.
New dependencies: scikit-learn, possibly `implicit` (ALS).
Dataset: introduce big_matrix for sparse-matrix realism.

### Phase 5 — ML Ranking and Serving (Future)
Deliverables: candidate generation, LightGBM/XGBoost ranking model, model serving API, MLflow experiment tracking.

### Phase 6 — Feedback Loop and Production Hardening (Future)
Deliverables: feedback collection, online learning, A/B testing, monitoring, Redis caching.

**Rule:** Do not implement features from a future phase. Do not create empty stub files for future
phases unless explicitly approved. The phase boundary is a hard scope limit.

---

## 9. Rules for Modifying Architecture

The following changes require explicit written approval from the project owner before implementation:

- Changing the backend web framework (FastAPI)
- Changing the ORM (SQLAlchemy)
- Changing the database engine (PostgreSQL)
- Adding a new infrastructure service (Redis, message queue, etc.)
- Changing the frontend framework (React)
- Moving from monolith to microservices
- Changing the migration tool (Alembic)
- Switching from Docker Compose to Kubernetes or any orchestrator
- Adding external API integrations not listed in the approved stack
- Changing the Python version
- Changing the branching strategy

Minor changes (adding a new library within the approved stack, adding an endpoint, refactoring within an existing module) do not require architecture approval but must follow coding conventions.

---

## 10. Approval Requirements Before Destructive Operations

The following operations MUST NOT be performed without explicit written approval:

| Operation | Why Approval Is Required |
|---|---|
| `git push --force` | Rewrites shared history |
| `alembic downgrade` | Rolls back database schema changes |
| `DROP TABLE` or `DROP DATABASE` | Permanent data loss |
| Deleting Docker volumes | Permanent data loss |
| `git clean -fd` | Deletes untracked files |
| Overwriting `.env` | May destroy local credentials |
| Deleting `backend/.venv` | Requires full reinstall |
| Resetting the database | Permanent data loss |
| Any bulk DELETE or UPDATE without WHERE clause | Data corruption risk |

---

## Quick Reference — Start the System

```powershell
# 1. Start the database
docker compose up -d db

# 2. Start the backend
cd backend
.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload

# 3. Start the frontend (new terminal)
cd frontend
npm run dev

# 4. Run ALL backend tests (from project root)
backend\.venv\Scripts\pytest.exe backend/tests/ -v

# 5. Run backend quality checks (from project root)
backend\.venv\Scripts\ruff.exe check backend/app/ scripts/ ml/ backend/tests/ml/
backend\.venv\Scripts\ruff.exe format --check backend/app/ scripts/ ml/

# 6. Run frontend checks
cd frontend
npm run lint
npx tsc --noEmit
```

## Quick Reference — Phase 2 Data Pipeline

```powershell
# From project root, all scripts require the backend venv:
backend\.venv\Scripts\python.exe scripts/data/download_dataset.py
backend\.venv\Scripts\python.exe scripts/data/inspect_dataset.py
backend\.venv\Scripts\python.exe scripts/data/validate_dataset.py
backend\.venv\Scripts\python.exe scripts/data/convert_to_parquet.py
backend\.venv\Scripts\python.exe scripts/data/generate_report.py
```

## Quick Reference — Phase 3 Baselines

```powershell
# From project root:
# Generate temporal splits (required once)
backend\.venv\Scripts\python.exe scripts/ml/prepare_splits.py

# Run all 5 baselines + evaluation
backend\.venv\Scripts\python.exe scripts/ml/run_baselines.py

# Run EDA report (optional)
backend\.venv\Scripts\python.exe scripts/ml/run_eda.py
```

---

*Last updated: Phase 3 completion*
*Owner: Project owner — approval required for all architectural changes*
