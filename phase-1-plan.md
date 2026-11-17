# Phase 1 — Hybrid Recommendation Intelligence Platform
## Complete Implementation Plan

---

## Top-Level Overview

**Goal:** Establish a complete, working local development foundation for the Hybrid Recommendation Intelligence Platform — covering workspace, version control, backend skeleton, frontend skeleton, database, Docker Compose, testing, and documentation.

**Scope:** Phase 1 only. No recommendation algorithms, ML pipelines, authentication, or advanced product features.

**Approach:** Implement one work package at a time in dependency order. Each work package is verified with tests or manual checks before moving to the next. No code is written until this plan is approved.

**Environment Confirmed:**
- OS: Windows 10/11 x64, PowerShell
- Git: installed at `C:\Program Files\Git`
- Node.js + npm: installed at `C:\Program Files\nodejs`
- Python: **3.13** installed at `C:\Users\Edunet Foundation\AppData\Local\Programs\Python\Python313` (see Decision D-01)
- Docker Desktop: installed at `C:\Program Files\Docker`
- Workspace: completely empty

---

## Architecture Decisions Requiring Approval

### D-01 — Python Version: 3.13 vs 3.11

**Situation:** The tech stack specifies Python 3.11. The machine has Python 3.13 installed. Python 3.11 is not present.

**Options:**
- **Option A (Recommended):** Use Python 3.13. All Phase 1 dependencies (FastAPI, SQLAlchemy, Alembic, Pydantic, Ruff, Pytest) are fully compatible with 3.13. Future-phase ML libraries (PyTorch, scikit-learn, LightGBM) now also support 3.13. This avoids installing a second Python version.
- **Option B:** Install Python 3.11 alongside 3.13. This matches the spec exactly but introduces two Python installations to manage on the same machine.

**Recommendation:** Option A — use Python 3.13, update the tech stack spec to 3.13, and proceed. This is a development environment only and there is no production deployment constraint in Phase 1.

**Decision required before implementation begins.**

---

### D-02 — Virtual Environment Location

**Situation:** Python virtual environments can live inside the project (e.g. `backend/.venv`) or at the workspace root. Frontend tooling (Vite) lives in `frontend/node_modules`.

**Recommendation:** Place the backend virtual environment at `backend/.venv`. This keeps it co-located with the backend code, makes it easy to identify, and is excluded from Git by `.gitignore`.

---

### D-03 — Docker Compose Version

**Situation:** Modern Docker Desktop ships with Compose V2 (the `docker compose` plugin, no hyphen). The older `docker-compose` (V1, hyphen) is a separate binary and is deprecated.

**Recommendation:** Use Compose V2 syntax (`docker compose up`) in all scripts and documentation.

---

## Sub-Tasks

---

### Task 1 — Environment Inspection and Prerequisites Report

**Status:** `[x] done` (completed during planning)

**Intent:** Confirm all required tools are present and identify any gaps before writing a single file.

**Findings:**
| Tool | Status | Location |
|---|---|---|
| Git | ✅ Installed | `C:\Program Files\Git` |
| Node.js + npm | ✅ Installed | `C:\Program Files\nodejs` |
| Python 3.13 | ✅ Installed | `C:\Users\Edunet Foundation\AppData\Local\Programs\Python\Python313` |
| Docker Desktop | ✅ Installed | `C:\Program Files\Docker\Docker` |
| Workspace | ✅ Empty | Ready for initialization |

**No prerequisites are missing.** See D-01 for the Python version decision.

---

### Task 2 — IBM Bob Workspace Instructions (AGENTS.md)

**Status:** `[ ] pending`

**Intent:** Create the root `AGENTS.md` that serves as the AI development contract for this project. Every AI-assisted session will read this file to understand conventions, boundaries, and rules.

**Expected Outcomes:**
- `AGENTS.md` exists at the repository root
- It documents: project objective, approved stack, architecture principles, repository structure, coding conventions, testing requirements, security rules, phase boundaries, change approval rules

**Todo List:**
1. Create `AGENTS.md` at the repository root with all required sections

**Relevant Context:** This is a standalone file. No code dependencies.

---

### Task 3 — Git and GitHub Version Control Setup

**Status:** `[ ] pending`

**Intent:** Initialize the Git repository, create the foundational tracked files, and establish commit and branch conventions. This locks in what is and is not committed to source control.

**Expected Outcomes:**
- `git init` completed
- `.gitignore` created (excludes `.env`, `__pycache__`, `.venv`, `node_modules`, `*.pyc`, IDE files, ML artifacts, datasets)
- `.env.example` created with placeholder values only (no real secrets)
- `.dockerignore` created
- `README.md` created with project overview and setup instructions
- Initial commit made on `main` branch

**Files to Create:**
- `README.md`
- `.gitignore`
- `.env.example`
- `.dockerignore`

**Commands Required:**
```powershell
cd "c:\Users\Edunet Foundation\Desktop\IBM Bob Projects\Recommendation System"
git init
git branch -M main
git add .
git commit -m "chore: initialize repository with project foundation"
```

**Security Rules:**
- `.env` must be in `.gitignore` before the first commit
- `.env.example` must contain only placeholder values such as `your-secret-key-here`

**Acceptance Criteria:**
- `git status` shows a clean working tree after the initial commit
- `.env` is not tracked
- `git log` shows one commit on `main`

---

### Task 4 — Backend Foundation

**Status:** `[ ] pending`

**Intent:** Stand up a working FastAPI application with SQLAlchemy, Alembic, Pydantic Settings, structured logging, and three health/info endpoints. This is the backbone all future phases build on.

**Expected Outcomes:**
- `backend/.venv` created and activated
- `backend/pyproject.toml` created with all Phase 1 dependencies pinned
- FastAPI application starts with `uvicorn app.main:app`
- Three endpoints respond correctly:
  - `GET /api/v1/health` → `{"status": "ok", "version": "0.1.0"}`
  - `GET /api/v1/health/db` → tests actual database connectivity, returns connected status or error
  - `GET /api/v1/system/info` → returns safe app metadata only
- SQLAlchemy engine configured from environment variables
- Alembic initialized and pointing at the correct database URL
- Structured JSON logging configured
- Centralized error handler returning consistent JSON error responses
- No secrets logged or exposed in any endpoint

**Files to Create:**
```
backend/
  app/
    __init__.py
    main.py                  # FastAPI app factory, middleware, router registration
    api/
      __init__.py
      v1/
        __init__.py
        router.py            # Aggregates v1 routes
        endpoints/
          __init__.py
          health.py          # /health and /health/db
          system.py          # /system/info
    core/
      __init__.py
      config.py              # Pydantic Settings — reads from .env
      logging.py             # Structured logging setup
      exceptions.py          # Centralized exception handlers
    db/
      __init__.py
      session.py             # SQLAlchemy engine, session factory, get_db dependency
      base.py                # Declarative base
  tests/
    __init__.py
    conftest.py              # Pytest fixtures, test client, test DB session
    test_health.py           # Tests for /health and /health/db
    test_system.py           # Tests for /system/info
  pyproject.toml             # Build system, dependencies, Ruff config, Pytest config
  requirements.txt           # Pinned requirements for reference
  alembic.ini                # Alembic configuration
  alembic/
    env.py                   # Points to app models and DB URL
    versions/                # Migration scripts (initially empty)
  .env                       # NOT committed — created locally by developer
```

**Phase 1 Backend Dependencies:**
- `fastapi`
- `uvicorn[standard]`
- `sqlalchemy`
- `alembic`
- `psycopg2-binary`
- `pydantic-settings`
- `python-dotenv`

**Dev Dependencies:**
- `pytest`
- `pytest-asyncio`
- `httpx`
- `ruff`

**Commands Required:**
```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
alembic init alembic
uvicorn app.main:app --reload   # manual smoke test
pytest tests/                   # run tests
ruff check app/                 # linting
ruff format app/                # formatting
```

**Acceptance Criteria:**
- `uvicorn` starts without errors
- All three endpoints return correct JSON
- `/api/v1/health/db` returns an error (not a crash) when the database is not running
- All backend tests pass
- `ruff check` reports zero issues

---

### Task 5 — Frontend Foundation

**Status:** `[ ] pending`

**Intent:** Create a working React + TypeScript + Vite application with a clean system health dashboard that shows the backend connection status. This validates the full frontend-to-backend communication path.

**Expected Outcomes:**
- `frontend/` Vite project bootstrapped
- TypeScript strict mode enabled
- Directory structure created: `src/components/`, `src/pages/`, `src/services/`, `src/hooks/`, `src/types/`
- API service layer configured with the backend base URL from an environment variable
- `SystemHealthPage` component that calls all three backend endpoints and displays results
- Health status displayed with visual indicators (green/red/loading)
- ESLint configured and passing
- TypeScript type check (`tsc --noEmit`) passing

**Files to Create / Modify:**
```
frontend/
  src/
    App.tsx                  # Root component, routing stub
    main.tsx                 # Entry point
    types/
      health.ts              # TypeScript interfaces for API responses
    services/
      api.ts                 # Axios or fetch wrapper, base URL config
      healthService.ts       # Calls /health, /health/db, /system/info
    hooks/
      useHealth.ts           # Custom hook for health data fetching
    components/
      StatusIndicator.tsx    # Green/red/loading dot + label
      StatusCard.tsx         # Card wrapping a status indicator
    pages/
      SystemHealthPage.tsx   # Main dashboard page
  public/
  index.html
  package.json
  vite.config.ts             # Proxy /api → backend:8000
  tsconfig.json
  .eslintrc.cjs
  .env.local                 # NOT committed — VITE_API_BASE_URL=http://localhost:8000
  .env.example               # Committed — with placeholder
```

**Phase 1 Frontend Dependencies:**
- `react`, `react-dom`
- `typescript`
- `vite`
- `@vitejs/plugin-react`
- `axios` (or native fetch — decision: use native fetch to minimize dependencies)

**Dev Dependencies:**
- `eslint`
- `@typescript-eslint/eslint-plugin`
- `@typescript-eslint/parser`

**Commands Required:**
```powershell
cd frontend
npm create vite@latest . -- --template react-ts
npm install
npm run dev        # manual smoke test
npm run build      # verify build
npm run lint       # ESLint
npx tsc --noEmit   # type check
```

**Acceptance Criteria:**
- `npm run dev` starts Vite dev server without errors
- Dashboard loads in browser and shows health status from backend
- `npm run lint` reports zero errors
- `npx tsc --noEmit` passes with zero errors
- No secrets in committed frontend files

---

### Task 6 — Database Foundation (PostgreSQL + Docker Compose)

**Status:** `[ ] pending`

**Intent:** Stand up a local PostgreSQL instance via Docker Compose with persistent volume, environment-based credentials, and health checks. Verify the backend can connect.

**Expected Outcomes:**
- `docker-compose.yml` at repository root defines the `db` service
- PostgreSQL starts with `docker compose up -d db`
- Data persists in a named Docker volume
- Database credentials come entirely from `.env`
- Health check verifies PostgreSQL is accepting connections before dependent services start
- FastAPI's `/api/v1/health/db` returns connected status when the database is running
- Alembic `upgrade head` runs successfully against the running database

**Files to Create / Modify:**
```
docker-compose.yml           # db service with PostgreSQL, volume, healthcheck
.env.example                 # DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD placeholders
```

**Docker Compose Service Definition (conceptual):**
```yaml
services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: ${DB_NAME}
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"           # localhost only, not exposed publicly
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER} -d ${DB_NAME}"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
```

**Commands Required:**
```powershell
docker compose up -d db
# wait for health check to pass
cd backend
alembic upgrade head
pytest tests/                # db connectivity test must now pass
```

**Acceptance Criteria:**
- `docker compose ps` shows `db` as healthy
- `GET /api/v1/health/db` returns `{"status": "connected"}`
- `alembic upgrade head` completes without errors
- Database data survives `docker compose stop` + `docker compose start`
- PostgreSQL port 5432 is only accessible on localhost

---

### Task 7 — Testing and Code Quality Setup

**Status:** `[ ] pending`

**Intent:** Confirm all test infrastructure is working and all quality gates pass. This task consolidates verification across backend and frontend.

**Expected Outcomes:**

**Backend:**
- `pytest tests/` passes all tests
- `test_health.py`: tests `/health` returns 200, correct schema
- `test_health.py`: tests `/health/db` returns 200 when DB connected, non-500 when not
- `test_system.py`: tests `/system/info` returns 200, no secrets in response
- `ruff check app/` → zero issues
- `ruff format --check app/` → zero issues

**Frontend:**
- `npm run lint` → zero errors
- `npx tsc --noEmit` → zero errors
- `npm run build` → clean production build

**Files to Create / Modify:**
```
backend/tests/conftest.py    # test client, override get_db with test session
backend/tests/test_health.py
backend/tests/test_system.py
backend/pyproject.toml       # [tool.pytest.ini_options] and [tool.ruff] sections
```

**Commands Required:**
```powershell
# Backend
cd backend
pytest tests/ -v
ruff check app/
ruff format --check app/

# Frontend
cd frontend
npm run lint
npx tsc --noEmit
npm run build
```

**Acceptance Criteria:**
- Zero test failures
- Zero linting errors in both backend and frontend
- Zero TypeScript errors

---

### Task 8 — Documentation

**Status:** `[ ] pending`

**Intent:** Create the documentation suite that lets any developer (or future AI session) understand the system, set it up from scratch, and understand why key decisions were made.

**Expected Outcomes:**
- All documentation files created and accurate
- Setup commands verified against actual implementation
- Architecture decisions recorded with rationale

**Files to Create:**
```
README.md                                     # Project overview, quick-start for Windows/PowerShell
AGENTS.md                                     # AI development contract (created in Task 2)
docs/
  architecture/
    system-architecture.md                    # Component diagram, data flow, module boundaries
  database/
    database-overview.md                      # Schema rationale, Alembic workflow, connection config
  api/
    api-overview.md                           # Endpoint reference with request/response examples
  phases/
    phase-1-setup-guide.md                    # Step-by-step setup from clean machine
  decisions/
    architecture-decisions.md                 # D-01 through D-03 and any others encountered
```

**Acceptance Criteria:**
- A developer following `phase-1-setup-guide.md` on a clean Windows machine can reach a fully working state
- All commands use PowerShell syntax
- No documentation references features that don't exist in Phase 1

---

## Dependency Order

The tasks must be executed in this order because each depends on the previous:

```
Task 2 (AGENTS.md)
    ↓
Task 3 (Git + GitHub)
    ↓
Task 4 (Backend Foundation)
    ↓
Task 6 (Database + Docker Compose)
    ↓
Task 5 (Frontend Foundation)
    ↓
Task 7 (Testing + Quality — full verification pass)
    ↓
Task 8 (Documentation)
```

Task 5 (Frontend) can begin as soon as Task 4 is running, but the full health dashboard requires the database endpoint (Task 6) to be live.

---

## Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Python 3.13 compatibility with a future ML library | Low for Phase 1 | All Phase 1 deps confirmed compatible; revisit at Phase 2 boundary |
| PowerShell execution policy blocking `.venv\Scripts\Activate.ps1` | Medium | Document `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` as a prerequisite step |
| Docker Desktop not running when tests execute | Medium | Document that Docker Desktop must be started before `docker compose up` |
| Port 5432 already occupied by a local PostgreSQL install | Low | Document how to check with `netstat -ano | findstr :5432` and change the port in `.env` |
| `psycopg2-binary` wheel availability for Python 3.13 | Low | Confirmed available on PyPI for Windows x64 / Python 3.13 |
| Vite proxy misconfiguration causing CORS errors | Low | Configure `/api` proxy in `vite.config.ts` to point to `http://localhost:8000` |

---

## Phase 1 Acceptance Checklist

| # | Criterion | Verified By |
|---|---|---|
| 1 | IBM Bob workspace configured | `AGENTS.md` exists and is complete |
| 2 | GitHub repository working | `git log` shows commits on `main` |
| 3 | `AGENTS.md` established | File reviewed |
| 4 | Frontend starts successfully | `npm run dev` with no errors |
| 5 | Backend starts successfully | `uvicorn` with no errors |
| 6 | Frontend communicates with backend | Dashboard shows live backend status |
| 7 | PostgreSQL starts via Docker Compose | `docker compose ps` shows `db` healthy |
| 8 | Backend connects to PostgreSQL | `/api/v1/health/db` returns connected |
| 9 | Alembic migrations work | `alembic upgrade head` succeeds |
| 10 | Backend tests pass | `pytest` zero failures |
| 11 | Frontend lint + typecheck pass | `eslint` and `tsc --noEmit` zero errors |
| 12 | Secrets excluded from Git | `.env` in `.gitignore`, not in `git log` |
| 13 | Documentation complete | All 5 doc files reviewed |
| 14 | Verified from clean workspace | Setup guide followed end-to-end |
