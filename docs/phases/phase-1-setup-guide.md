# Phase 1 Setup Guide — Windows PowerShell

This guide walks through setting up the Hybrid Recommendation Intelligence Platform
from a clean Windows machine. Follow every step in order.

---

## Prerequisites

Ensure the following are installed before starting:

| Tool | Required | How to verify |
|---|---|---|
| Git | Any recent | `git --version` |
| Python | 3.13+ | `python --version` |
| Node.js | 18+ | `node --version` |
| npm | 9+ | `npm --version` |
| Docker Desktop | Current | `docker --version` |

If any tool is missing, install it from its official website before continuing.

### PowerShell Execution Policy

If running `.ps1` scripts is blocked, run this once in PowerShell as your user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## Step 1 — Clone the Repository

```powershell
git clone https://github.com/0825aman/Recommendation-System-Works.git "Recommendation System"
cd "Recommendation System"
```

---

## Step 2 — Configure Environment Variables

```powershell
# Copy the template
Copy-Item .env.example .env

# Open and edit .env — set at minimum:
#   DB_PASSWORD    (choose a strong password)
#   SECRET_KEY     (generate a random string)
notepad .env
```

The `.env` file controls all application configuration. It is never committed to Git.

**Minimum required values in `.env`:**
```
APP_ENV=development
APP_VERSION=0.1.0
SECRET_KEY=<generate a random 32-character string>

DB_HOST=localhost
DB_PORT=5432
DB_NAME=recommendation_db
DB_USER=recommendation_user
DB_PASSWORD=<choose a strong password>

VITE_API_BASE_URL=
```

---

## Step 3 — Start the Database

Docker Desktop must be running before this step.

```powershell
# Start the PostgreSQL container
docker compose up -d db

# Wait ~15 seconds, then verify it is healthy
docker compose ps
```

Expected output:
```
NAME                IMAGE                STATUS              PORTS
recommendation_db   postgres:16-alpine   Up (healthy)        127.0.0.1:5432->5432/tcp
```

If the status shows `starting` instead of `healthy`, wait 15 more seconds and re-run `docker compose ps`.

---

## Step 4 — Set Up the Backend

Open a **new PowerShell terminal** in the project root.

```powershell
cd backend

# Create the Python virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\Activate.ps1

# Install all dependencies
pip install -e ".[dev]"

# Run database migrations (connects to PostgreSQL)
alembic upgrade head
```

Expected Alembic output:
```
INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
INFO  [alembic.runtime.migration] Will assume transactional DDL.
```

---

## Step 5 — Start the Backend Server

```powershell
# From backend/ with .venv activated
uvicorn app.main:app --reload
```

Expected output:
```
INFO  | app.main:on_startup | Starting Hybrid Recommendation Intelligence Platform v0.1.0 [development]
INFO  | uvicorn.error | Application startup complete.
INFO  | uvicorn.error | Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Verify the backend is working:
- Open `http://localhost:8000/docs` in your browser — Swagger UI should appear
- Or run: `curl http://localhost:8000/api/v1/health`

---

## Step 6 — Start the Frontend

Open a **second PowerShell terminal** in the project root.

```powershell
cd frontend
npm install
npm run dev
```

Expected output:
```
  VITE v8.x.x  ready in Xms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

Open `http://localhost:5173` in your browser. The System Health Dashboard should appear
showing green status indicators for Application, Database, and System.

---

## Step 7 — Run Tests and Quality Checks

### Backend

```powershell
cd backend
.venv\Scripts\Activate.ps1

# Run tests
pytest tests/ -v

# Lint
ruff check app/

# Format check
ruff format --check app/
```

Expected: `18 passed` with no errors.

### Frontend

```powershell
cd frontend

# Type check
npm run typecheck

# Lint
npm run lint

# Production build
npm run build
```

Expected: 0 errors, 0 warnings, clean build.

---

## Stopping the System

```powershell
# Stop the backend: press CTRL+C in the backend terminal
# Stop the frontend: press CTRL+C in the frontend terminal

# Stop the database container (data is preserved in the volume)
docker compose stop db

# To restart later:
docker compose start db
```

---

## Troubleshooting

### `Activate.ps1` blocked by execution policy
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Port 5432 already in use
Check what is using the port:
```powershell
netstat -ano | findstr :5432
```
If another PostgreSQL is running locally, change `DB_PORT` in `.env` to `5433`
and update `docker-compose.yml` to use `127.0.0.1:5433:5432`.

### Docker Desktop not starting
Ensure virtualization is enabled in BIOS/UEFI. On Windows, also check that
WSL2 is installed: `wsl --install`.

### `ModuleNotFoundError: No module named 'app'`
Make sure you are running commands from the `backend/` directory and the
virtual environment is activated (prompt shows `(.venv)`).

### Database health shows `error` in the dashboard
1. Check the database container is running: `docker compose ps`
2. Check the `.env` DB credentials match what was used to create the container
3. If you changed credentials after first run, the volume may have old data.
   Stop the container (`docker compose stop db`), inspect the logs
   (`docker compose logs db`), and ensure credentials are consistent.

---

## Quick Reference

```powershell
# Start everything (from project root)
docker compose up -d db
# Then in separate terminals:
cd backend; .venv\Scripts\Activate.ps1; uvicorn app.main:app --reload
cd frontend; npm run dev

# Run all quality checks
cd backend; pytest tests/ -v; ruff check app/; ruff format --check app/
cd frontend; npm run typecheck; npm run lint; npm run build
```
