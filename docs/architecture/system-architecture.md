# System Architecture — Hybrid Recommendation Intelligence Platform

## Overview

The platform is built as a **modular monolith** — a single deployable backend application
with logically separated internal modules. The frontend communicates with the backend through
versioned HTTP APIs. The database is owned exclusively by the backend.

This architecture is intentional for Phase 1 and early phases. When a module grows large
enough to justify independent deployment, it can be extracted at that point — not before.

---

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  Developer Machine                                              │
│                                                                 │
│  ┌──────────────────┐    HTTP (port 5173)                       │
│  │   Browser        │                                           │
│  │   localhost:5173 │                                           │
│  └────────┬─────────┘                                           │
│           │  Vite Proxy                                         │
│           │  /api/* → localhost:8000                            │
│           ▼                                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Frontend (React + TypeScript + Vite)                    │  │
│  │  localhost:5173                                          │  │
│  │                                                          │  │
│  │  pages/SystemHealthPage.tsx                              │  │
│  │  hooks/useHealth.ts                                      │  │
│  │  services/healthService.ts  →  services/api.ts (Axios)   │  │
│  │  components/StatusCard.tsx                               │  │
│  └──────────────────────────────────────────────────────────┘  │
│           │                                                     │
│           │  HTTP REST  /api/v1/*                               │
│           ▼                                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Backend (Python + FastAPI + Uvicorn)                    │  │
│  │  localhost:8000                                          │  │
│  │                                                          │  │
│  │  api/v1/endpoints/health.py   → GET /health, /health/db │  │
│  │  api/v1/endpoints/system.py   → GET /system/info        │  │
│  │  core/config.py               → Pydantic Settings       │  │
│  │  core/logging.py              → Structured logging      │  │
│  │  core/exceptions.py           → Centralized errors      │  │
│  │  db/session.py                → SQLAlchemy engine       │  │
│  │  db/base.py                   → Declarative base        │  │
│  └──────────────────────────┬───────────────────────────────┘  │
│                              │                                  │
│                              │  postgresql+psycopg2             │
│                              │  localhost:5432                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Database (PostgreSQL 16 — Docker Container)             │  │
│  │  127.0.0.1:5432  (localhost only)                        │  │
│  │                                                          │  │
│  │  Container: recommendation_db                            │  │
│  │  Volume:    recommendation_postgres_data                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow — Health Dashboard Request

```
Browser
  → Vite dev server (port 5173)
    → Vite proxy rewrites /api/v1/health → http://localhost:8000/api/v1/health
      → FastAPI router matches GET /api/v1/health
        → health.py endpoint handler
          → returns HealthResponse Pydantic model
        ← JSON: {"status": "ok", "version": "0.1.0", "environment": "development"}
      ← HTTP 200
    ← JSON response
  ← React useHealth hook updates state
← SystemHealthPage re-renders with live status
```

---

## Module Boundaries

| Module | Responsibility | May NOT |
|---|---|---|
| `frontend/` | Render UI, call APIs | Access database directly |
| `backend/app/api/` | Route handlers, request validation | Contain business logic |
| `backend/app/core/` | Configuration, logging, exceptions | Import from api/ or db/ |
| `backend/app/db/` | Database engine, sessions | Import from api/ or services/ |
| `backend/app/models/` | SQLAlchemy ORM models | Contain business logic |
| `backend/app/schemas/` | Pydantic request/response models | Import SQLAlchemy models |
| `backend/app/services/` | Business logic | Be imported by db/ |

---

## Technology Versions (Phase 1)

| Component | Technology | Version |
|---|---|---|
| Frontend framework | React | 19.x |
| Frontend language | TypeScript | 7.x |
| Frontend build tool | Vite | 8.x |
| Frontend HTTP client | Axios | 1.x |
| Frontend linter | oxlint | 1.x |
| Backend framework | FastAPI | 0.163.x |
| Backend language | Python | 3.13/3.14 |
| Backend server | Uvicorn | 0.30.x |
| ORM | SQLAlchemy | 2.x |
| Migrations | Alembic | 1.x |
| Configuration | Pydantic Settings | 2.x |
| Database | PostgreSQL | 16 (Alpine) |
| Containerization | Docker Compose | V2 |
| Backend linter | Ruff | 0.6.x |
| Backend tests | Pytest | 8.x |

---

## Security Boundaries

- PostgreSQL is bound to `127.0.0.1:5432` — not reachable from the network
- Backend CORS allows only `localhost:5173` (Vite dev server) in development
- No secrets are returned from any API endpoint
- Credentials live only in `.env` files, which are excluded from Git
- The Alembic migration runner reads credentials from Pydantic Settings, not `alembic.ini`

---

## Phase Boundaries

This document reflects **Phase 1 only**. Future phases will extend the architecture:

- **Phase 2:** Data ingestion models, Polars/DuckDB for data processing
- **Phase 3:** Recommendation engine modules under `ml/`
- **Phase 4:** ML model serving endpoint, ranking pipeline
- **Phase 5:** Feedback collection, A/B testing hooks, monitoring
