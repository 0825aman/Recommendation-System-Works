# Architecture Decision Records

This document records significant architectural decisions made during the development
of the Hybrid Recommendation Intelligence Platform, including context, options considered,
and the rationale for the chosen approach.

---

## ADR-001 — Modular Monolith over Microservices

**Status:** Accepted  
**Phase:** 1

**Context:**  
The platform will eventually consist of multiple concerns: API serving, data processing,
ML training, model serving, and feedback collection. A microservices architecture would
treat each as an independent deployable service.

**Decision:**  
Start with a modular monolith. Keep all backend logic in a single FastAPI application
with well-defined internal module boundaries. Extract services into independent processes
only when there is a demonstrated, specific need (e.g. ML training is too resource-intensive
to share a process with the API server).

**Rationale:**  
- Microservices add significant operational overhead (service discovery, inter-service auth,
  distributed tracing, independent CI/CD pipelines) before the system is well understood
- The team is a single developer — a monolith is proportionate to the team size
- Module boundaries within a monolith can be enforced with import rules; extraction later
  is straightforward when the boundaries are clean
- Martin Fowler's Monolith First pattern applies here

---

## ADR-002 — Python 3.13 (not 3.11 as originally specified)

**Status:** Accepted  
**Phase:** 1

**Context:**  
The original tech stack specification called for Python 3.11. The development machine has
Python 3.13 installed (at `C:\Users\Edunet Foundation\AppData\Local\Programs\Python\Python313`).
Python 3.11 was not installed.

**Decision:**  
Use Python 3.13. Update the tech stack specification accordingly.

**Rationale:**  
- All Phase 1 dependencies (FastAPI, SQLAlchemy, Alembic, Pydantic, psycopg2-binary, Ruff,
  Pytest, httpx) are fully compatible with Python 3.13
- Future-phase ML libraries (PyTorch, scikit-learn, LightGBM) now also support 3.13
- Installing Python 3.11 alongside 3.13 would require managing two Python installations
  with no technical benefit for Phase 1
- Python 3.13 is strictly newer and brings performance improvements

**Consequence:**  
If a future-phase ML library is found to be incompatible with 3.13 at that time, the
virtual environment can be recreated with a pyenv-managed 3.11 interpreter without
changing any application code.

---

## ADR-003 — Docker Compose V2 (`docker compose`, not `docker-compose`)

**Status:** Accepted  
**Phase:** 1

**Context:**  
Two versions of Docker Compose exist: V1 (standalone binary, invoked as `docker-compose`)
and V2 (Docker plugin, invoked as `docker compose` without hyphen). V1 reached end-of-life
in July 2023.

**Decision:**  
Use Docker Compose V2 (`docker compose`) in all scripts, documentation, and commands.

**Rationale:**  
- Docker Desktop on Windows ships V2 by default
- V1 is deprecated and no longer receives updates
- V2 is the standard going forward

---

## ADR-004 — Backend Virtual Environment at `backend/.venv`

**Status:** Accepted  
**Phase:** 1

**Context:**  
Python virtual environments can be placed at the repository root or inside the backend
directory.

**Decision:**  
Place the virtual environment at `backend/.venv`.

**Rationale:**  
- Co-located with the backend code it serves
- Clear ownership — no confusion about which `.venv` belongs to which component
- Excluded from Git by `.gitignore`
- Standard convention for Python projects

---

## ADR-005 — Axios as the Frontend HTTP Client

**Status:** Accepted  
**Phase:** 1

**Context:**  
The frontend needs to make HTTP requests to the backend. Options are native `fetch` API
(zero dependencies) or Axios (additional dependency, more ergonomic).

**Decision:**  
Use Axios. The decision was confirmed by the project owner during Phase 1 planning.

**Rationale:**  
- Axios is the established standard in React projects
- Provides a centralized instance with base URL, timeout, and default headers
- Interceptors allow easy request/response transformation in future phases
- More familiar to most developers than raw fetch with manual error handling

---

## ADR-006 — PostgreSQL Bound to localhost Only

**Status:** Accepted  
**Phase:** 1

**Context:**  
The Docker Compose port mapping can expose PostgreSQL to all interfaces (`0.0.0.0:5432`)
or only to localhost (`127.0.0.1:5432`).

**Decision:**  
Bind to `127.0.0.1:5432` only.

**Rationale:**  
- PostgreSQL should never be directly reachable from outside the development machine
- All database access goes through the FastAPI backend
- Principle of least privilege — expose only what is necessary
- No functional impact on local development

---

## ADR-007 — `postgresql+psycopg2` Dialect Explicitly in SQLAlchemy URL

**Status:** Accepted  
**Phase:** 1

**Context:**  
SQLAlchemy 2.x with a plain `postgresql://` URL attempts to use `psycopg` v3 by default
if installed, falling back to `psycopg2`. We install `psycopg2-binary` (v2).

**Decision:**  
Use `postgresql+psycopg2://` explicitly in the connection URL.

**Rationale:**  
- Makes the driver choice explicit and unambiguous
- Avoids SQLAlchemy silently picking a different driver if `psycopg` v3 is later installed
- `psycopg2-binary` has the best Windows support and broadest compatibility

---

## ADR-008 — oxlint as Frontend Linter (Vite 8 Default)

**Status:** Accepted  
**Phase:** 1

**Context:**  
Vite 8 scaffolds projects with `oxlint` as the default linter instead of ESLint.
`oxlint` is a Rust-based linter that is significantly faster than ESLint.

**Decision:**  
Accept the Vite 8 default and use `oxlint`. Do not replace it with traditional ESLint.

**Rationale:**  
- oxlint implements the same rule set as the most important ESLint rules
- 50–100x faster than ESLint on large codebases
- Zero configuration required for a React + TypeScript project
- Already included in `package.json` by the Vite scaffold
- TypeScript type checking is handled separately by `tsc --noEmit`, which is more
  thorough than any linter's type rules
