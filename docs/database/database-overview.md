# Database Overview — Hybrid Recommendation Intelligence Platform

## Phase 1 Database State

In Phase 1, the database is set up and connectivity is verified, but **no application tables
have been created yet**. The recommendation engine data model (users, items, interactions,
embeddings) is a Phase 2 deliverable.

The database infrastructure is fully in place and ready to accept migrations.

---

## Configuration

Database connection parameters are read from environment variables via Pydantic Settings.
The SQLAlchemy URL is composed at runtime — credentials are never hardcoded.

| Variable | Description | Default |
|---|---|---|
| `DB_HOST` | PostgreSQL host | `localhost` |
| `DB_PORT` | PostgreSQL port | `5432` |
| `DB_NAME` | Database name | `recommendation_db` |
| `DB_USER` | Application user | `recommendation_user` |
| `DB_PASSWORD` | Application password | *(required, no default)* |

The composed URL takes the form:
```
postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}
```

The `postgresql+psycopg2` dialect is used explicitly so SQLAlchemy 2.x uses
`psycopg2-binary` (not the newer `psycopg` v3 package).

---

## SQLAlchemy Configuration

The engine is created in [`backend/app/db/session.py`](../../backend/app/db/session.py):

```python
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,   # Verify connections before use — catches stale connections
    pool_size=5,          # Base pool size
    max_overflow=10,      # Maximum additional connections above pool_size
    echo=False,           # Set True temporarily for SQL query debugging
)
```

`pool_pre_ping=True` is important for local development where the database container
may be stopped and restarted — it prevents "connection already closed" errors.

---

## Alembic Migration Workflow

All schema changes must go through Alembic migrations. Manual DDL (`CREATE TABLE`, `ALTER TABLE`)
is not permitted.

### How Alembic is configured

- `alembic.ini` — Alembic configuration file (committed to Git)
- `alembic/env.py` — overrides `sqlalchemy.url` with `settings.database_url` from Pydantic Settings
- `alembic/versions/` — generated migration scripts (committed to Git)

The `alembic/env.py` file also imports `Base.metadata` from [`backend/app/db/base.py`](../../backend/app/db/base.py).
This means Alembic can autogenerate migrations by comparing `Base.metadata` to the live database schema.

### Common migration commands

```powershell
# From the backend/ directory with .venv activated

# Apply all pending migrations
alembic upgrade head

# Check current migration version
alembic current

# Generate a new migration (autogenerate from model changes)
alembic revision --autogenerate -m "add users table"

# View migration history
alembic history

# DESTRUCTIVE — requires explicit approval
alembic downgrade -1
```

### Adding a new model (future phases)

1. Create the SQLAlchemy model in `backend/app/models/your_model.py`
2. Import `Base` from `app.db.base` and inherit from it
3. Import the model in `alembic/env.py` so autogenerate discovers it
4. Run `alembic revision --autogenerate -m "description"`
5. Review the generated migration script in `alembic/versions/`
6. Apply with `alembic upgrade head`

---

## Docker Compose Database Service

The database runs as a Docker container defined in [`docker-compose.yml`](../../docker-compose.yml).

```yaml
services:
  db:
    image: postgres:16-alpine
    container_name: recommendation_db
    ports:
      - "127.0.0.1:5432:5432"   # localhost only
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER} -d ${DB_NAME}"]
      interval: 10s
      retries: 5
```

Key points:
- `postgres:16-alpine` — minimal Alpine-based image
- Port bound to `127.0.0.1` only — PostgreSQL is not reachable from outside the machine
- `postgres_data` named volume — data survives container restarts
- Health check ensures the container is fully ready before dependent services start

### Volume management

```powershell
# List volumes
docker volume ls

# Inspect the data volume
docker volume inspect recommendation_postgres_data

# DESTRUCTIVE — requires approval before running
# docker compose down -v    # Stops containers AND deletes the volume
```

---

## Phase 2 Data Model Preview

The following tables will be introduced in Phase 2 (not yet created):

- `users` — user profiles and preferences
- `items` — content items (movies, videos, products)
- `interactions` — user-item interaction events (clicks, views, ratings)
- `item_features` — extracted content features for content-based filtering

These will be designed and migrated in Phase 2 using Alembic autogenerate.
