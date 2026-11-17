# API Overview — Hybrid Recommendation Intelligence Platform

## Base URL

| Environment | Base URL |
|---|---|
| Local development | `http://localhost:8000` |
| Interactive docs | `http://localhost:8000/docs` |
| OpenAPI schema | `http://localhost:8000/openapi.json` |

All API routes are versioned under `/api/v1/`.

---

## Phase 1 Endpoints

### GET /api/v1/health

**Purpose:** Application liveness check. Always responds if the server process is running.
Does NOT test database connectivity.

**Response: 200 OK**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "environment": "development"
}
```

| Field | Type | Description |
|---|---|---|
| `status` | `string` | Always `"ok"` when the server is running |
| `version` | `string` | Application version from settings |
| `environment` | `string` | `APP_ENV` value from `.env` |

**Security:** No secrets, passwords, or configuration values are included.

---

### GET /api/v1/health/db

**Purpose:** Database connectivity check. Executes `SELECT 1` against PostgreSQL and
reports whether the connection succeeded. Returns HTTP 200 in both the success and
failure cases — the `status` field distinguishes them.

**Response: 200 OK — database connected**
```json
{
  "status": "connected"
}
```

**Response: 200 OK — database not reachable**
```json
{
  "status": "error",
  "detail": "Database is not reachable."
}
```

| Field | Type | Description |
|---|---|---|
| `status` | `"connected"` \| `"error"` | Live connectivity result |
| `detail` | `string?` | Present only when status is `"error"` |

**Security:** No connection strings, passwords, or hostnames are included in the response.
The error message is a safe, generic string — the actual exception is logged server-side only.

---

### GET /api/v1/system/info

**Purpose:** Returns safe, non-sensitive application metadata. Useful for verifying the
correct version is deployed and understanding the runtime environment.

**Response: 200 OK**
```json
{
  "app_name": "Hybrid Recommendation Intelligence Platform",
  "version": "0.1.0",
  "environment": "development",
  "python_version": "3.13.0",
  "platform_system": "Windows",
  "api_version": "v1"
}
```

| Field | Type | Description |
|---|---|---|
| `app_name` | `string` | Application name |
| `version` | `string` | Application version |
| `environment` | `string` | Runtime environment name |
| `python_version` | `string` | Python interpreter version |
| `platform_system` | `string` | OS name (Windows, Linux, Darwin) |
| `api_version` | `string` | API version string |

**Security:** No secret keys, database passwords, file paths, or internal identifiers
are included. This endpoint is safe to expose publicly.

---

## Error Responses

All endpoints return consistent JSON error responses. Unhandled exceptions are caught
by the centralized exception handler in `backend/app/core/exceptions.py`.

**500 Internal Server Error**
```json
{
  "error": "internal_server_error",
  "message": "An unexpected error occurred."
}
```

The full exception is logged server-side. The response never includes stack traces or
internal details.

---

## API Versioning Strategy

- Current version: `/api/v1/`
- Future versions will use `/api/v2/`, `/api/v3/`, etc.
- Old versions remain available during transition periods
- The `api_version` field in `/system/info` reflects the current active version

---

## Future Endpoints (Phase 2+)

The following endpoints will be added in future phases:

| Phase | Endpoint | Description |
|---|---|---|
| 2 | `GET /api/v1/items` | List available items |
| 2 | `POST /api/v1/interactions` | Record a user interaction |
| 3 | `GET /api/v1/recommendations/{user_id}` | Get recommendations for a user |
| 4 | `GET /api/v1/candidates` | Candidate generation |
| 5 | `POST /api/v1/feedback` | Submit recommendation feedback |
