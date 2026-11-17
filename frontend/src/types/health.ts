/**
 * TypeScript interfaces for backend API responses.
 * These match the Pydantic response schemas defined in the FastAPI backend.
 */

/** Response from GET /api/v1/health */
export interface HealthResponse {
  status: string;
  version: string;
  environment: string;
}

/** Response from GET /api/v1/health/db */
export interface DatabaseHealthResponse {
  status: "connected" | "error";
  detail?: string;
}

/** Response from GET /api/v1/system/info */
export interface SystemInfoResponse {
  app_name: string;
  version: string;
  environment: string;
  python_version: string;
  platform_system: string;
  api_version: string;
}

/** Unified health data shape used by the useHealth hook */
export interface HealthData {
  appHealth: HealthResponse | null;
  dbHealth: DatabaseHealthResponse | null;
  systemInfo: SystemInfoResponse | null;
  loading: boolean;
  error: string | null;
  lastChecked: Date | null;
}
