/**
 * Health service — calls all three backend health/info endpoints.
 */

import type {
  DatabaseHealthResponse,
  HealthResponse,
  SystemInfoResponse,
} from "../types/health";
import apiClient from "./api";

export async function fetchAppHealth(): Promise<HealthResponse> {
  const response = await apiClient.get<HealthResponse>("/api/v1/health");
  return response.data;
}

export async function fetchDatabaseHealth(): Promise<DatabaseHealthResponse> {
  const response = await apiClient.get<DatabaseHealthResponse>(
    "/api/v1/health/db",
  );
  return response.data;
}

export async function fetchSystemInfo(): Promise<SystemInfoResponse> {
  const response = await apiClient.get<SystemInfoResponse>(
    "/api/v1/system/info",
  );
  return response.data;
}
