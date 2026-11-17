/**
 * useHealth — custom hook that fetches data from all three health endpoints.
 *
 * Polls automatically every 30 seconds and exposes a manual refresh function.
 */

import { useCallback, useEffect, useState } from "react";
import type { HealthData } from "../types/health";
import {
  fetchAppHealth,
  fetchDatabaseHealth,
  fetchSystemInfo,
} from "../services/healthService";

const POLL_INTERVAL_MS = 30_000;

export function useHealth(): HealthData & { refresh: () => void } {
  const [data, setData] = useState<HealthData>({
    appHealth: null,
    dbHealth: null,
    systemInfo: null,
    loading: true,
    error: null,
    lastChecked: null,
  });

  const fetchAll = useCallback(async () => {
    setData((prev) => ({ ...prev, loading: true, error: null }));

    try {
      const [appHealth, dbHealth, systemInfo] = await Promise.all([
        fetchAppHealth(),
        fetchDatabaseHealth(),
        fetchSystemInfo(),
      ]);

      setData({
        appHealth,
        dbHealth,
        systemInfo,
        loading: false,
        error: null,
        lastChecked: new Date(),
      });
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Failed to reach backend.";
      setData((prev) => ({
        ...prev,
        loading: false,
        error: message,
        lastChecked: new Date(),
      }));
    }
  }, []);

  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect -- intentional: initial data fetch on mount
    void fetchAll();
    const interval = setInterval(() => void fetchAll(), POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [fetchAll]);

  return { ...data, refresh: fetchAll };
}
