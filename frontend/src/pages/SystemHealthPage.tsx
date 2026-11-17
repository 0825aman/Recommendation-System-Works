/**
 * SystemHealthPage — the Phase 1 system health dashboard.
 *
 * Displays live status from all three backend endpoints:
 *   - Application health (/api/v1/health)
 *   - Database connectivity (/api/v1/health/db)
 *   - System information (/api/v1/system/info)
 *
 * Auto-refreshes every 30 seconds. Manual refresh button also available.
 */

import { useHealth } from "../hooks/useHealth";
import { StatusCard } from "../components/StatusCard";

function formatTime(date: Date | null): string {
  if (!date) return "—";
  return date.toLocaleTimeString();
}

export default function SystemHealthPage() {
  const { appHealth, dbHealth, systemInfo, loading, error, lastChecked, refresh } =
    useHealth();

  const appStatus = loading
    ? "loading"
    : error
      ? "error"
      : appHealth?.status === "ok"
        ? "ok"
        : "error";

  const dbStatus = loading
    ? "loading"
    : dbHealth?.status === "connected"
      ? "connected"
      : "error";

  const appStatusLabel = loading
    ? "Checking…"
    : error
      ? "Unreachable"
      : (appHealth?.status ?? "unknown");

  const dbStatusLabel = loading
    ? "Checking…"
    : (dbHealth?.status ?? "unknown");

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#ffffff",
        fontFamily:
          '-apple-system, "Segoe UI", system-ui, sans-serif',
        color: "#1f2328",
        padding: "2.5rem 1.5rem",
      }}
    >
      {/* Header */}
      <div style={{ maxWidth: "900px", margin: "0 auto" }}>
        <div
          style={{
            borderBottom: "1px solid #e5e7eb",
            paddingBottom: "1.5rem",
            marginBottom: "2rem",
          }}
        >
          <h1
            style={{
              fontSize: "1.4rem",
              fontWeight: 600,
              margin: "0 0 0.25rem",
              color: "#1f2328",
            }}
          >
            Hybrid Recommendation Intelligence Platform
          </h1>
          <p style={{ fontSize: "0.9rem", color: "#57606a", margin: 0 }}>
            System Health Dashboard — Phase 1
          </p>
        </div>

        {/* Error banner */}
        {error && (
          <div
            style={{
              background: "#fef2f2",
              border: "1px solid #fecaca",
              borderRadius: "0.375rem",
              padding: "0.75rem 1rem",
              marginBottom: "1.5rem",
              fontSize: "0.875rem",
              color: "#991b1b",
            }}
          >
            <strong>Backend unreachable:</strong> {error}. Make sure the
            backend is running on port 8000.
          </div>
        )}

        {/* Status cards */}
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "1rem",
            marginBottom: "2rem",
          }}
        >
          <StatusCard
            title="Application"
            status={appStatus}
            statusLabel={appStatusLabel}
            details={
              appHealth
                ? [
                    { label: "Version", value: appHealth.version },
                    { label: "Environment", value: appHealth.environment },
                  ]
                : []
            }
          />

          <StatusCard
            title="Database"
            status={dbStatus}
            statusLabel={dbStatusLabel}
            details={
              dbHealth?.detail ? [{ label: "Detail", value: dbHealth.detail }] : []
            }
          />

          <StatusCard
            title="System"
            status={loading ? "loading" : error ? "error" : "ok"}
            statusLabel={loading ? "Checking…" : error ? "Unavailable" : "Available"}
            details={
              systemInfo
                ? [
                    { label: "Python", value: systemInfo.python_version },
                    { label: "Platform", value: systemInfo.platform_system },
                    { label: "API", value: systemInfo.api_version },
                  ]
                : []
            }
          />
        </div>

        {/* Footer bar */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexWrap: "wrap",
            gap: "0.75rem",
            borderTop: "1px solid #e5e7eb",
            paddingTop: "1rem",
          }}
        >
          <span style={{ fontSize: "0.8rem", color: "#9ca3af" }}>
            Last checked: {formatTime(lastChecked)} · Auto-refreshes every 30s
          </span>
          <button
            onClick={() => void refresh()}
            disabled={loading}
            style={{
              fontSize: "0.8rem",
              padding: "0.35rem 0.9rem",
              borderRadius: "0.375rem",
              border: "1px solid #e5e7eb",
              background: loading ? "#f3f4f6" : "#ffffff",
              color: loading ? "#9ca3af" : "#374151",
              cursor: loading ? "not-allowed" : "pointer",
            }}
          >
            {loading ? "Refreshing…" : "Refresh now"}
          </button>
        </div>
      </div>
    </div>
  );
}
