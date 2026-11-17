/**
 * SystemHealthPage — Phase 1 system health dashboard.
 *
 * Displays live status from all three backend endpoints.
 * Auto-refreshes every 30 seconds.
 */

import { useHealth } from "../hooks/useHealth";
import { StatusCard } from "../components/StatusCard";

function formatTime(date: Date | null): string {
  if (!date) return "—";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function OverallBadge({ loading, error, dbConnected }: { loading: boolean; error: string | null; dbConnected: boolean }) {
  if (loading) {
    return (
      <span style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem", fontSize: "0.78rem", fontWeight: 600, color: "#6b7280", background: "#f3f4f6", border: "1px solid #e5e7eb", borderRadius: "999px", padding: "0.25rem 0.75rem" }}>
        <span style={{ width: "7px", height: "7px", borderRadius: "50%", background: "#9ca3af", display: "inline-block" }} />
        CHECKING
      </span>
    );
  }
  if (error || !dbConnected) {
    return (
      <span style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem", fontSize: "0.78rem", fontWeight: 600, color: "#991b1b", background: "#fef2f2", border: "1px solid #fecaca", borderRadius: "999px", padding: "0.25rem 0.75rem" }}>
        <span style={{ width: "7px", height: "7px", borderRadius: "50%", background: "#ef4444", display: "inline-block" }} />
        DEGRADED
      </span>
    );
  }
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem", fontSize: "0.78rem", fontWeight: 600, color: "#166534", background: "#f0fdf4", border: "1px solid #bbf7d0", borderRadius: "999px", padding: "0.25rem 0.75rem" }}>
      <span style={{ width: "7px", height: "7px", borderRadius: "50%", background: "#22c55e", display: "inline-block" }} />
      ALL SYSTEMS OPERATIONAL
    </span>
  );
}

export default function SystemHealthPage() {
  const { appHealth, dbHealth, systemInfo, loading, error, lastChecked, refresh } =
    useHealth();

  const dbConnected = dbHealth?.status === "connected";

  const appStatus = loading ? "loading" : error ? "error" : appHealth?.status === "ok" ? "ok" : "error";
  const dbStatus = loading ? "loading" : dbConnected ? "connected" : "error";
  const sysStatus = loading ? "loading" : error ? "error" : "ok";

  return (
    <div style={{ minHeight: "100vh", background: "#f8fafc", fontFamily: '-apple-system, "Segoe UI", system-ui, sans-serif', color: "#1f2328" }}>

      {/* Top navigation bar */}
      <header style={{ background: "#ffffff", borderBottom: "1px solid #e5e7eb", padding: "0 2rem", height: "56px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          {/* Logo mark */}
          <svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="28" height="28" rx="6" fill="#1d4ed8" />
            <circle cx="10" cy="14" r="3" fill="white" opacity="0.9" />
            <circle cx="18" cy="9" r="2" fill="white" opacity="0.7" />
            <circle cx="18" cy="19" r="2" fill="white" opacity="0.7" />
            <line x1="13" y1="12.5" x2="16.2" y2="10.2" stroke="white" strokeWidth="1.2" opacity="0.6" />
            <line x1="13" y1="15.5" x2="16.2" y2="17.8" stroke="white" strokeWidth="1.2" opacity="0.6" />
          </svg>
          <div>
            <div style={{ fontSize: "0.9rem", fontWeight: 700, color: "#111827", lineHeight: 1.2 }}>HRIP</div>
            <div style={{ fontSize: "0.68rem", color: "#9ca3af", letterSpacing: "0.03em", lineHeight: 1.2 }}>Hybrid Recommendation Intelligence Platform</div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <span style={{ fontSize: "0.78rem", color: "#9ca3af" }}>Phase 1 — Foundation</span>
          <OverallBadge loading={loading} error={error} dbConnected={dbConnected} />
        </div>
      </header>

      {/* Page content */}
      <main style={{ maxWidth: "960px", margin: "0 auto", padding: "2.5rem 1.5rem" }}>

        {/* Page heading */}
        <div style={{ marginBottom: "2rem" }}>
          <h1 style={{ fontSize: "1.5rem", fontWeight: 700, margin: "0 0 0.3rem", color: "#111827" }}>
            System Health
          </h1>
          <p style={{ fontSize: "0.9rem", color: "#6b7280", margin: 0 }}>
            Real-time status of all platform services. Refreshes automatically every 30 seconds.
          </p>
        </div>

        {/* Error banner */}
        {error && (
          <div style={{ background: "#fef2f2", border: "1px solid #fecaca", borderRadius: "0.5rem", padding: "0.875rem 1.125rem", marginBottom: "1.5rem", display: "flex", alignItems: "flex-start", gap: "0.75rem" }}>
            <span style={{ fontSize: "1rem", lineHeight: 1.4 }}>⚠</span>
            <div>
              <div style={{ fontSize: "0.875rem", fontWeight: 600, color: "#991b1b", marginBottom: "0.1rem" }}>Backend unreachable</div>
              <div style={{ fontSize: "0.82rem", color: "#b91c1c" }}>{error} — make sure the backend server is running on port 8000.</div>
            </div>
          </div>
        )}

        {/* Status cards grid */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem", marginBottom: "2rem" }}>
          <StatusCard
            title="Application"
            status={appStatus}
            statusLabel={loading ? "Checking…" : error ? "Unreachable" : (appHealth?.status ?? "unknown")}
            details={appHealth ? [
              { label: "Version", value: appHealth.version },
              { label: "Environment", value: appHealth.environment },
            ] : []}
          />

          <StatusCard
            title="Database"
            status={dbStatus}
            statusLabel={loading ? "Checking…" : (dbHealth?.status ?? "unknown")}
            details={dbHealth?.detail ? [{ label: "Detail", value: dbHealth.detail }] : []}
          />

          <StatusCard
            title="Runtime"
            status={sysStatus}
            statusLabel={loading ? "Checking…" : error ? "Unavailable" : "Available"}
            details={systemInfo ? [
              { label: "Python", value: systemInfo.python_version },
              { label: "Platform", value: systemInfo.platform_system },
              { label: "API", value: systemInfo.api_version },
            ] : []}
          />
        </div>

        {/* Divider */}
        <hr style={{ border: "none", borderTop: "1px solid #e5e7eb", margin: "0 0 1.5rem" }} />

        {/* Endpoints reference table */}
        <section style={{ marginBottom: "2rem" }}>
          <h2 style={{ fontSize: "0.85rem", fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase", color: "#6b7280", margin: "0 0 0.875rem" }}>
            API Endpoints — v1
          </h2>
          <div style={{ background: "#ffffff", border: "1px solid #e5e7eb", borderRadius: "0.5rem", overflow: "hidden" }}>
            {[
              { method: "GET", path: "/api/v1/health", description: "Application liveness check" },
              { method: "GET", path: "/api/v1/health/db", description: "Database connectivity check" },
              { method: "GET", path: "/api/v1/system/info", description: "Safe system metadata" },
            ].map(({ method, path, description }, i, arr) => (
              <div key={path} style={{ display: "grid", gridTemplateColumns: "60px 1fr auto", alignItems: "center", padding: "0.65rem 1rem", gap: "1rem", borderBottom: i < arr.length - 1 ? "1px solid #f3f4f6" : "none" }}>
                <span style={{ fontSize: "0.72rem", fontWeight: 700, color: "#2563eb", background: "#eff6ff", border: "1px solid #bfdbfe", borderRadius: "4px", padding: "0.15rem 0.35rem", textAlign: "center", fontFamily: "monospace" }}>
                  {method}
                </span>
                <span style={{ fontSize: "0.82rem", fontFamily: "monospace", color: "#374151" }}>{path}</span>
                <span style={{ fontSize: "0.78rem", color: "#9ca3af" }}>{description}</span>
              </div>
            ))}
          </div>
        </section>

        {/* Footer row */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "0.75rem" }}>
          <div style={{ fontSize: "0.78rem", color: "#9ca3af" }}>
            Last checked: <strong style={{ color: "#6b7280" }}>{formatTime(lastChecked)}</strong>
            <span style={{ margin: "0 0.4rem" }}>·</span>
            Auto-refreshes every 30s
          </div>
          <button
            onClick={() => void refresh()}
            disabled={loading}
            style={{ fontSize: "0.8rem", fontWeight: 500, padding: "0.4rem 1rem", borderRadius: "0.375rem", border: "1px solid #d1d5db", background: loading ? "#f9fafb" : "#ffffff", color: loading ? "#9ca3af" : "#374151", cursor: loading ? "not-allowed" : "pointer", transition: "background 0.1s" }}
          >
            {loading ? "Refreshing…" : "↺ Refresh now"}
          </button>
        </div>
      </main>

      {/* Page footer */}
      <footer style={{ borderTop: "1px solid #e5e7eb", padding: "1.25rem 2rem", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.5rem", marginTop: "2rem" }}>
        <span style={{ fontSize: "0.75rem", color: "#9ca3af" }}>
          Hybrid Recommendation Intelligence Platform · Phase 1 Foundation
        </span>
        <span style={{ fontSize: "0.75rem", color: "#d1d5db" }}>
          Built with IBM Bob
        </span>
      </footer>
    </div>
  );
}
