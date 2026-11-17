/**
 * StatusCard — a surface card wrapping a status indicator and optional detail rows.
 */

import { StatusIndicator } from "./StatusIndicator";

interface DetailRow {
  label: string;
  value: string;
}

interface StatusCardProps {
  title: string;
  status: "ok" | "connected" | "error" | "loading";
  statusLabel: string;
  details?: DetailRow[];
}

export function StatusCard({
  title,
  status,
  statusLabel,
  details = [],
}: StatusCardProps) {
  const borderColor =
    status === "ok" || status === "connected"
      ? "#bbf7d0"
      : status === "error"
        ? "#fecaca"
        : "#e5e7eb";

  return (
    <div
      style={{
        background: "#ffffff",
        border: `1px solid ${borderColor}`,
        borderTop: `3px solid ${status === "ok" || status === "connected" ? "#22c55e" : status === "error" ? "#ef4444" : "#d1d5db"}`,
        borderRadius: "0.5rem",
        padding: "1.25rem 1.375rem",
        display: "flex",
        flexDirection: "column",
        gap: "0.875rem",
      }}
    >
      {/* Card title */}
      <div
        style={{
          fontSize: "0.72rem",
          fontWeight: 700,
          letterSpacing: "0.07em",
          textTransform: "uppercase",
          color: "#9ca3af",
        }}
      >
        {title}
      </div>

      {/* Status indicator */}
      <StatusIndicator status={status} label={statusLabel} />

      {/* Detail rows */}
      {details.length > 0 && (
        <div
          style={{
            borderTop: "1px solid #f3f4f6",
            paddingTop: "0.75rem",
            display: "flex",
            flexDirection: "column",
            gap: "0.35rem",
          }}
        >
          {details.map(({ label, value }) => (
            <div
              key={label}
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "baseline",
                gap: "0.5rem",
              }}
            >
              <span style={{ fontSize: "0.78rem", color: "#9ca3af", flexShrink: 0 }}>
                {label}
              </span>
              <span
                style={{
                  fontSize: "0.78rem",
                  color: "#374151",
                  fontFamily: "ui-monospace, monospace",
                  textAlign: "right",
                }}
              >
                {value}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
