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
  return (
    <div
      style={{
        background: "#f9fafb",
        border: "1px solid #e5e7eb",
        borderRadius: "0.5rem",
        padding: "1.25rem 1.5rem",
        display: "flex",
        flexDirection: "column",
        gap: "0.75rem",
        minWidth: "260px",
        flex: "1",
      }}
    >
      <div
        style={{
          fontSize: "0.75rem",
          fontWeight: 600,
          letterSpacing: "0.05em",
          textTransform: "uppercase",
          color: "#6b7280",
        }}
      >
        {title}
      </div>

      <StatusIndicator status={status} label={statusLabel} />

      {details.length > 0 && (
        <dl
          style={{
            margin: 0,
            display: "grid",
            gridTemplateColumns: "auto 1fr",
            gap: "0.25rem 1rem",
          }}
        >
          {details.map(({ label, value }) => (
            <div
              key={label}
              style={{ display: "contents" }}
            >
              <dt
                style={{
                  fontSize: "0.8rem",
                  color: "#9ca3af",
                  whiteSpace: "nowrap",
                }}
              >
                {label}
              </dt>
              <dd
                style={{
                  fontSize: "0.8rem",
                  color: "#374151",
                  margin: 0,
                  fontFamily: "monospace",
                }}
              >
                {value}
              </dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}
