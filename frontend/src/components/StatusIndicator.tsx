/**
 * StatusIndicator — a coloured dot with a text label.
 */

interface StatusIndicatorProps {
  status: "ok" | "connected" | "error" | "loading";
  label: string;
}

const DOT_COLOR: Record<StatusIndicatorProps["status"], string> = {
  ok: "#16a34a",
  connected: "#16a34a",
  error: "#dc2626",
  loading: "#9ca3af",
};

const LABEL_COLOR: Record<StatusIndicatorProps["status"], string> = {
  ok: "#15803d",
  connected: "#15803d",
  error: "#dc2626",
  loading: "#6b7280",
};

export function StatusIndicator({ status, label }: StatusIndicatorProps) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: "0.5rem" }}>
      <span
        style={{
          display: "inline-block",
          width: "0.6rem",
          height: "0.6rem",
          borderRadius: "50%",
          backgroundColor: DOT_COLOR[status],
          flexShrink: 0,
          boxShadow: status !== "loading" ? `0 0 0 3px ${DOT_COLOR[status]}22` : "none",
        }}
        aria-label={status}
      />
      <span style={{ fontSize: "1rem", fontWeight: 600, color: LABEL_COLOR[status] }}>
        {label}
      </span>
    </span>
  );
}
