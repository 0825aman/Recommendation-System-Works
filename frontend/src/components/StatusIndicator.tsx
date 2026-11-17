/**
 * StatusIndicator — a coloured dot with a text label.
 * Displays green for "ok"/"connected", red for "error", grey while loading.
 */

interface StatusIndicatorProps {
  status: "ok" | "connected" | "error" | "loading";
  label: string;
}

const STATUS_STYLES: Record<StatusIndicatorProps["status"], string> = {
  ok: "#22c55e",
  connected: "#22c55e",
  error: "#ef4444",
  loading: "#9ca3af",
};

export function StatusIndicator({ status, label }: StatusIndicatorProps) {
  const color = STATUS_STYLES[status];

  return (
    <span
      style={{ display: "inline-flex", alignItems: "center", gap: "0.5rem" }}
    >
      <span
        style={{
          display: "inline-block",
          width: "0.65rem",
          height: "0.65rem",
          borderRadius: "50%",
          backgroundColor: color,
          flexShrink: 0,
        }}
        aria-label={status}
      />
      <span style={{ fontSize: "0.9rem", color: "#374151" }}>{label}</span>
    </span>
  );
}
