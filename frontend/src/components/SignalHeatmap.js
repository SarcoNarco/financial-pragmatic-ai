export default function SignalHeatmap({ segments }) {
  const colorMap = {
    EXPANSION: "#4caf50",
    COST_PRESSURE: "#f44336",
    STRATEGIC_PROBING: "#ff9800",
    GENERAL_UPDATE: "#9e9e9e",
  };

  const counts = {};

  segments.forEach((s) => {
    counts[s.intent] = (counts[s.intent] || 0) + 1;
  });

  return (
    <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
      {Object.entries(counts).map(([intent, count]) => (
        <div
          key={intent}
          style={{
            minWidth: "140px",
            padding: "20px",
            background: colorMap[intent] || "#1e1e1e",
            border: "1px solid #333",
            borderRadius: "8px",
            textAlign: "center",
            cursor: "pointer",
          }}
          onClick={() => alert(intent)}
        >
          <div style={{ fontSize: "12px", color: "#f5f5f5" }}>{intent}</div>
          <div style={{ fontSize: "20px", color: "#d4d4d4" }}>{count}</div>
        </div>
      ))}
    </div>
  );
}
