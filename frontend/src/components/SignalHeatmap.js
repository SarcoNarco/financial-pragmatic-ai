export default function SignalHeatmap({ segments }) {
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
            background: "#1e1e1e",
            border: "1px solid #333",
            borderRadius: "8px",
            textAlign: "center",
            cursor: "pointer",
          }}
          onClick={() => alert(intent)}
        >
          <div style={{ fontSize: "12px", color: "#888" }}>{intent}</div>
          <div style={{ fontSize: "20px", color: "#d4d4d4" }}>{count}</div>
        </div>
      ))}
    </div>
  );
}
