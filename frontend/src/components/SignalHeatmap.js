export default function SignalHeatmap({ segments }) {
  const classMap = {
    EXPANSION: "intent-expansion",
    COST_PRESSURE: "intent-cost-pressure",
    STRATEGIC_PROBING: "intent-strategic-probing",
    GENERAL_UPDATE: "intent-general-update",
  };

  const counts = {};

  segments.forEach((s) => {
    counts[s.intent] = (counts[s.intent] || 0) + 1;
  });

  return (
    <div className="heatmap-grid">
      {Object.entries(counts).map(([intent, count]) => (
        <div
          key={intent}
          className={`heatmap-card ${classMap[intent] || ""}`}
          title={intent}
        >
          <div className="heatmap-title">{intent}</div>
          <div className="heatmap-count">{count}</div>
        </div>
      ))}
    </div>
  );
}
