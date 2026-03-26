function toneClass(value) {
  const normalized = String(value || "").toLowerCase();
  if (["growth", "up", "low"].includes(normalized)) return "growth";
  if (["risk", "down", "high"].includes(normalized)) return "risk";
  return "neutral";
}

export default function SummaryCard({
  signal,
  score,
  prediction,
  confidence,
  volatility,
  keyDriver,
  keyConcern,
}) {
  return (
    <section className="panel summary-card">
      <div className="panel-title">Analysis Summary</div>
      <div className="summary-row">
        <div className="summary-item">
          <span className="label">Signal</span>
          <span className={`value ${toneClass(signal)}`}>{String(signal || "neutral").toUpperCase()}</span>
        </div>
        <div className="summary-item">
          <span className="label">Risk Score</span>
          <span className={`value ${toneClass(score > 65 ? "risk" : score < 35 ? "growth" : "neutral")}`}>{score ?? "-"}</span>
        </div>
        <div className="summary-item">
          <span className="label">Prediction</span>
          <span className={`value ${toneClass(prediction)}`}>{String(prediction || "NEUTRAL").toUpperCase()}</span>
        </div>
        <div className="summary-item">
          <span className="label">Confidence</span>
          <span className="value">{confidence ?? "-"}%</span>
        </div>
        <div className="summary-item">
          <span className="label">Volatility</span>
          <span className={`value ${toneClass(volatility)}`}>{String(volatility || "-").toUpperCase()}</span>
        </div>
      </div>
      <div className="summary-notes">
        <div>
          <span className="label">Key Driver:</span> {keyDriver || "No growth driver detected"}
        </div>
        <div>
          <span className="label">Key Concern:</span> {keyConcern || "No risk concern detected"}
        </div>
      </div>
    </section>
  );
}
