import { useState } from "react";
import axios from "axios";
import TimelineChart from "./components/TimelineChart";
import SignalHeatmap from "./components/SignalHeatmap";
import "./App.css";

const SAMPLE = `CEO: We plan to expand operations globally.
CFO: Costs may rise due to supply chain issues.
Analyst: How will this impact margins?
CFO: We are monitoring cost structure carefully.`;

function App() {
  const [transcript, setTranscript] = useState(SAMPLE);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const data = result;

  const getPredictionClass = (prediction) => {
    if (prediction === "UP") return "badge-up";
    if (prediction === "DOWN") return "badge-down";
    return "badge-neutral";
  };

  const getVolatilityLabel = (volatility) => {
    if (volatility === "LOW") return "🟢 Stable";
    if (volatility === "MEDIUM") return "🟡 Moderate";
    if (volatility === "HIGH") return "🔴 Volatile";
    return "-";
  };

  const getVolatilityClass = (volatility) => {
    if (volatility === "LOW") return "volatility-low";
    if (volatility === "MEDIUM") return "volatility-medium";
    if (volatility === "HIGH") return "volatility-high";
    return "";
  };

  const getConfidenceValue = () => {
    const confidence = Number(data?.confidence ?? 0);
    if (Number.isNaN(confidence)) return 0;
    return Math.max(0, Math.min(100, confidence));
  };

  const confidenceValue = getConfidenceValue();
  const scoreValue = Number(data?.score ?? 0);
  const scoreDisplay = data?.score ?? "-";
  const growthDrivers = data?.drivers?.growth_drivers || [];
  const riskDrivers = data?.drivers?.risk_drivers || [];

  const getDriverItems = (items) => {
    if (items.length > 0) return items;
    return ["No drivers detected"];
  };

  const growthDriverItems = getDriverItems(growthDrivers);
  const riskDriverItems = getDriverItems(riskDrivers);

  const getDriverKey = (prefix, index, value) => `${prefix}-${index}-${value.slice(0, 12)}`;

  const getScoreTextColor = (score) => {
    if (typeof score !== "number" || Number.isNaN(score)) {
      return "#d4d4d4";
    }
    if (score < 30) return "#4CAF50";
    if (score <= 60) return "#FFC107";
    return "#F44336";
  };

  const getScoreShadow = (score) => {
    const color = getScoreTextColor(score);
    if (color === "#d4d4d4") return "none";
    return `0 0 10px ${color}B3`;
  };

  const distributionOrder = [
    "EXPANSION",
    "GENERAL_UPDATE",
    "STRATEGIC_PROBING",
    "COST_PRESSURE",
  ];

  const signalDistribution = distributionOrder.map((intent) => {
    const total = (data?.segments || []).length;
    const count = (data?.segments || []).filter((segment) => segment.intent === intent).length;
    const percentage = total > 0 ? Math.round((count / total) * 100) : 0;
    return { intent, percentage };
  });

  const analyzeTranscript = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await axios.post("http://127.0.0.1:8000/analyze", {
        transcript,
      });
      setResult(response.data);
    } catch (err) {
      setResult(null);
      setError(
        err?.response?.data?.detail ||
          "Unable to analyze transcript. Ensure backend is running on 127.0.0.1:8000"
      );
    } finally {
      setLoading(false);
    }
  };

  const uploadFile = async () => {
    if (!file) {
      setError("Please select a .txt transcript file first.");
      return;
    }

    setLoading(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post("http://127.0.0.1:8000/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      if (response.data?.error) {
        setResult(null);
        setError(response.data.error);
      } else {
        setResult(response.data);
      }
    } catch (err) {
      setResult(null);
      setError(
        err?.response?.data?.detail ||
          "Unable to upload transcript. Ensure backend is running on 127.0.0.1:8000"
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <aside className="left card fade-card">
        <h1>Financial Transcript Analyzer</h1>
        <label htmlFor="transcript">Transcript Input</label>
        <textarea
          id="transcript"
          value={transcript}
          onChange={(e) => setTranscript(e.target.value)}
          placeholder="Paste earnings call transcript here..."
        />
        <button onClick={analyzeTranscript} disabled={loading || !transcript.trim()}>
          {loading ? "Analyzing..." : "Analyze"}
        </button>

        <div className="upload-section">
          <label htmlFor="file-upload">Upload Transcript (.txt / .pdf)</label>
          <input
            id="file-upload"
            type="file"
            accept=".txt,.pdf,text/plain,application/pdf"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
          <button onClick={uploadFile} disabled={loading || !file}>
            {loading ? "Uploading..." : "Upload Transcript"}
          </button>
        </div>
        {error ? <p className="error">{error}</p> : null}
      </aside>

      <section className="right">
        <div className="right-content">
          <div className="metrics-card card fade-card">
            <div className="risk-card">
              <h2>Risk Score</h2>
              <h1
                className="score"
                style={{
                  color: getScoreTextColor(scoreValue),
                  textShadow: getScoreShadow(scoreValue),
                }}
              >
                {scoreDisplay}
              </h1>
            </div>

            <div className="metrics-grid">
              <div className="metric-item">
                <h3>Market Prediction</h3>
                <span className={`badge ${getPredictionClass(data?.prediction)}`}>
                  {data?.prediction || "NEUTRAL"}
                </span>
              </div>

              <div className="metric-item">
                <h3>Confidence</h3>
                <p>{data?.confidence ?? "-"}%</p>
                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{
                      width: `${confidenceValue}%`,
                      background: "#569cd6",
                    }}
                  />
                </div>
              </div>

              <div className="metric-item">
                <h3>Volatility</h3>
                <p className={`volatility-tag ${getVolatilityClass(data?.volatility)}`}>
                  {getVolatilityLabel(data?.volatility)}
                </p>
              </div>

              <div className="metric-item">
                <h3>Signal</h3>
                <p>{data?.signal || "-"}</p>
              </div>

              <div className="metric-item full-width">
                <h3>Insight</h3>
                <p>{data?.insight || "-"}</p>
              </div>
            </div>
          </div>

          <div className="analytics-card card fade-card">
            <div className="chart-panel">
              <h2>Timeline Graph</h2>
              <div className="timeline">
                <TimelineChart segments={data?.segments || []} />
              </div>
            </div>

            <div className="heatmap-panel">
              <h2>Signal Heatmap</h2>
              <SignalHeatmap segments={data?.segments || []} />
            </div>

            <div className="distribution-panel">
              <h3>Signal Distribution</h3>
              {signalDistribution.map((row) => (
                <div className="distribution-row" key={row.intent}>
                  <span className="distribution-label">{row.intent}</span>
                  <div className="distribution-track">
                    <div
                      className="distribution-fill"
                      style={{ width: `${row.percentage}%` }}
                    />
                  </div>
                  <span className="distribution-percent">{row.percentage}%</span>
                </div>
              ))}
            </div>

            <div className="drivers-grid">
              <div className="driver-column growth-column">
                <h3>Top Growth Drivers</h3>
                <ul className="driver-list">
                  {growthDriverItems.map((driver, index) => (
                    <li
                      key={getDriverKey("growth", index, driver)}
                      title={driver}
                      className="driver-item"
                    >
                      {driver}
                    </li>
                  ))}
                </ul>
              </div>

              <div className="driver-column risk-column">
                <h3>Top Risk Drivers</h3>
                <ul className="driver-list">
                  {riskDriverItems.map((driver, index) => (
                    <li
                      key={getDriverKey("risk", index, driver)}
                      title={driver}
                      className="driver-item"
                    >
                      {driver}
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            <h2>Segments</h2>
            <div className="segment-list">
              {(data?.segments || []).map((segment, index) => (
                <div
                  className="segment-item"
                  key={`${segment.speaker}-${segment.intent}-${index}`}
                  title={segment.text}
                >
                  {segment.speaker} - {segment.intent}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

export default App;
