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

  const getColor = (score) => {
    if (score > 70) return "#f44336";
    if (score > 40) return "#ff9800";
    return "#4caf50";
  };

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
      <h1>Financial Transcript Analyzer</h1>

      <div className="top-section">
        <div className="left-panel">
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
        </div>

        <div className="right-panel">
          <h2>Risk Overview</h2>
          <h1 style={{ fontSize: "48px", color: getColor(data?.score ?? 0) }}>
            {data?.score ?? "-"}
          </h1>
          <p>Risk Score</p>
          <h3>Market Prediction</h3>
          <p>{data?.prediction || "-"}</p>
          <h3>Confidence</h3>
          <p>{data?.confidence ?? "-"}%</p>
          <h3>Volatility</h3>
          <p>{data?.volatility || "-"}</p>

          <div className="result-row">
            <span className="label">Signal</span>
            <span className="value">{data?.signal || "-"}</span>
          </div>
          <div className="result-row">
            <span className="label">Insight</span>
            <span className="value">{data?.insight || "-"}</span>
          </div>

        </div>
      </div>

      <div className="bottom-section">
        <h2>Timeline Graph</h2>
        <TimelineChart segments={data?.segments || []} />

        <h2>Signal Heatmap</h2>
        <SignalHeatmap segments={data?.segments || []} />

        <h3>Top Growth Drivers</h3>
        <ul>
          {(data?.drivers?.growth_drivers || []).map((driver, index) => (
            <li key={`growth-${index}`}>{driver}</li>
          ))}
        </ul>

        <h3>Top Risk Drivers</h3>
        <ul>
          {(data?.drivers?.risk_drivers || []).map((driver, index) => (
            <li key={`risk-${index}`}>{driver}</li>
          ))}
        </ul>

        <h2>Segments</h2>
        <div className="segment-list">
          {(data?.segments || []).map((segment, index) => (
            <div className="segment-item" key={`${segment.speaker}-${segment.intent}-${index}`} title={segment.text}>
              {segment.speaker} - {segment.intent}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;
