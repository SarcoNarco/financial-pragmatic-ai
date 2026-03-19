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
    <main className="app">
      <section className="panel main-panel">
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
          <label htmlFor="file-upload">Upload Transcript (.txt)</label>
          <input
            id="file-upload"
            type="file"
            accept=".txt,text/plain"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
          <button onClick={uploadFile} disabled={loading || !file}>
            {loading ? "Uploading..." : "Upload Transcript"}
          </button>
        </div>
        {error ? <p className="error">{error}</p> : null}
      </section>

      <section className="panel result-panel">
        <h2>Results</h2>
        <div className="result-row">
          <span className="label">Signal</span>
          <span className="value">{result?.signal || "-"}</span>
        </div>
        <div className="result-row">
          <span className="label">Insight</span>
          <span className="value">{result?.insight || "-"}</span>
        </div>
        <h3>Risk Score</h3>
        <p>{result ? `${result.risk_score}/100` : "-"}</p>
        {result?.conflict ? <div style={{ color: "red" }}>⚠️ {result.conflict}</div> : null}

        <h3>Segments</h3>
        <pre>{JSON.stringify(result?.segments || [], null, 2)}</pre>

        {result ? (
          <>
            <h3>Conversation Timeline</h3>
            {(result.timeline || []).map((item, i) => (
              <div
                key={i}
                style={{
                  display: "flex",
                  gap: "20px",
                  padding: "6px",
                  borderBottom: "1px solid #333",
                }}
              >
                <span>{item.time}</span>
                <span>{item.speaker}</span>
                <span>{item.intent}</span>
              </div>
            ))}

            <h3>Timeline Graph</h3>
            <TimelineChart segments={result.segments || []} />

            <h3>Signal Heatmap</h3>
            <SignalHeatmap segments={result.segments || []} />
          </>
        ) : null}
      </section>
    </main>
  );
}

export default App;
