import { useState } from "react";
import axios from "axios";
import "./App.css";

const SAMPLE = `CEO: We plan to expand operations globally.
CFO: Costs may rise due to supply chain issues.
Analyst: How will this impact margins?
CFO: We are monitoring cost structure carefully.`;

function App() {
  const [transcript, setTranscript] = useState(SAMPLE);
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

        <h3>Segments</h3>
        <pre>{JSON.stringify(result?.segments || [], null, 2)}</pre>
      </section>
    </main>
  );
}

export default App;
