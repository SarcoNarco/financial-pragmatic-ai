import { useMemo, useState } from "react";
import axios from "axios";
import TimelineChart from "./components/TimelineChart";
import SignalHeatmap from "./components/SignalHeatmap";
import SummaryCard from "./components/SummaryCard";
import "./App.css";

const SAMPLE = `CEO: We plan to expand operations globally.
CFO: Costs may rise due to supply chain issues.
Analyst: How will this impact margins?
CFO: We are monitoring cost structure carefully.`;

const DEMO_TRANSCRIPTS = {
  growth: `CEO: We delivered record revenue and strong growth across core markets.
CFO: Operating leverage improved and margins were stable despite investments.
Analyst: How sustainable is demand in the next two quarters?`,
  risk: `CEO: Demand is softening in key regions.
CFO: We are seeing margin pressure from higher input costs and weaker pricing.
Analyst: Could you clarify downside risks to guidance?`,
  mixed: `CEO: We expanded in new regions with positive customer traction.
CFO: Cost inflation remains elevated and may pressure profitability.
Analyst: What is the timeline for margin recovery?`,
};

function getToneClass(value) {
  const normalized = String(value || "").toLowerCase();
  if (["growth", "up", "low"].includes(normalized)) return "growth";
  if (["risk", "down", "high"].includes(normalized)) return "risk";
  return "neutral";
}

function App() {
  const [activeTab, setActiveTab] = useState("analyze");
  const [transcript, setTranscript] = useState(SAMPLE);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [demoCase, setDemoCase] = useState("growth");

  const data = result;

  const analyzedSegments = data?.segments || [];

  const extractedDrivers = useMemo(() => {
    const growthFromSegments = analyzedSegments
      .filter((segment) => segment.intent === "EXPANSION")
      .map((segment) => segment.text)
      .slice(0, 3);

    const riskFromSegments = analyzedSegments
      .filter(
        (segment) =>
          segment.intent === "COST_PRESSURE" || segment.intent === "STRATEGIC_PROBING"
      )
      .map((segment) => segment.text)
      .slice(0, 3);

    return {
      growth: growthFromSegments.length
        ? growthFromSegments
        : (data?.drivers?.growth_drivers || []).slice(0, 3),
      risk: riskFromSegments.length
        ? riskFromSegments
        : (data?.drivers?.risk_drivers || []).slice(0, 3),
    };
  }, [analyzedSegments, data?.drivers]);

  const growthDrivers = extractedDrivers.growth;
  const riskDrivers = extractedDrivers.risk;

  const signalDistribution = useMemo(() => {
    const order = ["EXPANSION", "GENERAL_UPDATE", "STRATEGIC_PROBING", "COST_PRESSURE"];
    const total = analyzedSegments.length;

    return order.map((intent) => {
      const count = analyzedSegments.filter((segment) => segment.intent === intent).length;
      const percentage = total > 0 ? Math.round((count / total) * 100) : 0;
      return { intent, count, percentage };
    });
  }, [analyzedSegments]);

  const analyzeTranscript = async (overrideText) => {
    const payloadText = typeof overrideText === "string" ? overrideText : transcript;

    if (!payloadText.trim()) {
      setError("Please enter a transcript");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await axios.post("http://127.0.0.1:8000/analyze", {
        transcript: payloadText,
      });

      if (response.data?.error) {
        setResult(null);
        setError("Analysis failed. Try again.");
        return;
      }

      setResult(response.data);
      setActiveTab("analyze");
    } catch (_err) {
      setResult(null);
      setError("Analysis failed. Try again.");
    } finally {
      setLoading(false);
    }
  };

  const uploadFile = async () => {
    if (!file) {
      setError("Please enter a transcript");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post("http://127.0.0.1:8000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (response.data?.error) {
        setResult(null);
        setError("Analysis failed. Try again.");
        return;
      }

      setResult(response.data);
      setActiveTab("analyze");
    } catch (_err) {
      setResult(null);
      setError("Analysis failed. Try again.");
    } finally {
      setLoading(false);
    }
  };

  const loadSampleTranscript = async () => {
    const selected = DEMO_TRANSCRIPTS[demoCase];
    setTranscript(selected);
    await analyzeTranscript(selected);
  };

  return (
    <div className="main-container">
      <div className="content">
        <main className="container">
          <nav className="top-nav panel">
            <button
              className={activeTab === "analyze" ? "tab active" : "tab"}
              onClick={() => setActiveTab("analyze")}
              disabled={loading}
            >
              Analyze
            </button>
            <button
              className={activeTab === "insights" ? "tab active" : "tab"}
              onClick={() => setActiveTab("insights")}
              disabled={loading}
            >
              Insights
            </button>
            <button
              className={activeTab === "demo" ? "tab active" : "tab"}
              onClick={() => setActiveTab("demo")}
              disabled={loading}
            >
              Demo
            </button>
          </nav>

          {activeTab === "analyze" ? (
            <div className="main-grid">
              <aside className="left-panel">
                <section className="panel input-panel">
                  <div className="panel-title">Transcript Input</div>
                  <textarea
                    id="transcript"
                    value={transcript}
                    onChange={(event) => setTranscript(event.target.value)}
                    placeholder="Paste earnings call transcript here..."
                    disabled={loading}
                  />
                  <button onClick={() => analyzeTranscript()} disabled={loading}>
                    {loading ? "Analyzing..." : "Analyze"}
                  </button>
                  <div className="upload-section">
                    <label htmlFor="file-upload">Upload (.txt / .pdf)</label>
                    <input
                      id="file-upload"
                      type="file"
                      accept=".txt,.pdf,text/plain,application/pdf"
                      onChange={(event) => setFile(event.target.files?.[0] || null)}
                      disabled={loading}
                    />
                    <button onClick={uploadFile} disabled={loading || !file}>
                      {loading ? "Uploading..." : "Upload"}
                    </button>
                  </div>
                  {loading ? (
                    <div className="loading-row">
                      <span className="spinner" />
                      <span>Analyzing transcript...</span>
                    </div>
                  ) : null}
                  {error ? <p className="error">{error}</p> : null}
                </section>
              </aside>

              <section className="right-panel">
                <SummaryCard
                  signal={data?.signal}
                  score={data?.score}
                  prediction={data?.prediction}
                  confidence={data?.confidence}
                  volatility={data?.volatility}
                  keyDriver={growthDrivers[0]}
                  keyConcern={riskDrivers[0]}
                />

                <section className="panel graph-panel">
                  <div className="panel-title">Timeline</div>
                  <div className="chart-container">
                    <TimelineChart segments={analyzedSegments} />
                  </div>
                </section>

                <section className="panel">
                  <div className="panel-title">Signal Heatmap</div>
                  <SignalHeatmap segments={analyzedSegments} />
                </section>

                <section className="panel distribution-panel">
                  <div className="panel-title">Signal Distribution</div>
                  {signalDistribution.map((row) => (
                    <div className="distribution-row" key={row.intent}>
                      <span className="distribution-label">{row.intent}</span>
                      <div className="bar">
                        <div className="bar-fill" style={{ width: `${row.percentage}%` }} />
                      </div>
                      <span className="distribution-percent">{row.percentage}%</span>
                    </div>
                  ))}
                </section>

                <section className="panel drivers-panel">
                  <div className="panel-title">Drivers</div>
                  <div className="drivers">
                    <div className="driver-box">
                      <div className="sub-title growth">Growth Drivers</div>
                      <ul className="driver-list">
                        {(growthDrivers.length ? growthDrivers : ["No growth driver detected"]).map(
                          (driver, index) => (
                            <li
                              key={`growth-${index}-${driver.slice(0, 12)}`}
                              title={driver}
                              className="driver-item"
                            >
                              {driver}
                            </li>
                          )
                        )}
                      </ul>
                    </div>

                    <div className="driver-box">
                      <div className="sub-title risk">Risk Drivers</div>
                      <ul className="driver-list">
                        {(riskDrivers.length ? riskDrivers : ["No risk concern detected"]).map(
                          (driver, index) => (
                            <li
                              key={`risk-${index}-${driver.slice(0, 12)}`}
                              title={driver}
                              className="driver-item"
                            >
                              {driver}
                            </li>
                          )
                        )}
                      </ul>
                    </div>
                  </div>
                </section>

                <section className="panel">
                  <div className="panel-title">Segments</div>
                  <div className="segment-list">
                    {analyzedSegments.map((segment, index) => (
                      <div
                        className="segment"
                        key={`${segment.speaker}-${segment.intent}-${index}`}
                        title={segment.text}
                      >
                        [{segment.speaker}] {segment.intent} :: {segment.text}
                      </div>
                    ))}
                  </div>
                </section>
              </section>
            </div>
          ) : null}

          {activeTab === "insights" ? (
            <section className="panel insights-tab">
              <div className="panel-title">Explainability</div>
              <p className="insight-line">
                This system analyzes interactions between speakers, classifies intent at the
                segment level, and aggregates signals to estimate financial direction and risk.
              </p>

              <div className="insights-grid">
                <section className="panel nested-panel">
                  <div className="panel-title">Signal Breakdown</div>
                  {signalDistribution.map((row) => (
                    <div className="distribution-row" key={`insight-${row.intent}`}>
                      <span className="distribution-label">{row.intent}</span>
                      <div className="bar">
                        <div className="bar-fill" style={{ width: `${row.percentage}%` }} />
                      </div>
                      <span className="distribution-percent">{row.percentage}%</span>
                    </div>
                  ))}
                </section>

                <section className="panel nested-panel">
                  <div className="panel-title">Confidence</div>
                  <p className="insight-line">
                    Confidence is based on consistency of detected intent patterns across the
                    transcript. More agreement across segments yields higher confidence.
                  </p>
                </section>

                <section className="panel nested-panel">
                  <div className="panel-title">Volatility</div>
                  <p className="insight-line">
                    Volatility measures fluctuation in intent over time. Frequent shifts between
                    growth and risk signals indicate higher uncertainty.
                  </p>
                  <p className={`insight-line ${getToneClass(data?.volatility)}`}>
                    Current volatility: {(data?.volatility || "-").toUpperCase()}
                  </p>
                </section>
              </div>
            </section>
          ) : null}

          {activeTab === "demo" ? (
            <section className="panel demo-tab">
              <div className="panel-title">Demo Mode</div>
              <p className="insight-line">
                Choose a scenario, load the sample transcript, and run analysis instantly.
              </p>

              <div className="demo-controls">
                <select
                  value={demoCase}
                  onChange={(event) => setDemoCase(event.target.value)}
                  disabled={loading}
                >
                  <option value="growth">Growth case</option>
                  <option value="risk">Risk case</option>
                  <option value="mixed">Mixed case</option>
                </select>

                <button onClick={loadSampleTranscript} disabled={loading}>
                  {loading ? "Loading..." : "Load Sample Transcript"}
                </button>
              </div>

              <section className="panel nested-panel">
                <div className="panel-title">Sample Preview</div>
                <pre className="demo-preview">{DEMO_TRANSCRIPTS[demoCase]}</pre>
              </section>
            </section>
          ) : null}
        </main>
      </div>
    </div>
  );
}

export default App;
