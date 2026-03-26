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
  if (["risk", "down", "high", "volatile"].includes(normalized)) return "risk";
  return "neutral";
}

function extractDrivers(analysis) {
  const segments = analysis?.segments || [];
  const growthFromSegments = segments
    .filter((segment) => segment.intent === "EXPANSION")
    .map((segment) => segment.text)
    .slice(0, 3);

  const riskFromSegments = segments
    .filter(
      (segment) =>
        segment.intent === "COST_PRESSURE" || segment.intent === "STRATEGIC_PROBING"
    )
    .map((segment) => segment.text)
    .slice(0, 3);

  return {
    growth: growthFromSegments.length
      ? growthFromSegments
      : (analysis?.drivers?.growth_drivers || []).slice(0, 3),
    risk: riskFromSegments.length
      ? riskFromSegments
      : (analysis?.drivers?.risk_drivers || []).slice(0, 3),
  };
}

function CompactCompareSummary({ title, analysis }) {
  const drivers = extractDrivers(analysis);
  return (
    <section className="panel compare-card">
      <div className="panel-title">{title}</div>
      <div className="compare-metrics">
        <div>
          <span className="metric-label">Signal:</span>{" "}
          <span className={getToneClass(analysis?.signal)}>
            {(analysis?.signal || "neutral").toUpperCase()}
          </span>
        </div>
        <div>
          <span className="metric-label">Risk:</span> {analysis?.score ?? "-"}
        </div>
        <div>
          <span className="metric-label">Prediction:</span>{" "}
          <span className={getToneClass(analysis?.prediction)}>
            {(analysis?.prediction || "NEUTRAL").toUpperCase()}
          </span>
        </div>
        <div>
          <span className="metric-label">Confidence:</span> {analysis?.confidence ?? "-"}%
        </div>
        <div>
          <span className="metric-label">Volatility:</span>{" "}
          <span className={getToneClass(analysis?.volatility)}>
            {(analysis?.volatility || "-").toUpperCase()}
          </span>
        </div>
      </div>
      <div className="compare-note">
        <span className="metric-label">Top Driver:</span>{" "}
        {drivers.growth[0] || "No growth driver detected"}
      </div>
      <div className="compare-note">
        <span className="metric-label">Top Concern:</span>{" "}
        {drivers.risk[0] || "No risk concern detected"}
      </div>
    </section>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState("analyze");
  const [transcript, setTranscript] = useState(SAMPLE);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [demoCase, setDemoCase] = useState("growth");

  const [compareTranscript1, setCompareTranscript1] = useState(DEMO_TRANSCRIPTS.growth);
  const [compareTranscript2, setCompareTranscript2] = useState(DEMO_TRANSCRIPTS.risk);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState("");
  const [compareResult, setCompareResult] = useState(null);

  const data = result;
  const analyzedSegments = data?.segments || [];

  const extractedDrivers = useMemo(() => extractDrivers(data), [data]);
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

  const compareTranscripts = async () => {
    if (!compareTranscript1.trim() || !compareTranscript2.trim()) {
      setCompareError("Please enter both transcripts");
      return;
    }

    setCompareLoading(true);
    setCompareError("");

    try {
      const response = await axios.post("http://127.0.0.1:8000/compare", {
        transcript_1: compareTranscript1,
        transcript_2: compareTranscript2,
      });

      if (response.data?.error) {
        setCompareResult(null);
        setCompareError("Comparison failed. Try again.");
        return;
      }

      setCompareResult(response.data);
    } catch (_err) {
      setCompareResult(null);
      setCompareError("Comparison failed. Try again.");
    } finally {
      setCompareLoading(false);
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
              disabled={loading || compareLoading}
            >
              Analyze
            </button>
            <button
              className={activeTab === "insights" ? "tab active" : "tab"}
              onClick={() => setActiveTab("insights")}
              disabled={loading || compareLoading}
            >
              Insights
            </button>
            <button
              className={activeTab === "demo" ? "tab active" : "tab"}
              onClick={() => setActiveTab("demo")}
              disabled={loading || compareLoading}
            >
              Demo
            </button>
            <button
              className={activeTab === "compare" ? "tab active" : "tab"}
              onClick={() => setActiveTab("compare")}
              disabled={loading || compareLoading}
            >
              Compare
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
                  predictionExplanation={data?.prediction_explanation}
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
                    Confidence is based on consistency of dominant conversation signals across
                    segments. Higher agreement means stronger confidence in the final outlook.
                  </p>
                </section>

                <section className="panel nested-panel">
                  <div className="panel-title">Volatility</div>
                  <p className="insight-line">
                    Volatility is derived from variation in signal trajectory over time. Higher
                    variance indicates unstable communication patterns.
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

          {activeTab === "compare" ? (
            <section className="panel compare-tab">
              <div className="panel-title">Multi-Transcript Comparison</div>

              <div className="compare-inputs">
                <section className="panel nested-panel">
                  <div className="panel-title">Transcript A</div>
                  <textarea
                    value={compareTranscript1}
                    onChange={(event) => setCompareTranscript1(event.target.value)}
                    disabled={compareLoading}
                  />
                </section>

                <section className="panel nested-panel">
                  <div className="panel-title">Transcript B</div>
                  <textarea
                    value={compareTranscript2}
                    onChange={(event) => setCompareTranscript2(event.target.value)}
                    disabled={compareLoading}
                  />
                </section>
              </div>

              <button onClick={compareTranscripts} disabled={compareLoading}>
                {compareLoading ? "Comparing..." : "Compare"}
              </button>

              {compareLoading ? (
                <div className="loading-row">
                  <span className="spinner" />
                  <span>Comparing transcripts...</span>
                </div>
              ) : null}

              {compareError ? <p className="error">{compareError}</p> : null}

              {compareResult ? (
                <>
                  <div className="compare-summary-grid">
                    <CompactCompareSummary
                      title="Transcript A Summary"
                      analysis={compareResult.transcript_1}
                    />
                    <CompactCompareSummary
                      title="Transcript B Summary"
                      analysis={compareResult.transcript_2}
                    />
                  </div>

                  <section className="panel nested-panel compare-delta-panel">
                    <div className="panel-title">Delta Metrics</div>
                    <div className="compare-metrics">
                      <div>
                        <span className="metric-label">Signal Shift:</span>{" "}
                        {(compareResult.signal_difference?.from || "neutral").toUpperCase()} →{" "}
                        {(compareResult.signal_difference?.to || "neutral").toUpperCase()}
                      </div>
                      <div>
                        <span className="metric-label">Risk Delta:</span>{" "}
                        <span className={compareResult.risk_delta > 0 ? "risk" : compareResult.risk_delta < 0 ? "growth" : "neutral"}>
                          {compareResult.risk_delta > 0 ? "+" : ""}
                          {compareResult.risk_delta}%
                        </span>
                      </div>
                      <div>
                        <span className="metric-label">Confidence Delta:</span>{" "}
                        <span className={compareResult.confidence_delta >= 0 ? "growth" : "risk"}>
                          {compareResult.confidence_delta > 0 ? "+" : ""}
                          {compareResult.confidence_delta}%
                        </span>
                      </div>
                      <div>
                        <span className="metric-label">Trend:</span>{" "}
                        <span className={getToneClass(compareResult.trend === "UP" ? "risk" : compareResult.trend === "DOWN" ? "growth" : "neutral")}>
                          {compareResult.trend}
                        </span>
                      </div>
                    </div>
                    <p className="insight-line">{compareResult.comparison}</p>
                  </section>
                </>
              ) : null}
            </section>
          ) : null}
        </main>
      </div>
    </div>
  );
}

export default App;
