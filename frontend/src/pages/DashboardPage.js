import { useEffect, useMemo, useState } from "react";

import SummaryCard from "../components/SummaryCard";
import TimelineChart from "../components/TimelineChart";
import SignalHeatmap from "../components/SignalHeatmap";
import {
  analyzeTranscript,
  compareAnalyses,
  getAnalysisById,
  getHistory,
  saveAnalysis,
  uploadTranscript,
} from "../api/client";

const SAMPLE = `CEO: We plan to expand operations globally.
CFO: Costs may rise due to supply chain issues.
Analyst: How will this impact margins?
CFO: We are monitoring cost structure carefully.`;

export default function DashboardPage({ token, onLogout }) {
  const [activeTab, setActiveTab] = useState("analyze");
  const [transcript, setTranscript] = useState(SAMPLE);
  const [file, setFile] = useState(null);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  const [compareA, setCompareA] = useState("");
  const [compareB, setCompareB] = useState("");
  const [compareResult, setCompareResult] = useState(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState("");

  const segments = result?.segments || [];

  const drivers = useMemo(() => {
    const growth = (result?.drivers?.growth_drivers || []).slice(0, 3);
    const risk = (result?.drivers?.risk_drivers || []).slice(0, 3);
    return { growth, risk };
  }, [result]);

  const signalDistribution = useMemo(() => {
    const order = ["EXPANSION", "GENERAL_UPDATE", "STRATEGIC_PROBING", "COST_PRESSURE"];
    const total = segments.length || 1;

    return order.map((intent) => {
      const count = segments.filter((segment) => segment.intent === intent).length;
      return { intent, count, percentage: Math.round((count / total) * 100) };
    });
  }, [segments]);

  const refreshHistory = async () => {
    setHistoryLoading(true);
    try {
      const response = await getHistory(token);
      setHistory(response.items || []);
    } catch (_err) {
      setHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  };

  useEffect(() => {
    refreshHistory();
  }, []);

  const handleAnalyze = async () => {
    if (!transcript.trim()) {
      setError("Please enter a transcript");
      return;
    }

    setLoading(true);
    setError("");
    try {
      const analysis = await analyzeTranscript(transcript);
      if (analysis?.error) {
        setError("Analysis failed. Try again.");
        setResult(null);
        return;
      }

      setResult(analysis);

      await saveAnalysis(token, transcript);
      await refreshHistory();
    } catch (_err) {
      setError("Analysis failed. Try again.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a .txt or .pdf file");
      return;
    }

    setLoading(true);
    setError("");
    try {
      const analysis = await uploadTranscript(token, file);
      if (analysis?.error) {
        setError("Analysis failed. Try again.");
        setResult(null);
        return;
      }
      setResult(analysis);
      setActiveTab("analyze");
    } catch (_err) {
      setError("Upload failed. Try again.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const openHistoryItem = async (analysisId) => {
    setLoading(true);
    setError("");
    try {
      const detail = await getAnalysisById(token, analysisId);
      const fresh = await analyzeTranscript(detail.transcript || "");
      setTranscript(detail.transcript || transcript);
      setResult({
        ...fresh,
        signal: detail.signal,
        prediction: detail.prediction,
        confidence: detail.confidence,
        volatility: detail.volatility,
        score: detail.score,
        drivers: detail.drivers,
      });
      setActiveTab("analyze");
    } catch (_err) {
      setError("Could not load history item.");
    } finally {
      setLoading(false);
    }
  };

  const runComparison = async () => {
    if (!compareA || !compareB) {
      setCompareError("Select two analyses to compare.");
      return;
    }
    setCompareLoading(true);
    setCompareError("");
    try {
      const compared = await compareAnalyses(token, Number(compareA), Number(compareB));
      setCompareResult(compared);
    } catch (_err) {
      setCompareResult(null);
      setCompareError("Comparison failed. Try again.");
    } finally {
      setCompareLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      <header className="border-b border-slate-800 bg-slate-900/70 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
          <h1 className="text-sm font-semibold tracking-wide text-slate-100">
            Financial Pragmatic AI
          </h1>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setActiveTab("analyze")}
              className={`rounded px-3 py-1 text-xs ${
                activeTab === "analyze" ? "bg-sky-600 text-white" : "bg-slate-800 text-slate-300"
              }`}
            >
              Analyze
            </button>
            <button
              onClick={() => setActiveTab("history")}
              className={`rounded px-3 py-1 text-xs ${
                activeTab === "history" ? "bg-sky-600 text-white" : "bg-slate-800 text-slate-300"
              }`}
            >
              History
            </button>
            <button
              onClick={() => setActiveTab("compare")}
              className={`rounded px-3 py-1 text-xs ${
                activeTab === "compare" ? "bg-sky-600 text-white" : "bg-slate-800 text-slate-300"
              }`}
            >
              Compare
            </button>
            <button
              onClick={onLogout}
              className="rounded bg-rose-700 px-3 py-1 text-xs font-medium text-white"
            >
              Logout
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-4 px-4 py-4 lg:grid-cols-12">
        <aside className="space-y-4 lg:col-span-4">
          <section className="rounded-lg border border-slate-700 bg-slate-900 p-4">
            <h2 className="mb-3 text-xs uppercase tracking-wider text-slate-400">Transcript Input</h2>
            <textarea
              value={transcript}
              onChange={(event) => setTranscript(event.target.value)}
              rows={14}
              className="w-full rounded border border-slate-700 bg-slate-950 p-3 text-xs outline-none ring-0"
              disabled={loading}
            />
            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              <button
                onClick={handleAnalyze}
                disabled={loading}
                className="rounded bg-sky-600 px-3 py-2 text-xs font-semibold text-white disabled:opacity-50"
              >
                {loading ? "Analyzing..." : "Analyze & Save"}
              </button>
              <button
                onClick={refreshHistory}
                disabled={historyLoading}
                className="rounded bg-slate-800 px-3 py-2 text-xs font-semibold text-slate-200 disabled:opacity-50"
              >
                {historyLoading ? "Refreshing..." : "Refresh History"}
              </button>
            </div>

            <div className="mt-3 space-y-2 rounded border border-slate-800 bg-slate-950 p-3">
              <p className="text-[11px] uppercase tracking-wide text-slate-500">Upload Transcript</p>
              <input
                type="file"
                accept=".txt,.pdf,text/plain,application/pdf"
                onChange={(event) => setFile(event.target.files?.[0] || null)}
                disabled={loading}
                className="w-full text-xs text-slate-300"
              />
              <button
                onClick={handleUpload}
                disabled={loading || !file}
                className="rounded bg-slate-700 px-3 py-2 text-xs font-semibold text-slate-100 disabled:opacity-50"
              >
                {loading ? "Uploading..." : "Upload & Analyze"}
              </button>
            </div>

            {error ? <p className="mt-3 text-xs text-rose-400">{error}</p> : null}
          </section>

          <section className="rounded-lg border border-slate-700 bg-slate-900 p-4">
            <h2 className="mb-3 text-xs uppercase tracking-wider text-slate-400">History</h2>
            <div className="max-h-80 space-y-2 overflow-y-auto pr-1">
              {history.map((item) => (
                <button
                  key={item.id}
                  onClick={() => openHistoryItem(item.id)}
                  className="w-full rounded border border-slate-700 bg-slate-950 p-2 text-left text-xs hover:border-sky-500"
                >
                  <p className="font-semibold text-slate-200">
                    {item.signal?.toUpperCase()} • {item.prediction?.toUpperCase()} • {item.score}
                  </p>
                  <p className="mt-1 text-slate-400">{item.transcript_preview}</p>
                </button>
              ))}
              {!history.length && !historyLoading ? (
                <p className="text-xs text-slate-500">No saved analyses yet.</p>
              ) : null}
            </div>
          </section>
        </aside>

        <section className="space-y-4 lg:col-span-8">
          {activeTab === "analyze" ? (
            <>
              <SummaryCard
                signal={result?.signal}
                score={result?.score}
                prediction={result?.prediction}
                confidence={result?.confidence}
                volatility={result?.volatility}
                keyDriver={drivers.growth[0]}
                keyConcern={drivers.risk[0]}
              />

              <section className="rounded-lg border border-slate-700 bg-slate-900 p-4">
                <h2 className="mb-3 text-xs uppercase tracking-wider text-slate-400">Timeline</h2>
                <div className="h-72 w-full">
                  <TimelineChart segments={segments} />
                </div>
              </section>

              <section className="rounded-lg border border-slate-700 bg-slate-900 p-4">
                <h2 className="mb-3 text-xs uppercase tracking-wider text-slate-400">Signal Heatmap</h2>
                <SignalHeatmap segments={segments} />
              </section>

              <section className="rounded-lg border border-slate-700 bg-slate-900 p-4">
                <h2 className="mb-3 text-xs uppercase tracking-wider text-slate-400">Signal Distribution</h2>
                <div className="space-y-2">
                  {signalDistribution.map((row) => (
                    <div key={row.intent} className="grid grid-cols-[150px_1fr_50px] items-center gap-2 text-xs">
                      <span className="text-slate-400">{row.intent}</span>
                      <div className="h-1.5 rounded bg-slate-800">
                        <div className="h-full rounded bg-sky-500" style={{ width: `${row.percentage}%` }} />
                      </div>
                      <span className="text-right text-slate-400">{row.percentage}%</span>
                    </div>
                  ))}
                </div>
              </section>

              <section className="grid gap-4 md:grid-cols-2">
                <div className="rounded-lg border border-emerald-500/40 bg-slate-900 p-4">
                  <h3 className="mb-2 text-xs uppercase tracking-wider text-emerald-300">Growth Drivers</h3>
                  <ul className="max-h-48 list-disc space-y-1 overflow-y-auto pl-4 text-xs text-slate-300">
                    {(drivers.growth.length ? drivers.growth : ["No growth driver detected"]).map(
                      (driver, index) => (
                        <li key={`g-${index}`} title={driver}>
                          {driver}
                        </li>
                      )
                    )}
                  </ul>
                </div>

                <div className="rounded-lg border border-rose-500/40 bg-slate-900 p-4">
                  <h3 className="mb-2 text-xs uppercase tracking-wider text-rose-300">Risk Drivers</h3>
                  <ul className="max-h-48 list-disc space-y-1 overflow-y-auto pl-4 text-xs text-slate-300">
                    {(drivers.risk.length ? drivers.risk : ["No risk concern detected"]).map(
                      (driver, index) => (
                        <li key={`r-${index}`} title={driver}>
                          {driver}
                        </li>
                      )
                    )}
                  </ul>
                </div>
              </section>

              <section className="rounded-lg border border-slate-700 bg-slate-900 p-4">
                <h2 className="mb-3 text-xs uppercase tracking-wider text-slate-400">Segments</h2>
                <div className="max-h-52 space-y-1 overflow-y-auto text-xs">
                  {segments.map((segment, index) => (
                    <div
                      key={`${segment.speaker}-${index}`}
                      className="rounded border border-slate-800 bg-slate-950 px-2 py-1"
                      title={segment.text}
                    >
                      [{segment.speaker}] {segment.intent} :: {segment.text}
                    </div>
                  ))}
                </div>
              </section>
            </>
          ) : null}

          {activeTab === "history" ? (
            <section className="rounded-lg border border-slate-700 bg-slate-900 p-4">
              <h2 className="mb-3 text-xs uppercase tracking-wider text-slate-400">Transcript History</h2>
              <div className="space-y-2 text-xs">
                {history.map((item) => (
                  <div key={`history-${item.id}`} className="rounded border border-slate-700 bg-slate-950 p-2">
                    <p className="font-semibold text-slate-200">
                      #{item.id} • {item.signal?.toUpperCase()} • {item.prediction?.toUpperCase()} • {item.score}
                    </p>
                    <p className="mt-1 text-slate-400">{item.transcript_preview}</p>
                    <p className="mt-1 text-[11px] text-slate-500">{item.created_at}</p>
                  </div>
                ))}
              </div>
            </section>
          ) : null}

          {activeTab === "compare" ? (
            <section className="space-y-4 rounded-lg border border-slate-700 bg-slate-900 p-4">
              <h2 className="text-xs uppercase tracking-wider text-slate-400">Compare Saved Analyses</h2>
              <div className="grid gap-2 md:grid-cols-2">
                <select
                  value={compareA}
                  onChange={(event) => setCompareA(event.target.value)}
                  className="rounded border border-slate-700 bg-slate-950 p-2 text-xs"
                >
                  <option value="">Select analysis A</option>
                  {history.map((item) => (
                    <option key={`a-${item.id}`} value={item.id}>
                      #{item.id} • {item.signal} • {item.score}
                    </option>
                  ))}
                </select>
                <select
                  value={compareB}
                  onChange={(event) => setCompareB(event.target.value)}
                  className="rounded border border-slate-700 bg-slate-950 p-2 text-xs"
                >
                  <option value="">Select analysis B</option>
                  {history.map((item) => (
                    <option key={`b-${item.id}`} value={item.id}>
                      #{item.id} • {item.signal} • {item.score}
                    </option>
                  ))}
                </select>
              </div>

              <button
                onClick={runComparison}
                disabled={compareLoading}
                className="rounded bg-sky-600 px-3 py-2 text-xs font-semibold text-white disabled:opacity-50"
              >
                {compareLoading ? "Comparing..." : "Compare"}
              </button>

              {compareError ? <p className="text-xs text-rose-400">{compareError}</p> : null}

              {compareResult ? (
                <div className="space-y-2 rounded border border-slate-700 bg-slate-950 p-3 text-xs">
                  <p>
                    Signal shift: {compareResult.signal_difference?.from} → {compareResult.signal_difference?.to}
                  </p>
                  <p>Risk delta: {compareResult.risk_delta}%</p>
                  <p>Confidence delta: {compareResult.confidence_delta}%</p>
                  <p>Trend: {compareResult.trend}</p>
                  <p className="text-slate-400">{compareResult.comparison}</p>
                </div>
              ) : null}
            </section>
          ) : null}
        </section>
      </main>
    </div>
  );
}
