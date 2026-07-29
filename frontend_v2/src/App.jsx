import { useEffect, useRef, useState } from "react";
import { Clock3, FileText, Loader2, Upload } from "lucide-react";
import { supabase } from "./supabaseClient";
import Auth from "./components/Auth";
import Sidebar from "./components/Sidebar";
import Navbar from "./components/Navbar";
import Tabs from "./components/Tabs";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"
).replace(/\/+$/, "");
const API_URL = `${API_BASE_URL}/analyze`;
const HEALTH_URL = `${API_BASE_URL}/health`;
const ANALYZE_TIMEOUT_MS = 90_000;
const SAMPLE_TRANSCRIPT =
  "CEO: Revenue grew 12 percent this quarter as enterprise demand improved. " +
  "CFO: Margins improved due to better operating discipline, although cloud costs remain a pressure point. " +
  "Analyst: What drove the growth and how sustainable is it?";

function getResponseError(status) {
  if (status === 413) {
    return "This transcript is too long for the demo. Please shorten it and try again.";
  }
  if ([408, 502, 503, 504].includes(status)) {
    return "Backend is reachable but analysis took too long. Please try a shorter transcript.";
  }
  return "Analysis failed on the backend. Please try again.";
}

export default function App() {
  const [transcript, setTranscript] = useState("");
  const [loading, setLoading] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [analysisError, setAnalysisError] = useState("");
  const [saveWarning, setSaveWarning] = useState("");
  const [backendStatus, setBackendStatus] = useState("warming-up");
  const [result, setResult] = useState(null);
  const [uploadedFileName, setUploadedFileName] = useState("");
  const [session, setSession] = useState(null);
  const [history, setHistory] = useState([]);
  const [compareA, setCompareA] = useState(null);
  const [compareB, setCompareB] = useState(null);
  const [compareMode, setCompareMode] = useState(false);
  const [activeTab, setActiveTab] = useState("overview");
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);
  const [isFromHistory, setIsFromHistory] = useState(false);
  const fileInputRef = useRef(null);
  const statusCheckedForUserRef = useRef(null);

  useEffect(() => {
    if (compareA && compareB) {
      setActiveTab("compare");
    }
  }, [compareA, compareB]);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      if (session) fetchHistory(session.user.id);
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      setSession(nextSession);
      if (nextSession) fetchHistory(nextSession.user.id);
      else {
        setHistory([]);
        statusCheckedForUserRef.current = null;
      }
    });

    return () => subscription.unsubscribe();
  }, []);

  useEffect(() => {
    const userId = session?.user?.id;
    if (!userId || statusCheckedForUserRef.current === userId) return;
    statusCheckedForUserRef.current = userId;

    const checkBackendStatus = async () => {
      setBackendStatus("warming-up");
      const controller = new AbortController();
      const timeoutId = window.setTimeout(() => controller.abort(), 8_000);

      try {
        const response = await fetch(HEALTH_URL, {
          signal: controller.signal,
        });
        const data = response.ok ? await response.json() : null;
        setBackendStatus(data?.status === "ok" ? "online" : "unavailable");
      } catch {
        setBackendStatus("unavailable");
      } finally {
        window.clearTimeout(timeoutId);
      }
    };

    checkBackendStatus();
  }, [session]);

  useEffect(() => {
    if (!loading) {
      setElapsedSeconds(0);
      return undefined;
    }

    const startedAt = Date.now();
    const intervalId = window.setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
    }, 1_000);

    return () => window.clearInterval(intervalId);
  }, [loading]);

  const fetchHistory = async (userId) => {
    const { data, error } = await supabase
      .from("analyses")
      .select("*")
      .eq("user_id", userId)
      .order("created_at", { ascending: false });

    if (!error && data) {
      setHistory(data);
      return true;
    }
    return false;
  };

  const handleHistoryClick = (item) => {
    setSelectedAnalysis(item);
    setIsFromHistory(true);
    setTranscript(item.transcript);
    setAnalysisError("");
    setSaveWarning("");
    setResult({
      signal: item.signal,
      score: item.score,
      distribution: item.distribution,
      drivers: {
        growth: item.growth_drivers,
        risk: item.risk_drivers,
      },
      timeline: item.timeline || [],
    });
  };

  const signal = (result?.signal || "neutral").toLowerCase();

  const scoreText =
    typeof result?.score === "number" ? result.score.toFixed(2) : "--";
  const confidenceText =
    typeof result?.confidence === "number"
      ? `${(result.confidence <= 1 ? result.confidence * 100 : result.confidence).toFixed(1)}%`
      : "85.0%";

  const heroGlowClass =
    signal === "growth"
      ? "shadow-[0_0_20px_rgba(0,255,156,0.2)]"
      : signal === "risk"
        ? "shadow-[0_0_20px_rgba(255,77,79,0.2)]"
        : "shadow-[0_0_20px_rgba(88,166,255,0.2)]";

  const saveAnalysis = async (mapped) => {
    if (!session?.user) return;

    let error;
    if (isFromHistory && selectedAnalysis?.id) {
      ({ error } = await supabase
        .from("analyses")
        .update({
          signal: mapped.signal,
          score: mapped.score,
          distribution: mapped.distribution,
          growth_drivers: mapped.drivers.growth,
          risk_drivers: mapped.drivers.risk,
          timeline: mapped.timeline,
        })
        .eq("id", selectedAnalysis.id));
    } else {
      ({ error } = await supabase.from("analyses").insert({
        user_id: session.user.id,
        transcript,
        signal: mapped.signal,
        score: mapped.score,
        distribution: mapped.distribution,
        growth_drivers: mapped.drivers.growth,
        risk_drivers: mapped.drivers.risk,
        timeline: mapped.timeline,
      }));
    }

    if (error) {
      console.error(
        "Supabase analysis save failed:",
        error.message || "Unknown error",
      );
      setSaveWarning("Analysis completed, but saving to history failed.");
      return;
    }

    setIsFromHistory(false);
    setSelectedAnalysis(null);
    const refreshed = await fetchHistory(session.user.id);
    if (!refreshed) {
      setSaveWarning("Analysis saved, but history could not refresh.");
    }
  };

  const handleAnalyze = async () => {
    if (loading) return;
    if (!transcript.trim()) {
      setAnalysisError("Enter a transcript or use the sample transcript before analyzing.");
      return;
    }

    setLoading(true);
    setAnalysisError("");
    setSaveWarning("");

    const controller = new AbortController();
    const timeoutId = window.setTimeout(
      () => controller.abort(),
      ANALYZE_TIMEOUT_MS,
    );

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(getResponseError(response.status));
      }

      const data = await response.json();
      if (data?.error) {
        throw new Error("The backend could not analyze this transcript. Please try a shorter sample.");
      }

      const intentDist = data.intent_distribution || {};
      const getIntentCount = (intent) => {
        const value = Number(intentDist[intent]);
        return Number.isFinite(value) && value > 0 ? value : 0;
      };
      const growthCount = getIntentCount("EXPANSION");
      const riskCount = getIntentCount("COST_PRESSURE");
      const neutralCount =
        getIntentCount("GENERAL_UPDATE") +
        getIntentCount("STRATEGIC_PROBING");
      const total = growthCount + riskCount + neutralCount || 1;
      const mapped = {
        signal: data.final_signal || data.signal,
        score: data.score,
        confidence: data.confidence || 0.8,
        distribution: {
          growth: growthCount / total,
          risk: riskCount / total,
          neutral: neutralCount / total,
        },
        drivers: {
          growth: data.growth_drivers || [],
          risk: data.risk_drivers || [],
        },
        timeline: data.timeline || [],
      };

      setResult(mapped);
      setBackendStatus("online");
      await saveAnalysis(mapped);
    } catch (error) {
      setResult(null);
      if (error?.name === "AbortError") {
        setAnalysisError(
          "Backend is reachable but analysis took too long. Please try a shorter transcript.",
        );
      } else if (error instanceof TypeError) {
        setBackendStatus("unavailable");
        setAnalysisError("Could not reach the backend. Check deployment status.");
      } else {
        setAnalysisError(
          error?.message || "Analysis failed. Please try again.",
        );
      }
    } finally {
      window.clearTimeout(timeoutId);
      setLoading(false);
    }
  };

  const handleUseSample = () => {
    if (loading) return;
    setTranscript(SAMPLE_TRANSCRIPT);
    setUploadedFileName("");
    setAnalysisError("");
    setSaveWarning("");
    setIsFromHistory(false);
    setSelectedAnalysis(null);
  };

  const handleUploadClick = () => {
    if (!loading) fileInputRef.current?.click();
  };

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      const content = typeof reader.result === "string" ? reader.result : "";
      setTranscript(content);
      setUploadedFileName(file.name);
      setAnalysisError("");
      setSaveWarning("");
      setIsFromHistory(false);
      setSelectedAnalysis(null);
    };
    reader.readAsText(file);
  };

  if (!session) return <Auth />;

  return (
    <div className="flex h-screen overflow-hidden bg-[#0d1117] text-[#c9d1d9]">
      <Sidebar
        history={history}
        onHistoryClick={handleHistoryClick}
        userEmail={session.user.email}
        compareMode={compareMode}
        compareA={compareA}
        setCompareA={setCompareA}
        compareB={compareB}
        setCompareB={setCompareB}
      />

      <div className="flex-1 flex flex-col overflow-hidden">
        <Navbar
          userEmail={session.user.email}
          backendStatus={backendStatus}
        />

        <div className="p-4 flex-1 overflow-y-auto">
          <div className="bg-[rgba(22,27,34,0.6)] backdrop-blur-xl border border-[#30363d] p-4 rounded-lg mb-4 transition-all duration-300">
            <input
              ref={fileInputRef}
              type="file"
              accept=".txt"
              className="hidden"
              onChange={handleFileChange}
            />
            <textarea
              className="w-full h-32 bg-[#0d1117] border border-[#30363d] focus:border-blue-500 p-2 rounded transition-all duration-300 outline-none resize-none disabled:opacity-70"
              value={transcript}
              disabled={loading}
              onChange={(event) => {
                setTranscript(event.target.value);
                setAnalysisError("");
                setSaveWarning("");
                setIsFromHistory(false);
                setSelectedAnalysis(null);
              }}
              placeholder="Enter financial transcript..."
            />

            <div className="mt-3 flex flex-wrap items-center gap-2">
              <button
                className="min-w-[138px] bg-gradient-to-r from-blue-500 to-blue-700 hover:scale-[1.02] active:scale-95 transition-all duration-300 px-5 py-2 rounded font-semibold disabled:opacity-60 disabled:hover:scale-100 flex items-center justify-center gap-2"
                onClick={handleAnalyze}
                disabled={loading}
                type="button"
              >
                {loading ? (
                  <>
                    <Loader2 className="animate-spin" size={16} />
                    Analyzing...
                  </>
                ) : (
                  "Analyze"
                )}
              </button>
              <button
                className="bg-[#30363d] px-4 py-2 rounded hover:bg-[#444c56] transition-all duration-300 text-sm disabled:opacity-60 flex items-center gap-2"
                onClick={handleUseSample}
                disabled={loading}
                type="button"
              >
                <FileText size={15} />
                Use sample transcript
              </button>
              <button
                className="bg-[#30363d] px-4 py-2 rounded hover:bg-[#444c56] transition-all duration-300 text-sm disabled:opacity-60 flex items-center gap-2"
                onClick={handleUploadClick}
                disabled={loading}
                type="button"
              >
                <Upload size={15} />
                Upload file
              </button>
            </div>

            <div className="mt-3 flex items-center gap-2 text-xs text-[#8b949e]">
              <Clock3 size={13} />
              <span>
                {loading
                  ? `Analyzing transcript... ${elapsedSeconds}s elapsed.`
                  : "The Railway demo backend may take up to 60 seconds for model inference."}
              </span>
            </div>

            {uploadedFileName ? (
              <div className="mt-2 text-xs text-[#8b949e]">
                Uploaded: {uploadedFileName}
              </div>
            ) : null}

            {analysisError ? (
              <div
                className="mt-3 p-3 rounded border border-red-500/30 bg-red-500/10 text-red-300 text-sm"
                role="alert"
              >
                {analysisError}
              </div>
            ) : null}

            {saveWarning ? (
              <div
                className="mt-3 p-3 rounded border border-yellow-500/30 bg-yellow-500/10 text-yellow-200 text-sm"
                role="status"
              >
                {saveWarning}
              </div>
            ) : null}
          </div>

          <div
            className={`bg-[rgba(22,27,34,0.6)] backdrop-blur-xl border border-[#30363d] p-6 rounded-lg mb-4 text-center transition-all duration-300 ${heroGlowClass}`}
          >
            <div
              className={`text-3xl font-bold transition-all duration-300 ${
                (result?.signal || "NEUTRAL").toUpperCase() === "GROWTH"
                  ? "text-green-400"
                  : (result?.signal || "NEUTRAL").toUpperCase() === "RISK"
                    ? "text-red-400"
                    : "text-blue-400"
              }`}
            >
              {(result?.signal || "NEUTRAL").toUpperCase()}
            </div>
            <div className="text-xl mt-1">{scoreText}</div>
            <div className="text-xs text-[#8b949e] mt-2 uppercase tracking-widest">
              Confidence:{" "}
              <span className="text-[#c9d1d9]">{confidenceText}</span>
            </div>
          </div>

          <div className="flex justify-between items-center mb-3 mt-4">
            <div className="text-sm font-medium text-[#8b949e] italic transition-opacity tracking-wide pl-2">
              {compareMode &&
                history.length < 2 &&
                "Run at least two analyses before comparing results"}
              {compareMode &&
                history.length >= 2 &&
                !compareA &&
                !compareB &&
                "Select 2 analyses to compare from the sidebar"}
              {compareMode &&
                history.length >= 2 &&
                ((compareA && !compareB) || (!compareA && compareB)) &&
                "Select one more analysis"}
            </div>
            <button
              onClick={() => setCompareMode(!compareMode)}
              className={`px-5 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider transition-all duration-300 ${
                compareMode
                  ? "bg-blue-500/20 text-blue-400 border border-blue-500/50 shadow-[0_0_12px_rgba(59,130,246,0.3)] scale-[1.02]"
                  : "bg-[#30363d] text-[#8b949e] hover:bg-[#444c56] border border-transparent"
              }`}
              type="button"
            >
              Compare Mode
            </button>
          </div>

          <Tabs
            active={activeTab}
            onTabChange={setActiveTab}
            result={result}
            compareA={compareA}
            compareB={compareB}
            historyCount={history.length}
            onClearCompare={() => {
              setCompareA(null);
              setCompareB(null);
            }}
          />
        </div>
      </div>
    </div>
  );
}
