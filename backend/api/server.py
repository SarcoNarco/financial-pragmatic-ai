import logging
import os
import threading
from time import perf_counter

os.environ.setdefault("HF_HOME", "/tmp/hf_cache")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import CompareRequest, TranscriptRequest
from financial_pragmatic_ai.analysis.earnings_call_analyzer import EarningsCallAnalyzer
from financial_pragmatic_ai.analysis.segment_sampler import select_representative_segments
from financial_pragmatic_ai.analysis.financial_signal_engine import (
    compute_confidence,
    compute_intent_distribution,
    compute_risk_score,
    compute_signal_distribution,
    compute_signal_std,
    detect_volatility,
    derive_signal,
    generate_insight,
)
from financial_pragmatic_ai.analysis.insight_engine import extract_key_drivers
from financial_pragmatic_ai.analysis.market_predictor import predict_market_outlook

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

APP_NAME = "financial-pragmatic-ai-api"
API_ENTRYPOINT = "backend/api/server.py"
MODEL_NAME = "SarcoNarco/finbert_intent_v3"
DEFAULT_MAX_DIRECT_TRANSCRIPT_CHARS = 20_000
DEFAULT_MAX_FULL_TRANSCRIPT_CHARS = 250_000
DEFAULT_FULL_TRANSCRIPT_SEGMENT_BUDGET = 32


def _positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Invalid %s=%r; using default=%s", name, raw_value, default)
        return default
    if value <= 0:
        logger.warning("Invalid %s=%r; using default=%s", name, raw_value, default)
        return default
    return value


MAX_DIRECT_TRANSCRIPT_CHARS = _positive_int_env(
    "MAX_DIRECT_TRANSCRIPT_CHARS",
    _positive_int_env("MAX_TRANSCRIPT_CHARS", DEFAULT_MAX_DIRECT_TRANSCRIPT_CHARS),
)
MAX_FULL_TRANSCRIPT_CHARS = _positive_int_env(
    "MAX_FULL_TRANSCRIPT_CHARS",
    DEFAULT_MAX_FULL_TRANSCRIPT_CHARS,
)
FULL_TRANSCRIPT_SEGMENT_BUDGET = _positive_int_env(
    "FULL_TRANSCRIPT_SEGMENT_BUDGET",
    DEFAULT_FULL_TRANSCRIPT_SEGMENT_BUDGET,
)

app = FastAPI(title="Financial Pragmatic AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = None
_analyzer_lock = threading.Lock()


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": APP_NAME,
    }


@app.get("/version")
def version():
    return {
        "app": APP_NAME,
        "api_entrypoint": API_ENTRYPOINT,
        "model": MODEL_NAME,
        "status": "portfolio-hardening",
    }


def _get_analyzer() -> EarningsCallAnalyzer:
    global analyzer
    if analyzer is not None:
        return analyzer

    with _analyzer_lock:
        if analyzer is not None:
            return analyzer

        started = perf_counter()
        initialized = False
        try:
            analyzer = EarningsCallAnalyzer()
            initialized = True
            return analyzer
        finally:
            logger.info(
                "timing stage=analyzer_lazy_init duration_ms=%.1f success=%s",
                (perf_counter() - started) * 1000,
                initialized,
            )


def _validate_transcript(transcript: str) -> None:
    transcript_chars = len(transcript)
    if transcript_chars > MAX_FULL_TRANSCRIPT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=(
                "Transcript exceeds the hosted demo's absolute "
                f"{MAX_FULL_TRANSCRIPT_CHARS} character safety limit "
                f"({transcript_chars} received)."
            ),
        )


def _run_analysis(transcript: str):
    _validate_transcript(transcript)
    service_analyzer = _get_analyzer()
    transcript_chars = len(transcript)
    sampled = transcript_chars > MAX_DIRECT_TRANSCRIPT_CHARS

    if sampled:
        full_segments = service_analyzer.segment_transcript(transcript)
        selected_segments = select_representative_segments(
            full_segments,
            FULL_TRANSCRIPT_SEGMENT_BUDGET,
        )
        result = service_analyzer.analyze_segments(
            selected_segments,
            include_details=False,
        )
        segments_total = len(full_segments)
        segments_analyzed = len(selected_segments)
        actually_sampled = segments_analyzed < segments_total
        analysis_mode = "sampled_full_transcript"
        sampling_note = (
            "Hosted demo analyzed a representative subset of the full transcript "
            "for performance."
            if actually_sampled
            else None
        )
    else:
        result = service_analyzer.analyze(transcript, include_details=False)
        segments_total = len(result["segments"])
        segments_analyzed = segments_total
        actually_sampled = False
        analysis_mode = "standard"
        sampling_note = None

    segments = result["segments"]
    fallback_used = bool(result.get("fallback_used", False))
    if len(segments) == 0:
        return {"error": "Could not parse transcript"}

    # --- Raw driver extraction removed ---
    # Path B (insight_engine.extract_key_drivers) is the single source of truth.

    scoring_started = perf_counter()
    score = compute_risk_score(segments)
    signal = derive_signal(score)
    confidence = compute_confidence(segments)
    volatility = detect_volatility(segments)
    volatility_std = round(compute_signal_std(segments), 4)
    intent_distribution = compute_intent_distribution(segments)
    signal_distribution = compute_signal_distribution(segments)
    logger.info(
        "timing stage=scoring_signal duration_ms=%.1f segments=%s",
        (perf_counter() - scoring_started) * 1000,
        len(segments),
    )
    logger.debug("intent_distribution=%s", intent_distribution)
    logger.debug("signal_distribution=%s", signal_distribution)

    market_insight_started = perf_counter()
    market = predict_market_outlook(
        signal=signal,
        risk_score=score,
        volatility=volatility,
        intent_distribution=intent_distribution,
    )
    insight = generate_insight(score, segments)
    logger.info(
        "timing stage=market_prediction_insight duration_ms=%.1f segments=%s",
        (perf_counter() - market_insight_started) * 1000,
        len(segments),
    )

    drivers_started = perf_counter()
    drivers = extract_key_drivers(segments)
    logger.info(
        "timing stage=driver_extraction duration_ms=%.1f segments=%s",
        (perf_counter() - drivers_started) * 1000,
        len(segments),
    )

    _INTENT_VALUE = {
        "EXPANSION": 1,
        "COST_PRESSURE": -1,
        "GENERAL_UPDATE": 0,
        "STRATEGIC_PROBING": 0,
    }
    timeline_started = perf_counter()
    timeline = [
        {
            "step": seg.get("source_index", i),
            "value": _INTENT_VALUE.get(seg["intent"], 0),
            "intent": seg["intent"],
            "label": seg["text"][:60],
        }
        for i, seg in enumerate(segments)
    ]
    logger.info(
        "timing stage=timeline_generation duration_ms=%.1f segments=%s",
        (perf_counter() - timeline_started) * 1000,
        len(segments),
    )

    return {
        "score": score,
        "signal": signal,
        "prediction": market["prediction"],
        "prediction_explanation": market["explanation"],
        "confidence": confidence,
        "volatility": volatility,
        "volatility_std": volatility_std,
        "intent_distribution": intent_distribution,
        "insight": insight,
        "segments": segments,
        "growth_drivers": drivers["growth_drivers"],
        "risk_drivers": drivers["risk_drivers"],
        "drivers": drivers,
        "timeline": timeline,
        "fallback_used": fallback_used,
        "analysis_mode": analysis_mode,
        "sampled": actually_sampled,
        "segments_total": segments_total,
        "segments_analyzed": segments_analyzed,
        "segment_budget": FULL_TRANSCRIPT_SEGMENT_BUDGET if sampled else None,
        "sampling_note": sampling_note,
        "transcript_chars": transcript_chars,
    }


@app.post("/analyze")
def analyze_transcript(request: TranscriptRequest):
    started = perf_counter()
    outcome = "ok"
    try:
        return _run_analysis(request.transcript)
    except Exception:
        outcome = "error"
        raise
    finally:
        logger.info(
            "timing stage=request_total endpoint=/analyze duration_ms=%.1f "
            "transcript_chars=%s outcome=%s",
            (perf_counter() - started) * 1000,
            len(request.transcript),
            outcome,
        )


@app.post("/upload")
async def upload_transcript(
    file: UploadFile = File(...),
):
    content = await file.read()
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        import io
        import pdfplumber

        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                pages = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text)

            text = "\n".join(pages)
        except Exception as exc:
            return {"error": f"PDF parsing failed: {str(exc)}"}
    elif filename.endswith(".txt"):
        try:
            text = content.decode("utf-8")
        except Exception:
            return {"error": "TXT file must be UTF-8 encoded"}
    else:
        return {"error": "Only .txt and .pdf files are supported"}

    text = text.replace("\n\n", "\n")
    text = text.strip()
    return _run_analysis(text)


@app.post("/compare")
async def compare_transcripts(request: CompareRequest):
    if not request.transcript_1 or not request.transcript_2:
        raise HTTPException(
            status_code=400,
            detail="Provide transcript_1 and transcript_2",
        )

    first = _run_analysis(request.transcript_1)
    if "error" in first:
        return first
    second = _run_analysis(request.transcript_2)
    if "error" in second:
        return second

    risk_delta = round(float(second["score"]) - float(first["score"]), 2)
    confidence_delta = round(float(second["confidence"]) - float(first["confidence"]), 2)
    signal_changed = first["signal"] != second["signal"]

    if risk_delta > 0:
        comparison_text = f"Risk increased by {abs(risk_delta):.2f}% compared to previous call."
        trend = "UP"
    elif risk_delta < 0:
        comparison_text = f"Risk decreased by {abs(risk_delta):.2f}% compared to previous call."
        trend = "DOWN"
    else:
        comparison_text = "Risk is unchanged compared to previous call."
        trend = "FLAT"

    return {
        "transcript_1": first,
        "transcript_2": second,
        "signal_difference": {
            "from": first["signal"],
            "to": second["signal"],
            "changed": signal_changed,
        },
        "risk_delta": risk_delta,
        "confidence_delta": confidence_delta,
        "trend": trend,
        "comparison": comparison_text,
    }
