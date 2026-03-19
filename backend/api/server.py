from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from financial_pragmatic_ai.analysis.earnings_call_analyzer import EarningsCallAnalyzer
from financial_pragmatic_ai.analysis.financial_signal_engine import (
    compute_risk_score,
    detect_conflict,
    generate_advanced_insight,
    normalize_score,
)
from financial_pragmatic_ai.analysis.signal_statistics import compute_signal_stats
from financial_pragmatic_ai.analysis.timeline_builder import build_timeline


app = FastAPI(title="Financial Pragmatic AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = EarningsCallAnalyzer()


class TranscriptRequest(BaseModel):
    transcript: str


@app.post("/analyze")
def analyze_transcript(request: TranscriptRequest):
    result = analyzer.analyze(request.transcript)
    segments = result["segments"]
    timeline = build_timeline(segments)
    stats = compute_signal_stats(segments)
    dominant_signal = result.get("dominant_signal")
    if dominant_signal is None:
        dominant_signal = result["aggregation"]["dominant_signal"]
    insight = generate_advanced_insight(segments)
    score = compute_risk_score(segments)
    risk_score = normalize_score(score)
    conflict = detect_conflict(segments)

    return {
        "segments": segments,
        "timeline": timeline,
        "signal_stats": stats,
        "signal": dominant_signal,
        "insight": insight,
        "risk_score": risk_score,
        "conflict": conflict,
    }


@app.post("/upload")
async def upload_transcript(file: UploadFile = File(...)):
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

    result = analyzer.analyze(text)
    segments = result["segments"]
    timeline = build_timeline(segments)
    stats = compute_signal_stats(segments)
    dominant_signal = result.get("dominant_signal")
    if dominant_signal is None:
        dominant_signal = result["aggregation"]["dominant_signal"]
    insight = generate_advanced_insight(segments)
    score = compute_risk_score(segments)
    risk_score = normalize_score(score)
    conflict = detect_conflict(segments)

    return {
        "segments": segments,
        "timeline": timeline,
        "signal_stats": stats,
        "signal": dominant_signal,
        "insight": insight,
        "risk_score": risk_score,
        "conflict": conflict,
    }
