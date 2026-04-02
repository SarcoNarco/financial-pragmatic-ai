from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import CompareRequest, TranscriptRequest
from financial_pragmatic_ai.analysis.earnings_call_analyzer import EarningsCallAnalyzer
from financial_pragmatic_ai.analysis.financial_signal_engine import (
    compute_confidence,
    compute_intent_distribution,
    compute_risk_score,
    compute_signal_std,
    detect_volatility,
    derive_signal,
    generate_insight,
)
from financial_pragmatic_ai.analysis.insight_engine import extract_key_drivers
from financial_pragmatic_ai.analysis.market_predictor import predict_market_outlook


app = FastAPI(title="Financial Pragmatic AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = EarningsCallAnalyzer()


def _run_analysis(transcript: str):
    result = analyzer.analyze(transcript)
    results = result["segments"]
    if len(results) == 0:
        return {"error": "Could not parse transcript"}

    score = compute_risk_score(results)
    signal = result["aggregation"]["dominant_signal"]
    if signal == "risk":
        score = max(score, 65)
    elif signal == "growth":
        score = min(score, 35)
    else:
        score = min(max(score, 36), 64)

    signal = derive_signal(score)
    confidence = compute_confidence(results)
    volatility = detect_volatility(results)
    volatility_std = round(compute_signal_std(results), 4)
    intent_distribution = compute_intent_distribution(results)
    market = predict_market_outlook(
        signal=signal,
        risk_score=score,
        volatility=volatility,
        intent_distribution=intent_distribution,
    )
    insight = generate_insight(score, results)
    drivers = extract_key_drivers(results)

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
        "segments": results,
        "drivers": drivers,
    }


@app.post("/analyze")
def analyze_transcript(request: TranscriptRequest):
    return _run_analysis(request.transcript)


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
