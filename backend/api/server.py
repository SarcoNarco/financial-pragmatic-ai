from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from financial_pragmatic_ai.analysis.earnings_call_analyzer import EarningsCallAnalyzer
from financial_pragmatic_ai.analysis.financial_signal_engine import (
    compute_confidence,
    compute_risk_score,
    detect_volatility,
    derive_market_prediction,
    derive_signal,
    generate_insight,
)
from financial_pragmatic_ai.analysis.insight_engine import extract_key_drivers


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
    prediction = derive_market_prediction(score)
    confidence = compute_confidence(results)
    volatility = detect_volatility(results)
    insight = generate_insight(score, results)
    drivers = extract_key_drivers(results)

    return {
        "score": score,
        "signal": signal,
        "prediction": prediction,
        "confidence": confidence,
        "volatility": volatility,
        "insight": insight,
        "segments": results,
        "drivers": drivers,
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
    prediction = derive_market_prediction(score)
    confidence = compute_confidence(results)
    volatility = detect_volatility(results)
    insight = generate_insight(score, results)
    drivers = extract_key_drivers(results)

    return {
        "score": score,
        "signal": signal,
        "prediction": prediction,
        "confidence": confidence,
        "volatility": volatility,
        "insight": insight,
        "segments": results,
        "drivers": drivers,
    }
