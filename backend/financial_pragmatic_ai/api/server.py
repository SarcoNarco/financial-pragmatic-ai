from fastapi import FastAPI
from pydantic import BaseModel

from financial_pragmatic_ai.analysis.earnings_call_analyzer import EarningsCallAnalyzer
from financial_pragmatic_ai.analysis.financial_insight_generator import generate_insight

app = FastAPI()

analyzer = EarningsCallAnalyzer()

class TranscriptRequest(BaseModel):
    transcript: str


@app.post("/analyze")

def analyze_transcript(request: TranscriptRequest):

    result = analyzer.analyze(request.transcript)

    insight = generate_insight(result["dominant_signal"])

    return {
        "segments": result["segments"],
        "signal": result["dominant_signal"],
        "insight": insight
    }