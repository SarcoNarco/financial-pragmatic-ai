import json
from datetime import timedelta

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from api.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from api.db import Base, engine, get_db
from api.models import Analysis, Transcript, User
from api.schemas import AuthRequest, AuthResponse, CompareRequest, SaveAnalysisRequest, TranscriptRequest
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


Base.metadata.create_all(bind=engine)

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


def _serialize_analysis_row(analysis: Analysis, transcript_text: str | None = None):
    return {
        "id": analysis.id,
        "transcript_id": analysis.transcript_id,
        "transcript": transcript_text,
        "signal": analysis.signal,
        "prediction": analysis.prediction,
        "confidence": analysis.confidence,
        "volatility": analysis.volatility,
        "score": analysis.score,
        "drivers": json.loads(analysis.drivers),
        "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
    }


def _save_analysis_for_user(db: Session, user: User, transcript_text: str, analysis_result: dict):
    transcript = Transcript(user_id=user.id, text=transcript_text)
    db.add(transcript)
    db.commit()
    db.refresh(transcript)

    analysis_row = Analysis(
        transcript_id=transcript.id,
        signal=str(analysis_result.get("signal", "neutral")),
        prediction=str(analysis_result.get("prediction", "NEUTRAL")),
        confidence=float(analysis_result.get("confidence", 0.0)),
        volatility=str(analysis_result.get("volatility", "LOW")),
        drivers=json.dumps(analysis_result.get("drivers", {})),
        score=float(analysis_result.get("score", 50.0)),
    )
    db.add(analysis_row)
    db.commit()
    db.refresh(analysis_row)

    return transcript, analysis_row


@app.post("/auth/signup", response_model=AuthResponse)
def signup(payload: AuthRequest, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email")

    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(email=email, password_hash=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return AuthResponse(access_token=token)


@app.post("/auth/login", response_model=AuthResponse)
def login(payload: AuthRequest, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    user = db.query(User).filter(User.email == email).first()
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return AuthResponse(access_token=token)


@app.post("/analyze")
def analyze_transcript(request: TranscriptRequest):
    return _run_analysis(request.transcript)


@app.post("/save-analysis")
def save_analysis(
    request: SaveAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    analysis = _run_analysis(request.transcript)
    if "error" in analysis:
        return analysis

    transcript, analysis_row = _save_analysis_for_user(db, current_user, request.transcript, analysis)
    return {
        "analysis_id": analysis_row.id,
        "transcript_id": transcript.id,
        "analysis": analysis,
    }


@app.get("/history")
def get_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(Analysis, Transcript)
        .join(Transcript, Analysis.transcript_id == Transcript.id)
        .filter(Transcript.user_id == current_user.id)
        .order_by(Analysis.created_at.desc())
        .all()
    )

    history = []
    for analysis, transcript in rows:
        history.append(
            {
                "id": analysis.id,
                "transcript_id": transcript.id,
                "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
                "signal": analysis.signal,
                "prediction": analysis.prediction,
                "confidence": analysis.confidence,
                "volatility": analysis.volatility,
                "score": analysis.score,
                "transcript_preview": transcript.text[:220],
            }
        )
    return {"items": history}


@app.get("/analysis/{analysis_id}")
def get_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    row = (
        db.query(Analysis, Transcript)
        .join(Transcript, Analysis.transcript_id == Transcript.id)
        .filter(Analysis.id == analysis_id, Transcript.user_id == current_user.id)
        .first()
    )

    if row is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    analysis, transcript = row
    return _serialize_analysis_row(analysis, transcript_text=transcript.text)


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

    return _run_analysis(text)


@app.post("/compare")
def compare_transcripts(
    request: CompareRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if request.analysis_id_1 is not None and request.analysis_id_2 is not None:
        first_row = (
            db.query(Analysis, Transcript)
            .join(Transcript, Analysis.transcript_id == Transcript.id)
            .filter(Analysis.id == request.analysis_id_1, Transcript.user_id == current_user.id)
            .first()
        )
        second_row = (
            db.query(Analysis, Transcript)
            .join(Transcript, Analysis.transcript_id == Transcript.id)
            .filter(Analysis.id == request.analysis_id_2, Transcript.user_id == current_user.id)
            .first()
        )
        if first_row is None or second_row is None:
            raise HTTPException(status_code=404, detail="One or both analyses not found")

        first = _serialize_analysis_row(first_row[0], transcript_text=first_row[1].text)
        second = _serialize_analysis_row(second_row[0], transcript_text=second_row[1].text)
    elif request.transcript_1 and request.transcript_2:
        first = _run_analysis(request.transcript_1)
        if "error" in first:
            return first
        second = _run_analysis(request.transcript_2)
        if "error" in second:
            return second
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either analysis_id_1/analysis_id_2 or transcript_1/transcript_2",
        )

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
