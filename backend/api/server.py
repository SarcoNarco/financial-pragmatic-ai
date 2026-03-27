import json
from datetime import datetime, timedelta, timezone

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pymongo.errors import DuplicateKeyError

from api.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from api.database import analyses_collection, init_database, transcripts_collection, users_collection
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


app = FastAPI(title="Financial Pragmatic AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = EarningsCallAnalyzer()


@app.on_event("startup")
async def startup_event():
    await init_database()


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


def _safe_iso(value):
    if isinstance(value, datetime):
        return value.isoformat()
    return None


def _serialize_analysis_doc(doc: dict, include_text: bool = False):
    payload = {
        "id": str(doc["_id"]),
        "transcript_id": doc.get("transcript_id"),
        "signal": doc.get("signal"),
        "prediction": doc.get("prediction"),
        "confidence": doc.get("confidence"),
        "volatility": doc.get("volatility"),
        "score": doc.get("score"),
        "drivers": doc.get("drivers", {}),
        "created_at": _safe_iso(doc.get("created_at")),
    }
    if include_text:
        payload["transcript"] = doc.get("text", "")
    return payload


def _object_id_or_404(value: str, message: str = "Invalid ID") -> ObjectId:
    try:
        return ObjectId(value)
    except (InvalidId, TypeError) as exc:
        raise HTTPException(status_code=404, detail=message) from exc


async def _save_analysis_for_user(user_id: str, transcript_text: str, analysis_result: dict):
    now = datetime.now(timezone.utc)
    transcript_doc = {
        "user_id": user_id,
        "text": transcript_text,
        "created_at": now,
    }
    transcript_result = await transcripts_collection.insert_one(transcript_doc)
    transcript_id = str(transcript_result.inserted_id)

    analysis_doc = {
        "user_id": user_id,
        "transcript_id": transcript_id,
        "text": transcript_text,
        "signal": str(analysis_result.get("signal", "neutral")),
        "prediction": str(analysis_result.get("prediction", "NEUTRAL")),
        "confidence": float(analysis_result.get("confidence", 0.0)),
        "volatility": str(analysis_result.get("volatility", "LOW")),
        "drivers": analysis_result.get("drivers", {}),
        "score": float(analysis_result.get("score", 50.0)),
        "created_at": now,
    }
    analysis_result_db = await analyses_collection.insert_one(analysis_doc)
    return transcript_id, str(analysis_result_db.inserted_id)


async def _analyze_and_save_for_user(user_id: str, transcript_text: str):
    analysis = _run_analysis(transcript_text)
    if "error" in analysis:
        return analysis

    transcript_id, analysis_id = await _save_analysis_for_user(user_id, transcript_text, analysis)
    return {
        "analysis": analysis,
        "analysis_id": analysis_id,
        "transcript_id": transcript_id,
    }


@app.post("/auth/signup", response_model=AuthResponse)
async def signup(payload: AuthRequest):
    email = payload.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email")

    existing = await users_collection.find_one({"email": email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_doc = {
        "email": email,
        "password_hash": hash_password(payload.password),
        "created_at": datetime.now(timezone.utc),
    }

    try:
        insert_result = await users_collection.insert_one(user_doc)
    except DuplicateKeyError as exc:
        raise HTTPException(status_code=400, detail="Email already registered") from exc

    token = create_access_token(
        data={"sub": str(insert_result.inserted_id)},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return AuthResponse(access_token=token)


@app.post("/auth/login", response_model=AuthResponse)
async def login(payload: AuthRequest):
    email = payload.email.strip().lower()
    user = await users_collection.find_one({"email": email})
    if user is None or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(
        data={"sub": str(user["_id"])},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return AuthResponse(access_token=token)


@app.post("/analyze")
def analyze_transcript(request: TranscriptRequest):
    return _run_analysis(request.transcript)


@app.post("/save-analysis")
async def save_analysis(
    request: SaveAnalysisRequest,
    current_user: dict = Depends(get_current_user),
):
    return await _analyze_and_save_for_user(current_user["id"], request.transcript)


@app.get("/history")
async def get_history(current_user: dict = Depends(get_current_user)):
    docs = (
        await analyses_collection.find({"user_id": current_user["id"]})
        .sort("created_at", -1)
        .to_list(length=100)
    )

    history = []
    for doc in docs:
        history.append(
            {
                "id": str(doc["_id"]),
                "transcript_id": doc.get("transcript_id"),
                "created_at": _safe_iso(doc.get("created_at")),
                "signal": doc.get("signal"),
                "prediction": doc.get("prediction"),
                "confidence": doc.get("confidence"),
                "volatility": doc.get("volatility"),
                "score": doc.get("score"),
                "transcript_preview": str(doc.get("text", ""))[:220],
            }
        )
    return {"items": history}


@app.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: str, current_user: dict = Depends(get_current_user)):
    object_id = _object_id_or_404(analysis_id, message="Analysis not found")
    doc = await analyses_collection.find_one({"_id": object_id, "user_id": current_user["id"]})

    if doc is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return _serialize_analysis_doc(doc, include_text=True)


@app.post("/upload")
async def upload_transcript(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
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

    saved = await _analyze_and_save_for_user(current_user["id"], text)
    if "error" in saved:
        return saved

    return saved["analysis"]


@app.post("/compare")
async def compare_transcripts(
    request: CompareRequest,
    current_user: dict = Depends(get_current_user),
):
    if request.analysis_id_1 is not None and request.analysis_id_2 is not None:
        first_id = _object_id_or_404(request.analysis_id_1, message="One or both analyses not found")
        second_id = _object_id_or_404(request.analysis_id_2, message="One or both analyses not found")

        first_doc = await analyses_collection.find_one({"_id": first_id, "user_id": current_user["id"]})
        second_doc = await analyses_collection.find_one({"_id": second_id, "user_id": current_user["id"]})
        if first_doc is None or second_doc is None:
            raise HTTPException(status_code=404, detail="One or both analyses not found")

        first = _serialize_analysis_doc(first_doc, include_text=True)
        second = _serialize_analysis_doc(second_doc, include_text=True)
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
