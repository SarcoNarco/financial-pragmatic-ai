# AGENT_HANDOFF.md

## 1. Project Snapshot (Verified)

This repository is a full-stack financial transcript analysis system with:
- FastAPI backend (`backend/api/server.py`) as the active runtime API
- React + Tailwind frontend (`frontend/src/*`)
- Financial NLP/ML modules under `backend/financial_pragmatic_ai/*`

Current core product behavior:
- User authentication (signup/login) with JWT
- Transcript analysis from text and file upload (`.txt`, `.pdf`)
- Saved analysis history per user
- Comparison between two saved analyses or two ad-hoc transcripts
- Visual dashboard (summary, timeline, heatmap, drivers)

---

## 2. Active Runtime Architecture

### Backend (active)
- Entry point: `backend/api/server.py`
- DB adapter: `backend/api/database.py`
- Auth helpers: `backend/api/auth.py`
- Request schemas: `backend/api/schemas.py`

### Backend (legacy / not used by README run path)
- `backend/financial_pragmatic_ai/api/server.py` exists but is a simplified older API.
- README run command points to `uvicorn api.server:app --reload` (active server).

### Frontend
- App shell: `frontend/src/App.js`
- Pages:
  - `frontend/src/pages/LoginPage.js`
  - `frontend/src/pages/SignupPage.js`
  - `frontend/src/pages/DashboardPage.js`
- API client: `frontend/src/api/client.js`

No React Router is used; page/view switching is state-driven.

---

## 3. Database & Auth (Current State)

## MongoDB (Motor)
- `backend/api/database.py` uses Motor async client:
  - `AsyncIOMotorClient(MONGO_URI, server_api=ServerApi("1"))`
- Collections:
  - `users_collection`
  - `transcripts_collection`
  - `analyses_collection`
  - `test_collection`

## Required env var
- `MONGO_URI` is mandatory.
- If missing, import-time `RuntimeError` is raised in `database.py`.

## Startup behavior
- On startup (`server.py`):
  - `ping_database()`
  - prints success/failure log
  - initializes indexes via `init_database()`

## Auth
- Password hashing: `pbkdf2_sha256` via Passlib (`backend/api/auth.py`)
- JWT:
  - `HS256`
  - `ACCESS_TOKEN_EXPIRE_MINUTES = 1440`
  - token subject (`sub`) stores Mongo user `_id` string
- `get_current_user` decodes token, converts `sub` to `ObjectId`, fetches Mongo user.

## Security note
- `SECRET_KEY` is hardcoded in `backend/api/auth.py`.

---

## 4. API Endpoints (Exact Current Contract)

All endpoints are in `backend/api/server.py`.

### Public
- `POST /auth/signup`
- `POST /auth/login`
- `POST /analyze`
- `GET /test-db`

### Authenticated (Bearer token required)
- `POST /save-analysis`
- `GET /history`
- `GET /analysis/{analysis_id}`
- `POST /upload`
- `POST /compare`

### Notes on behavior
- `/analyze`: returns analysis payload; does **not** persist.
- `/save-analysis`: re-runs analysis and persists, returns:
  - `analysis` (full analysis payload)
  - `analysis_id` (Mongo ObjectId string)
  - `transcript_id` (Mongo ObjectId string)
- `/upload`: extracts text, runs same analyze+save helper, then returns only `analysis` payload.
- `/history`: reads from `analyses_collection` for current user.
- `/analysis/{id}`: returns saved doc fields + transcript text; does **not** include segments.
- `/compare`:
  - mode 1: compare saved analyses via `analysis_id_1`, `analysis_id_2`
  - mode 2: compare ad-hoc `transcript_1`, `transcript_2` (not persisted)

---

## 5. ML / Analysis Inference Pipeline

Main orchestration:
- `backend/financial_pragmatic_ai/analysis/earnings_call_analyzer.py`
- `backend/financial_pragmatic_ai/analysis/transcript_analyzer.py`
- `backend/financial_pragmatic_ai/analysis/financial_signal_engine.py`
- `backend/financial_pragmatic_ai/analysis/market_predictor.py`
- `backend/financial_pragmatic_ai/analysis/insight_engine.py`

## Transcript parsing
- `analysis/transcript_parser.py`
  - Parses `NAME:` blocks
  - Role inference: `CEO`, `CFO`, `ANALYST`, `EXECUTIVE`, `OPERATOR`
  - Fallback chunking if no speaker blocks

## Segment intent prediction
- `TranscriptAnalyzer.predict_intent()`
  - Primary: `FinBERTIntentModel` (`finbert_intent_model.py`)
  - Fallback: `FinancialPragmaticTransformer` (`financial_pragmatic_transformer_v2.py`)
  - Then applies heuristic overrides in `TranscriptAnalyzer.analyze()`
  - Then applies weighted smoothing (`smooth_intents`)

## Conversation signal
- If `conversation_attention.pt` exists:
  - uses `ConversationAttentionModel`
- Else fallback:
  - computes risk score from intents and derives signal via `financial_signal_engine`

## Final API output
`server.py` computes final score/signal/prediction/confidence/volatility/insight/drivers and returns them to frontend.

---

## 6. Model Files & Artifact Status (Cross-checked)

Directory checked: `backend/financial_pragmatic_ai/models`

### Present
- `pragmatic_transformer_trained.pt`
- `intent_classifier.pt`
- `conversation_signal_model.pt`

### Missing
- `finbert_intent.pt` (expected by `TranscriptAnalyzer`)
- `conversation_attention.pt` (expected by `TranscriptAnalyzer`)

### Important consequence
`TranscriptAnalyzer` calls `self.intent_model.load_weights(FINBERT_INTENT_PATH)` but does not branch on `False` return.
- If `finbert_intent.pt` is missing, the in-memory classifier head in `FinBERTIntentModel` remains whatever was initialized in code (not loaded from disk).
- This is current runtime behavior.

---

## 7. Training Pipelines (Current)

### A) FinBERT intent training
- File: `backend/financial_pragmatic_ai/models/finbert_intent_model.py`
- Uses frozen `AutoModel("yiyanghkust/finbert-tone")` on CPU
- Trains classifier head on CLS embeddings
- Current implementation precomputes embeddings in memory (`X`, `y`) for one run

### B) V2 unified pipeline
- File: `backend/financial_pragmatic_ai/training/train_v2_pipeline.py`
- Flow:
  1. train finbert intent model (`train_finbert_intent_model`)
  2. build conversation windows from pragmatic dataset
  3. stream sequence embeddings to disk (`./embeddings/*.pt`)
  4. train `ConversationAttentionModel` from lazy file dataset
- Memory/throughput improvements implemented:
  - disk streaming of embeddings
  - `EmbeddingDataset` lazy loading
  - classifier training loader with `num_workers=2`, `pin_memory` on CUDA
  - max_length=64, batch_size=32 for embedding phase

### C) Legacy conversation model
- File: `backend/financial_pragmatic_ai/training/train_conversation_model.py`
- Trains `ConversationInteractionModel` and writes `conversation_signal_model.pt`
- This model is not wired into active API path in `backend/api/server.py`.

---

## 8. Frontend Behavior (Current)

### Auth flow
- Token stored in `localStorage` key: `financial_pragmatic_ai_token`
- `App.js` shows login/signup if no token, dashboard if token exists

### Dashboard tabs
- `Analyze`
- `History`
- `Compare`

### Analyze tab
- Text area + `Analyze & Save`
  - calls `/analyze`
  - then `/save-analysis`
  - then refreshes history
- File upload + `Upload & Analyze`
  - calls `/upload` with bearer token
  - backend saves internally
  - frontend refreshes history

### History tab
- Reads `/history`
- Clicking an item calls `/analysis/{id}` then calls `/analyze` on transcript text to regenerate segments for chart views.

### Compare tab
- Select 2 saved analysis IDs
- calls `/compare` with `analysis_id_1`, `analysis_id_2`

---

## 9. Exact Known Limitations / Risks

1. `MONGO_URI` hard requirement
- Server crashes at import/startup if unset.

2. Hardcoded JWT secret
- `SECRET_KEY` is in source code (`api/auth.py`).

3. Inference artifact gap
- Missing `finbert_intent.pt` and `conversation_attention.pt` means hybrid/fallback behavior.

4. Potential intent model quality issue
- Because `load_weights()` return value is not checked, missing intent weights do not disable primary model object.

5. `/test-db` side effect
- Every call inserts a doc in `test_connection` collection (no cleanup).

6. Duplicate API module still present
- `backend/financial_pragmatic_ai/api/server.py` exists and can confuse maintainers; active API is `backend/api/server.py`.

7. Docs drift
- Root `README.md` response example is older/simplified and does not fully match current response payloads.

---

## 10. Verification Performed For This Handoff Update

The following checks were run successfully before finalizing this file:

- Python compile checks:
  - `backend/api/server.py`
  - `backend/api/auth.py`
  - `backend/api/database.py`
  - `backend/api/schemas.py`
  - `backend/financial_pragmatic_ai/analysis/*.py`
  - `backend/financial_pragmatic_ai/models/*.py`
  - `backend/financial_pragmatic_ai/training/train_v2_pipeline.py`

- Frontend production build:
  - `cd frontend && npm run build` (compiled successfully)

No assumptions in this handoff were made beyond the current repository state and these checks.

---

## 11. Run Instructions (Current)

## Backend
```bash
cd backend
pip install -r requirements.txt
export MONGO_URI="<your_mongodb_uri>"
uvicorn api.server:app --reload
```

## Frontend
```bash
cd frontend
npm install
npm start
```

## Quick DB test
Open in browser:
- `http://127.0.0.1:8000/test-db`

Expected response shape:
```json
{
  "status": "success",
  "inserted_id": "...",
  "retrieved": true
}
```
