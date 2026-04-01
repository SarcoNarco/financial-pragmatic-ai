# AGENT_HANDOFF.md

## 1. PROJECT OVERVIEW (VERIFIED)

This repository is a full-stack financial transcript analysis system.

Primary runtime flow:
- Frontend (React) sends transcript text/file to FastAPI backend.
- Backend runs financial NLP inference over parsed speaker segments.
- Backend returns analysis outputs (signal, score, prediction, confidence, volatility, drivers, segments).
- Authenticated users can save analyses and compare prior runs.

Current product-level capabilities:
- JWT auth (signup/login)
- transcript analysis via text and file upload (`.txt`, `.pdf`)
- per-user analysis history (MongoDB)
- comparison of two saved analyses (and API support for ad-hoc transcript comparison)
- dashboard visualization (summary, timeline, heatmap, distributions, drivers, segments)

---

## 2. VERIFIED PROJECT STRUCTURE

```text
NLP_Proj/
├── AGENT_HANDOFF.md
├── README.md
├── requirements.txt
├── .gitattributes
├── .gitignore
├── backend/
│   ├── requirements.txt
│   ├── api/
│   │   ├── server.py                  # Active FastAPI app
│   │   ├── auth.py
│   │   ├── database.py
│   │   └── schemas.py
│   ├── financial_pragmatic_ai/
│   │   ├── analysis/
│   │   │   ├── transcript_parser.py
│   │   │   ├── transcript_analyzer.py
│   │   │   ├── earnings_call_analyzer.py
│   │   │   ├── financial_signal_engine.py
│   │   │   ├── market_predictor.py
│   │   │   ├── insight_engine.py
│   │   │   ├── timeline_signal_analyzer.py
│   │   │   ├── timeline_builder.py
│   │   │   ├── signal_statistics.py
│   │   │   ├── financial_insight_generator.py
│   │   │   └── conversation_vectorizer.py
│   │   ├── models/
│   │   │   ├── finbert_intent_model.py
│   │   │   ├── financial_pragmatic_transformer_v2.py
│   │   │   ├── conversation_attention_model.py
│   │   │   ├── conversation_interaction_model.py
│   │   │   ├── finbert_base.py
│   │   │   ├── intent_classifier.py
│   │   │   ├── speaker_embedding.py
│   │   │   ├── pragmatic_input_layer.py
│   │   │   ├── pragmatic_attention.py
│   │   │   ├── financial_pragmatic_transformer.py
│   │   │   ├── conversation_signal_model.pt
│   │   │   ├── intent_classifier.pt
│   │   │   └── pragmatic_transformer_trained.pt
│   │   ├── data/
│   │   │   ├── pragmatic_intent_dataset_clean.csv
│   │   │   ├── pragmatic_intent_dataset.csv
│   │   │   ├── conversation_signal_dataset.csv
│   │   │   ├── intent_dataset.csv
│   │   │   ├── build_pragmatic_training_dataset.py
│   │   │   ├── build_conversation_dataset.py
│   │   │   ├── clean_pragmatic_dataset.py
│   │   │   └── pragmatic_training_dataset/combined_pragmatic_transcripts.jsonl
│   │   ├── training/
│   │   │   ├── train_v2_pipeline.py
│   │   │   ├── train_pragmatic_transformer.py
│   │   │   ├── train_pragmatic_transformwer.py  # compatibility wrapper
│   │   │   ├── train_intent_classifier.py
│   │   │   ├── train_conversation_model.py
│   │   │   └── test_load_model.py
│   │   ├── testing/
│   │   │   ├── test_transcript_analyzer.py
│   │   │   ├── test_earnings_call_analyzer.py
│   │   │   ├── test_trained_model.py
│   │   │   └── evaluate_model.py
│   │   ├── evaluation/
│   │   │   └── better_than_fin/
│   │   │       ├── evaluate.py
│   │   │       ├── metrics.py
│   │   │       ├── utils.py
│   │   │       ├── visualize.py
│   │   │       └── results/            # committed smoke-run outputs
│   │   ├── inference/
│   │   │   ├── signal_extractor.py
│   │   │   └── decision_engine.py
│   │   ├── utils/
│   │   │   ├── device.py
│   │   │   ├── check_mps.py
│   │   │   ├── transcript_parser.py
│   │   │   ├── financial_event_tokenizer.py
│   │   │   └── pragmatic_analyzer.py
│   │   └── api/
│   │       └── server.py               # legacy simplified API module
│   └── financial_pragmatic_ai.db       # legacy sqlite artifact (not used)
└── frontend/
    ├── package.json
    ├── tailwind.config.js
    ├── postcss.config.js
    ├── src/
    │   ├── App.js
    │   ├── index.js
    │   ├── index.css
    │   ├── App.css                     # present but not imported in index.js/App.js
    │   ├── api/client.js
    │   ├── pages/
    │   │   ├── LoginPage.js
    │   │   ├── SignupPage.js
    │   │   └── DashboardPage.js
    │   └── components/
    │       ├── SummaryCard.js
    │       ├── TimelineChart.js
    │       └── SignalHeatmap.js
    └── build/                          # committed build artifacts
```

---

## 3. ACTIVE RUNTIME ARCHITECTURE

### Backend (active)
- Entry point: `backend/api/server.py`
- Framework: FastAPI
- Database: MongoDB via Motor (`backend/api/database.py`)
- Auth: JWT + Passlib PBKDF2 (`backend/api/auth.py`)

### Backend (legacy, not primary runtime)
- `backend/financial_pragmatic_ai/api/server.py` exists (older minimal API).
- It is not the service started by README command.

### Frontend
- Framework: React (CRA) + Tailwind utility classes
- Main app gate: `frontend/src/App.js`
- No React Router; auth and page switching are local React state.

---

## 4. BACKEND API CONTRACT (EXACT CURRENT)

All routes are in `backend/api/server.py`.

### Public routes
- `POST /auth/signup`
  - body: `{ "email": str, "password": str }`
  - password validation from schema: min 6, max 256 chars
- `POST /auth/login`
  - body: `{ "email": str, "password": str }`
- `POST /analyze`
  - body: `{ "transcript": str }`
  - returns analysis payload (does not persist)
- `GET /test-db`
  - writes one test doc and reads it back

### Authenticated routes (Bearer token)
- `POST /save-analysis`
  - body: `{ "transcript": str }`
  - runs analysis + persists transcript + analysis
  - returns:
    - `analysis` (full analysis payload)
    - `analysis_id`
    - `transcript_id`
- `GET /history`
  - returns last 100 saved analyses for user
- `GET /analysis/{analysis_id}`
  - returns saved analysis fields + transcript text
- `POST /upload`
  - multipart file upload (`.txt` or `.pdf`)
  - extracts text, analyzes, persists (same save helper), returns analysis payload only
- `POST /compare`
  - mode A: compare by `analysis_id_1`, `analysis_id_2`
  - mode B: compare by `transcript_1`, `transcript_2` (not persisted)
  - returns deltas and trend

### Current analysis payload fields
- `score`
- `signal`
- `prediction`
- `prediction_explanation`
- `confidence`
- `volatility`
- `volatility_std`
- `intent_distribution`
- `insight`
- `segments`
- `drivers`

---

## 5. DATABASE + AUTH (VERIFIED)

### Database
- `backend/api/database.py`
- Motor client with `ServerApi("1")`
- `MONGO_URI` is required; startup/import fails if unset.
- DB name default: `financial_ai` (override via `MONGO_DB_NAME`)

Collections:
- `users`
- `transcripts`
- `analyses`
- `test_connection`

Indexes created on startup:
- `users.email` unique
- `analyses.user_id`
- `analyses.created_at`
- `transcripts.user_id`

### Auth
- JWT algorithm: `HS256`
- access token expiry: 24h
- password hashing: `pbkdf2_sha256` via Passlib
- bearer auth dependency resolves Mongo user via `sub` ObjectId

Security status:
- `SECRET_KEY` is hardcoded in source (`backend/api/auth.py`), not env-driven.

---

## 6. INFERENCE PIPELINE (ACTUAL CODE PATH)

### 6.1 Request path
`backend/api/server.py::_run_analysis()`
1. calls `EarningsCallAnalyzer.analyze(transcript)`
2. consumes `result["segments"]` + aggregation signal
3. computes score/signal/confidence/volatility/distribution/insight/prediction/drivers
4. returns final payload

### 6.2 Earnings-call orchestration
`analysis/earnings_call_analyzer.py`
1. `TranscriptAnalyzer.analyze()` -> segment intents
2. `predict_conversation_signal(segments)` -> `model_signal`
3. `TimelineSignalAnalyzer.analyze_timeline(segments)` -> window signals
4. aggregate counts + use `model_signal` as dominant signal
5. generate simple insight text from dominant signal

### 6.3 Transcript parsing
`analysis/transcript_parser.py`
- Cleans text
- Extracts `NAME:` blocks (supports title case and uppercase names)
- infers roles: `CEO`, `CFO`, `ANALYST`, `OPERATOR`, else `EXECUTIVE`
- fallback semantic chunking when no speaker blocks

### 6.4 Intent prediction
`analysis/transcript_analyzer.py`
- Primary object: `FinBERTIntentModel`
  - expected weights file: `models/finbert_intent.pt`
- Fallback object: `FinancialPragmaticTransformer` loaded from `pragmatic_transformer_trained.pt`
- Additional heuristic overrides for `GENERAL_UPDATE/EXECUTIVE` intents
- Weighted smoothing (`smooth_intents`) over predicted intent sequence

### 6.5 Conversation signal
`TranscriptAnalyzer.predict_conversation_signal()`
- If `conversation_attention.pt` is available and embedding lengths align:
  - runs `ConversationAttentionModel`
- Else:
  - fallback rule path: `compute_risk_score` -> `derive_signal`

Important behavior:
- For full-sequence call (from `EarningsCallAnalyzer`), conversation model can be used (if weights file exists).
- For timeline windows (size=3), embedding-length check usually fails when transcript has >3 segments, so those windows generally use fallback scoring.

### 6.6 Scoring + prediction
`analysis/financial_signal_engine.py`
- score initialized at 45
- +3 `COST_PRESSURE`
- +1 `STRATEGIC_PROBING`
- -1 `EXPANSION`
- clipped to `[5, 95]`
- signal thresholds:
  - `>=65` risk
  - `<=35` growth
  - else neutral
- confidence from dominant mapped signal share
- volatility from signal std dev thresholds
- intent distribution percentages

`analysis/market_predictor.py`
- maps signal/risk_score/volatility/intent_distribution to:
  - `UP`, `DOWN`, `VOLATILE`, or `NEUTRAL`
- returns prediction + explanation

`analysis/insight_engine.py`
- extracts top growth/risk driver snippets from segment intents + keyword filters

---

## 7. MODEL INVENTORY (VERIFIED FILE STATE)

In `backend/financial_pragmatic_ai/models`:

Present artifacts:
- `pragmatic_transformer_trained.pt`
- `intent_classifier.pt`
- `conversation_signal_model.pt`

Missing expected artifacts:
- `finbert_intent.pt` (expected by `TranscriptAnalyzer` primary intent path)
- `conversation_attention.pt` (expected by conversation attention path)

Consequence:
- Runtime logs warnings and enters fallback/hybrid behavior when these files are absent.
- `FinBERTIntentModel.load_weights(...)` return value is not checked in `TranscriptAnalyzer`; object remains active even when weights file is missing.

---

## 8. TRAINING PIPELINES (CURRENT)

### 8.1 `models/finbert_intent_model.py`
- Defines `FinBERTIntentModel`
- Frozen FinBERT encoder (`yiyanghkust/finbert-tone`) + trainable classifier head
- training function: `train_finbert_intent_model(...)`
- pre-tokenized `IntentTextDataset` (tokenization done in `__init__`)
- precomputes CLS embeddings then trains classifier head
- saves classifier state into `finbert_intent.pt` format (`{"classifier": ...}`)

### 8.2 `training/train_v2_pipeline.py`
Unified pipeline:
1. trains FinBERT intent head (`train_finbert_intent_model`)
2. builds conversation sequences from pragmatic dataset (`CEO/CFO/ANALYST` presence in 3-row windows)
3. streams 3-step speaker-aware embeddings to disk (`./embeddings/*.pt`)
4. trains `ConversationAttentionModel` from lazy `EmbeddingDataset`
5. writes `conversation_attention.pt`

Notes:
- sampling line present: `df = pd.read_csv(DATA_PATH).sample(10000, random_state=42)` for sequence building
- FinBERT training still uses full `DATA_PATH` passed to trainer

### 8.3 Other training scripts
- `training/train_pragmatic_transformer.py` (supervised intent classifier training for v2 transformer)
- `training/train_conversation_model.py` (legacy interaction model -> `conversation_signal_model.pt`)
- `training/train_intent_classifier.py` (older `ExecutiveIntentClassifier` on `intent_dataset.csv`)
- `training/train_pragmatic_transformwer.py` is a wrapper to the correctly spelled script

---

## 9. EVALUATION SUITE (BETTER-THAN-FIN)

Location:
- `backend/financial_pragmatic_ai/evaluation/better_than_fin/`

Files:
- `evaluate.py`
- `metrics.py`
- `utils.py`
- `visualize.py`

Command:
- `python -m financial_pragmatic_ai.evaluation.better_than_fin.evaluate`

Implemented outputs:
- accuracy / precision / recall / F1 / confusion matrix (both systems)
- agreement rate + disagreement logs
- confidence by class
- top-10 examples where custom system is correct and FinBERT is wrong
- plots and csv/json artifacts under `results/`

Current committed `results/` status:
- artifacts exist from a smoke run with `dataset_rows = 2` (see `results/summary.json`)
- this is a sanity run, not a full benchmark.

---

## 10. FRONTEND IMPLEMENTATION (VERIFIED)

### Stack
- React + Axios + Tailwind utility classes
- charts: `react-chartjs-2`/`chart.js` and `recharts` dependency present

### App shell
- `App.js`
  - token gate via localStorage key: `financial_pragmatic_ai_token`
  - unauthenticated views: login/signup
  - authenticated view: dashboard

### Dashboard tabs (actual)
- `Analyze`
- `History`
- `Compare`

### Analyze tab behavior
- Text flow:
  - calls `/analyze`
  - then `/save-analysis`
  - refreshes history
- File upload flow:
  - calls `/upload` with bearer token
  - backend already persists
  - refreshes history

### History tab behavior
- loads `/history`
- selecting an item:
  - fetches `/analysis/{id}`
  - then calls `/analyze` on transcript text to regenerate segments/charts
  - overlays saved summary fields from `/analysis/{id}`

### Compare tab behavior
- UI compares two saved analysis IDs only
- calls `/compare` with `analysis_id_1`, `analysis_id_2`

### Styling status
- Tailwind utility styling is active (`index.css` includes Tailwind directives)
- `App.css` exists but is not imported by `index.js`/`App.js` (currently unused stylesheet)

---

## 11. KNOWN LIMITATIONS / ARCHITECTURAL INCONSISTENCIES

1. Missing V2 weight artifacts
- `finbert_intent.pt` and `conversation_attention.pt` are missing from repo currently.

2. Timeline signal windows and conversation model
- Windowed calls in timeline analyzer usually fallback to rule scoring due embedding-length guard in `predict_conversation_signal`.

3. Hardcoded JWT secret
- security risk for production.

4. `MONGO_URI` hard requirement at import time
- backend fails early if unset.

5. Legacy API + legacy sqlite artifact still present
- `backend/financial_pragmatic_ai/api/server.py`
- `backend/financial_pragmatic_ai.db`

6. `/test-db` has write side effect
- each call inserts into `test_connection`.

7. README response example is outdated
- README shows simplified `/analyze` response (signal/insight only), but actual API returns richer payload.

8. Evaluation dependency drift
- `backend/requirements.txt` does not include `pandas`, `matplotlib`, `scikit-learn` required by evaluation scripts.

9. Repository hygiene
- `frontend/build/` and `frontend/node_modules/` are present in working tree (large, generally not source-of-truth for dev flow).

---

## 12. HOW TO RUN (CURRENT PRACTICAL STEPS)

### Backend
```bash
cd /Users/saroshnadaf/Documents/NLP_Proj/backend
export MONGO_URI="<your_mongo_uri>"
pip install -r requirements.txt
uvicorn api.server:app --reload
```

### Frontend
```bash
cd /Users/saroshnadaf/Documents/NLP_Proj/frontend
npm install
npm start
```

### Evaluation
```bash
cd /Users/saroshnadaf/Documents/NLP_Proj/backend
python -m financial_pragmatic_ai.evaluation.better_than_fin.evaluate
```

---

## 13. PRIORITIZED NEXT STEPS

1. Train and persist missing V2 artifacts:
- `finbert_intent.pt`
- `conversation_attention.pt`

2. Make `TranscriptAnalyzer` robust to missing intent weights:
- explicitly branch when `load_weights()` is false.

3. Align timeline analyzer with conversation model embeddings:
- support per-window embedding extraction so attention model can score each window.

4. Move JWT secret to environment variable and rotate.

5. Update `backend/requirements.txt` with evaluation dependencies.

6. Clean repo hygiene:
- exclude/remove `frontend/node_modules` and generated build artifacts from tracked source branch if desired.

---

## 14. VERIFICATION SCOPE FOR THIS UPDATE

This handoff update was generated by direct source inspection of:
- backend API/auth/database/schemas
- analysis/model/training/testing/evaluation modules
- frontend app/pages/components/client
- repo artifact inventory and model/data file presence

No assumptions were made about unimplemented components.
If a component was ambiguous, it is explicitly called out above as legacy, missing, or fallback behavior.
