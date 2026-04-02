# AGENT_HANDOFF.md

## 1. PROJECT OVERVIEW

This repository is a full-stack financial transcript analysis system. It accepts earnings-call style transcripts (text or uploaded file), parses speaker-level segments, predicts segment intents, aggregates conversation-level financial signal, and returns structured inference outputs.

Core product goal:
- Provide fast, explainable financial conversation analysis for V2 model development.
- Current runtime is **stateless inference**: no authentication, no database, no user session persistence.

Key capabilities currently implemented:
- Transcript parsing (structured and fallback chunking)
- Segment-level intent prediction (hybrid model + heuristics)
- Conversation-level signal and risk scoring
- Market prediction + explanation
- Driver extraction (growth/risk snippets)
- React dashboard with Analyze + Compare workflows

---

## 2. CURRENT SYSTEM ARCHITECTURE

### Backend
- Framework: FastAPI
- Entry point: `/Users/saroshnadaf/Documents/NLP_Proj/backend/api/server.py`
- Active endpoints:
  - `POST /analyze`
  - `POST /upload`
  - `POST /compare`
- Architecture type: **stateless inference service**
- Removed from active system:
  - MongoDB layer
  - JWT/auth layer
  - history/save-analysis APIs

### Frontend
- Framework: React (CRA-style app)
- API client: `/Users/saroshnadaf/Documents/NLP_Proj/frontend/src/api/client.js`
- Main page: `/Users/saroshnadaf/Documents/NLP_Proj/frontend/src/pages/DashboardPage.js`
- Tabs currently active:
  - Analyze
  - Compare
- Removed from active system:
  - Login page
  - Signup page
  - token/session logic
  - history API integration

### Data flow
`Transcript text/file -> FastAPI /analyze or /upload -> EarningsCallAnalyzer -> TranscriptAnalyzer -> signal engines -> formatted JSON -> React dashboard`

---

## 3. VERIFIED PROJECT STRUCTURE (CURRENT)

```text
NLP_Proj/
├── AGENT_HANDOFF.md
├── README.md
├── requirements.txt
├── backend/
│   ├── requirements.txt
│   ├── api/
│   │   ├── server.py                  # active FastAPI app
│   │   └── schemas.py
│   └── financial_pragmatic_ai/
│       ├── analysis/
│       │   ├── transcript_parser.py
│       │   ├── transcript_analyzer.py
│       │   ├── earnings_call_analyzer.py
│       │   ├── financial_signal_engine.py
│       │   ├── market_predictor.py
│       │   ├── insight_engine.py
│       │   ├── timeline_signal_analyzer.py
│       │   ├── timeline_builder.py
│       │   ├── signal_statistics.py
│       │   ├── financial_insight_generator.py
│       │   └── conversation_vectorizer.py
│       ├── models/
│       │   ├── finbert_intent_model.py
│       │   ├── financial_pragmatic_transformer_v2.py
│       │   ├── conversation_attention_model.py
│       │   ├── conversation_interaction_model.py
│       │   ├── financial_pragmatic_transformer.py
│       │   ├── finbert_base.py
│       │   ├── intent_classifier.py
│       │   ├── speaker_embedding.py
│       │   ├── pragmatic_input_layer.py
│       │   ├── pragmatic_attention.py
│       │   ├── conversation_signal_model.pt
│       │   ├── intent_classifier.pt
│       │   └── pragmatic_transformer_trained.pt
│       ├── data/
│       │   ├── pragmatic_intent_dataset_clean.csv
│       │   ├── pragmatic_intent_dataset.csv
│       │   ├── conversation_signal_dataset.csv
│       │   ├── intent_dataset.csv
│       │   └── pragmatic_training_dataset/combined_pragmatic_transcripts.jsonl
│       ├── training/
│       │   ├── train_v2_pipeline.py
│       │   ├── train_pragmatic_transformer.py
│       │   ├── train_pragmatic_transformwer.py
│       │   ├── train_intent_classifier.py
│       │   ├── train_conversation_model.py
│       │   └── test_load_model.py
│       ├── testing/
│       │   ├── test_transcript_analyzer.py
│       │   ├── test_earnings_call_analyzer.py
│       │   ├── test_trained_model.py
│       │   └── evaluate_model.py
│       ├── evaluation/
│       │   └── better_than_fin/
│       │       ├── evaluate.py
│       │       ├── metrics.py
│       │       ├── utils.py
│       │       └── visualize.py
│       ├── inference/
│       │   ├── signal_extractor.py
│       │   └── decision_engine.py
│       └── utils/
│           ├── device.py
│           ├── check_mps.py
│           ├── transcript_parser.py
│           ├── financial_event_tokenizer.py
│           └── pragmatic_analyzer.py
└── frontend/
    ├── package.json
    ├── src/
    │   ├── App.js
    │   ├── api/client.js
    │   ├── pages/DashboardPage.js
    │   └── components/
    │       ├── SummaryCard.js
    │       ├── TimelineChart.js
    │       └── SignalHeatmap.js
    └── build/                           # committed build artifacts
```

Notes:
- `/Users/saroshnadaf/Documents/NLP_Proj/backend/api/auth.py` was removed.
- `/Users/saroshnadaf/Documents/NLP_Proj/backend/api/database.py` was removed.
- `/Users/saroshnadaf/Documents/NLP_Proj/frontend/src/pages/LoginPage.js` was removed.
- `/Users/saroshnadaf/Documents/NLP_Proj/frontend/src/pages/SignupPage.js` was removed.

---

## 4. BACKEND API CONTRACT (ACTIVE)

### `POST /analyze`
- Request:
```json
{ "transcript": "CEO: ..." }
```
- Response (from `_run_analysis`):
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
- Error mode:
  - `{ "error": "Could not parse transcript" }`

### `POST /upload`
- Input: multipart file (`.txt` or `.pdf`)
- Flow: decode/extract text -> `_run_analysis`
- Response: same payload as `/analyze`
- Error responses:
  - unsupported file type
  - pdf parsing failure
  - txt decode failure

### `POST /compare`
- Request:
```json
{ "transcript_1": "...", "transcript_2": "..." }
```
- Response:
  - `transcript_1` (analysis payload)
  - `transcript_2` (analysis payload)
  - `signal_difference`
  - `risk_delta`
  - `confidence_delta`
  - `trend`
  - `comparison`
- No persisted IDs are used.

---

## 5. FRONTEND IMPLEMENTATION (ACTIVE)

### App composition
- `/Users/saroshnadaf/Documents/NLP_Proj/frontend/src/App.js`
  - always renders `DashboardPage`
  - no auth guard

### Dashboard behavior
- `/Users/saroshnadaf/Documents/NLP_Proj/frontend/src/pages/DashboardPage.js`
- Analyze tab:
  - textarea input
  - analyze button (`/analyze`)
  - file upload (`/upload`)
  - displays summary, timeline, heatmap, signal distribution, drivers
- Compare tab:
  - two transcript textareas
  - compare button (`/compare`)
  - displays deltas and trend

### API client
- `/Users/saroshnadaf/Documents/NLP_Proj/frontend/src/api/client.js`
- exported functions:
  - `analyzeTranscript`
  - `uploadTranscript`
  - `compareTranscripts`
- no token/header injection logic

---

## 6. INFERENCE PIPELINE DETAILS

### Core orchestration
- `_run_analysis()` in `/Users/saroshnadaf/Documents/NLP_Proj/backend/api/server.py`
- Calls `EarningsCallAnalyzer.analyze(transcript)` and then computes:
  - risk score normalization
  - derived signal
  - confidence
  - volatility (+ std)
  - intent distribution
  - market prediction
  - insight text
  - key drivers

### Analyzer stack
1. `analysis/transcript_parser.py`
   - cleans text
   - extracts `NAME:` speaker blocks when possible
   - infers role (`CEO`, `CFO`, `ANALYST`, `OPERATOR`, `EXECUTIVE`)
   - fallback chunking when explicit speaker tags are missing

2. `analysis/transcript_analyzer.py`
   - predicts intent per segment via `FinBERTIntentModel` when available
   - fallback to `FinancialPragmaticTransformer` when needed
   - applies keyword-based override logic
   - applies weighted smoothing (`smooth_intents`)
   - attempts conversation-attention inference if model file exists and embeddings align

3. `analysis/earnings_call_analyzer.py`
   - builds timeline window signals
   - aggregates dominant signal
   - returns `segments`, `timeline_signals`, `aggregation`, `insight`

4. `analysis/financial_signal_engine.py`
   - computes score/signal/confidence/volatility/distribution

5. `analysis/market_predictor.py`
   - computes `UP | DOWN | VOLATILE | NEUTRAL` + explanation

6. `analysis/insight_engine.py`
   - extracts top growth and risk drivers

---

## 7. MODEL + ARTIFACT STATUS

### Present `.pt` files
- `conversation_signal_model.pt`
- `intent_classifier.pt`
- `pragmatic_transformer_trained.pt`

### Referenced but may be missing at runtime
- `finbert_intent.pt` (used by `FinBERTIntentModel.load_weights`)
- `conversation_attention.pt` (used by attention-based conversation signal path)

Runtime behavior when missing:
- system logs warnings and uses fallback/hybrid inference paths.

---

## 8. TRAINING + EVALUATION STATUS

Training scripts exist under:
- `/Users/saroshnadaf/Documents/NLP_Proj/backend/financial_pragmatic_ai/training/`

Evaluation suite exists under:
- `/Users/saroshnadaf/Documents/NLP_Proj/backend/financial_pragmatic_ai/evaluation/better_than_fin/`

Important:
- This handoff reflects runtime app wiring only.
- No retraining is required for current stateless API operation.

---

## 9. KNOWN LIMITATIONS (CURRENT)

- No persistence layer: outputs are not saved.
- No user authentication: API is open for local/dev usage.
- Inference quality depends on available local model weights.
- Conversation attention path may fallback when `conversation_attention.pt` is absent.
- Compare endpoint is transcript-to-transcript only (no saved analysis IDs).

---

## 10. HOW TO RUN

### Backend
```bash
cd /Users/saroshnadaf/Documents/NLP_Proj/backend
pip install -r requirements.txt
uvicorn api.server:app --reload
```

### Frontend
```bash
cd /Users/saroshnadaf/Documents/NLP_Proj/frontend
npm install
npm start
```

### URLs
- UI: `http://localhost:3000`
- API: `http://127.0.0.1:8000`

---

## 11. IMPORTANT NOTES FOR NEXT AGENT

- Do not reintroduce MongoDB or JWT unless explicitly requested.
- Preserve `/analyze` response schema to avoid frontend breakage.
- Keep model/training/evaluation modules untouched for app-layer work.
- Prefer extending analysis modules rather than rewriting runtime API contract.
- Validate both Analyze and Compare flows after any UI/API changes.
