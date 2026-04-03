# AGENT_HANDOFF.md

## 1. PROJECT OVERVIEW

This is a full-stack financial transcript analysis system.

Runtime flow:
- React frontend sends transcript text or uploaded file to FastAPI.
- Backend parses speaker segments, predicts intent per segment, aggregates conversation signal, then returns structured analysis.
- The app is currently **stateless**: no authentication, no DB persistence, no user sessions.

Primary capabilities currently implemented:
- Transcript parsing for structured and unstructured earnings-call text.
- Segment intent prediction using a finetuned FinBERT intent head (with fallback model path).
- Conversation signal derivation (`growth | neutral | risk`).
- Risk score, confidence, volatility, market prediction, and driver extraction.
- UI for Analyze and Compare workflows.

---

## 2. CURRENT SYSTEM ARCHITECTURE

### Backend (active)
- Framework: FastAPI
- Entry point: `/Users/saroshnadaf/Documents/NLP_Proj/backend/api/server.py`
- Active endpoints:
  - `POST /analyze`
  - `POST /upload`
  - `POST /compare`
- No DB/auth modules are used by runtime.

### Frontend (active)
- Framework: React
- Entry point: `/Users/saroshnadaf/Documents/NLP_Proj/frontend/src/App.js`
- Main page: `/Users/saroshnadaf/Documents/NLP_Proj/frontend/src/pages/DashboardPage.js`
- Tabs active: `Analyze`, `Compare`

### End-to-end flow
`text/file -> /analyze or /upload -> EarningsCallAnalyzer -> TranscriptAnalyzer -> financial_signal_engine + market_predictor + insight_engine -> JSON response -> dashboard render`

---

## 3. VERIFIED PROJECT STRUCTURE (CURRENT)

```text
NLP_Proj/
├── AGENT_HANDOFF.md
├── README.md
├── backend/
│   ├── requirements.txt
│   ├── api/
│   │   ├── server.py
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
│       │   ├── conversation_attention_model.py
│       │   ├── financial_pragmatic_transformer_v2.py
│       │   ├── financial_pragmatic_transformer.py
│       │   ├── intent_classifier.py
│       │   ├── speaker_embedding.py
│       │   ├── pragmatic_input_layer.py
│       │   ├── pragmatic_attention.py
│       │   ├── conversation_interaction_model.py
│       │   ├── finbert_base.py
│       │   ├── finbert_intent.pt
│       │   ├── conversation_signal_model.pt
│       │   ├── intent_classifier.pt
│       │   └── pragmatic_transformer_trained.pt
│       ├── data/
│       ├── training/
│       ├── testing/
│       ├── evaluation/better_than_fin/
│       │   ├── evaluate.py
│       │   ├── metrics.py
│       │   ├── utils.py
│       │   └── visualize.py
│       ├── inference/
│       └── utils/
└── frontend/
    ├── src/
    │   ├── App.js
    │   ├── index.js
    │   ├── index.css
    │   ├── App.css
    │   ├── api/client.js
    │   ├── pages/DashboardPage.js
    │   └── components/
    │       ├── SummaryCard.js
    │       ├── TimelineChart.js
    │       └── SignalHeatmap.js
    └── package.json
```

Removed app-layer files (confirmed absent):
- `backend/api/auth.py`
- `backend/api/database.py`
- `frontend/src/pages/LoginPage.js`
- `frontend/src/pages/SignupPage.js`

---

## 4. ACTIVE API CONTRACT

### `POST /analyze`
Request:
```json
{ "transcript": "CEO: ..." }
```
Response keys:
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

### `POST /upload`
- Input: multipart file (`.txt` or `.pdf`)
- Flow: text extraction -> `_run_analysis`
- Output schema matches `/analyze`

### `POST /compare`
Request:
```json
{ "transcript_1": "...", "transcript_2": "..." }
```
Response keys:
- `transcript_1`
- `transcript_2`
- `signal_difference`
- `risk_delta`
- `confidence_delta`
- `trend`
- `comparison`

Not implemented:
- `/auth/*`
- `/save-analysis`
- `/history`
- `/analysis/{id}`

---

## 5. MODEL DETAILS (CURRENT)

### FinBERT intent model (`finbert_intent_model.py`)
- Uses: `BertForSequenceClassification` (`yiyanghkust/finbert-tone`)
- Labels:
  - `EXPANSION`
  - `COST_PRESSURE`
  - `STRATEGIC_PROBING`
  - `GENERAL_UPDATE`
- Classifier head is replaced with:
  - `Linear(hidden, 256) -> ReLU -> Linear(256, 4)`
- `load_weights()` behavior:
  - expects `{"classifier": state_dict}` in `finbert_intent.pt`
  - validates required keys: `0.weight`, `0.bias`, `2.weight`, `2.bias`
  - prints `missing_keys` and `unexpected_keys`
  - raises RuntimeError if classifier weights are missing
- `predict()`:
  - uses model logits directly
  - returns `intent`, `logits`, `embedding`, `confidence`
  - currently prints debug:
    - `LOGITS: ...`
    - `PRED CLASS: ...`

### Transcript analyzer (`transcript_analyzer.py`)
- Tries to load `finbert_intent.pt`; if missing, falls back to pragmatic transformer.
- If classifier load fails with RuntimeError, exception is re-raised.
- Current debug prints in `analyze()`:
  - `🔥 NEW VERSION LOADED 🔥`
  - `MODEL INTENT: ...`
- Heuristic intent override: **currently disabled**.
- Intent smoothing call: **currently commented out** (`# results = smooth_intents(results)`).

### Conversation model usage
- Uses `ConversationAttentionModel` only if `conversation_attention.pt` exists.
- `conversation_attention.pt` is **not present** in models folder.
- Runtime therefore falls back to score-based signal path for conversation signal.

---

## 6. SIGNAL AGGREGATION LOGIC (CURRENT)

File: `analysis/financial_signal_engine.py`

### `compute_risk_score(intents)`
- Current mapping:
  - `EXPANSION` -> `+1.0`
  - `STRONG_GROWTH` -> `+1.0`
  - `COST_PRESSURE` -> `-1.0`
  - `RISK` -> `-1.0`
  - `GENERAL_UPDATE` -> `0.0`
  - `STRATEGIC_PROBING` -> `0.0`
- Returns mean score across intents.
- Current debug prints:
  - `INTENTS: ...`
  - `SCORE: ...`

### `derive_signal(score)`
- Current thresholds:
  - `score > 0.2` -> `growth`
  - `score < -0.2` -> `risk`
  - else `neutral`
- Current debug prints:
  - `FINAL SCORE: ...`
  - `FINAL SIGNAL: ...`

### Important consistency note (verified from code)
`backend/api/server.py::_run_analysis()` currently clamps score to ranges `<=35`, `>=65`, or `36..64` before calling `derive_signal()`, while `derive_signal()` now expects a small-range score around `[-1, 1]`.

This is a live logic mismatch and can skew final API `signal` output.

---

## 7. FRONTEND STATE (CURRENT)

### API client
- `frontend/src/api/client.js`
- Functions:
  - `analyzeTranscript(transcript)` -> `/analyze`
  - `uploadTranscript(file)` -> `/upload`
  - `compareTranscripts(transcript1, transcript2)` -> `/compare`

### Dashboard UI
- `DashboardPage.js` is the only page.
- Analyze tab:
  - transcript textarea
  - upload control
  - summary, timeline, heatmap, distribution, growth/risk driver panels
- Compare tab:
  - two transcript textareas
  - compare result panel

### Visual components
- `SummaryCard.js`: score/prediction/confidence/volatility cards and confidence bar
- `TimelineChart.js`: smoothed line chart with tooltips
- `SignalHeatmap.js`: intent count cards

---

## 8. EVALUATION PIPELINE (CURRENT)

Path: `backend/financial_pragmatic_ai/evaluation/better_than_fin/evaluate.py`

- `_safe_analyze()` no longer suppresses stdout.
- Evaluation prints prediction distributions:
  - `FinBERT prediction distribution: ...`
  - `Our system prediction distribution: ...`
- Collapse guard:
  - prints `[ERROR] Model collapse detected` if one class >80% of custom predictions.
- No model retraining happens in evaluation.

---

## 9. KNOWN LIMITATIONS / RISKS

1. **Score scale mismatch in API path**
   - `_run_analysis()` clamps score to 35/65 style range, but `derive_signal()` expects near-zero thresholds.

2. **Conversation attention weights missing**
   - `conversation_attention.pt` not present, so rule-based fallback is used.

3. **Debug logging is very verbose**
   - logits, intents, and score prints are active in runtime.

4. **Fallback behavior exists**
   - if finetuned checkpoint is missing, system falls back to pragmatic transformer intent model.

---

## 10. HOW TO RUN

Backend:
```bash
cd /Users/saroshnadaf/Documents/NLP_Proj/backend
pip install -r requirements.txt
uvicorn api.server:app --reload
```

Frontend:
```bash
cd /Users/saroshnadaf/Documents/NLP_Proj/frontend
npm install
npm start
```

URLs:
- API: `http://127.0.0.1:8000`
- UI: `http://localhost:3000`

---

## 11. RECENT DEBUGGING CHANGES (LATEST)

- FinBERT intent loader switched to `BertForSequenceClassification` with explicit classifier head checks.
- Transcript analyzer heuristic override disabled.
- Transcript analyzer smoothing disabled.
- `_safe_analyze()` in evaluation no longer redirects stdout.
- Aggregation debug prints added in `financial_signal_engine.py`.

---

## 12. IMPORTANT NOTES FOR NEXT AGENT

- Do not assume legacy auth/database still exists; runtime is stateless.
- If fixing final signal correctness, start with `_run_analysis()` + `compute_risk_score()`/`derive_signal()` scale alignment.
- Keep changes modular: model loading, parser, and API aggregation are separate concerns.
- If reducing log noise, remove temporary debug prints only after validating intent/signal distributions.
- Preserve `/analyze` response schema to avoid frontend breakage.
