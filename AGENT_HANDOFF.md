# AGENT_HANDOFF.md

## 1. PROJECT OVERVIEW

Financial Pragmatic AI is a full-stack transcript intelligence tool for earnings calls. It ingests transcript text (or uploaded `.txt`/`.pdf` files), parses speaker turns, predicts segment intents, derives conversation-level financial risk/signal metrics, and serves structured outputs to a React dashboard.

The system focuses on pragmatic financial interpretation rather than generic sentiment. Core outputs include intent distribution, risk score, signal (`growth` / `neutral` / `risk`), confidence, volatility, prediction, driver snippets, and transcript-to-transcript comparison deltas.

---

## 2. CURRENT SYSTEM ARCHITECTURE

### Backend

- Framework: **FastAPI**
- Canonical runtime API: `backend/api/server.py`
- Secondary API module (simplified, not used by README run path): `backend/financial_pragmatic_ai/api/server.py`

Core backend modules:

- `backend/financial_pragmatic_ai/analysis/transcript_parser.py`
  - Cleans transcript text
  - Extracts speaker blocks using `NAME:` patterns
  - Infers speaker role (`CEO`, `CFO`, `ANALYST`, `EXECUTIVE`, `OPERATOR`)

- `backend/financial_pragmatic_ai/analysis/transcript_analyzer.py`
  - Segment intent inference
  - Speaker-aware embeddings for interaction modeling
  - Attention-model signal inference **if model file exists**
  - Rule-based signal fallback

- `backend/financial_pragmatic_ai/analysis/earnings_call_analyzer.py`
  - Wraps transcript analyzer
  - Produces dominant signal aggregation and timeline windows

- `backend/financial_pragmatic_ai/analysis/financial_signal_engine.py`
  - Risk score, signal derivation, confidence, volatility, intent distribution

- `backend/financial_pragmatic_ai/analysis/market_predictor.py`
  - Market prediction (`UP` / `DOWN` / `NEUTRAL` / `VOLATILE`) + explanation

- `backend/financial_pragmatic_ai/analysis/insight_engine.py`
  - Growth/risk driver extraction from segment text

### Frontend

- Framework: **React**
- Main entry UI: `frontend/src/App.js`
- Key components:
  - Input panel (text + file upload)
  - `SummaryCard`
  - `TimelineChart`
  - `SignalHeatmap`
  - Drivers and segment panels
  - Tabs: **Analyze / Insights / Demo / Compare**

### Data Flow

1. Transcript input (text or file)
2. `transcript_parser` creates segments with inferred speakers
3. `transcript_analyzer` predicts intents per segment
4. `earnings_call_analyzer` provides dominant conversation signal candidate
5. API computes final metrics using `financial_signal_engine` + `market_predictor`
6. API response drives summary/timeline/drivers/compare UI

Important runtime detail:
- Final API `signal` is recomputed from score (`derive_signal`) in `backend/api/server.py`, not taken directly from analyzer dominant signal.

---

## 3. MODEL DETAILS

### FinBERT usage

1. Base embedding usage:
- `backend/financial_pragmatic_ai/models/finbert_base.py`
- Model: `ProsusAI/finbert` (`AutoModel`)

2. Transformer intent model:
- `backend/financial_pragmatic_ai/models/financial_pragmatic_transformer_v2.py`
- Uses FinBERT token embeddings + speaker embedding + pragmatic input/attention + linear intent head

3. Finetuned intent classifier wrapper:
- `backend/financial_pragmatic_ai/models/finbert_intent_model.py`
- Model: `yiyanghkust/finbert-tone` (`AutoModelForSequenceClassification`)
- Returns intent + logits + CLS embedding

### Intent labels (canonical)

- `EXPANSION`
- `COST_PRESSURE`
- `STRATEGIC_PROBING`
- `GENERAL_UPDATE`

### Interaction modeling

Two separate conversation model stacks exist:

1. **Legacy interaction model**
- `backend/financial_pragmatic_ai/models/conversation_interaction_model.py`
- Trained by `training/train_conversation_model.py`
- Artifact: `conversation_signal_model.pt` (exists)
- Not integrated into active `backend/api/server.py` path

2. **Attention interaction model (V2 target)**
- `backend/financial_pragmatic_ai/models/conversation_attention_model.py`
- Integrated in `transcript_analyzer.py` **if** `conversation_attention.pt` exists
- Artifact: `conversation_attention.pt` (currently missing)

### Speaker handling

- Parser infers role heuristically.
- `transcript_analyzer.py` uses one-hot speaker vectors for interaction embedding concatenation:
  - `CEO = [1, 0, 0]`
  - `CFO = [0, 1, 0]`
  - `ANALYST = [0, 0, 1]`
- Combined sequence embedding shape for attention model: `771` (`768 + 3`).

### Training implementation status

Implemented training code exists for:
- Pragmatic transformer intent model
- MLP intent classifier
- Legacy conversation model
- FinBERT intent model (module function)
- Unified V2 pipeline script

Current artifact state on disk:
- `backend/financial_pragmatic_ai/models/pragmatic_transformer_trained.pt` ✅
- `backend/financial_pragmatic_ai/models/intent_classifier.pt` ✅
- `backend/financial_pragmatic_ai/models/conversation_signal_model.pt` ✅
- `backend/financial_pragmatic_ai/models/finbert_intent.pt` ❌ (missing)
- `backend/financial_pragmatic_ai/models/conversation_attention.pt` ❌ (missing)

---

## 4. CURRENT FEATURES (STATE OF PROJECT)

Implemented:

- Transcript parsing + role inference
- Segment intent prediction
- Rule-assisted intent overrides and smoothing
- Risk score computation
- Final signal derivation
- Market prediction + explanation
- Confidence metric
- Volatility metric (+ volatility std)
- Intent distribution output
- Driver extraction (growth/risk)
- Timeline chart + heatmap UI
- `.txt` and `.pdf` upload
- Tabs: Analyze / Insights / Demo / Compare
- Demo transcript presets + one-click analyze
- Multi-transcript comparison endpoint + UI

Not implemented:
- Persistent storage (DB)
- Auth/authorization
- Background jobs / async task queue

---

## 5. KNOWN LIMITATIONS

1. Missing V2 model artifacts at runtime:
- `finbert_intent.pt` missing
- `conversation_attention.pt` missing
- This forces fallback/hybrid behavior.

2. Final API signal logic is partly rule-based:
- API recalculates signal from score after analyzer output.

3. `transcript_analyzer` still applies post-model heuristics:
- Rule overrides and smoothing can dominate model outputs.

4. Parser is heuristic:
- May mis-segment unstructured or noisy transcripts.

5. `finbert_intent_model.py` default dataset points first to `conversation_dataset.csv` (not present), then falls back to `pragmatic_intent_dataset_clean.csv`.

6. V2 training pipeline currently samples only 10,000 rows before sequence build:
- `training/train_v2_pipeline.py`
- This is an implementation choice, not documented hyperparameter metadata.

7. Conversation sequence builder in V2 pipeline:
- Uses 3-row sliding windows requiring all roles present
- Does **not** enforce strict order `CEO -> CFO -> ANALYST`.

---

## 6. NEXT DEVELOPMENT TASKS

1. Generate missing V2 artifacts
- Train and save `finbert_intent.pt`
- Train and save `conversation_attention.pt`
- Validate that active inference path picks them up

2. Stabilize V2 training pipeline
- Add train/val split and evaluation metrics for conversation attention model
- Add reproducible artifact metadata (config + metrics)
- Decide whether strict role order is required for sequence construction

3. Reduce heuristic dominance
- Quantify effect of intent override/smoothing in `transcript_analyzer.py`
- Gate or tune rules after model artifacts exist

4. Parser hardening
- Add tests for edge transcript formats
- Improve speaker block extraction robustness

5. Operational maturity
- API schema tests (especially `/compare`)
- Logging/monitoring and deployment docs

---

## 7. HOW TO RUN THE PROJECT

Backend:

```bash
cd backend
source ../.venv/bin/activate
pip install -r requirements.txt
uvicorn api.server:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm start
```

---

## 8. API CONTRACT

### `POST /analyze`

Input:

```json
{
  "transcript": "CEO: ... CFO: ..."
}
```

Output fields:

```json
{
  "score": 52,
  "signal": "neutral",
  "prediction": "NEUTRAL",
  "prediction_explanation": "...",
  "confidence": 67.5,
  "volatility": "MEDIUM",
  "volatility_std": 0.42,
  "intent_distribution": {
    "EXPANSION": 20.0,
    "GENERAL_UPDATE": 40.0,
    "STRATEGIC_PROBING": 15.0,
    "COST_PRESSURE": 25.0
  },
  "insight": "...",
  "segments": [],
  "drivers": {
    "growth_drivers": [],
    "risk_drivers": []
  }
}
```

### `POST /upload`

- Multipart upload key: `file`
- Supported: `.txt`, `.pdf`
- Returns same schema as `/analyze`

### `POST /compare`

Input:

```json
{
  "transcript_1": "...",
  "transcript_2": "..."
}
```

Output fields:

```json
{
  "transcript_1": { "...analyze response..." },
  "transcript_2": { "...analyze response..." },
  "signal_difference": {
    "from": "neutral",
    "to": "risk",
    "changed": true
  },
  "risk_delta": 15.0,
  "confidence_delta": -6.5,
  "trend": "UP",
  "comparison": "Risk increased by 15.00% compared to previous call."
}
```

---

## 9. DESIGN PRINCIPLES

- Conversation context is treated as first-class (not isolated sentence sentiment only).
- Intent-centric interpretation over generic sentiment labels.
- Explainability required in user outputs (drivers, confidence, volatility, rationale).
- Modular composition: parser, intent model, interaction model, scoring, predictor can be swapped.
- Backward compatibility via fallback logic is intentionally preserved.

Current practical reality:
- Production path is hybrid model + heuristic until missing V2 artifacts are trained and loaded.

---

## 10. IMPORTANT NOTES FOR NEXT AGENT

- Do not rewrite the system from scratch.
- Extend existing modules; preserve API contracts.
- Treat `backend/api/server.py` as source of truth for runtime outputs.
- Keep frontend aligned with current response keys (`prediction_explanation`, `/compare` deltas, etc.).
- When changing inference behavior, keep fallback compatibility unless explicitly removing legacy support.
- Prefer adding tests and evaluators before changing model/rule thresholds.
- If behavior is uncertain, mark as **Not implemented** rather than guessing.
