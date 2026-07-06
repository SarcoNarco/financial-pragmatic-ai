# Hugging Face Spaces Backend Deployment

Hugging Face Spaces is the preferred free and ML-native deployment target for the Financial Pragmatic AI backend demo.

Spaces is a good fit because:

- It supports Docker SDK apps for custom FastAPI services.
- It is designed for ML demos and Hugging Face-hosted model artifacts.
- It provides a public portfolio URL with minimal infrastructure work.
- It can sleep or cold start, which is acceptable for a free demo backend.

## Canonical Backend

- Canonical backend entrypoint: `backend/api/server.py`
- Docker runtime command: `uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-7860}`
- Legacy backend API: `backend/financial_pragmatic_ai/api/server.py`
  - Do not deploy or extend this legacy module.

## Recommended Space Settings

Create a new Hugging Face Space with:

- Owner: `SarcoNarco`
- Space name: `financial-pragmatic-ai-backend`
- SDK: Docker
- App port: `7860`
- Hardware: CPU Basic initially
- Visibility: Public for portfolio demo

The Space README frontmatter should include:

```yaml
---
title: Financial Pragmatic AI Backend
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
```

## Files To Push To The Space

The Space repo should contain a backend runtime package, not the whole project workspace.

Required files and folders:

- `README.md` from `deployment/huggingface/README.md`
- `Dockerfile` from `backend/Dockerfile`
- `.dockerignore` from `backend/.dockerignore`
- `requirements.txt` from `backend/requirements.txt`
- `api/`
- `financial_pragmatic_ai/`

Excluded from the Space package:

- local datasets
- training code
- tests
- generated report assets
- local virtual environments
- model weight files

Use the helper script from the repo root:

```bash
bash scripts/prepare_hf_space_backend.sh
```

It creates:

```text
dist/hf-space-backend
```

Then push the contents of that folder to the Hugging Face Space repo.

## Local Verification Before Deploying

From the project repo root:

```bash
docker build --platform linux/amd64 -t financial-pragmatic-ai-backend ./backend
docker run --rm -p 7860:7860 -e PORT=7860 financial-pragmatic-ai-backend
```

In another shell:

```bash
curl http://localhost:7860/health
curl http://localhost:7860/version
BASE_URL=http://localhost:7860 python backend/scripts/smoke_backend.py
```

Expected `/health` response:

```json
{"status":"ok","service":"financial-pragmatic-ai-api"}
```

Expected `/version` response:

```json
{"app":"financial-pragmatic-ai-api","api_entrypoint":"backend/api/server.py","model":"SarcoNarco/finbert_intent_v3","status":"portfolio-hardening"}
```

`GET /health` must not load model artifacts.

## Frontend V2 Configuration

After the Space is deployed, configure `frontend_v2` with:

```text
VITE_API_BASE_URL=https://sarconarco-financial-pragmatic-ai-backend.hf.space
```

Then rebuild/redeploy the frontend so it calls the Space backend.

## Runtime Behavior

- Hugging Face Spaces should inject `PORT`, but the backend defaults to `7860`.
- The container must bind to `0.0.0.0`.
- Model artifacts download from Hugging Face at runtime.
- The first `/analyze` request may be slow.
- The Space may sleep and cold start on the free tier.
- The Space must allow outbound network access for model downloads.

## Troubleshooting

### Build Fails Due To Large Dependencies

The image includes PyTorch, Transformers, Pandas, Datasets, and PDF parsing dependencies. CPU Basic builds may take time. Keep datasets, training outputs, and generated reports out of the Space package.

### App Is Not Reachable

Confirm:

- README frontmatter has `sdk: docker`.
- README frontmatter has `app_port: 7860`.
- Docker command binds to `0.0.0.0`.
- The app runs on `${PORT:-7860}`.

### Model Download Is Slow

This is expected on the first `/analyze` request. Validate `/health` and `/version` first, then test `/analyze`.

### CORS Or Frontend Connection Issue

Confirm `frontend_v2` uses:

```text
VITE_API_BASE_URL=https://sarconarco-financial-pragmatic-ai-backend.hf.space
```

The backend currently allows all origins for portfolio/demo simplicity.

### Memory Issue During First Inference

Start on CPU Basic, then upgrade hardware if first inference fails due to memory. Do not commit model binaries to git as a workaround.
