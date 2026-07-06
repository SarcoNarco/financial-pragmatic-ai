---
title: Financial Pragmatic AI Backend
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Financial Pragmatic AI Backend

FastAPI backend for Financial Pragmatic AI, packaged as a Hugging Face Docker Space.

## API Endpoints

- `GET /health`
- `GET /version`
- `POST /analyze`
- `POST /upload`
- `POST /compare`

## Runtime Notes

- The backend runs the canonical API entrypoint: `api.server:app`.
- The container binds to `0.0.0.0`.
- Hugging Face Spaces serves the app on port `7860`.
- `GET /health` and `GET /version` are lightweight checks.
- The first `/analyze` request may be slow because model artifacts download from Hugging Face at runtime.

## Frontend Configuration

Set `frontend_v2` to call this Space by configuring:

```text
VITE_API_BASE_URL=https://sarconarco-financial-pragmatic-ai-backend.hf.space
```
