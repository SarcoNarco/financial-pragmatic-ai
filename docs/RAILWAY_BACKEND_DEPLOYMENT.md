# Railway Backend Deployment

This guide deploys the canonical backend only.

## Canonical Service

- Backend service root/source directory: `backend`
- Dockerfile path from that service root: `Dockerfile`
- Railway config file: `backend/railway.json`
- Runtime entrypoint inside the image: `api.server:app`
- Legacy API module: `financial_pragmatic_ai/api/server.py`
  - Do not deploy this module.
  - Do not point Railway at this module.

## Railway Service Settings

Create a Railway service from the GitHub repo and set the service root/source directory to:

```text
backend
```

Use the Dockerfile builder. With service root set to `backend`, Railway can use:

```text
Dockerfile
```

The committed `backend/railway.json` sets:

- Builder: `DOCKERFILE`
- Dockerfile path: `Dockerfile`
- Health check path: `/health`
- Health check timeout: `300`
- Restart policy: `ON_FAILURE`
- Restart max retries: `3`

Do not create a Railway service from the repo root unless you also adjust the Docker build context. The backend Dockerfile expects `backend/` as its build context.

## Start Command Behavior

The container command is defined in `backend/Dockerfile`:

```bash
uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-7860}
```

Railway should inject `PORT`. If it does not, the container falls back to `7860` for Hugging Face Spaces compatibility.

No separate Railway start command is required.

## Environment Variables

Required backend environment variables:

- None currently required.

Optional backend environment variables:

```text
PORT=8000
HF_HOME=/tmp/hf_cache
```

Notes:

- `PORT` is normally injected by Railway.
- `HF_HOME` defaults to `/tmp/hf_cache` in `backend/api/server.py` and in the Docker image.
- No fake secrets are required for the backend service.

## Health and Version Checks

After Railway deploys the service, validate it with:

```bash
curl https://YOUR-RAILWAY-BACKEND-DOMAIN/health
curl https://YOUR-RAILWAY-BACKEND-DOMAIN/version
```

Expected `/health` response:

```json
{"status":"ok","service":"financial-pragmatic-ai-api"}
```

Expected `/version` response:

```json
{"app":"financial-pragmatic-ai-api","api_entrypoint":"backend/api/server.py","model":"SarcoNarco/finbert_intent_v3","status":"portfolio-hardening"}
```

`GET /health` is intentionally lightweight and must not load model artifacts.

## Model Download Behavior

The Docker image does not include model weight files.

At runtime:

- The first real `/analyze` request may be slow.
- Model artifacts download from Hugging Face at runtime.
- The deployed service must allow outbound network access.
- `GET /health` and `GET /version` should work before any model download.

## Local Verification Before Deploying

From the repo root:

```bash
docker build --platform linux/amd64 -t financial-pragmatic-ai-backend ./backend
docker run --rm -p 8000:8000 financial-pragmatic-ai-backend
```

In another shell:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/version
BASE_URL=http://localhost:8000 python backend/scripts/smoke_backend.py
```

On Apple Silicon, Docker may print an expected platform warning because the image targets `linux/amd64` for compatibility with the pinned CPU PyTorch wheel.

## Troubleshooting

### Railway builds the wrong app

Confirm the Railway service root/source directory is:

```text
backend
```

The root `frontend/` and `frontend_v2/` directories are not backend services.

### Railway cannot find the Dockerfile

With service root set to `backend`, the Dockerfile path should be:

```text
Dockerfile
```

The file should resolve to:

```text
backend/Dockerfile
```

### Container starts but health check fails

Check deploy logs for Uvicorn startup. The expected command is:

```bash
uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-8000}
```

Confirm Railway is injecting `PORT` and that `/health` is configured as the health check path.

### First analysis request is slow

This can be expected on a fresh deployment because model artifacts are downloaded from Hugging Face on first inference path use.

### Hugging Face download fails

Confirm the deployed service has outbound network access. If needed later, configure a persistent cache or deploy with a warm-up strategy, but do not add model files to git.
