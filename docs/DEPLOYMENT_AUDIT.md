# Deployment Audit

Sprint: Sprint 1 - Repo Cleanup + Deployment Audit

## Canonical Application Paths

- Canonical backend API entrypoint: `backend/api/server.py`
- Canonical frontend: `frontend_v2/`
- Legacy backend API module: `backend/financial_pragmatic_ai/api/server.py`
  - Do not extend this module.
  - Keep it only for historical reference until a later cleanup pass.

## Local Run Commands

Backend:

```bash
cd /Users/saroshnadaf/Documents/NLP_Proj/backend
pip install -r requirements.txt
uvicorn api.server:app --reload
```

Frontend V2:

```bash
cd /Users/saroshnadaf/Documents/NLP_Proj/frontend_v2
npm install
npm run dev
```

## Required Environment Variables

Backend:

- No backend environment variables are required for local startup at this time.
- `HF_HOME` is set in `backend/api/server.py` with a default of `/tmp/hf_cache`.
- Model artifacts are downloaded from Hugging Face at runtime.

Frontend V2:

- `VITE_SUPABASE_URL`
- `VITE_SUPABASE_ANON_KEY`
- `VITE_API_BASE_URL`
  - Local default: `http://localhost:8000`

## Backend API Surface

Canonical backend endpoints:

- `GET /health`
- `GET /version`
- `POST /analyze`
- `POST /upload`
- `POST /compare`

Health check behavior:

- `GET /health` returns static service status.
- It does not instantiate `EarningsCallAnalyzer`.
- It does not load model weights.

Version endpoint behavior:

- `GET /version` returns app name, canonical API entrypoint, model name, and hardening status.

## Railway Notes

Detected deployment clue:

- `backend/runtime.txt` pins Python to `python-3.11.9`.

Sprint 3 Railway deployment guide:

- See `docs/RAILWAY_BACKEND_DEPLOYMENT.md`.

Recommended Railway backend deployment path:

- Use Docker, not Nixpacks/Railpack, for the backend service.
- Service root/source directory: `backend`
- Dockerfile path from service root: `Dockerfile`
- Railway config file from service root: `railway.json`
- Health check path: `/health`

Recommended Railway backend settings:

- Root directory: `backend`
- Builder: Dockerfile
- Dockerfile path: `Dockerfile`
- Start command: use the Dockerfile `CMD`
- Ensure outbound network access is available for Hugging Face model downloads.

Validate deployed backend with:

```bash
curl https://YOUR-RAILWAY-BACKEND-DOMAIN/health
curl https://YOUR-RAILWAY-BACKEND-DOMAIN/version
```

## Docker Notes

Backend image build command from the repo root:

```bash
docker build --platform linux/amd64 -t financial-pragmatic-ai-backend ./backend
```

Backend container run command:

```bash
docker run --rm -p 8000:8000 financial-pragmatic-ai-backend
```

Health check command:

```bash
curl http://localhost:8000/health
```

Version check command:

```bash
curl http://localhost:8000/version
```

Docker behavior notes:

- The image runs the canonical backend entrypoint: `api.server:app`.
- The image targets `linux/amd64` because the pinned `torch==2.2.2+cpu` wheel is not available for Linux ARM64.
- The container binds Uvicorn to `0.0.0.0`.
- The container uses `${PORT:-8000}` so managed hosts can inject a port.
- Docker build should not trigger model loading.
- `GET /health` should stay lightweight and must not load model artifacts.
- The first real `/analyze` request may download Hugging Face model artifacts.
- The backend API image excludes local dataset, training, testing, and generated evaluation artifacts. It keeps evaluation utility source because current model modules import a helper from `financial_pragmatic_ai.evaluation.better_than_fin.utils`.

## Known Deployment Risks

- The backend imports the ML stack at app import time, but model inference objects are lazily instantiated only when analysis runs.
- First real `/analyze` request can be slow because Hugging Face model artifacts may need to download.
- Runtime depends on external Hugging Face availability for model artifacts.
- `backend/financial_pragmatic_ai/api/server.py` still exists and can confuse deployment if the wrong uvicorn target is used.
- `frontend_v2` depends on Supabase auth and an `analyses` table.
- `frontend_v2` previously hardcoded the backend URL; it now supports `VITE_API_BASE_URL` with a localhost fallback.
- CORS is currently permissive. This is convenient for local deployment testing but should be narrowed before a production portfolio launch.
- Some generated report artifacts exist in the workspace and should stay out of source control unless intentionally published.

## Current Limitations

- Backend Dockerfile is present at `backend/Dockerfile`.
- No multi-service container orchestration file is present.
- No CI workflow was detected in this pass.
- No production frontend host configuration was detected.
- No Supabase schema/migration file is present for the `analyses` table.
- `frontend/` still exists as the older CRA app, but `frontend_v2/` is the portfolio canonical UI.
- Model training and evaluation files are intentionally frozen for now.

## Next Recommended Deployment Steps

1. Create the Railway backend service using `docs/RAILWAY_BACKEND_DEPLOYMENT.md`.
2. Decide the frontend host target, such as Vercel, Netlify, or Railway static hosting.
3. Add a Supabase setup note or SQL schema for the `analyses` table.
4. Confirm `GET /health` works in the deployed backend before testing `/analyze`.
5. Configure `VITE_API_BASE_URL` in the frontend host to point at the deployed backend base URL.
6. Narrow CORS origins once the deployed frontend URL is known.
7. Add a short root README refresh that points portfolio visitors to `frontend_v2/`.
