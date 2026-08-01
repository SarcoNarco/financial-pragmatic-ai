# Documentation Index

This directory contains the operating, deployment, and portfolio documentation for Financial Pragmatic AI.

| Document | Purpose |
| --- | --- |
| [Deployment Audit](DEPLOYMENT_AUDIT.md) | Canonical entrypoints, environment variables, deployment risks, and current architecture decisions. |
| [Railway Backend Deployment](RAILWAY_BACKEND_DEPLOYMENT.md) | Docker-based deployment of the canonical FastAPI backend to Railway. |
| [Frontend Vercel Deployment](FRONTEND_VERCEL_DEPLOYMENT.md) | Vite build settings, frontend environment variables, and post-deploy checks for Vercel. |
| [Supabase Setup](SUPABASE_SETUP.md) | Auth URL configuration, Vercel variables, database schema, and per-user RLS verification. |
| [Hugging Face Spaces Deployment](HUGGINGFACE_SPACES_DEPLOYMENT.md) | Optional fallback packaging and deployment path for a Docker Space. |
| [Portfolio Checklist](PORTFOLIO_CHECKLIST.md) | Final recruiter-demo, deployment, visual asset, and presentation checks. |
| [Portfolio Screenshots](assets/screenshots/) | Clean public captures of authentication, analysis, timeline/drivers, and comparison workflows. |
| [Full-Transcript Strategy](FULL_TRANSCRIPT_STRATEGY.md) | Representative segment sampling for long hosted-demo inputs, metadata, and benchmarking. |

Additional source-of-truth files:

- [`../supabase/schema.sql`](../supabase/schema.sql): `analyses` table, index, grants, and Row Level Security policies.
- [`../backend/api/server.py`](../backend/api/server.py): canonical FastAPI entrypoint.
- [`../frontend_v2/`](../frontend_v2/): canonical frontend application.
- [`../AGENT_HANDOFF.md`](../AGENT_HANDOFF.md): current engineering decisions and project context.
