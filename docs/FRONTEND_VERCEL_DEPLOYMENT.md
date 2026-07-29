# Frontend Vercel Deployment

This guide deploys the canonical `frontend_v2` application to Vercel while the canonical FastAPI backend remains on Railway.

## Architecture

- Frontend host: Vercel
- Frontend source: `frontend_v2/`
- Backend host: Railway
- Backend URL: `https://financial-pragmatic-ai-production.up.railway.app`
- Authentication and analysis history: Supabase

## Import the Project

1. In Vercel, create a new project and import the GitHub repository.
2. Set the project root directory to:

```text
frontend_v2
```

3. Use the Vite framework preset.
4. Confirm these build settings:

```text
Build Command: npm run build
Output Directory: dist
```

The committed `frontend_v2/vercel.json` provides a minimal SPA rewrite so direct navigation and browser refreshes resolve to the Vite application.

## Environment Variables

Add these required variables in the Vercel project settings:

```text
VITE_API_BASE_URL=https://financial-pragmatic-ai-production.up.railway.app
VITE_SUPABASE_URL=YOUR_SUPABASE_PROJECT_URL
VITE_SUPABASE_ANON_KEY=YOUR_SUPABASE_ANON_KEY
```

Vite environment variables are injected at build time. Changing any `VITE_*` value requires a new Vercel deployment.

Do not add a Supabase service-role key to Vercel or expose it through a `VITE_*` variable. The browser application should use only the public Supabase URL and anon key.

Complete the schema, RLS policy, and authentication redirect setup in `docs/SUPABASE_SETUP.md` before testing history persistence.

## Demo Runtime Behavior

- `GET /health` is checked once after authentication to show backend availability.
- `POST /analyze` runs Torch and Transformers inference on Railway.
- A demo analysis may take up to about 60 seconds.
- The frontend allows up to 90 seconds before aborting an analysis request.
- The frontend does not automatically retry `/analyze`.

## Post-Deploy Verification

1. Open the Vercel URL and confirm the authentication screen loads.
2. Sign up or sign in through Supabase.
3. Confirm the header reports `Backend online`.
4. Click `Use sample transcript`.
5. Run one analysis and confirm the elapsed timer appears.
6. Confirm the result renders without a browser timeout.
7. Confirm the analysis appears in recent history.
8. Create a second analysis and verify Compare mode guidance and selection.
9. Refresh the deployed page and confirm the SPA still loads.

## CORS Note

The Railway backend currently allows permissive CORS for deployment testing. After the final Vercel domain is known, a future hardening pass should restrict backend CORS origins to the production Vercel frontend and approved local development origins.
