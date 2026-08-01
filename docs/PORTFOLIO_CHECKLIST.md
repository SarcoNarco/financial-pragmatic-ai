# Portfolio Release Checklist

Use this list before sharing Financial Pragmatic AI with a recruiter, interviewer, or portfolio reviewer.

## Live Application

- [x] Live Vercel frontend opens successfully.
- [x] Railway `/health` endpoint responds successfully.
- [x] Railway `/version` endpoint reports the canonical API and model.
- [x] Sample transcript analysis has been tested end to end.
- [x] Supabase authentication and saved history have been tested.
- [x] README live links point to the active services.

## Deployment Configuration

- [x] Vercel includes `VITE_API_BASE_URL`, `VITE_SUPABASE_URL`, and `VITE_SUPABASE_ANON_KEY`.
- [ ] Confirm Supabase Site URL and production/local redirect URLs before a public demo.
- [ ] Check current Railway usage, cost, and spending limits before sharing widely.
- [ ] Confirm Railway remains at one service and one replica unless scaling is intentional.
- [ ] Confirm no service-role key or other secret appears in frontend variables or committed files.

## Demo Readiness

- [x] Add a clean live frontend authentication screenshot.
- [x] Add a completed analysis dashboard screenshot with distribution and timeline.
- [x] Add a saved-analysis comparison screenshot.
- [x] Confirm public screenshots exclude account identity.
- [ ] Test the live demo once in a private browser session.
- [ ] Confirm the first-analysis wait message remains visible during a cold backend start.
- [ ] Optional: record a short demo video or GIF.

## Portfolio Presentation

- [x] README explains the problem, architecture, features, setup, deployment, and limitations.
- [x] Official custom-system and FinBERT baseline metrics are documented.
- [x] The research/demo disclaimer is visible.
- [ ] Add the published paper link when a stable public URL is available.
- [ ] Optional: prepare a concise LinkedIn post.
- [ ] Optional: add a resume bullet with the deployed architecture and verified evaluation improvement.

## Final Technical Check

- [ ] Re-run the frontend production build after any application change.
- [ ] Run `git diff --check` before committing documentation changes.
- [ ] Confirm `main` is clean and pushed.
- [ ] Confirm Vercel and Railway deployments correspond to the latest intended commit.
