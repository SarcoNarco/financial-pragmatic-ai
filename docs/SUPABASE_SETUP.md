# Supabase Setup

Supabase provides authentication and per-user analysis history for the canonical `frontend_v2` application. Railway does not store users or analysis history.

## Create the Project

1. Create a project in the Supabase dashboard.
2. Open **Project Settings -> API**.
3. Copy the Project URL and the publishable/anon key.
4. Never use the service-role key in frontend code or Vercel environment variables.

## Create the Analyses Table

1. Open **SQL Editor** in the Supabase dashboard.
2. Copy and run `supabase/schema.sql` from this repository.
3. In **Table Editor**, confirm that `public.analyses` exists.
4. Confirm Row Level Security is enabled for the table.
5. Confirm these four policies exist:
   - `analyses_select_own`
   - `analyses_insert_own`
   - `analyses_update_own`
   - `analyses_delete_own`

The policies allow authenticated users to read and modify only rows whose `user_id` matches their authenticated Supabase user ID.

## Configure Authentication URLs

Open **Authentication -> URL Configuration** and set:

Site URL:

```text
https://financial-pragmatic-ai.vercel.app
```

Redirect URLs:

```text
https://financial-pragmatic-ai.vercel.app
https://financial-pragmatic-ai.vercel.app/*
http://localhost:5173
http://localhost:5173/*
```

These URLs allow production email confirmations to return to Vercel and local confirmations to return to the Vite development server.

## Configure Vercel

Add these variables to the Vercel project:

```text
VITE_SUPABASE_URL=YOUR_SUPABASE_PROJECT_URL
VITE_SUPABASE_ANON_KEY=YOUR_SUPABASE_PUBLISHABLE_OR_ANON_KEY
```

Both variables are required. Vite injects them at build time, so redeploy the frontend after adding or changing either value.

For local development, add the same values to `frontend_v2/.env.local`. That file is ignored by Git.

## Verify the Setup

1. Sign up or sign in through the deployed frontend.
2. Run one transcript analysis.
3. Confirm the result remains visible.
4. Confirm a matching row appears in `public.analyses`.
5. Confirm the analysis appears under **Recent Analyses** after refresh.
6. Sign in as a different test user and confirm the first user's rows are not visible.

## Troubleshooting

### Login works but history does not save

- Confirm `public.analyses` exists.
- Check the browser console for the concise Supabase save error.
- Confirm the signed-in user ID matches the row's `user_id`.
- Confirm the insert and select RLS policies are present.

### Email confirmation redirects to the wrong URL

- Confirm the production Site URL is `https://financial-pragmatic-ai.vercel.app`.
- Confirm the production and local wildcard redirect URLs are listed under Authentication URL Configuration.

### The analyses table is missing

Run `supabase/schema.sql` in the Supabase SQL Editor. The script is safe to rerun and recreates the expected policies.

### RLS blocks inserts

- Confirm the user is authenticated.
- Confirm `analyses_insert_own` exists.
- Confirm the frontend insert includes `user_id: session.user.id`.
- Do not disable RLS as a workaround.

### Vercel still uses old environment values

Vite environment variables are build-time values. Trigger a new Vercel deployment after changing any `VITE_*` variable.
