create table if not exists public.analyses (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  transcript text not null,
  signal text,
  score double precision,
  distribution jsonb not null default '{}'::jsonb,
  growth_drivers jsonb not null default '[]'::jsonb,
  risk_drivers jsonb not null default '[]'::jsonb,
  timeline jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists analyses_user_created_at_idx
  on public.analyses (user_id, created_at desc);

alter table public.analyses enable row level security;

drop policy if exists analyses_select_own on public.analyses;
create policy analyses_select_own
  on public.analyses
  for select
  to authenticated
  using ((select auth.uid()) = user_id);

drop policy if exists analyses_insert_own on public.analyses;
create policy analyses_insert_own
  on public.analyses
  for insert
  to authenticated
  with check ((select auth.uid()) = user_id);

drop policy if exists analyses_update_own on public.analyses;
create policy analyses_update_own
  on public.analyses
  for update
  to authenticated
  using ((select auth.uid()) = user_id)
  with check ((select auth.uid()) = user_id);

drop policy if exists analyses_delete_own on public.analyses;
create policy analyses_delete_own
  on public.analyses
  for delete
  to authenticated
  using ((select auth.uid()) = user_id);

grant select, insert, update, delete on public.analyses to authenticated;
