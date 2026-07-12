-- Daily usage limits and post-export feedback.

create table if not exists public.usage_events (
  id uuid primary key default uuid_generate_v4(),
  user_id uuid not null references auth.users(id) on delete cascade,
  project_id uuid references public.projects(id) on delete set null,
  action text not null check (action in ('project', 'beat_generation', 'mix')),
  created_at timestamptz not null default now()
);

create index if not exists idx_usage_events_user_action_created
  on public.usage_events (user_id, action, created_at desc);

alter table public.usage_events enable row level security;

drop policy if exists "Users can read own usage events" on public.usage_events;
create policy "Users can read own usage events"
  on public.usage_events for select
  using (auth.uid() = user_id);

create table if not exists public.project_feedback (
  id uuid primary key default uuid_generate_v4(),
  user_id uuid not null references auth.users(id) on delete cascade,
  project_id uuid not null references public.projects(id) on delete cascade,
  beat_quality int check (beat_quality between 1 and 5),
  vocal_preservation int check (vocal_preservation between 1 and 5),
  overall_satisfaction int check (overall_satisfaction between 1 and 5),
  created_at timestamptz not null default now()
);

create index if not exists idx_project_feedback_project
  on public.project_feedback (project_id, created_at desc);

alter table public.project_feedback enable row level security;

drop policy if exists "Users can insert own project feedback" on public.project_feedback;
create policy "Users can insert own project feedback"
  on public.project_feedback for insert
  with check (auth.uid() = user_id);

drop policy if exists "Users can read own project feedback" on public.project_feedback;
create policy "Users can read own project feedback"
  on public.project_feedback for select
  using (auth.uid() = user_id);
