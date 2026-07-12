-- Producer Cuts, beat telemetry, and tempo persistence.

alter table public.projects
  add column if not exists producer_cuts jsonb not null default '[]'::jsonb,
  add column if not exists beat_genre_history jsonb not null default '[]'::jsonb,
  add column if not exists tempo_bpm float,
  add column if not exists rhythmic_metadata jsonb,
  add column if not exists coach_feedback jsonb;

comment on column public.projects.producer_cuts is
  'Every generated Producer Cut for this project; keeps beat history deterministic.';

comment on column public.projects.tempo_bpm is
  'Last generated vocal-derived tempo used by beat generation and final mixing.';

create table if not exists public.beat_telemetry (
  id uuid primary key default uuid_generate_v4(),
  project_id uuid references public.projects(id) on delete cascade,
  tier_used text not null,
  tier_attempts jsonb not null default '[]'::jsonb,
  duration_ms int,
  analysis_summary jsonb,
  selected_genre text,
  selected_bpm int,
  candidate_scores jsonb,
  final_score float,
  success boolean not null default false,
  failure_reason text,
  created_at timestamptz not null default now()
);

alter table public.beat_telemetry enable row level security;

drop policy if exists "Users can read own beat telemetry" on public.beat_telemetry;
create policy "Users can read own beat telemetry"
  on public.beat_telemetry for select
  using (
    exists (
      select 1 from public.projects p
      where p.id = beat_telemetry.project_id and p.user_id = auth.uid()
    )
  );
