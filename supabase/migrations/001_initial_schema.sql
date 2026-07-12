-- Enable UUID extension
create extension if not exists "uuid-ossp";

-- ============================================================
-- VOICE PROFILES
-- Stores extracted vocal characteristics from voice training.
-- Structured data only — no audio files stored here.
-- ============================================================
create table public.voice_profiles (
  id          uuid primary key default uuid_generate_v4(),
  user_id     uuid not null references auth.users(id) on delete cascade,
  language    text not null check (language in ('en', 'ar')),
  min_freq_hz float,
  max_freq_hz float,
  tone_type   text,             -- e.g. 'warm', 'bright', 'raspy'
  tempo_bpm   float,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now(),
  unique (user_id)
);

alter table public.voice_profiles enable row level security;

create policy "Users can read own voice profile"
  on public.voice_profiles for select
  using (auth.uid() = user_id);

create policy "Users can insert own voice profile"
  on public.voice_profiles for insert
  with check (auth.uid() = user_id);

create policy "Users can update own voice profile"
  on public.voice_profiles for update
  using (auth.uid() = user_id);


-- ============================================================
-- PROJECTS
-- One row per song. Tracks status through the pipeline.
-- Only final_mp3_key lives long-term in R2.
-- All intermediary R2 keys are deleted after final mix.
-- ============================================================
create table public.projects (
  id              uuid primary key default uuid_generate_v4(),
  user_id         uuid not null references auth.users(id) on delete cascade,
  name            text not null default 'Untitled Song',
  status          text not null default 'created' check (
                    status in (
                      'created',
                      'voice_training',
                      'recording',
                      'processing_vocal',
                      'beat_generation',
                      'coaching',
                      'mixing',
                      'completed'
                    )
                  ),
  language        text not null default 'en' check (language in ('en', 'ar')),

  -- Autotune setting chosen by artist
  autotune_level  text default 'subtle' check (
                    autotune_level in ('natural', 'subtle', 'modern', 'heavy')
                  ),

  -- Beat generation tracking (max 3 attempts per session)
  beat_attempts   int not null default 0,

  -- R2 storage keys — only final files kept long-term
  final_mp3_key   text,
  final_wav_key   text,

  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now()
);

alter table public.projects enable row level security;

create policy "Users can read own projects"
  on public.projects for select
  using (auth.uid() = user_id);

create policy "Users can insert own projects"
  on public.projects for insert
  with check (auth.uid() = user_id);

create policy "Users can update own projects"
  on public.projects for update
  using (auth.uid() = user_id);

create policy "Users can delete own projects"
  on public.projects for delete
  using (auth.uid() = user_id);


-- ============================================================
-- USER SETTINGS
-- Stores language preference and notification settings.
-- ============================================================
create table public.user_settings (
  user_id       uuid primary key references auth.users(id) on delete cascade,
  language      text not null default 'en' check (language in ('en', 'ar')),
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now()
);

alter table public.user_settings enable row level security;

create policy "Users can read own settings"
  on public.user_settings for select
  using (auth.uid() = user_id);

create policy "Users can insert own settings"
  on public.user_settings for insert
  with check (auth.uid() = user_id);

create policy "Users can update own settings"
  on public.user_settings for update
  using (auth.uid() = user_id);


-- ============================================================
-- AUTO-UPDATE updated_at on row change
-- ============================================================
create or replace function public.handle_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create trigger trg_voice_profiles_updated_at
  before update on public.voice_profiles
  for each row execute procedure public.handle_updated_at();

create trigger trg_projects_updated_at
  before update on public.projects
  for each row execute procedure public.handle_updated_at();

create trigger trg_user_settings_updated_at
  before update on public.user_settings
  for each row execute procedure public.handle_updated_at();


-- ============================================================
-- AUTO-CREATE user_settings row on signup
-- ============================================================
create or replace function public.handle_new_user()
returns trigger language plpgsql security definer as $$
begin
  insert into public.user_settings (user_id)
  values (new.id)
  on conflict do nothing;
  return new;
end;
$$;

create trigger trg_on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();
