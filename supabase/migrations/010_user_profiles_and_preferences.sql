-- Complete user profile and preference persistence for DreamStage.
-- Supabase Auth owns credential storage and password hashing in auth.users.

create table if not exists public.user_profiles (
  id uuid primary key default uuid_generate_v4(),
  user_id uuid not null references auth.users(id) on delete cascade,
  username text unique,
  display_name text,
  email text not null,
  google_id text,
  avatar_url text,
  bio text not null default '',
  role text not null default 'artist' check (role in ('artist', 'admin')),
  provider text not null default 'email' check (provider in ('email', 'google')),
  email_verified boolean not null default false,
  subscription jsonb not null default '{"plan":"free","status":"active"}'::jsonb,
  generation_credits int not null default 25 check (generation_credits >= 0),
  preferences jsonb not null default '{}'::jsonb,
  settings jsonb not null default '{}'::jsonb,
  last_login timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint username_format check (
    username is null or username ~ '^[A-Za-z0-9_]{3,20}$'
  )
);

create unique index if not exists idx_user_profiles_user_id
  on public.user_profiles (user_id);

create index if not exists idx_user_profiles_email
  on public.user_profiles (lower(email));

create or replace function public.is_username_available(candidate text)
returns boolean
language sql
security definer
set search_path = public
as $$
  select candidate ~ '^[A-Za-z0-9_]{3,20}$'
    and not exists (
      select 1
      from public.user_profiles
      where lower(username) = lower(candidate)
    );
$$;

grant execute on function public.is_username_available(text) to anon, authenticated;

alter table public.user_profiles enable row level security;

drop policy if exists "Users can read own profile" on public.user_profiles;
create policy "Users can read own profile"
  on public.user_profiles for select
  using (auth.uid() = user_id);

drop policy if exists "Users can insert own profile" on public.user_profiles;
create policy "Users can insert own profile"
  on public.user_profiles for insert
  with check (auth.uid() = user_id);

drop policy if exists "Users can update own profile" on public.user_profiles;
create policy "Users can update own profile"
  on public.user_profiles for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

alter table public.user_settings
  add column if not exists theme text not null default 'system' check (theme in ('dark', 'light', 'system')),
  add column if not exists accent_color text not null default 'emerald',
  add column if not exists animation_intensity text not null default 'balanced' check (animation_intensity in ('reduced', 'balanced', 'expressive')),
  add column if not exists reduce_motion boolean not null default false,
  add column if not exists notifications jsonb not null default '{"product":true,"generation":true,"security":true}'::jsonb,
  add column if not exists privacy jsonb not null default '{"profileVisibility":"private","trainingDataOptIn":false}'::jsonb,
  add column if not exists music_preferences jsonb not null default '{"genres":[],"moods":[]}'::jsonb;

drop trigger if exists trg_user_profiles_updated_at on public.user_profiles;
create trigger trg_user_profiles_updated_at
  before update on public.user_profiles
  for each row execute procedure public.handle_updated_at();

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  provider_name text;
begin
  provider_name := coalesce(new.raw_app_meta_data ->> 'provider', 'email');

  insert into public.user_settings (user_id)
  values (new.id)
  on conflict do nothing;

  insert into public.user_profiles (
    user_id,
    username,
    email,
    display_name,
    avatar_url,
    provider,
    google_id,
    email_verified,
    last_login
  )
  values (
    new.id,
    nullif(lower(new.raw_user_meta_data ->> 'username'), ''),
    coalesce(new.email, ''),
    coalesce(new.raw_user_meta_data ->> 'display_name', new.raw_user_meta_data ->> 'full_name'),
    new.raw_user_meta_data ->> 'avatar_url',
    case when provider_name = 'google' then 'google' else 'email' end,
    case when provider_name = 'google' then new.raw_user_meta_data ->> 'provider_id' else null end,
    new.email_confirmed_at is not null,
    now()
  )
  on conflict (user_id) do update set
    email = excluded.email,
    display_name = coalesce(public.user_profiles.display_name, excluded.display_name),
    avatar_url = coalesce(public.user_profiles.avatar_url, excluded.avatar_url),
    provider = excluded.provider,
    google_id = coalesce(public.user_profiles.google_id, excluded.google_id),
    email_verified = excluded.email_verified,
    last_login = now();

  return new;
end;
$$;

drop trigger if exists trg_on_auth_user_created on auth.users;
create trigger trg_on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();
