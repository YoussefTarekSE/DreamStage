alter table public.projects
  add column if not exists processed_vocal_key text,
  add column if not exists beat_key text,
  add column if not exists merged_key text;
