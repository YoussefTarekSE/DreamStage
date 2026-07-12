-- 007: expand projects.autotune_level to the 8 user-facing vocal styles.
--
-- The original CHECK only allowed ('natural','subtle','modern','heavy'), so when
-- the app added the styles Modern Pop / R&B / Rap / Melodic / No Autotune, any
-- insert with a new value (e.g. 'rnb') was rejected by Postgres and the studio
-- endpoint returned "Couldn't create your project."
--
-- Widen the constraint. Additive — existing rows (incl. legacy 'modern') stay
-- valid; no data migration needed. Idempotent via DROP ... IF EXISTS.
--
-- NOTE: run as two separate statements (the Supabase Management API
-- /database/query endpoint rejects anonymous DO/PL-pgSQL blocks). The constraint
-- name 'projects_autotune_level_check' is Postgres's auto-generated name for the
-- original inline column CHECK in 001_initial_schema.sql.

alter table projects drop constraint if exists projects_autotune_level_check;

alter table projects add constraint projects_autotune_level_check
  check (autotune_level in (
    'natural', 'subtle', 'modern_pop', 'rnb',
    'rap', 'melodic', 'heavy', 'none',
    'modern'   -- legacy value kept valid for older projects
  ));
