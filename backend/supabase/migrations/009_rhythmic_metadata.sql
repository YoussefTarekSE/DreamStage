-- 009: rhythmic_metadata column referenced by beat.py's project update.
-- The code shipped before the column existed; a missing column made the
-- PostgREST update fail and 500'd every beat generation (found 2026-07-12).
-- Applied 2026-07-12 via the management API.
ALTER TABLE public.projects ADD COLUMN IF NOT EXISTS rhythmic_metadata JSONB;
-- Also missing (same schema-drift audit, 2026-07-12):
ALTER TABLE public.projects ADD COLUMN IF NOT EXISTS tempo_bpm INTEGER;
ALTER TABLE public.projects ADD COLUMN IF NOT EXISTS beat_genre_history JSONB DEFAULT '[]';
