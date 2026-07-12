-- 010: async job queue (applied 2026-07-12 via management API).
-- Beat generation moves from one fragile long HTTP request (browser aborts
-- at ~300s; neural cuts take 1-2 min) to submit-then-poll. The same table is
-- the future GPU-worker queue: a remote worker can claim 'queued' rows.
CREATE TABLE IF NOT EXISTS public.jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL,
  project_id UUID NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('beat', 'mix')),
  status TEXT NOT NULL DEFAULT 'queued'
    CHECK (status IN ('queued', 'running', 'done', 'failed')),
  payload JSONB DEFAULT '{}',
  result JSONB,
  error JSONB,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS jobs_project_idx
  ON public.jobs (project_id, created_at DESC);
-- RLS on with no policies: only the service role (the backend) touches jobs.
ALTER TABLE public.jobs ENABLE ROW LEVEL SECURITY;
