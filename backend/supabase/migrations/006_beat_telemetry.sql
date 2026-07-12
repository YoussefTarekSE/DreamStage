-- Beat generation telemetry
-- Records every beat generation event: tier used, duration, analysis, score, success/failure.
-- Used by /admin/beat-metrics to compute real production distributions.

CREATE TABLE IF NOT EXISTS beat_telemetry (
    id           UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    project_id   UUID        NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT NOW() NOT NULL,

    -- Which tier finally produced the beat
    tier_used    TEXT,        -- 'musicgen_local' | 'musicgen_hf_api' | 'musicgen_gradio'
                              -- | '<synthesizer_genre>' | 'failed'

    -- Per-tier attempt log: [{tier, name, duration_ms, success, reason?}]
    tier_attempts JSONB,

    -- Wall-clock time from start to final output (ms)
    duration_ms  INTEGER,

    -- Compact snapshot of the vocal analysis that drove this generation
    analysis_summary JSONB,  -- {tempo, key, mode, emotion, valence, rms}

    selected_genre TEXT,
    selected_bpm   INTEGER,

    -- Final beat scorer output (total + per-dimension breakdown)
    candidate_scores JSONB,
    final_score      NUMERIC(5, 2),

    success        BOOLEAN NOT NULL DEFAULT TRUE,
    failure_reason TEXT
);

CREATE INDEX IF NOT EXISTS beat_telemetry_created_at_idx  ON beat_telemetry (created_at DESC);
CREATE INDEX IF NOT EXISTS beat_telemetry_project_id_idx  ON beat_telemetry (project_id);
CREATE INDEX IF NOT EXISTS beat_telemetry_tier_used_idx   ON beat_telemetry (tier_used);
CREATE INDEX IF NOT EXISTS beat_telemetry_success_idx     ON beat_telemetry (success);
