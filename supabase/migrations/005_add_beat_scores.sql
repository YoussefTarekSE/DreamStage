-- Beat candidate scoring: stores best candidate's score breakdown
ALTER TABLE projects
ADD COLUMN IF NOT EXISTS beat_scores jsonb;

COMMENT ON COLUMN projects.beat_scores IS
  'Quality scores from beat candidate system: total, tempo_match, key_match, energy_match, mood_match, dynamic_quality, rhythm_quality';
