-- Migration: add vocal_analysis column to projects table
-- Used by: beat router (stores analysis for mixer dynamic EQ carve)
-- Stores: JSON with centroid, key, mode, emotion, valence fields

ALTER TABLE projects
ADD COLUMN IF NOT EXISTS vocal_analysis jsonb;

COMMENT ON COLUMN projects.vocal_analysis IS
  'Vocal spectral analysis from beat generation. Used by mixer for dynamic EQ carving.';
