"""
Feature extraction pipeline for DreamStage training datasets.

Extracts 136-dimensional feature vectors from each audio file and saves
them as a Parquet file for fast training. The feature schema must match
backend/app/services/ml_analyzer.py:extract_feature_vector() exactly —
that function runs the same extraction at inference time.

Features:
  MFCC × 40         mean + std → 80 dims
  Chroma × 12       mean + std → 24 dims
  Spectral centroid  mean + std →  2 dims
  Spectral rolloff   mean + std →  2 dims
  Zero crossing rate mean + std →  2 dims
  Spectral contrast ×7 mean+std → 14 dims
  Tonnetz × 6       mean + std → 12 dims
  ─────────────────────────────────────────
  Total:                          136 dims

Additional metadata columns:
  file_path, dataset, genre_label, mood_label, bpm_approx, split

Usage:
  python extract_features.py --dataset gtzan
  python extract_features.py --dataset fma_small --workers 4
  python extract_features.py --dataset all --output features_all.parquet
"""
from __future__ import annotations

import os
import sys
import json
import logging
import argparse
import warnings
import traceback
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import librosa

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

DATA_DIR    = Path(os.environ.get("DATA_DIR",    Path(__file__).resolve().parents[1] / "data"))
OUTPUT_DIR  = Path(os.environ.get("OUTPUT_DIR",  Path(__file__).resolve().parents[1] / "features"))
SAMPLE_RATE = 22050   # standard for librosa feature extraction
CLIP_SEC    = 30      # seconds to use per clip (trim or pad)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── GTZAN genre → mood + valence ground truth ─────────────────────────────────
# Derived from music psychology literature and crowd-sourced annotation studies.
# Reference: Eerola & Vuoskoski (2011) — "A comparison of the discrete and
# dimensional models of emotion in music"

GTZAN_GENRE_META = {
    "blues":     {"mood": "melancholic", "valence": 0.32, "arousal": 0.40, "energy": "medium"},
    "classical": {"mood": "intimate",    "valence": 0.55, "arousal": 0.25, "energy": "low"},
    "country":   {"mood": "uplifting",   "valence": 0.65, "arousal": 0.55, "energy": "medium"},
    "disco":     {"mood": "euphoric",    "valence": 0.80, "arousal": 0.85, "energy": "high"},
    "hiphop":    {"mood": "energetic",   "valence": 0.50, "arousal": 0.70, "energy": "high"},
    "jazz":      {"mood": "smooth",      "valence": 0.60, "arousal": 0.45, "energy": "medium"},
    "metal":     {"mood": "dark",        "valence": 0.25, "arousal": 0.90, "energy": "high"},
    "pop":       {"mood": "uplifting",   "valence": 0.72, "arousal": 0.65, "energy": "medium"},
    "reggae":    {"mood": "smooth",      "valence": 0.68, "arousal": 0.50, "energy": "medium"},
    "rock":      {"mood": "energetic",   "valence": 0.48, "arousal": 0.78, "energy": "high"},
}


# ── Core feature extraction ───────────────────────────────────────────────────

def extract_features_from_array(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract 136-dimensional feature vector from audio array.
    Identical to ml_analyzer.extract_feature_vector() — keep in sync.
    """
    try:
        mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        chroma   = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zcr      = librosa.feature.zero_crossing_rate(y)[0]
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
        y_harm   = librosa.effects.harmonic(y)
        tonnetz  = librosa.feature.tonnetz(y=y_harm, sr=sr)

        return np.concatenate([
            np.mean(mfcc, axis=1),     np.std(mfcc, axis=1),
            np.mean(chroma, axis=1),   np.std(chroma, axis=1),
            [np.mean(centroid)],       [np.std(centroid)],
            [np.mean(rolloff)],        [np.std(rolloff)],
            [np.mean(zcr)],            [np.std(zcr)],
            np.mean(contrast, axis=1), np.std(contrast, axis=1),
            np.mean(tonnetz, axis=1),  np.std(tonnetz, axis=1),
        ], dtype=np.float32)
    except Exception:
        return np.full(136, np.nan, dtype=np.float32)


def _load_and_trim(path: str) -> Optional[np.ndarray]:
    """Load audio, resample to SAMPLE_RATE, trim/pad to CLIP_SEC seconds."""
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=CLIP_SEC)
        target_len = SAMPLE_RATE * CLIP_SEC
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
        return y
    except Exception:
        return None


# ── Dataset-specific loaders ──────────────────────────────────────────────────

def _iter_gtzan() -> list[dict]:
    """
    Iterate GTZAN genre collection.
    Expected structure: gtzan/genres/{genre}/{genre}.00001.wav
    """
    gtzan_dir = DATA_DIR / "gtzan"
    if not gtzan_dir.exists():
        raise FileNotFoundError(f"GTZAN not found at {gtzan_dir}. Run download.py first.")

    records = []
    genres_root = gtzan_dir / "genres"
    if not genres_root.exists():
        # Some archives unpack directly into gtzan/ with no genres/ subfolder
        genres_root = gtzan_dir
    for genre_dir in sorted(genres_root.iterdir()):
        if not genre_dir.is_dir():
            continue
        genre = genre_dir.name
        if genre not in GTZAN_GENRE_META:
            continue
        meta = GTZAN_GENRE_META[genre]
        # Exclude macOS AppleDouble sidecar files (._*.wav, always 211 bytes)
        for wav in sorted(f for f in genre_dir.glob("*.wav")
                          if not f.name.startswith("._") and f.stat().st_size > 10_000):
            records.append({
                "file_path":   str(wav),
                "dataset":     "gtzan",
                "genre_label": genre,
                "mood_label":  meta["mood"],
                "valence":     meta["valence"],
                "arousal":     meta["arousal"],
                "energy_label": meta["energy"],
                "split":       "train" if int(wav.stem.split(".")[-1]) < 80 else "test",
            })
    return records


def _iter_fma_small(fma_metadata_dir: Path) -> list[dict]:
    """
    Iterate FMA-small with metadata labels.
    Requires: fma_small/ (audio) + fma_metadata/ (CSV labels)
    """
    fma_dir   = DATA_DIR / "fma_small"
    meta_dir  = fma_metadata_dir or DATA_DIR / "fma_metadata"

    if not fma_dir.exists():
        raise FileNotFoundError(f"FMA-small not found at {fma_dir}.")
    if not meta_dir.exists():
        raise FileNotFoundError(f"FMA metadata not found at {meta_dir}. Download fma_metadata first.")

    tracks_csv = meta_dir / "fma_metadata" / "tracks.csv"
    if not tracks_csv.exists():
        raise FileNotFoundError(f"tracks.csv not found at {tracks_csv}.")

    # FMA CSV has multi-level header — read with two-level columns
    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])

    # Extract top-level genre and split
    try:
        genre_col  = tracks["track"]["genre_top"]
        split_col  = tracks["set"]["split"]
    except KeyError:
        raise RuntimeError("Unexpected FMA tracks.csv structure — check FMA docs.")

    # Mood tag mapping from FMA genre → DreamStage
    FMA_GENRE_MAP = {
        "Electronic": {"mood": "energetic",  "valence": 0.55, "arousal": 0.70, "energy": "high"},
        "Experimental": {"mood": "intimate", "valence": 0.45, "arousal": 0.35, "energy": "low"},
        "Folk": {"mood": "intimate",         "valence": 0.60, "arousal": 0.30, "energy": "low"},
        "Hip-Hop": {"mood": "energetic",     "valence": 0.50, "arousal": 0.72, "energy": "high"},
        "Instrumental": {"mood": "smooth",   "valence": 0.55, "arousal": 0.40, "energy": "medium"},
        "International": {"mood": "uplifting","valence": 0.65, "arousal": 0.60, "energy": "medium"},
        "Pop": {"mood": "uplifting",         "valence": 0.70, "arousal": 0.65, "energy": "medium"},
        "Rock": {"mood": "energetic",        "valence": 0.48, "arousal": 0.75, "energy": "high"},
    }

    records = []
    for track_id, row in zip(tracks.index, zip(genre_col, split_col)):
        genre_top, split = row
        if pd.isna(genre_top):
            continue

        meta = FMA_GENRE_MAP.get(genre_top)
        if meta is None:
            continue

        # FMA file path: fma_small/{folder}/{track_id:06d}.mp3
        folder = f"{track_id // 1000:03d}"
        mp3    = fma_dir / "fma_small" / folder / f"{track_id:06d}.mp3"
        if not mp3.exists():
            continue

        records.append({
            "file_path":    str(mp3),
            "dataset":      "fma_small",
            "genre_label":  genre_top,
            "mood_label":   meta["mood"],
            "valence":      meta["valence"],
            "arousal":      meta["arousal"],
            "energy_label": meta["energy"],
            "split":        split if isinstance(split, str) else "train",
        })
    return records


# ── Parallel worker ───────────────────────────────────────────────────────────

def _process_one(record: dict) -> Optional[dict]:
    """Extract features from one audio file. Returns None on failure."""
    try:
        y = _load_and_trim(record["file_path"])
        if y is None:
            return None
        feats = extract_features_from_array(y, SAMPLE_RATE)
        if np.isnan(feats).any():
            return None

        out = dict(record)
        for i, v in enumerate(feats):
            out[f"feat_{i:03d}"] = float(v)
        return out
    except Exception:
        return None


# ── Main extraction loop ──────────────────────────────────────────────────────

def extract_dataset(dataset_name: str, workers: int = 2,
                    output_path: Optional[Path] = None) -> Path:
    """
    Extract features from a full dataset and save to Parquet.

    Args:
        dataset_name: 'gtzan' | 'fma_small' | 'all'
        workers:      number of parallel processes
        output_path:  where to save .parquet file

    Returns:
        Path to output Parquet file
    """
    if output_path is None:
        output_path = OUTPUT_DIR / f"features_{dataset_name}.parquet"

    print(f"\nExtracting features: {dataset_name}")
    print(f"Workers: {workers} | Sample rate: {SAMPLE_RATE} Hz | Clip: {CLIP_SEC}s")

    # Gather records
    records = []
    if dataset_name in ("gtzan", "all"):
        try:
            r = _iter_gtzan()
            records.extend(r)
            print(f"  GTZAN:     {len(r)} files")
        except FileNotFoundError as e:
            print(f"  GTZAN: SKIP ({e})")

    if dataset_name in ("fma_small", "all"):
        try:
            r = _iter_fma_small(None)
            records.extend(r)
            print(f"  FMA-small: {len(r)} files")
        except FileNotFoundError as e:
            print(f"  FMA-small: SKIP ({e})")

    if not records:
        raise RuntimeError("No audio files found. Run download.py first.")

    print(f"\nProcessing {len(records)} files...")

    rows = []
    failed = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one, rec): rec for rec in records}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 100 == 0:
                pct = done / len(records) * 100
                print(f"\r  {done}/{len(records)} ({pct:.0f}%) | failed: {failed}", end="", flush=True)
            result = fut.result()
            if result is not None:
                rows.append(result)
            else:
                failed += 1

    print(f"\n  Completed: {len(rows)} OK | {failed} failed")

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False, compression="snappy")
    print(f"  Saved: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Print class distribution
    if "genre_label" in df.columns:
        print("\nGenre distribution:")
        for label, count in df["genre_label"].value_counts().items():
            print(f"  {label:<20} {count:>5}")

    if "mood_label" in df.columns:
        print("\nMood distribution:")
        for label, count in df["mood_label"].value_counts().items():
            print(f"  {label:<20} {count:>5}")

    return output_path


# ── Dataset quality tracking ──────────────────────────────────────────────────

def compute_quality_report(parquet_path: Path) -> dict:
    """
    Compute dataset quality metrics from extracted features.
    Checks for corrupted files, feature outliers, class imbalance.
    """
    df = pd.read_parquet(parquet_path)
    feat_cols = [c for c in df.columns if c.startswith("feat_")]

    feat_matrix = df[feat_cols].values
    nan_rows    = np.isnan(feat_matrix).any(axis=1).sum()
    inf_rows    = np.isinf(feat_matrix).any(axis=1).sum()

    # Z-score outliers (|z| > 5 in any feature)
    from scipy.stats import zscore
    z = np.abs(zscore(feat_matrix, nan_policy="omit"))
    outlier_rows = (z > 5).any(axis=1).sum()

    # Per-class counts
    genre_counts = df["genre_label"].value_counts().to_dict() if "genre_label" in df.columns else {}
    mood_counts  = df["mood_label"].value_counts().to_dict()  if "mood_label"  in df.columns else {}

    # Imbalance ratio (max/min class count)
    if genre_counts and len(genre_counts) > 1:
        imbalance = max(genre_counts.values()) / max(min(genre_counts.values()), 1)
    else:
        imbalance = 1.0

    report = {
        "total_samples":   len(df),
        "feature_dims":    len(feat_cols),
        "nan_rows":        int(nan_rows),
        "inf_rows":        int(inf_rows),
        "outlier_rows":    int(outlier_rows),
        "valid_rows":      int(len(df) - nan_rows - inf_rows),
        "genre_counts":    genre_counts,
        "mood_counts":     mood_counts,
        "genre_imbalance_ratio": round(imbalance, 2),
        "feature_means":   feat_matrix.mean(axis=0).tolist(),
        "feature_stds":    feat_matrix.std(axis=0).tolist(),
    }

    print("\nDataset Quality Report")
    print(f"  Total samples:    {report['total_samples']}")
    print(f"  Valid:            {report['valid_rows']}")
    print(f"  NaN rows:         {report['nan_rows']}")
    print(f"  Outlier rows:     {report['outlier_rows']}")
    print(f"  Imbalance ratio:  {report['genre_imbalance_ratio']}x")

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract audio features for ML training")
    parser.add_argument("--dataset", choices=["gtzan", "fma_small", "all"], default="gtzan")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quality-report", action="store_true")
    args = parser.parse_args()

    output = Path(args.output) if args.output else None
    parquet = extract_dataset(args.dataset, workers=args.workers, output_path=output)

    if args.quality_report:
        compute_quality_report(parquet)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
