"""
Build combined training dataset from multiple sources.

Merges GTZAN and FMA-small feature files, handles class imbalance via
SMOTE or class weights, applies quality filters, and produces a clean
training-ready Parquet file.

Usage:
  python build_training_data.py
  python build_training_data.py --balance --min-samples-per-class 80
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURES_DIR = Path(__file__).resolve().parents[1] / "features"
OUTPUT_DIR   = Path(__file__).resolve().parents[1] / "features"


def load_and_merge(paths: list[Path]) -> pd.DataFrame:
    """Load multiple feature Parquet files and concatenate."""
    dfs = []
    for p in paths:
        if p.exists():
            df = pd.read_parquet(p)
            dfs.append(df)
            print(f"  Loaded {p.name}: {len(df)} rows")
        else:
            print(f"  Skipping (not found): {p}")
    if not dfs:
        raise FileNotFoundError("No feature files found. Run extract_features.py first.")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Combined: {len(combined)} rows")
    return combined


def quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with NaN features, inf values, or near-zero variance."""
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    X = df[feat_cols].values.astype(np.float32)

    # Remove NaN / inf
    good = (~np.isnan(X).any(axis=1)) & (~np.isinf(X).any(axis=1))
    df = df[good].reset_index(drop=True)
    print(f"  After NaN/inf filter: {len(df)} rows")

    # Remove zero-variance samples (silent audio that extracted all zeros)
    X2 = df[feat_cols].values.astype(np.float32)
    row_var = np.var(X2, axis=1)
    nonzero = row_var > 1e-6
    df = df[nonzero].reset_index(drop=True)
    print(f"  After zero-variance filter: {len(df)} rows")

    return df


def balance_classes(df: pd.DataFrame, label_col: str = "genre_label",
                    min_samples: int = 80) -> pd.DataFrame:
    """
    Balance class distribution by:
    1. Dropping classes with < min_samples (too few to learn from)
    2. Undersampling dominant classes to max_samples = 2× min class
    """
    counts = df[label_col].value_counts()
    print(f"\nClass distribution before balancing:")
    for cls, cnt in counts.items():
        print(f"  {cls:<20} {cnt}")

    # Drop undersized classes
    valid_classes = counts[counts >= min_samples].index
    df = df[df[label_col].isin(valid_classes)].reset_index(drop=True)

    # Undersample to 2× minimum class size
    min_count = df[label_col].value_counts().min()
    max_count  = min_count * 3

    balanced = []
    for cls in df[label_col].unique():
        subset = df[df[label_col] == cls]
        if len(subset) > max_count:
            subset = subset.sample(max_count, random_state=42)
        balanced.append(subset)

    df = pd.concat(balanced, ignore_index=True).sample(frac=1, random_state=42)

    print(f"\nClass distribution after balancing:")
    for cls, cnt in df[label_col].value_counts().items():
        print(f"  {cls:<20} {cnt}")
    return df


def create_train_test_split(df: pd.DataFrame,
                             label_col: str = "genre_label") -> pd.DataFrame:
    """
    Ensure stratified train/test split is represented in the 'split' column.
    Uses file-level split from GTZAN (first 80 = train, last 20 = test).
    For FMA-small, uses the official split column.
    """
    if "split" not in df.columns:
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(
            df.index, test_size=0.20, random_state=42,
            stratify=df[label_col] if label_col in df.columns else None
        )
        df.loc[train_idx, "split"] = "training"
        df.loc[test_idx,  "split"] = "test"

    counts = df["split"].value_counts()
    print(f"\nSplit distribution: {dict(counts)}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Build training dataset")
    parser.add_argument("--balance", action="store_true", help="Balance class distribution")
    parser.add_argument("--min-samples-per-class", type=int, default=50)
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR / "features_all.parquet"))
    args = parser.parse_args()

    print("Building training dataset...")

    # Find available feature files
    candidates = [
        FEATURES_DIR / "features_gtzan.parquet",
        FEATURES_DIR / "features_fma_small.parquet",
    ]

    df = load_and_merge(candidates)
    df = quality_filter(df)

    if args.balance and "genre_label" in df.columns:
        df = balance_classes(df, min_samples=args.min_samples_per_class)

    df = create_train_test_split(df)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, compression="snappy")

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    size_mb = out.stat().st_size / 1024 / 1024

    print(f"\nSaved: {out} ({size_mb:.1f} MB)")
    print(f"Rows: {len(df)} | Features: {len(feat_cols)} | Columns: {len(df.columns)}")
    print(f"\nNext step: python backend/ai/training/train_classifiers.py")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
