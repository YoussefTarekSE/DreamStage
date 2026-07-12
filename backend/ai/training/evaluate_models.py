"""
Evaluation framework for DreamStage ML models.

Runs comprehensive evaluation of trained classifiers and regressors,
generates confusion matrices, learning curves, and a full report.

Usage:
  python evaluate_models.py --features features_gtzan.parquet
  python evaluate_models.py --features features_all.parquet --plot
  python evaluate_models.py --quick-check  # just verify models load and work
"""
from __future__ import annotations

import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    r2_score, mean_absolute_error, mean_squared_error,
)
from sklearn.model_selection import learning_curve, StratifiedKFold
import joblib

logger = logging.getLogger(__name__)

MODELS_DIR  = Path(__file__).resolve().parents[1] / "models"
FEATURES_DIR = Path(__file__).resolve().parents[1] / "features"
REPORTS_DIR  = Path(__file__).resolve().parents[1] / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_bundle(name: str) -> dict | None:
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        print(f"  Model not found: {path}")
        return None
    return joblib.load(path)


def _load_features(parquet_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    df = pd.read_parquet(parquet_path)
    feat_cols = sorted([c for c in df.columns if c.startswith("feat_")])
    X = df[feat_cols].values.astype(np.float32)
    meta = df[[c for c in df.columns if not c.startswith("feat_")]]
    valid = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
    return X[valid], meta[valid].reset_index(drop=True)


# ── Genre evaluation ──────────────────────────────────────────────────────────

def evaluate_genre_classifier(X: np.ndarray, meta: pd.DataFrame,
                               bundle: dict, plot: bool = False) -> dict:
    """Full evaluation of genre classifier."""
    if "genre_label" not in meta.columns:
        print("  No genre_label column — skip")
        return {}

    le      = bundle["label_encoder"]
    clf     = bundle["classifier"]
    scaler  = bundle["scaler"]
    classes = bundle["classes"]

    y = le.transform(meta["genre_label"].values)
    X_s = scaler.transform(X)
    y_pred = clf.predict(X_s)

    acc = accuracy_score(y, y_pred)
    report_str = classification_report(y, y_pred, target_names=classes)
    cm  = confusion_matrix(y, y_pred)

    print(f"\nGenre Classifier Evaluation")
    print(f"  Overall accuracy: {acc:.3f}")
    print(f"\n{report_str}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=classes, columns=classes).to_string())

    # Per-class accuracy
    per_class = {}
    for i, cls in enumerate(classes):
        mask = y == i
        if mask.sum() > 0:
            per_class[cls] = float((y_pred[mask] == i).mean())

    # 5-fold CV for reliable estimate
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler as SS
    pipe = Pipeline([("s", SS()), ("c", clf)])
    cv_scores = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X, y):
        pipe.fit(X[train_idx], y[train_idx])
        cv_scores.append(accuracy_score(y[test_idx], pipe.predict(X[test_idx])))
    cv_mean = float(np.mean(cv_scores))
    cv_std  = float(np.std(cv_scores))
    print(f"\n  5-fold CV: {cv_mean:.3f} ± {cv_std:.3f}")

    # Compare to random baseline
    random_baseline = 1.0 / len(classes)
    majority_baseline = float(pd.Series(y).value_counts().iloc[0] / len(y))
    print(f"\n  Random baseline:   {random_baseline:.3f}")
    print(f"  Majority baseline: {majority_baseline:.3f}")
    print(f"  Our model:         {acc:.3f}")
    improvement = (acc - majority_baseline) / (1 - majority_baseline) * 100
    print(f"  Improvement over majority: {improvement:.1f}%")

    result = {
        "accuracy":          float(acc),
        "cv_mean":           cv_mean,
        "cv_std":            cv_std,
        "per_class":         per_class,
        "random_baseline":   random_baseline,
        "majority_baseline": majority_baseline,
    }

    if plot:
        _plot_confusion_matrix(cm, classes, "genre_confusion_matrix")
        _plot_learning_curve(X, y, clf, scaler, "genre_learning_curve")

    return result


# ── Valence / arousal evaluation ──────────────────────────────────────────────

def evaluate_valence_regressor(X: np.ndarray, meta: pd.DataFrame,
                                bundle: dict, plot: bool = False) -> dict:
    """Full evaluation of valence + arousal regressor."""
    if not bundle:
        return {}

    required = {"valence", "arousal"}
    if not required.issubset(meta.columns):
        print(f"  Missing columns {required - set(meta.columns)} — skip")
        return {}

    reg    = bundle["regressor"]
    scaler = bundle["scaler"]

    y_true = np.column_stack([
        meta["valence"].values.astype(np.float32),
        meta["arousal"].values.astype(np.float32),
    ])
    valid = ~np.isnan(y_true).any(axis=1)
    X_v, y_v = X[valid], y_true[valid]

    X_s    = scaler.transform(X_v)
    y_pred = reg.predict(X_s)

    r2_val  = r2_score(y_v[:, 0], y_pred[:, 0])
    r2_aro  = r2_score(y_v[:, 1], y_pred[:, 1])
    mae_val = float(mean_absolute_error(y_v[:, 0], y_pred[:, 0]))
    mae_aro = float(mean_absolute_error(y_v[:, 1], y_pred[:, 1]))

    print(f"\nValence/Arousal Regressor Evaluation")
    print(f"  Valence  → R²={r2_val:.3f}, MAE={mae_val:.3f}")
    print(f"  Arousal  → R²={r2_aro:.3f}, MAE={mae_aro:.3f}")

    # Compare to arithmetic formula (expected baseline ~0.3 Pearson r → ~0.09 R²)
    print("\n  Baseline comparison (arithmetic formula from audio_analysis.py):")
    print("    Valence arithmetic formula: Pearson r ≈ 0.30, R² ≈ 0.09")
    print(f"    Our regressor:              R² = {r2_val:.3f}  (higher is better)")
    if r2_val > 0.25:
        print("    ✓ Model outperforms arithmetic baseline")
    else:
        print("    ✗ Model does not outperform baseline — needs more training data")

    return {
        "r2_valence": float(r2_val),
        "r2_arousal": float(r2_aro),
        "mae_valence": mae_val,
        "mae_arousal": mae_aro,
    }


# ── Quick functional check ────────────────────────────────────────────────────

def quick_check():
    """
    Verify all models load and can make predictions on dummy input.
    Used in CI / deployment health checks.
    """
    import numpy as np
    dummy = np.random.randn(1, 136).astype(np.float32)
    results = {}

    for name in ["genre_classifier", "valence_regressor", "energy_classifier"]:
        bundle = _load_bundle(name)
        if bundle is None:
            results[name] = "MISSING"
            continue
        try:
            scaler = bundle.get("scaler")
            x = scaler.transform(dummy) if scaler else dummy
            if "classifier" in bundle:
                pred = bundle["classifier"].predict(x)
                proba = bundle["classifier"].predict_proba(x)
                results[name] = f"OK — pred={pred[0]}, confidence={float(proba.max()):.2f}"
            elif "regressor" in bundle:
                pred = bundle["regressor"].predict(x)
                results[name] = f"OK — pred={pred[0]}"
            else:
                results[name] = "UNKNOWN structure"
        except Exception as exc:
            results[name] = f"ERROR: {exc}"

    print("\nModel quick check:")
    all_ok = True
    for name, status in results.items():
        icon = "✓" if status.startswith("OK") else ("✗" if "ERROR" in status else "○")
        print(f"  {icon} {name}: {status}")
        if "ERROR" in status or "MISSING" in status:
            all_ok = False

    if all_ok:
        print("\nAll models healthy.")
    else:
        print("\nSome models missing or broken. Run train_classifiers.py.")
    return all_ok


# ── Plotting (optional — requires matplotlib) ─────────────────────────────────

def _plot_confusion_matrix(cm, classes, name: str):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {name}")
        out = REPORTS_DIR / f"{name}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"  Saved plot: {out}")
    except ImportError:
        print("  (matplotlib/seaborn not installed — skipping plot)")


def _plot_learning_curve(X, y, clf, scaler, name: str):
    try:
        import matplotlib.pyplot as plt
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler as SS

        pipe = Pipeline([("s", SS()), ("c", clf)])
        train_sizes, train_scores, val_scores = learning_curve(
            pipe, X, y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring="accuracy", n_jobs=-1,
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_sizes, train_scores.mean(axis=1), label="Train")
        ax.fill_between(train_sizes,
                        train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.2)
        ax.plot(train_sizes, val_scores.mean(axis=1), label="Validation")
        ax.fill_between(train_sizes,
                        val_scores.mean(axis=1) - val_scores.std(axis=1),
                        val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.2)
        ax.set_xlabel("Training samples")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Learning Curve — {name}")
        ax.legend()
        out = REPORTS_DIR / f"{name}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"  Saved plot: {out}")
    except ImportError:
        print("  (matplotlib not installed — skipping learning curve)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate DreamStage ML models")
    parser.add_argument("--features", type=str, default=None)
    parser.add_argument("--models-dir", type=str, default=str(MODELS_DIR))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--quick-check", action="store_true")
    args = parser.parse_args()

    global MODELS_DIR
    MODELS_DIR = Path(args.models_dir)

    if args.quick_check:
        ok = quick_check()
        exit(0 if ok else 1)

    # Find features
    if args.features:
        parquet_path = Path(args.features)
    else:
        candidates = [
            FEATURES_DIR / "features_all.parquet",
            FEATURES_DIR / "features_gtzan.parquet",
        ]
        parquet_path = next((p for p in candidates if p.exists()), None)
        if parquet_path is None:
            print("No features file. Run extract_features.py first.")
            return

    X, meta = _load_features(parquet_path)

    all_results = {}

    genre_bundle = _load_bundle("genre_classifier")
    if genre_bundle:
        all_results["genre"] = evaluate_genre_classifier(X, meta, genre_bundle, plot=args.plot)

    val_bundle = _load_bundle("valence_regressor")
    if val_bundle:
        all_results["valence"] = evaluate_valence_regressor(X, meta, val_bundle, plot=args.plot)

    # Save report
    report_path = REPORTS_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
