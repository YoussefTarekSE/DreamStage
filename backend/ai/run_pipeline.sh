#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# DreamStage AI Training Pipeline — Complete Run
#
# Runs the full pipeline from raw datasets to deployed model files.
# Expected total time: ~25 minutes on CPU (mostly download + extraction).
#
# Prerequisites:
#   pip install -r backend/requirements.txt
#   pip install -r backend/ai/requirements-ai.txt
#
# Usage:
#   cd backend
#   bash ai/run_pipeline.sh
#
#   # Quick run (GTZAN only, ~15 min):
#   bash ai/run_pipeline.sh --quick
#
#   # Full run (GTZAN + FMA, ~45 min):
#   bash ai/run_pipeline.sh --full
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

QUICK=false
FULL=false
for arg in "$@"; do
    case $arg in
        --quick) QUICK=true ;;
        --full)  FULL=true ;;
    esac
done

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI_DIR="$BACKEND_DIR/ai"
DATA_DIR="$AI_DIR/data"
MODELS_DIR="$AI_DIR/models"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         DreamStage AI Training Pipeline                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Backend: $BACKEND_DIR"
echo "Models will be saved to: $MODELS_DIR"
echo ""

# ── Step 1: Download datasets ─────────────────────────────────────────────────
echo "─── Step 1/4: Downloading datasets ──────────────────────────────────"

if $FULL; then
    python -m backend.ai.dataset_pipeline.download --datasets gtzan fma_metadata fma_small
elif $QUICK; then
    python -m backend.ai.dataset_pipeline.download --datasets gtzan
else
    echo "Defaulting to GTZAN only. Use --full for FMA-small as well."
    python -m backend.ai.dataset_pipeline.download --datasets gtzan
fi

# ── Step 2: Extract features ──────────────────────────────────────────────────
echo ""
echo "─── Step 2/4: Extracting audio features ─────────────────────────────"

if $FULL; then
    python -m backend.ai.dataset_pipeline.extract_features --dataset fma_small --workers 4
fi

python -m backend.ai.dataset_pipeline.extract_features --dataset gtzan --workers 4

# ── Step 3: Build combined training data ──────────────────────────────────────
echo ""
echo "─── Step 3/4: Building training dataset ─────────────────────────────"

python -m backend.ai.dataset_pipeline.build_training_data --balance

# ── Step 4: Train classifiers ─────────────────────────────────────────────────
echo ""
echo "─── Step 4/4: Training ML classifiers ───────────────────────────────"

python -m backend.ai.training.train_classifiers

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "─── Verification ─────────────────────────────────────────────────────"

python -m backend.ai.training.evaluate_models --quick-check

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Pipeline complete! Models saved to backend/ai/models/       ║"
echo "║                                                               ║"
echo "║  Deploy: copy *.joblib files alongside your backend code      ║"
echo "║  Test:   restart FastAPI and check /health endpoint           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
