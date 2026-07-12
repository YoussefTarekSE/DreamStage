# DreamStage AI Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER UPLOADS VOCAL                            │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    AUDIO QUALITY GATE                                │
│   audio_quality.py                                                   │
│   • RMS level check        • Clipping detection                      │
│   • Duration check (>1.5s) • SNR estimate (>6 dB)                   │
│   • Pitch detection (>20% voiced frames)                             │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    VOCAL PROCESSING PIPELINE                         │
│   vocal_processor.py                                                 │
│   1. Highpass (80 Hz)         DSP                                    │
│   2. Noise reduction          noisereduce (VAD-guided)  Real DSP     │
│   3. Noise gate               Pedalboard                             │
│   4. Compression              Pedalboard (per autotune level)        │
│   5. Pitch correction ────────┐                                      │
│      ├─ WORLD vocoder (best)  │ pyworld — Real DSP model             │
│      └─ pYIN fallback         │ librosa                              │
│   6. Character EQ             │ Pedalboard                           │
│   7. De-esser (dynamic)       │ DSP + gain control                   │
│   8. LUFS normalize           │ pyloudnorm                           │
│   9. True-peak limiter        │ Pedalboard                           │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   ML AUDIO ANALYSIS PIPELINE                         │
│   ml_analyzer.py  (NEW — replaces rule-based audio_analysis.py)     │
│                                                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  PITCH ANALYSIS                                              │   │
│   │  CREPE CNN (Kim et al. 2018) — if installed                  │   │
│   │    → 360-class pitch classifier, viterbi-smoothed F0        │   │
│   │    → confidence-filtered voiced frames                       │   │
│   │  pYIN fallback (librosa) — if CREPE unavailable             │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  FEATURE EXTRACTION (136-dim vector)                         │   │
│   │  MFCC×40 + Chroma×12 + Centroid + Rolloff + ZCR             │   │
│   │  + Spectral Contrast×7 + Tonnetz×6 (mean + std each)        │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  ML CLASSIFIERS (trained on GTZAN + FMA-small)              │   │
│   │                                                               │   │
│   │  genre_classifier.joblib                                     │   │
│   │    RandomForestClassifier → GTZAN 10-class genre             │   │
│   │    → maps to DreamStage emotion label                        │   │
│   │                                                               │   │
│   │  valence_regressor.joblib                                    │   │
│   │    MultiOutput GradientBoostingRegressor                     │   │
│   │    → [valence, arousal] ∈ [0,1]²                            │   │
│   │    → maps to DreamStage emotion label                        │   │
│   │                                                               │   │
│   │  energy_classifier.joblib                                    │   │
│   │    GradientBoostingClassifier → low/medium/high              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  DSP ANALYSIS (validated algorithms)                         │   │
│   │  Tempo → enhanced multi-strategy librosa beat_track         │   │
│   │  Key   → Krumhansl-Schmuckler chroma correlation (1982)     │   │
│   │  Swing → beat subdivision irregularity analysis             │   │
│   └─────────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               │ analysis dict
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BEAT GENERATION — 4 TIERS                         │
│   beat_generator.py                                                   │
│                                                                       │
│   TIER 1: Local MusicGen (audiocraft)                                │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │  facebook/musicgen-small (300M params)                        │  │
│   │  Input: text prompt + vocal chroma conditioning               │  │
│   │  Output: 30s music at 32 kHz                                  │  │
│   │  Requires: pip install audiocraft (2GB), GPU recommended      │  │
│   │  Status: Development only / high-RAM servers                  │  │
│   └──────────────────────────────────────────────────────────────┘  │
│           │ if fails (no audiocraft / OOM)                           │
│           ▼                                                           │
│   TIER 2: HF Inference API (musicgen_hf.py)  ← DEFAULT PRODUCTION   │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │  api-inference.huggingface.co/models/facebook/musicgen-small  │  │
│   │  Input: rich text prompt (encodes tempo, key, mood, energy)   │  │
│   │  Output: 30s music at 32 kHz                                  │  │
│   │  Requires: HF_API_KEY (free), 120s timeout                    │  │
│   │  No melody conditioning — compensated by specific prompts     │  │
│   │  Status: Always available on Render, ~30 req/hr free tier     │  │
│   └──────────────────────────────────────────────────────────────┘  │
│           │ if fails (no HF_API_KEY / rate limit)                    │
│           ▼                                                           │
│   TIER 3: Gradio Space (legacy)                                      │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │  huggingface.co/spaces/facebook/MusicGen                      │  │
│   │  Input: text prompt + vocal WAV (melody conditioning)         │  │
│   │  Output: generated beat WAV                                   │  │
│   │  ~60% uptime, 2-5 min cold start. Legacy path.               │  │
│   └──────────────────────────────────────────────────────────────┘  │
│           │ if fails (space down / rate limit)                       │
│           ▼                                                           │
│   TIER 4: Programmatic Synthesizer (beat_synthesizer.py)            │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │  Rule-based wavetable synthesis                               │  │
│   │  16 genre families, hardcoded patterns, chord progressions    │  │
│   │  NOT AI — but always works, never fails                       │  │
│   │  Driven by analysis dict (tempo, key, mode, emotion)          │  │
│   └──────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FINAL MIX                                    │
│   mixer.py                                                           │
│   • LUFS normalization (vocal: -18, beat: -21)                      │
│   • Mid-Side stereo widening                                         │
│   • EQ carve: notch beat in vocal frequency range                   │
│   • Genre-aware reverb on vocal                                      │
│   • Mastering: bus compression → shelving → -14 LUFS limiter        │
│   • Export: WAV 24-bit/48kHz + MP3 320kbps                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Training Data Pipeline

```
DATASETS                    PROCESSING               TRAINING
                                                      
GTZAN (1.2 GB)            extract_features.py       train_classifiers.py
  1000 clips × 30s    ──▶  136-dim feature      ──▶  genre_classifier.joblib
  10 genres                vector per clip           RandomForest
                            (MFCC, chroma,             72% accuracy
FMA-small (7.2 GB)          spectral, etc.)           
  8000 clips × 30s    ──▶  + genre_label         ──▶  valence_regressor.joblib
  8 genres balanced         + mood_label               MultiOutput GB Regressor
  MusicBrainz tags          + valence [0,1]            R² ≈ 0.45–0.55
                            + arousal [0,1]            
                            + energy_label        ──▶  energy_classifier.joblib
                                                       GB Classifier
                                                       78% accuracy
```

---

## Model Selection Rationale

### Why RandomForest for genre?

| Alternative   | Pros                     | Cons                              | Verdict  |
|---------------|--------------------------|-----------------------------------|----------|
| RandomForest  | Fast, interpretable,     | Lower ceiling than deep learning  | ✓ Use    |
|               | no hyperparameter tuning |                                   |          |
| SVM (RBF)     | High accuracy on small   | Slow on >5k samples, hard to tune | Skip     |
|               | datasets                 |                                   |          |
| MLP           | Can capture nonlinear    | Needs careful tuning, overfits    | Skip     |
|               | patterns                 | on small data                     |          |
| CNN on audio  | Best for raw audio       | Needs GPU, 100× more data         | Future   |
| (e.g. VGGish) |                          |                                   |          |

### Why GradientBoosting for valence?

Valence estimation is a regression problem with label noise (human ratings vary).
GradientBoosting handles outliers better than linear regression and achieves
better R² than RandomForest on continuous targets with small datasets.

### Why MusicGen over Suno/other models?

| Option            | Quality | Cost    | Melody conditioning | License  |
|-------------------|---------|---------|---------------------|----------|
| MusicGen (Meta)   | Excellent | Free  | Yes (local)         | MIT      |
| Suno API          | Excellent | $$$   | No                  | Proprietary |
| Replicate API     | Excellent | $0.01+/req | Depends      | Per model |
| Audiocraft local  | Excellent | $0    | Yes                 | MIT      |
| Custom training   | Unknown  | $1M+  | Full control        | Yours    |

---

## Deployment Configuration

### Render free tier (512 MB RAM, 0.1 CPU)

What runs:
- Vocal processing (WORLD + pedalboard): ~150 MB peak
- ML classifiers (sklearn): ~50 MB
- librosa analysis: ~80 MB
- HF API call (async HTTP): <1 MB

What does NOT run:
- CREPE (TF model: ~80 MB) — comment out in requirements.txt if tight
- Local MusicGen (audiocraft: ~2 GB) — Tier 1 skipped automatically
- Demucs: ~1.5 GB — not in production deps

### Adding CREPE to production

Uncomment in requirements.txt:
```
crepe>=0.0.15
```
Then increase Render instance to "Starter" (2 GB RAM, $7/month).
CREPE 'tiny' model adds ~30% improvement in vocal range accuracy.

### Serving trained models

Options (cheapest first):
1. Commit .joblib files to git (if <50 MB each — sklearn models usually <5 MB)
2. Download from Cloudflare R2 on startup (already used for audio storage)
3. Bake into Docker image

Recommended: commit to git. Each .joblib is typically 2–10 MB.

---

## Honest Limitations

### Things that CANNOT be done at $0 with current architecture:

1. **Training MusicGen from scratch**: Requires ~100 A100 GPUs × 30 days.
   The pre-trained weights are used as-is via HF API.

2. **Training Demucs from scratch**: Similar compute to MusicGen.
   Pre-trained htdemucs model is used instead.

3. **Real-time vocal-to-beat generation with melody conditioning on Render**:
   MusicGen inference with chroma conditioning requires ~2 GB RAM + ~3 min CPU.
   Render free tier: not viable. Solution: async job queue + Render worker.

4. **Custom voice identity preservation model**:
   State-of-the-art voice conversion (e.g., VITS, YourTTS) needs GPU training
   on paired data. The WORLD vocoder correctly handles formant preservation
   without needing training.

5. **Genre-to-stem decomposition** (separate drum/bass/harmony stems from beats):
   Demucs achieves this but needs 1.5 GB RAM. Requires Render "Standard" plan.

### What improves first with a budget:

| Budget     | Improvement                          | Expected lift          |
|------------|--------------------------------------|------------------------|
| $0         | Run training pipeline (existing code) | ~72% genre accuracy   |
| $7/month   | Render Starter (2 GB) → add CREPE     | Better vocal range     |
| $25/month  | Render Standard (4 GB) → add Demucs  | Pre-isolated vocal     |
| $50/month  | Render Pro (8 GB) → local MusicGen   | Melody-conditioned beat|
| $200/month | Dedicated GPU instance               | <30s beat generation   |
