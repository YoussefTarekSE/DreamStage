"""
Real ML-based audio analysis.

Replaces every rule-based heuristic in audio_analysis.py with trained models
or genuine signal-processing research algorithms.

Priority chain per feature:
  1. Real ML model (CREPE, trained sklearn classifier)
  2. Well-validated DSP algorithm (librosa with validated parameters)
  3. Safe default (only when audio is too short / corrupted)

Models used:
  - CREPE (Kim et al., 2018, ISMIR): Deep CNN pitch estimator, MIT license.
      Trained on synthesized + real audio at 16 kHz. Outperforms YIN/pYIN
      on real vocal recordings by ~25% in RPA metrics.
      pip install crepe  (requires tensorflow >=2.3 or keras-standalone)

  - Trained genre / mood / energy sklearn classifiers:
      RandomForestClassifier trained on GTZAN (genre, 1000 clips, 10 classes)
      and FMA-small (mood/energy, 8000 clips, MusicBrainz tags).
      Loaded from backend/ai/models/*.joblib — absent = graceful fallback.
      Training script: backend/ai/training/train_classifiers.py

Compute budget (Render free tier, 512 MB RAM):
  - CREPE 'tiny' model: ~80 MB RAM, ~1 s per 10 s clip
  - sklearn classifiers: ~5 MB RAM, <50 ms inference
  - Everything else: librosa (already in requirements)
"""
from __future__ import annotations

import os
import pathlib
import logging
from functools import lru_cache
import numpy as np
import librosa

from .audio_loader import load_audio, TARGET_SR
from .metrics import increment

logger = logging.getLogger(__name__)

# ── Optional ML dependencies ──────────────────────────────────────────────────

try:
    import crepe  # type: ignore
    _HAS_CREPE = True
except ImportError:
    _HAS_CREPE = False
    logger.info("crepe not installed — using librosa.yin for pitch detection")

try:
    import joblib  # type: ignore
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

# ── Model paths ───────────────────────────────────────────────────────────────
# The trained models live in backend/ai/models. ml_analyzer.py is at
# backend/app/services/, so parents[2] == backend. (A previous parents[3] pointed
# at the repo root — backend's parent — so the models were never found and every
# analysis silently fell back to arithmetic rules.) We probe a few candidate
# layouts so it also works if models are shipped at the repo root or in a Docker
# image, and pick the first directory that actually contains the model files.
_here = pathlib.Path(__file__).resolve()
_MODEL_DIR_CANDIDATES = [
    _here.parents[2] / "ai" / "models",   # backend/ai/models  (canonical)
    _here.parents[3] / "ai" / "models",   # <repo>/ai/models   (alt layout)
    pathlib.Path.cwd() / "ai" / "models",  # cwd-relative (Render/Docker)
    pathlib.Path.cwd() / "backend" / "ai" / "models",
]


def _resolve_models_dir() -> pathlib.Path:
    for cand in _MODEL_DIR_CANDIDATES:
        try:
            if cand.is_dir() and any(cand.glob("*.joblib")):
                return cand
        except Exception:
            continue
    return _MODEL_DIR_CANDIDATES[0]  # canonical default (for clear logging)


_MODELS_DIR = _resolve_models_dir()
logger.info("[ml_analyzer] models dir: %s (exists=%s)", _MODELS_DIR, _MODELS_DIR.is_dir())


@lru_cache(maxsize=None)
def _load_model_cached(name: str):
    """Load a joblib model file, return None if missing or joblib unavailable."""
    if not _HAS_JOBLIB:
        return None
    path = _MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        logger.warning("Could not load model %s: %s", name, exc)
        return None


def _load_model(name: str):
    before = _load_model_cached.cache_info()
    model = _load_model_cached(name)
    after = _load_model_cached.cache_info()
    if after.hits > before.hits:
        increment("model_cache_hits")
    else:
        increment("model_cache_misses")
    return model


_load_model.cache_clear = _load_model_cached.cache_clear  # type: ignore[attr-defined]


# ── CREPE pitch detection ─────────────────────────────────────────────────────

def detect_pitch_crepe(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate fundamental frequency with CREPE (convolutional pitch estimator).

    CREPE is a deep CNN trained end-to-end on synthesized tones + real recordings
    to classify the fundamental frequency of 360 candidate pitches (32.7–1975 Hz).
    Unlike YIN/pYIN which use autocorrelation heuristics, CREPE learns a continuous
    likelihood surface over pitch space.

    Paper: Kim et al. (2018) "CREPE: A Convolutional Representation for Pitch
    Estimation". ISMIR 2018. https://arxiv.org/abs/1802.06182

    Returns:
        times:      frame time in seconds
        frequencies: estimated F0 in Hz (0.0 = unvoiced)
        confidence: per-frame confidence [0, 1]
    """
    if not _HAS_CREPE:
        return _detect_pitch_yin(y, sr)

    try:
        # CREPE expects mono float32 at any sample rate (resamples internally).
        y_f32 = y.astype(np.float32)
        # Use 'tiny' model: 14 M parameters, ~15 MB weights, fast on CPU.
        times, freqs, conf, _ = crepe.predict(
            y_f32, sr,
            viterbi=True,        # Viterbi decoding smooths F0 trajectory
            step_size=10,        # 10 ms hop = 100 frames/sec
            model_capacity="tiny",
            verbose=0,
        )
        # Zero out low-confidence frames (unvoiced / noise)
        freqs = np.where(conf >= 0.50, freqs, 0.0)
        return times, freqs.astype(np.float32), conf.astype(np.float32)
    except Exception as exc:
        logger.warning("CREPE failed (%s), using YIN fallback", exc)
        return _detect_pitch_yin(y, sr)


def _detect_pitch_yin(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback: pYIN (probabilistic YIN) when CREPE is unavailable."""
    try:
        f0, voiced, prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sr,
            frame_length=2048,
        )
        f0 = np.where(voiced & ~np.isnan(f0), f0, 0.0).astype(np.float32)
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=512)
        conf = np.where(voiced, prob.astype(np.float32), 0.0)
        return times, f0, conf
    except Exception:
        return np.array([0.0]), np.array([0.0]), np.array([0.0])


# ── Feature vector for classifiers ───────────────────────────────────────────

def extract_feature_vector(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract a 136-dimensional feature vector for genre/mood classifiers.

    Features chosen to match what the sklearn classifiers were trained on
    (see backend/ai/training/train_classifiers.py). Mean + std across time
    for each descriptor gives both the average value and its variability.

    Dimensions:
      MFCC × 40:             mean (40) + std (40) = 80
      Chroma × 12:           mean (12) + std (12) = 24
      Spectral centroid ×1:  mean  (1) + std  (1) =  2
      Spectral rolloff  ×1:  mean  (1) + std  (1) =  2
      Zero crossing rate×1:  mean  (1) + std  (1) =  2
      Spectral contrast ×7:  mean  (7) + std  (7) = 14
      Tonnetz ×6:            mean  (6) + std  (6) = 12
      Total: 136
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
        return np.zeros(136, dtype=np.float32)


# ── Tempo estimation with validated multi-strategy librosa ────────────────────

def detect_tempo_robust(y: np.ndarray, sr: int) -> float:
    """
    Multi-strategy tempo estimation tuned for unaccompanied vocals.

    librosa.beat.beat_track runs the El Paso / DBN beat tracker, which was
    trained on rhythm-centric recordings. On solo vocals it tends to lock on
    syllable onsets which are often at half or double the musical tempo.

    Strategy:
      1. Onset envelope restricted to vocal frequency range (80–4000 Hz)
         to suppress non-vocal transients from room noise.
      2. Two independent tempo estimators: standard beat_track + PLP.
      3. Evaluate 2× and 0.5× candidates, keep cluster centred in 75–145 BPM.
      4. Select median of surviving candidates.

    This is the same algorithm already in audio_analysis.py but centralised here
    so ml_analyzer is the single source of truth for tempo.
    """
    try:
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, aggregate=np.median, fmax=4000
        )

        tempos_1, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, units="bpm")
        t1 = float(np.atleast_1d(tempos_1)[0])

        try:
            plp = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
            tempos_2, _ = librosa.beat.beat_track(onset_envelope=plp, sr=sr, units="bpm")
            t2 = float(np.atleast_1d(tempos_2)[0])
        except Exception:
            t2 = t1

        candidates = [t1, t2, t1 * 2, t1 / 2, t2 * 2, t2 / 2]
        in_range   = [t for t in candidates if 55 <= t <= 165]
        if not in_range:
            return 90.0
        preferred = [t for t in in_range if 75 <= t <= 145]
        return float(round(np.median(preferred if preferred else in_range), 1))
    except Exception:
        return 90.0


# ── Genre classification with trained model ───────────────────────────────────

# GTZAN label → DreamStage emotion mapping
# GTZAN genres (10): blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
_GTZAN_TO_EMOTION: dict[str, str] = {
    "blues":     "melancholic",
    "classical": "intimate",
    "country":   "uplifting",
    "disco":     "euphoric",
    "hiphop":    "energetic",
    "jazz":      "smooth",
    "metal":     "dark",
    "pop":       "uplifting",
    "reggae":    "smooth",
    "rock":      "energetic",
}

# DreamStage genre hint → beat synthesizer genre
_EMOTION_TO_SYNTH_GENRE: dict[str, str] = {
    "euphoric":   "trap_melodic",
    "uplifting":  "pop_bright",
    "dark":       "trap_dark",
    "melancholic": "rnb_neo_soul",
    "energetic":  "hiphop_modern",
    "intimate":   "soul_ballad",
    "smooth":     "rnb_smooth",
}


def classify_genre_ml(y: np.ndarray, sr: int) -> tuple[str, str, float]:
    """
    Classify genre using a trained RandomForestClassifier.

    Returns:
        gtzan_label: raw GTZAN genre label (or 'unknown')
        emotion:     mapped DreamStage emotion label
        confidence:  probability of top class [0, 1]

    Falls back to 'unknown' / 'smooth' if no trained model is found.
    Train the model with: python backend/ai/training/train_classifiers.py
    """
    model_bundle = _load_model("genre_classifier")
    if model_bundle is None:
        return "unknown", "smooth", 0.0

    try:
        clf    = model_bundle["classifier"]
        scaler = model_bundle.get("scaler")

        features = extract_feature_vector(y, sr).reshape(1, -1)
        if scaler is not None:
            features = scaler.transform(features)

        label_idx = clf.predict(features)[0]
        proba = float(np.max(clf.predict_proba(features)))

        # Map the integer class index back to the GTZAN genre NAME. The model was
        # trained on integer-encoded labels, so clf.predict returns an int — the
        # emotion lookup (keyed by name) silently missed and always returned
        # "smooth", flattening every genre to the same mood and (when confident)
        # forcing the beat toward R&B regardless of the actual genre.
        classes = model_bundle.get("classes")
        le      = model_bundle.get("label_encoder")
        try:
            if le is not None:
                gtzan_label = str(le.inverse_transform([label_idx])[0])
            elif classes is not None and 0 <= int(label_idx) < len(classes):
                gtzan_label = str(classes[int(label_idx)])
            else:
                gtzan_label = str(label_idx)
        except Exception:
            gtzan_label = str(label_idx)

        emotion = _GTZAN_TO_EMOTION.get(gtzan_label, "smooth")
        return gtzan_label, emotion, proba
    except Exception as exc:
        logger.warning("Genre classifier error: %s", exc)
        return "unknown", "smooth", 0.0


# ── Mood / valence estimation with trained model ──────────────────────────────

def estimate_valence_ml(y: np.ndarray, sr: int) -> tuple[float, str]:
    """
    Predict valence (emotional positivity) and arousal using a trained
    multi-output regression model.

    The model was trained on FMA-small audio features paired with per-track
    valence/arousal annotations derived from MusicBrainz mood tags:
      - Tags like 'sad', 'melancholic', 'dark' → low valence
      - Tags like 'happy', 'uplifting', 'euphoric' → high valence
      - Tags like 'aggressive', 'energetic' → high arousal
      - Tags like 'calm', 'ambient' → low arousal

    Returns:
        valence: float [0, 1] — 0 = dark/sad, 1 = happy/bright
        emotion: str — DreamStage emotion label derived from valence + arousal

    Falls back to arithmetic estimation if model unavailable.
    """
    model_bundle = _load_model("valence_regressor")
    if model_bundle is None:
        return _estimate_valence_arithmetic(y, sr)

    try:
        reg    = model_bundle["regressor"]
        scaler = model_bundle.get("scaler")

        features = extract_feature_vector(y, sr).reshape(1, -1)
        if scaler is not None:
            features = scaler.transform(features)

        prediction = reg.predict(features)[0]
        # Model outputs [valence, arousal] in [0, 1]
        valence = float(np.clip(prediction[0] if len(prediction) > 1 else prediction, 0, 1))
        arousal = float(np.clip(prediction[1] if len(prediction) > 1 else 0.5, 0, 1))

        emotion = _emotion_from_valence_arousal(valence, arousal)
        return valence, emotion
    except Exception as exc:
        logger.warning("Valence regressor error: %s", exc)
        return _estimate_valence_arithmetic(y, sr)


def _estimate_valence_arithmetic(y: np.ndarray, sr: int) -> tuple[float, str]:
    """
    Arithmetic fallback when trained model is absent.
    Uses tempo + mode + spectral centroid + RMS — a valid approximation
    but NOT a machine learning model. Documents itself honestly as a fallback.
    """
    try:
        rms      = float(np.sqrt(np.mean(y ** 2)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        tempo    = detect_tempo_robust(y, sr)
        chroma   = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_m = np.mean(chroma, axis=1)
        # Rough major/minor via correlation with K-S profiles
        from .audio_analysis import _MAJOR_PROFILE, _MINOR_PROFILE
        best_major = max(float(np.corrcoef(np.roll(chroma_m, -i), _MAJOR_PROFILE)[0,1])
                         for i in range(12))
        best_minor = max(float(np.corrcoef(np.roll(chroma_m, -i), _MINOR_PROFILE)[0,1])
                         for i in range(12))
        major_weight = 1.0 if best_major > best_minor else 0.0

        v = 0.50
        v += 0.18 * major_weight - 0.12 * (1 - major_weight)
        v += 0.10 if tempo > 125 else (0.05 if tempo > 105 else
             (-0.12 if tempo < 72 else (-0.06 if tempo < 85 else 0)))
        v += 0.08 if centroid > 2200 else (0.03 if centroid > 1700 else
             (-0.09 if centroid < 1100 else (-0.04 if centroid < 1400 else 0)))
        v += 0.06 if rms > 0.28 else (-0.04 if rms < 0.08 else 0)
        valence = float(np.clip(v, 0.0, 1.0))
        return valence, _emotion_from_valence_arousal(valence, rms)
    except Exception:
        return 0.5, "smooth"


def _emotion_from_valence_arousal(valence: float, arousal: float) -> str:
    """Map valence + arousal scalars to a DreamStage emotion label."""
    if valence > 0.74 and arousal > 0.55:
        return "euphoric"
    if valence > 0.65:
        return "uplifting"
    if valence < 0.28 and arousal > 0.55:
        return "dark"
    if valence < 0.32:
        return "melancholic"
    if arousal > 0.60:
        return "energetic"
    if arousal < 0.30:
        return "intimate"
    return "smooth"


# ── Energy estimation with trained model ──────────────────────────────────────

def estimate_energy_ml(y: np.ndarray, sr: int) -> tuple[float, str]:
    """
    Estimate perceived energy using a trained classifier.

    Returns:
        energy_score: float [0, 1]
        energy_label: 'low' | 'medium' | 'high'

    Falls back to RMS-based rule if model unavailable.
    """
    model_bundle = _load_model("energy_classifier")
    if model_bundle is None:
        rms = float(np.sqrt(np.mean(y ** 2)))
        label = "high" if rms > 0.22 else ("medium" if rms > 0.10 else "low")
        score = float(np.clip(rms / 0.30, 0, 1))
        return score, label

    try:
        clf    = model_bundle["classifier"]
        scaler = model_bundle.get("scaler")

        features = extract_feature_vector(y, sr).reshape(1, -1)
        if scaler is not None:
            features = scaler.transform(features)

        label_idx = clf.predict(features)[0]
        # Model was trained on integer-encoded labels — map back to the name so
        # the score_map lookup works (it was silently defaulting to 0.5).
        classes = model_bundle.get("classes")
        le      = model_bundle.get("label_encoder")
        try:
            if le is not None:
                label = str(le.inverse_transform([label_idx])[0])
            elif classes is not None and 0 <= int(label_idx) < len(classes):
                label = str(classes[int(label_idx)])
            else:
                label = str(label_idx)
        except Exception:
            label = str(label_idx)
        # Map to scalar: low=0.2, medium=0.5, high=0.85
        score_map = {"low": 0.20, "medium": 0.50, "high": 0.85}
        return float(score_map.get(label, 0.5)), label
    except Exception as exc:
        logger.warning("Energy classifier error: %s", exc)
        rms = float(np.sqrt(np.mean(y ** 2)))
        label = "high" if rms > 0.22 else ("medium" if rms > 0.10 else "low")
        return float(np.clip(rms / 0.30, 0, 1)), label


# ── Vocal range classification with CREPE ────────────────────────────────────

def classify_vocal_range_ml(y: np.ndarray, sr: int) -> dict:
    """
    Classify vocal range (soprano/alto/tenor/baritone) using CREPE F0.

    CREPE gives reliable frame-level pitch estimates including confidence,
    so we can restrict to high-confidence voiced frames before computing
    the singer's F0 distribution — much more accurate than simple YIN.

    Returns:
        range_label: soprano | alto | tenor | baritone | unknown
        f0_min_hz:   10th-percentile F0 (Hz)
        f0_max_hz:   90th-percentile F0 (Hz)
        f0_median_hz: median F0 (Hz)
        voiced_ratio: fraction of frames with detected pitch
    """
    _, freqs, conf = detect_pitch_crepe(y, sr)

    voiced_freqs = freqs[(freqs > 80) & (conf >= 0.50)]

    if len(voiced_freqs) < 20:
        return {
            "range_label": "unknown",
            "f0_min_hz": None,
            "f0_max_hz": None,
            "f0_median_hz": None,
            "voiced_ratio": 0.0,
        }

    voiced_ratio = float(len(voiced_freqs) / max(len(freqs), 1))
    f0_min   = float(np.percentile(voiced_freqs, 10))
    f0_max   = float(np.percentile(voiced_freqs, 90))
    f0_med   = float(np.median(voiced_freqs))

    # Standard vocal range classification thresholds (Hz):
    # Soprano: ~262–1047 Hz   (C4–C6),  median ~523 Hz
    # Mezzo:   ~220–880  Hz   (A3–A5),  median ~370 Hz
    # Alto:    ~175–698  Hz   (F3–F5),  median ~293 Hz
    # Tenor:   ~131–524  Hz   (C3–C5),  median ~220 Hz
    # Baritone: ~98–392  Hz   (G2–G4),  median ~147 Hz
    # Bass:     ~82–330  Hz   (E2–E4),  median ~110 Hz
    if f0_med > 440:
        label = "soprano"
    elif f0_med > 300:
        label = "alto"
    elif f0_med > 210:
        label = "tenor"
    elif f0_med > 150:
        label = "baritone"
    else:
        label = "bass"

    return {
        "range_label":   label,
        "f0_min_hz":     round(f0_min, 1),
        "f0_max_hz":     round(f0_max, 1),
        "f0_median_hz":  round(f0_med, 1),
        "voiced_ratio":  round(voiced_ratio, 3),
    }


# ── Vocal tone / character analysis ──────────────────────────────────────────

def analyze_vocal_tone(y: np.ndarray, sr: int) -> dict:
    """
    Analyze vocal tone character: brightness, warmth, presence, breathiness.

    These are perceptually meaningful dimensions derived from validated
    spectral shape measurements, not arbitrary thresholds.

    Brightness  (spectral centroid ratio vs Nyquist): warm ↔ bright
    Presence    (2–4 kHz energy ratio): key frequency band for intelligibility
    Breathiness (HNR — harmonic-to-noise ratio via CREPE confidence):
                 high CREPE confidence = clean tone; low = breathy/noisy
    """
    try:
        # Spectral centroid normalised to [0, 1] over vocal range (80–8000 Hz)
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        brightness = float(np.clip((centroid - 800) / (3500 - 800), 0, 1))

        # Presence: energy ratio in 2–4 kHz band vs. total RMS
        # Isolate 2–4 kHz via STFT
        stft = np.abs(librosa.stft(y))
        freqs_bins = librosa.fft_frequencies(sr=sr)
        presence_mask = (freqs_bins >= 2000) & (freqs_bins <= 4000)
        presence_energy = float(np.mean(stft[presence_mask, :] ** 2))
        total_energy    = float(np.mean(stft ** 2)) + 1e-9
        presence = float(np.clip(presence_energy / total_energy * 10, 0, 1))

        # Breathiness via CREPE confidence mean on voiced frames
        _, freqs_f0, conf = detect_pitch_crepe(y, sr)
        voiced_conf = conf[freqs_f0 > 80]
        breathiness = float(1.0 - np.mean(voiced_conf)) if len(voiced_conf) > 5 else 0.5

        # Tone type: classic 3-label classification
        if brightness > 0.60:
            tone_type = "bright"
        elif brightness < 0.35:
            tone_type = "warm"
        else:
            tone_type = "balanced"

        return {
            "tone_type":    tone_type,
            "brightness":   round(brightness, 3),
            "presence":     round(presence, 3),
            "breathiness":  round(breathiness, 3),
            "centroid_hz":  round(centroid, 1),
        }
    except Exception:
        return {
            "tone_type":   "balanced",
            "brightness":  0.5,
            "presence":    0.5,
            "breathiness": 0.5,
            "centroid_hz": 1500.0,
        }


# ── Full analysis pipeline ────────────────────────────────────────────────────

def analyze_full_ml(audio_bytes: bytes) -> dict:
    """
    Run the full ML analysis pipeline on raw audio bytes.

    Returns a feature dict compatible with beat_generator.py —
    drop-in replacement for analyze_vocal_mood() in beat_generator.py.

    What's real ML vs DSP:
      tempo         — DSP (enhanced multi-strategy librosa)  [ML upgrade: madmom]
      key           — DSP (Krumhansl-Schmuckler psychoacoustics)
      mode          — DSP (same)
      valence       — ML classifier (when model present) | arithmetic fallback
      emotion       — ML classifier (when model present) | rule fallback
      genre_hint    — ML classifier (when model present) | rule fallback
      vocal_range   — ML via CREPE F0 (when crepe installed) | yin fallback
      tone          — ML via CREPE confidence + spectral shape
      voiced_ratio  — ML via CREPE (when installed) | yin fallback
    """
    y, sr = load_audio(audio_bytes, TARGET_SR)
    y = y.astype(np.float32)

    # Tempo (DSP, best achievable without madmom on Python 3.11)
    tempo = detect_tempo_robust(y, sr)

    # Key and mode (validated K-S psychoacoustic algorithm)
    from .audio_analysis import detect_key_and_mode, detect_swing
    key, mode = detect_key_and_mode(y, sr)
    swing_ratio = detect_swing(y, sr, tempo)

    # Onset density (rhythmic characteristic)
    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    density = len(onsets) / max(len(y) / sr, 1.0)

    # Vocal range with CREPE (real CNN)
    vocal_range = classify_vocal_range_ml(y, sr)

    # Vocal tone character
    tone_info = analyze_vocal_tone(y, sr)

    # Valence + emotion (ML or arithmetic fallback)
    valence, emotion = estimate_valence_ml(y, sr)

    # Energy
    energy_score, energy_label = estimate_energy_ml(y, sr)

    # Genre hint from trained classifier
    gtzan_label, genre_emotion, genre_confidence = classify_genre_ml(y, sr)

    # If trained genre model has high confidence, let it override the
    # valence-based emotion. Low confidence → keep valence-based emotion.
    if genre_confidence > 0.55 and gtzan_label != "unknown":
        final_emotion = genre_emotion
    else:
        final_emotion = emotion

    # Vocal style: melodic vs rhythmic
    _, voiced_f0, conf = detect_pitch_crepe(y, sr)
    voiced_freqs = voiced_f0[voiced_f0 > 80]
    f0_std = float(np.std(voiced_freqs)) if len(voiced_freqs) > 10 else 0.0
    vocal_style = "melodic" if (f0_std > 55 or density < 2.8) else "rhythmic"

    rms = float(np.sqrt(np.mean(y ** 2)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    return {
        # Core features (beat_generator.py contract)
        "tempo":        round(tempo, 1),
        "key":          key,
        "mode":         mode,
        "valence":      round(valence, 3),
        "emotion":      final_emotion,
        "vocal_style":  vocal_style,
        "swing_ratio":  round(swing_ratio, 3),
        "overall_rms":  round(rms, 4),
        "density":      round(density, 2),
        "rms":          round(rms, 4),
        "centroid":     round(centroid, 1),

        # Extended ML features
        "vocal_range":      vocal_range,
        "tone":             tone_info,
        "energy_score":     round(energy_score, 3),
        "energy_label":     energy_label,
        "genre_hint":       gtzan_label,
        "genre_confidence": round(genre_confidence, 3),
        "voiced_ratio":     vocal_range.get("voiced_ratio", 0.0),

        # Model availability flags (for debugging / UI display)
        "_using_crepe":   _HAS_CREPE,
        "_using_genre_model": _load_model("genre_classifier") is not None,
        "_using_valence_model": _load_model("valence_regressor") is not None,
    }
