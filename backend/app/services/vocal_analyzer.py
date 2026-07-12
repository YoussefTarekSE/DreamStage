"""
Advanced vocal analysis for beat generation.
Extracts musical key, groove feel, emotional content, style, and detailed
spectral features — everything the beat generator needs to make a perfect match.
"""
import numpy as np
import librosa
from .audio_loader import load_audio


CHROMATIC_KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# For each root, major/minor scale degrees (relative to root)
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]


def analyze_for_coaching(processed_bytes: bytes) -> dict:
    """Detailed analysis for the AI coach."""
    return _full_analysis(processed_bytes)


def analyze_for_beat(processed_bytes: bytes) -> dict:
    """
    Full analysis for beat generation:
    tempo, key, mode, groove, energy, emotional valence, vocal style, range.
    """
    return _full_analysis(processed_bytes)


def _full_analysis(audio_bytes: bytes) -> dict:
    try:
        y, sr = load_audio(audio_bytes)
    except Exception:
        return _defaults()

    duration = len(y) / sr

    # ── Tempo + beat tracking ─────────────────────────────────────────────────
    # Use vocal-range onset envelope for tempo — standard beat_track halves/doubles
    # tempo on a cappella vocals because it chases percussion transients
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median, fmax=4000)
    tempo_arr, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, trim=False)
    tempo_raw = float(np.atleast_1d(tempo_arr)[0])
    # Apply same range-clamping heuristic as audio_analysis.py
    candidates = [tempo_raw, tempo_raw * 2.0, tempo_raw / 2.0]
    in_range = [t for t in candidates if 55.0 <= t <= 165.0]
    preferred = [t for t in in_range if 75.0 <= t <= 145.0]
    tempo = float(round(np.median(preferred if preferred else (in_range or [90.0])), 1))
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # ── Groove feel (swing ratio) ─────────────────────────────────────────────
    # Detect if beats land early/late relative to a strict grid = swing feel
    swing_ratio = _detect_swing(beat_times, tempo)

    # ── Musical key detection via chroma + Krumhansl-Schmuckler ──────────────
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)
    mean_chroma = np.mean(chroma, axis=1)
    key_name, mode, key_confidence = _detect_key(mean_chroma)

    # ── Pitch analysis ────────────────────────────────────────────────────────
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'), sr=sr
        )
        voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
        pitch_accuracy = float(1.0 - np.clip(np.mean(np.abs(librosa.hz_to_midi(voiced_f0) - np.round(librosa.hz_to_midi(voiced_f0)))) * 2, 0, 1)) if len(voiced_f0) > 10 else 0.75
        pitch_stability = float(1.0 - np.clip(np.std(np.abs(librosa.hz_to_midi(voiced_f0) - np.round(librosa.hz_to_midi(voiced_f0)))), 0, 1)) if len(voiced_f0) > 10 else 0.70
        voiced_pct = float(len(voiced_f0) / max(len(f0[~np.isnan(f0)]), 1))

        # Vocal range classification
        if len(voiced_f0) > 10:
            median_hz = float(np.median(voiced_f0))
            if median_hz > 350:
                vocal_range = "soprano"
            elif median_hz > 260:
                vocal_range = "alto"
            elif median_hz > 196:
                vocal_range = "tenor"
            else:
                vocal_range = "baritone"
        else:
            vocal_range = "unknown"
    except Exception:
        pitch_accuracy, pitch_stability, voiced_pct = 0.75, 0.70, 0.55
        vocal_range = "unknown"

    # ── Energy + dynamics ─────────────────────────────────────────────────────
    rms_frames = librosa.feature.rms(y=y)[0]
    rms_db = librosa.amplitude_to_db(rms_frames + 1e-9)
    overall_rms = float(np.sqrt(np.mean(y ** 2)))
    dynamic_range = float(np.percentile(rms_db, 90) - np.percentile(rms_db, 10))

    # Section energy profile (4 quarters)
    parts = np.array_split(y, 4)
    section_rms = [float(np.sqrt(np.mean(p ** 2))) for p in parts]
    energy_arc = _classify_energy_arc(section_rms)
    energy_consistency = float(1.0 - np.std(section_rms) / (np.mean(section_rms) + 1e-9))

    # ── Spectral features for mood/tone ───────────────────────────────────────
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spectral_rolloff  = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    spectral_contrast = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
    zero_crossing     = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

    # ── Emotional valence proxy ───────────────────────────────────────────────
    # High centroid + major key + fast tempo → brighter/happier
    # Low centroid + minor key + slow tempo → darker/sadder
    valence = _estimate_valence(spectral_centroid, mode, tempo, overall_rms)
    emotion = _classify_emotion(valence, tempo, dynamic_range)

    # ── Onset density (vocal style: sparse vs dense) ──────────────────────────
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    onset_rate = float(len(onsets) / max(duration, 1.0))
    vocal_style = "flowing" if onset_rate < 2.5 else ("rhythmic" if onset_rate < 5.0 else "rapid")

    # ── Sibilance ─────────────────────────────────────────────────────────────
    spec = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    sib_mask = freqs > 6000
    sibilance_ratio = float(np.mean(spec[sib_mask]) / (np.mean(spec) + 1e-9))

    return {
        # Core music theory
        "key": key_name,
        "mode": mode,
        "key_confidence": round(key_confidence, 3),
        "tempo": round(tempo, 1),
        "swing_ratio": round(swing_ratio, 3),

        # Vocal characteristics
        "vocal_range": vocal_range,
        "vocal_style": vocal_style,
        "pitch_accuracy": round(pitch_accuracy, 3),
        "pitch_stability": round(pitch_stability, 3),
        "voiced_pct": round(voiced_pct, 3),

        # Energy + emotion
        "overall_rms": round(overall_rms, 4),
        "dynamic_range_db": round(dynamic_range, 1),
        "energy_arc": energy_arc,
        "energy_consistency": round(energy_consistency, 3),
        "section_rms": [round(r, 4) for r in section_rms],
        "valence": round(valence, 3),
        "emotion": emotion,

        # Spectral
        "spectral_centroid": round(spectral_centroid, 1),
        "spectral_rolloff": round(spectral_rolloff, 1),
        "spectral_contrast": round(spectral_contrast, 3),
        "zero_crossing_rate": round(zero_crossing, 4),
        "sibilance_ratio": round(sibilance_ratio, 4),
        "onset_rate_per_sec": round(onset_rate, 2),

        "duration_sec": round(duration, 1),
    }


def _detect_key(mean_chroma: np.ndarray) -> tuple:
    """
    Krumhansl-Schmuckler key-finding algorithm.
    Returns (key_name, 'major'|'minor', confidence).
    """
    # Major and minor key profiles (Krumhansl & Kessler 1982)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    best_key, best_mode, best_corr = 0, "major", -1.0

    for i in range(12):
        # Rotate chroma to align with key i
        rotated = np.roll(mean_chroma, -i)

        corr_major = float(np.corrcoef(rotated, major_profile)[0, 1])
        corr_minor = float(np.corrcoef(rotated, minor_profile)[0, 1])

        if corr_major > best_corr:
            best_corr, best_key, best_mode = corr_major, i, "major"
        if corr_minor > best_corr:
            best_corr, best_key, best_mode = corr_minor, i, "minor"

    return CHROMATIC_KEYS[best_key], best_mode, best_corr


def _detect_swing(beat_times: np.ndarray, tempo: float) -> float:
    """
    Estimate swing ratio from inter-beat intervals.
    Returns value near 0.5 = straight, near 0.67 = triplet swing.
    """
    if len(beat_times) < 4:
        return 0.5
    ibi = np.diff(beat_times)
    # Split into even/odd pairs
    even_ibi = ibi[::2]
    odd_ibi  = ibi[1::2]
    n = min(len(even_ibi), len(odd_ibi))
    if n == 0:
        return 0.5
    ratio = float(np.mean(even_ibi[:n]) / (np.mean(even_ibi[:n]) + np.mean(odd_ibi[:n]) + 1e-9))
    return np.clip(ratio, 0.4, 0.7)


def _classify_energy_arc(section_rms: list) -> str:
    if len(section_rms) < 4:
        return "steady"
    start, *mid, end = section_rms
    avg_mid = float(np.mean(mid))
    if end > start * 1.3 and end > avg_mid * 1.2:
        return "builds"
    elif start > end * 1.3:
        return "fades"
    elif avg_mid > start * 1.3 and avg_mid > end * 1.3:
        return "peaks_middle"
    elif float(np.std(section_rms)) < 0.02:
        return "steady"
    else:
        return "dynamic"


def _estimate_valence(centroid: float, mode: str, tempo: float, rms: float) -> float:
    """Rough emotional valence: 0 = dark/sad, 1 = bright/happy."""
    v = 0.5
    v += 0.15 if mode == "major" else -0.15
    v += 0.10 if tempo > 110 else (-0.10 if tempo < 75 else 0)
    v += 0.10 if centroid > 2000 else (-0.10 if centroid < 1200 else 0)
    v += 0.05 if rms > 0.20 else (-0.05 if rms < 0.08 else 0)
    return float(np.clip(v, 0.0, 1.0))


def _classify_emotion(valence: float, tempo: float, dynamic_range: float) -> str:
    if valence > 0.65 and tempo > 110:
        return "euphoric"
    elif valence > 0.60:
        return "uplifting"
    elif valence > 0.50 and tempo > 95:
        return "confident"
    elif valence > 0.45:
        return "smooth"
    elif valence > 0.38 and dynamic_range > 15:
        return "intense"
    elif valence > 0.35:
        return "melancholic"
    else:
        return "dark"


def _defaults() -> dict:
    return {
        "key": "C", "mode": "major", "key_confidence": 0.5,
        "tempo": 90.0, "swing_ratio": 0.5,
        "vocal_range": "unknown", "vocal_style": "rhythmic",
        "pitch_accuracy": 0.75, "pitch_stability": 0.70, "voiced_pct": 0.55,
        "overall_rms": 0.15, "dynamic_range_db": 12.0,
        "energy_arc": "steady", "energy_consistency": 0.80,
        "section_rms": [0.15, 0.16, 0.15, 0.14],
        "valence": 0.50, "emotion": "smooth",
        "spectral_centroid": 1500.0, "spectral_rolloff": 3000.0,
        "spectral_contrast": 20.0, "zero_crossing_rate": 0.08,
        "sibilance_ratio": 0.08, "onset_rate_per_sec": 2.5,
        "duration_sec": 30.0,
    }
