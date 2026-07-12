"""
Deep vocal analysis — key/mode detection, emotion, valence, swing, vocal style.
All features feed directly into the beat generator's genre selector.
"""
import numpy as np
import librosa
from .audio_loader import load_audio, TARGET_SR

# Krumhansl-Schmuckler tonal hierarchy profiles (psychoacoustic standard)
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                            2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                            2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def _detect_tempo_from_vocal(y: np.ndarray, sr: int) -> float:
    """
    Multi-strategy tempo estimation designed for a cappella vocals.

    librosa.beat.beat_track() chases percussion transients — on a solo vocal
    it often halves or doubles the true tempo, or locks onto the wrong onset
    density entirely. This function uses onset-envelope analysis (filtered to
    the vocal frequency range) plus tempo-range filtering to produce a stable
    estimate regardless of accompaniment.
    """
    try:
        # Onset envelope focused on vocal frequency range (not full spectrum)
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, aggregate=np.median, fmax=4000
        )

        # Strategy 1: standard beat tracking on vocal-range onset envelope
        tempos_1, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, units='bpm')
        t1 = float(np.atleast_1d(tempos_1)[0])

        # Strategy 2: PLP (periodicity likelihood proxy) for smoother estimation
        try:
            plp = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
            tempos_2, _ = librosa.beat.beat_track(onset_envelope=plp, sr=sr, units='bpm')
            t2 = float(np.atleast_1d(tempos_2)[0])
        except Exception:
            t2 = t1

        # Consider doubled/halved candidates — beat tracker commonly errors by 2×
        candidates = [t1, t2, t1 * 2.0, t1 / 2.0, t2 * 2.0, t2 / 2.0]
        in_range = [t for t in candidates if 55.0 <= t <= 165.0]

        if not in_range:
            return 90.0

        # Prefer the natural range for modern music (75–145 BPM)
        preferred = [t for t in in_range if 75.0 <= t <= 145.0]
        if preferred:
            return float(round(np.median(preferred), 1))
        return float(round(np.median(in_range), 1))
    except Exception:
        return 90.0


def detect_key_and_mode(y: np.ndarray, sr: int) -> tuple:
    """Detect musical key and mode (major/minor) via chroma + K-S profiles."""
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)
        chroma_mean = np.mean(chroma, axis=1)

        best_score = -np.inf
        best_key = 'C'
        best_mode = 'major'

        for i in range(12):
            rotated = np.roll(chroma_mean, -i)
            major_score = float(np.corrcoef(rotated, _MAJOR_PROFILE)[0, 1])
            minor_score = float(np.corrcoef(rotated, _MINOR_PROFILE)[0, 1])
            if major_score > best_score:
                best_score = major_score
                best_key = NOTE_NAMES[i]
                best_mode = 'major'
            if minor_score > best_score:
                best_score = minor_score
                best_key = NOTE_NAMES[i]
                best_mode = 'minor'

        return best_key, best_mode
    except Exception:
        return 'C', 'major'


def detect_swing(y: np.ndarray, sr: int, tempo: float) -> float:
    """
    Estimate groove swing ratio from beat subdivision irregularity.
    0.50 = perfectly straight (no swing)
    0.67 = maximum triplet swing
    """
    try:
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, bpm=tempo)
        if len(beat_frames) < 6:
            return 0.50
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        intervals = np.diff(beat_times)
        if len(intervals) < 4:
            return 0.50
        even = intervals[0::2]
        odd  = intervals[1::2]
        min_len = min(len(even), len(odd))
        if min_len == 0:
            return 0.50
        ratio = float(np.mean(even[:min_len]) /
                      (np.mean(even[:min_len]) + np.mean(odd[:min_len])))
        return float(np.clip(ratio, 0.44, 0.70))
    except Exception:
        return 0.50


def estimate_valence(tempo: float, mode: str, centroid: float, rms: float) -> float:
    """
    Estimate emotional valence (0 = dark/sad, 1 = happy/bright).
    Based on Thayer's arousal-valence model and music psychology research.
    """
    v = 0.50
    v += 0.18 if mode == 'major' else -0.12
    if tempo > 125:
        v += 0.10
    elif tempo > 105:
        v += 0.05
    elif tempo < 72:
        v -= 0.12
    elif tempo < 85:
        v -= 0.06
    if centroid > 2200:
        v += 0.08
    elif centroid > 1700:
        v += 0.03
    elif centroid < 1100:
        v -= 0.09
    elif centroid < 1400:
        v -= 0.04
    if rms > 0.28:
        v += 0.06
    elif rms < 0.08:
        v -= 0.04
    return float(np.clip(v, 0.0, 1.0))


def classify_emotion(mode: str, valence: float, rms: float, tempo: float) -> str:
    """Map features to an emotion label consumed by the beat genre selector."""
    if valence > 0.74 and rms > 0.20:
        return "euphoric"
    if valence > 0.65 and tempo > 100:
        return "uplifting"
    if valence < 0.28 and tempo >= 120:
        return "dark"
    if valence < 0.32:
        return "melancholic"
    if rms > 0.22 and tempo >= 112:
        return "energetic"
    if tempo < 80:
        return "intimate"
    return "smooth"


def classify_vocal_style(f0_array: np.ndarray, onset_density: float) -> str:
    """
    Determine if the vocal delivery is more melodic (singing) or
    rhythmic (rap/spoken word) — influences groove and swing selection.
    """
    if len(f0_array) < 10:
        return "rhythmic"
    f0_std = float(np.std(f0_array))
    # High pitch variance + lower onset density = melodic singer
    if f0_std > 55 or onset_density < 2.8:
        return "melodic"
    return "rhythmic"


def _classify_tone(centroid_hz: float) -> str:
    if centroid_hz > 2200:
        return "bright"
    elif centroid_hz < 1300:
        return "warm"
    return "balanced"


def _most_common(lst: list):
    return max(set(lst), key=lst.count) if lst else None


def extract_voice_profile(recordings: list, language: str) -> dict:
    """
    Analyze multiple vocal recordings and return a rich Voice Profile dict.
    recordings: list of raw audio bytes (webm/opus/wav — any format)
    """
    all_f0       = []
    all_tempos   = []
    all_centroids = []
    all_keys     = []
    all_modes    = []
    all_swings   = []
    all_rms      = []
    all_densities = []

    for audio_bytes in recordings:
        try:
            y, sr = load_audio(audio_bytes, TARGET_SR)
        except Exception:
            continue

        # Fundamental frequency (YIN algorithm — robust for singing)
        f0 = librosa.yin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=TARGET_SR,
        )
        voiced = f0[f0 > 80]
        if len(voiced) > 10:
            all_f0.extend(voiced.tolist())

        # Tempo (multi-strategy — beat_track alone is unreliable on a cappella vocals)
        tempo_val = _detect_tempo_from_vocal(y, TARGET_SR)
        all_tempos.append(tempo_val)

        # Spectral centroid (voice brightness/warmth)
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=TARGET_SR)))
        all_centroids.append(centroid)

        # Key and mode detection
        key, mode = detect_key_and_mode(y, TARGET_SR)
        all_keys.append(key)
        all_modes.append(mode)

        # Swing ratio
        all_swings.append(detect_swing(y, TARGET_SR, tempo_val))

        # Energy (RMS)
        all_rms.append(float(np.sqrt(np.mean(y ** 2))))

        # Onset density (rhythmic density of the vocal)
        onsets = librosa.onset.onset_detect(y=y, sr=TARGET_SR)
        all_densities.append(len(onsets) / max(len(y) / TARGET_SR, 1.0))

    # ── Aggregate ──────────────────────────────────────────────────────────────
    f0_arr = np.array(all_f0)
    if len(f0_arr) > 10:
        min_freq = float(np.percentile(f0_arr, 10))
        max_freq = float(np.percentile(f0_arr, 90))
    else:
        min_freq = max_freq = None

    tempo_bpm   = float(np.mean(all_tempos))   if all_tempos   else None
    avg_centroid = float(np.mean(all_centroids)) if all_centroids else 1500.0
    avg_rms      = float(np.mean(all_rms))      if all_rms      else 0.15
    avg_swing    = float(np.mean(all_swings))   if all_swings   else 0.50
    avg_density  = float(np.mean(all_densities)) if all_densities else 2.0

    key  = _most_common(all_keys)  or "C"
    mode = _most_common(all_modes) or "major"

    valence     = estimate_valence(tempo_bpm or 90, mode, avg_centroid, avg_rms)
    emotion     = classify_emotion(mode, valence, avg_rms, tempo_bpm or 90)
    vocal_style = classify_vocal_style(f0_arr, avg_density)
    tone_type   = _classify_tone(avg_centroid)

    return {
        "language":   language,
        "min_freq_hz": round(min_freq, 2) if min_freq else None,
        "max_freq_hz": round(max_freq, 2) if max_freq else None,
        "tempo_bpm":   round(tempo_bpm, 1) if tempo_bpm else None,
        "tone_type":   tone_type,
        # Deep features used by beat generator
        "key":         key,
        "mode":        mode,
        "valence":     round(valence, 3),
        "emotion":     emotion,
        "vocal_style": vocal_style,
        "swing_ratio": round(avg_swing, 3),
        "overall_rms": round(avg_rms, 4),
        "onset_density": round(avg_density, 2),
    }
