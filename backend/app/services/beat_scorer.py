"""
Beat quality scoring for candidate selection and rejection.

Scores a generated beat across 6 dimensions:
  tempo_match     — how closely beat tempo matches vocal tempo
  key_match       — does beat harmonic content match vocal key/mode
  energy_match    — does beat energy level suit the vocal energy
  mood_match      — does the beat genre/emotion suit the vocal mood
  dynamic_quality — does the beat have internal dynamic variation (not flat)
  rhythmic_quality — rhythmic complexity and arrangement structure

Used by the candidate system to select the best of 5 generated beats
and by the rejection gate to re-generate if the winner scores below 45/100.
"""
from __future__ import annotations

import numpy as np
import librosa
from .audio_loader import load_audio, TARGET_SR


# ── Public API ────────────────────────────────────────────────────────────────

def score_beat(beat_bytes: bytes, vocal_analysis: dict,
               genre_used: str = "") -> dict:
    """
    Score a beat against the vocal analysis.
    Returns dict with individual component scores and a total 0–100.
    """
    try:
        y, sr = load_audio(beat_bytes, TARGET_SR)
    except Exception:
        return _zero_score()

    tempo_s  = _score_tempo_match(y, sr, vocal_analysis)
    key_s    = _score_key_match(y, sr, vocal_analysis)
    energy_s = _score_energy_match(y, sr, vocal_analysis)
    mood_s   = _score_mood_match(genre_used, vocal_analysis)
    dyn_s    = _score_dynamic_quality(y, sr)
    rhythm_s = _score_rhythmic_quality(y, sr)

    total = (
        tempo_s  * 0.25 +
        key_s    * 0.15 +
        energy_s * 0.15 +
        mood_s   * 0.15 +
        dyn_s    * 0.15 +
        rhythm_s * 0.15
    )

    return {
        "total":           round(float(total), 1),
        "tempo_match":     round(float(tempo_s), 1),
        "key_match":       round(float(key_s), 1),
        "energy_match":    round(float(energy_s), 1),
        "mood_match":      round(float(mood_s), 1),
        "dynamic_quality": round(float(dyn_s), 1),
        "rhythm_quality":  round(float(rhythm_s), 1),
        "genre_used":      genre_used,
    }


REJECTION_THRESHOLD = 45.0   # re-generate if winner scores below this


def should_reject(score_dict: dict) -> bool:
    return score_dict.get("total", 0) < REJECTION_THRESHOLD


# ── Tempo match ───────────────────────────────────────────────────────────────

def _score_tempo_match(y: np.ndarray, sr: int, analysis: dict) -> float:
    vocal_tempo = float(analysis.get("tempo", 90.0))
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
        tempos, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, units="bpm")
        beat_tempo = float(np.atleast_1d(tempos)[0])
        # Consider half-time / double-time
        candidates = [beat_tempo, beat_tempo * 2, beat_tempo / 2]
        best_diff = min(abs(c - vocal_tempo) for c in candidates)
        # 0 BPM diff = 100; 5 BPM = 80; 10 BPM = 60; 20+ BPM = 0
        return float(max(0, 100 - best_diff * 4))
    except Exception:
        return 50.0


# ── Key match ─────────────────────────────────────────────────────────────────

_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                            2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                            2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']


def _score_key_match(y: np.ndarray, sr: int, analysis: dict) -> float:
    vocal_key  = analysis.get("key", "C")
    vocal_mode = analysis.get("mode", "major")
    target_idx = _NOTES.index(vocal_key) if vocal_key in _NOTES else 0
    try:
        chroma   = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)
        cm       = np.mean(chroma, axis=1)
        best_score, best_key, best_mode = -999, 0, "major"
        for i in range(12):
            rot = np.roll(cm, -i)
            mj  = float(np.corrcoef(rot, _MAJOR_PROFILE)[0, 1])
            mn  = float(np.corrcoef(rot, _MINOR_PROFILE)[0, 1])
            if mj > best_score: best_score, best_key, best_mode = mj, i, "major"
            if mn > best_score: best_score, best_key, best_mode = mn, i, "minor"

        semitone_diff = min(abs(best_key - target_idx),
                            12 - abs(best_key - target_idx))
        # Exact key + mode = 100; relative key (3 semitones) = 70; wrong = 30
        if semitone_diff == 0 and best_mode == vocal_mode:
            return 100.0
        if semitone_diff == 0:
            return 80.0     # right root, wrong mode
        if semitone_diff == 3:
            return 70.0     # relative major/minor
        if semitone_diff <= 2:
            return 55.0
        return max(30.0, 100.0 - semitone_diff * 10)
    except Exception:
        return 70.0  # synthesizer always uses vocal key, so default high


# ── Energy match ─────────────────────────────────────────────────────────────

def _score_energy_match(y: np.ndarray, sr: int, analysis: dict) -> float:
    vocal_rms = float(analysis.get("overall_rms", 0.18))
    beat_rms  = float(np.sqrt(np.mean(y ** 2)))
    # Beat should be slightly quieter or equal (beat -3 dB below vocal in the mix)
    # Ideal beat RMS is 70–90% of vocal RMS
    ideal_lo = vocal_rms * 0.65
    ideal_hi = vocal_rms * 1.10
    if ideal_lo <= beat_rms <= ideal_hi:
        return 100.0
    diff = min(abs(beat_rms - ideal_lo), abs(beat_rms - ideal_hi))
    return float(max(30.0, 100.0 - (diff / vocal_rms) * 200))


# ── Mood match ────────────────────────────────────────────────────────────────

_MOOD_GENRE_MAP = {
    "euphoric":    {"trap_melodic", "pop_bright", "afrobeats", "reggaeton"},
    "uplifting":   {"pop_bright", "afrobeats", "hiphop_modern", "rnb_smooth"},
    "dark":        {"trap_dark", "drill", "uk_drill", "phonk"},
    "melancholic": {"rnb_neo_soul", "soul_ballad", "lofi_chill"},
    "energetic":   {"hiphop_modern", "hiphop_boom_bap", "drill", "trap_dark"},
    "intimate":    {"soul_ballad", "lofi_chill", "rnb_neo_soul"},
    "smooth":      {"rnb_smooth", "rnb_neo_soul", "hiphop_modern"},
}

_AI_GENRES = {"musicgen_ai", "musicgen_hf_api", "musicgen_local", "musicgen_gradio"}


def _score_mood_match(genre_used: str, analysis: dict) -> float:
    if genre_used in _AI_GENRES:
        return 85.0  # AI beat is conditioned on the vocal mood — give benefit of doubt
    emotion = analysis.get("emotion", "smooth")
    ideal   = _MOOD_GENRE_MAP.get(emotion, set())
    if genre_used in ideal:
        return 100.0
    # Check if it's at least in the same energy tier
    for mood, genres in _MOOD_GENRE_MAP.items():
        if genre_used in genres and mood == emotion:
            return 100.0
    # Partial match: some genres work for adjacent moods
    adjacent = {
        "euphoric": {"uplifting"},
        "uplifting": {"euphoric", "smooth"},
        "dark": {"energetic"},
        "melancholic": {"intimate", "smooth"},
        "energetic": {"dark", "uplifting"},
        "intimate": {"melancholic"},
        "smooth": {"uplifting", "melancholic"},
    }
    for adj_mood in adjacent.get(emotion, set()):
        if genre_used in _MOOD_GENRE_MAP.get(adj_mood, set()):
            return 65.0
    return 30.0


# ── Dynamic quality ───────────────────────────────────────────────────────────

def _score_dynamic_quality(y: np.ndarray, sr: int) -> float:
    """
    Score how much the beat changes dynamically over time.
    A flat-energy beat scores low; a beat with distinct sections (intro/
    chorus/outro energy levels) scores high.
    """
    try:
        # Split into 8 segments, measure RMS of each
        segments = np.array_split(y, 8)
        rms_vals  = [float(np.sqrt(np.mean(s ** 2))) for s in segments if len(s) > 0]
        if len(rms_vals) < 4:
            return 50.0

        # Coefficient of variation (std/mean): 0 = flat, higher = more dynamic
        cv = float(np.std(rms_vals) / (np.mean(rms_vals) + 1e-9))
        # Also check for clear peak (chorus drop)
        has_peak = max(rms_vals) > np.mean(rms_vals) * 1.25
        # Score: CV of 0.10–0.25 is ideal for hip-hop production
        cv_score = min(100, cv * 600)  # 0.10 → 60, 0.20 → 120 (capped at 100)
        peak_bonus = 15.0 if has_peak else 0.0
        return float(min(100, cv_score + peak_bonus))
    except Exception:
        return 50.0


# ── Rhythmic quality ──────────────────────────────────────────────────────────

def _score_rhythmic_quality(y: np.ndarray, sr: int) -> float:
    """
    Score rhythmic complexity and consistency.
    A beat with varied onset patterns and consistent tempo scores high.
    """
    try:
        # Onset density
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
        duration = len(y) / sr
        onset_rate = len(onsets) / max(duration, 1.0)

        # Good hip-hop beat: 4–10 onsets/sec
        rate_score = 0.0
        if 4 <= onset_rate <= 10:
            rate_score = 100.0
        elif 2 <= onset_rate < 4:
            rate_score = 65.0
        elif 10 < onset_rate <= 15:
            rate_score = 70.0
        else:
            rate_score = 30.0

        # Onset regularity (low std of inter-onset intervals = steady groove)
        if len(onsets) > 8:
            ioi = np.diff(onsets)
            regularity = 1.0 - float(np.clip(np.std(ioi) / (np.mean(ioi) + 1e-9), 0, 1))
            reg_score = regularity * 100
        else:
            reg_score = 50.0

        return float(rate_score * 0.6 + reg_score * 0.4)
    except Exception:
        return 50.0


def _zero_score() -> dict:
    return {
        "total": 0.0, "tempo_match": 0.0, "key_match": 0.0,
        "energy_match": 0.0, "mood_match": 0.0,
        "dynamic_quality": 0.0, "rhythm_quality": 0.0, "genre_used": "",
    }
