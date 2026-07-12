"""
Vocal harmony analysis — make the beat follow the singer's actual tune.

This transcribes the recorded vocal's pitch, detects the real musical key from
the notes the artist actually sang, and — bar by bar — chooses the chord that
best supports the vocal's prominent notes. The beat generator then builds its
bass + chords from those choices, so the harmony moves WITH the melody instead
of using a random progression in a roughly-guessed key.

It also exposes scale helpers so the vocal can be tuned to the song's key
(scale-aware autotune) — vocal and beat then sit in the exact same key.
"""
from __future__ import annotations

import numpy as np
import librosa
from .audio_loader import load_audio

_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Krumhansl-Schmuckler key profiles
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]   # natural minor

# Per-degree musicality prior (0-indexed scale degrees I..vii). Favour the
# strong functional chords, discourage the diminished vii°.
_DEGREE_PRIOR_MAJOR = np.array([1.00, 0.55, 0.60, 0.85, 0.90, 0.80, 0.30])
_DEGREE_PRIOR_MINOR = np.array([1.00, 0.40, 0.70, 0.80, 0.75, 0.85, 0.60])


def scale_pitch_classes(key: str, mode: str) -> set:
    root = _NOTE_NAMES.index(key) if key in _NOTE_NAMES else 0
    intervals = MAJOR_SCALE if mode == "major" else MINOR_SCALE
    return {(root + i) % 12 for i in intervals}


def snap_midi_to_scale(midi_value: float, key: str, mode: str) -> int:
    """Nearest MIDI note whose pitch class is in the key's scale (for tuning)."""
    pcs = scale_pitch_classes(key, mode)
    target = int(round(midi_value))
    for d in range(0, 7):
        for cand in (target - d, target + d):
            if cand % 12 in pcs:
                return cand
    return target


def scale_step_up(midi_value: float, key: str, mode: str, steps: int) -> int:
    """
    Move a note up (or down) by `steps` DIATONIC scale degrees within the key.
    e.g. steps=2 → a diatonic third, steps=4 → a diatonic fifth. Used to build
    in-key vocal harmonies. Returns an integer MIDI note.
    """
    root = _NOTE_NAMES.index(key) if key in _NOTE_NAMES else 0
    intervals = MAJOR_SCALE if mode == "major" else MINOR_SCALE
    m = int(round(midi_value))
    pc = (m - root) % 12
    if pc not in intervals:                      # snap to nearest scale tone first
        pc = min(intervals, key=lambda iv: min(abs(iv - pc), 12 - abs(iv - pc)))
    deg = intervals.index(pc)
    base = m - root - pc                          # octave base (a multiple of 12)
    new_deg = deg + steps
    new_pc = intervals[new_deg % 7]
    new_base = base + 12 * (new_deg // 7)
    return root + new_pc + new_base


def detect_key_from_histogram(pc_hist: np.ndarray) -> tuple[str, str]:
    """Key + mode from a 12-bin pitch-class histogram via KS correlation."""
    if float(np.sum(pc_hist)) < 1e-6:
        return "C", "minor"
    best_score, best_key, best_mode = -2.0, "C", "minor"
    for i in range(12):
        rot = np.roll(pc_hist, -i)
        for prof, mode in ((_KS_MAJOR, "major"), (_KS_MINOR, "minor")):
            c = np.corrcoef(rot, prof)[0, 1]
            if np.isfinite(c) and c > best_score:
                best_score, best_key, best_mode = float(c), _NOTE_NAMES[i], mode
    return best_key, best_mode


def _bar_chord_degree(bar_hist: np.ndarray, key: str, mode: str,
                      prev_degree: int | None) -> int:
    """
    Pick the diatonic triad (0-indexed scale degree) that best supports the
    vocal pitch-classes in this bar. Score = energy on the triad's tones minus
    energy on out-of-chord tones, weighted by a functional prior and a mild
    bias to stay on the previous chord (smoother progression).
    """
    root_pc = _NOTE_NAMES.index(key) if key in _NOTE_NAMES else 0
    intervals = MAJOR_SCALE if mode == "major" else MINOR_SCALE
    prior = _DEGREE_PRIOR_MAJOR if mode == "major" else _DEGREE_PRIOR_MINOR
    total = float(np.sum(bar_hist))

    if total < 1e-6:
        return prev_degree if prev_degree is not None else 0   # silence → hold/tonic

    best_score, best_deg = -1e9, 0
    for d in range(7):
        triad_pcs = {(root_pc + intervals[(d + k) % 7]) % 12 for k in (0, 2, 4)}
        in_chord = sum(bar_hist[pc] for pc in triad_pcs)
        out_chord = total - in_chord
        score = (in_chord - 0.6 * out_chord) / total
        score *= prior[d]
        if prev_degree is not None and d == prev_degree:
            score += 0.08   # gentle continuity bias
        if score > best_score:
            best_score, best_deg = score, d
    return best_deg


def _segment_melody(midi: np.ndarray, voiced: np.ndarray, times: np.ndarray,
                    key: str, mode: str, tempo: float) -> tuple[list, float]:
    """
    Turn the frame-level F0 into a clean note list the beat can play back:
    snap each voiced frame to the key's scale, group runs of the same note into
    note events, drop blips, then quantise onsets + durations to a 1/8 grid.

    Returns (notes, loop_beats) where notes = [(beat_pos, midi, dur_beats)] and
    loop_beats is the melody's length rounded up to whole bars (so it tiles
    cleanly across the beat).
    """
    beat_dur = 60.0 / max(tempo, 1.0)
    snapped = np.full(len(midi), -1, dtype=int)
    for i in range(len(midi)):
        if voiced[i] and np.isfinite(midi[i]):
            snapped[i] = snap_midi_to_scale(midi[i], key, mode)

    # Group consecutive equal-pitch frames into note segments (tolerate 1-frame gaps)
    raw = []
    i, n = 0, len(snapped)
    while i < n:
        if snapped[i] < 0:
            i += 1
            continue
        note = snapped[i]
        start = times[i]
        j = i + 1
        gap = 0
        while j < n and (snapped[j] == note or (snapped[j] < 0 and gap < 2)):
            if snapped[j] < 0:
                gap += 1
            else:
                gap = 0
            j += 1
        end = times[min(j, n - 1)]
        raw.append((start, end, note))
        i = j

    # Quantise to 1/8 grid, drop notes shorter than a 16th
    grid = beat_dur / 2.0           # 1/8 note
    min_dur = beat_dur / 4.0        # 1/16 note
    notes = []
    for start, end, note in raw:
        if end - start < min_dur:
            continue
        beat_pos = round((start / beat_dur) / 0.5) * 0.5          # snap onset to 1/8
        dur_beats = max(0.5, round(((end - start) / beat_dur) / 0.5) * 0.5)
        notes.append((beat_pos, int(note), float(dur_beats)))

    if not notes:
        return [], 4.0
    span_beats = notes[-1][0] + notes[-1][2]
    loop_beats = float(max(4, int(np.ceil(span_beats / 4.0)) * 4))  # round up to bars
    return notes, loop_beats


def transcribe_harmony(audio_bytes: bytes, tempo: float, bars: int) -> dict | None:
    """
    Transcribe the vocal and return the harmony the beat should follow:

        {
          "key":  str, "mode": str,
          "bar_degrees": [int]*bars   # 0-indexed scale degree of the chord/bar
          "confidence": float          # how pitched/clear the vocal was (0-1)
        }

    The vocal's own chord sequence (over however many bars it spans) is tiled
    across the beat's `bars`, so the beat repeats the singer's harmonic motion.
    Returns None when the recording has too little pitched content to trust.
    """
    try:
        y, sr = load_audio(audio_bytes)            # mono, 22050 Hz
        if len(y) < sr // 2:
            return None

        f0, vflag, prob = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"),
            sr=sr, frame_length=2048,
        )
        times = librosa.times_like(f0, sr=sr)
        voiced = vflag & ~np.isnan(f0) & (prob > 0.5)
        if int(np.sum(voiced)) < 15:
            return None

        midi = np.full_like(f0, np.nan)
        midi[voiced] = librosa.hz_to_midi(f0[voiced])
        pcs = (np.round(midi[voiced]).astype(int)) % 12

        # Overall key from the duration-weighted pitch-class histogram
        global_hist = np.bincount(pcs, minlength=12).astype(float)
        key, mode = detect_key_from_histogram(global_hist)

        # How many bars the vocal actually spans
        beat_dur = 60.0 / max(tempo, 1.0)
        bar_dur = 4.0 * beat_dur
        vocal_dur = len(y) / sr
        vocal_bars = max(1, int(np.ceil(vocal_dur / bar_dur)))

        # Chord degree per vocal bar, then tile across the beat's bars
        vocal_degrees: list[int] = []
        prev = None
        for b in range(vocal_bars):
            t0, t1 = b * bar_dur, (b + 1) * bar_dur
            m = voiced & (times >= t0) & (times < t1)
            bar_hist = (np.bincount(np.round(midi[m]).astype(int) % 12, minlength=12)
                        .astype(float) if np.any(m) else np.zeros(12))
            deg = _bar_chord_degree(bar_hist, key, mode, prev)
            vocal_degrees.append(deg)
            prev = deg

        bar_degrees = [vocal_degrees[b % len(vocal_degrees)] for b in range(bars)]
        confidence = float(np.clip(np.mean(prob[voiced]) if np.any(voiced) else 0.0, 0, 1))

        # Note-for-note lead melody the beat can double
        melody, loop_beats = _segment_melody(midi, voiced, times, key, mode, tempo)

        return {"key": key, "mode": mode, "bar_degrees": bar_degrees,
                "confidence": round(confidence, 3),
                "melody": melody, "melody_loop_beats": loop_beats}
    except Exception:
        return None
