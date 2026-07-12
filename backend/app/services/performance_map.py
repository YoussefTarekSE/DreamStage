"""
Performance map — read what the singer is DOING, moment to moment.

The doctrine: the vocal is the composition. The beat is not a template the voice
is dropped onto; it is accompaniment that reacts to THIS performance. This module
turns the recorded vocal into a time-aligned map of the performance so the
synthesizer can build with the singer, relax with the singer, and answer the
singer's pauses — instead of laying out intro/verse/chorus by bar-count fractions.

What it extracts (cheap, no pitch tracking — RMS energy, spectral brightness and
onset density are enough and fast):

    {
    CONSUMED by the synthesizer (these shape the audio):
      "sections":    [str]*bars     # arrangement DERIVED FROM the performance's
                                    #   energy shape (chorus lands on the real peak),
                                    #   not a fixed template
      "bar_tension": [float]*bars   # 0..1 per-bar intensity — drives "build/relax
                                    #   with them" instead of one global density knob
      "pauses":      [(start_beat, end_beat, kind)]   # gaps the band ANSWERS;
                                    #   kind ∈ {fill, bass_move, silence, chord_change}
      "energy_arc":  str            # builds | fades | peaks_middle | steady | dynamic
                                    #   (shapes the fallback generated lead)
      "seed":        int            # performance-derived → structure varies per take
                                    #   (anti-fingerprint), reproducible for the same take
      "confidence":  float          # 0..1; caller ignores the map below ~0.35

    ADVISORY (computed but NOT consumed by the synth — exposed for telemetry, the
    mixer's future vocal-reactive moves, and UI structure display):
      "bar_active":  [float]*bars   # voiced fraction per bar (0 = singer silent)
      "bar_energy":  [float]*bars   # 0..1 loudness per bar
      "phrase_bars": [int]          # bars where a sung phrase begins (after a breath)
      "peaks":       [int]          # the climactic (chorus) bars
      "struct_bars": int            # how many vocal bars the structure was read over
    }

Bar alignment MIRRORS vocal_harmony.transcribe_harmony exactly: structure is
derived over the vocal's own `struct_bars` and tiled across the beat's `bars`, so
the arrangement and the vocal-derived harmony stay phase-locked, and at final mix
(beat head-aligned + looped to the vocal) the right production lands under the
right moment of the take.

Returns None when the recording is too short/quiet to read — the caller then falls
back to the bar-count template, so generation never breaks.
"""
from __future__ import annotations

import numpy as np
import librosa

from .audio_loader import load_audio

_HOP = 512

# Tension = how hard the singer is going right now. Loudness dominates; brightness
# (vocal effort/strain pushes energy up the spectrum) and onset density refine it.
_W_ENERGY, _W_BRIGHT, _W_DENSITY = 0.55, 0.27, 0.18

# A bar counts as "sung" once the voice is present for at least this fraction of it.
_ACTIVE_BAR = 0.12

# Shortest silence (in beats) worth reacting to. Below this it's just phrasing.
_MIN_PAUSE_BEATS = 0.5


def _moving_avg(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if k <= 1 or len(x) <= 1:
        return x
    k = min(k, len(x))
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(k) / k
    return np.convolve(xp, ker, mode="same")[pad:pad + len(x)]


def _norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _smooth_bool(mask: np.ndarray, k: int = 5) -> np.ndarray:
    """Median-ish smoothing: removes single-frame blips and bridges tiny gaps so
    phrase/pause boundaries are stable, not chattering on every frame."""
    if len(mask) <= 2:
        return np.asarray(mask, dtype=bool)
    return _moving_avg(np.asarray(mask, dtype=float), k) >= 0.5


def _runs(times: np.ndarray, voiced: np.ndarray) -> tuple[list, list]:
    """Split the voiced timeline into sung phrases and the pauses between them.
    Leading/trailing silence is excluded (that becomes intro/outro, not a pause).
    Returns (phrases, pauses) as [(start_sec, end_sec)] each."""
    idx = np.where(voiced)[0]
    if len(idx) == 0:
        return [], []
    first_i, last_i = int(idx[0]), int(idx[-1])
    phrases, pauses = [], []
    b = first_i
    n = len(times)

    def _t(i: int) -> float:
        return float(times[min(i, n - 1)])

    while b <= last_i:
        if voiced[b]:
            s = b
            while b <= last_i and voiced[b]:
                b += 1
            phrases.append((_t(s), _t(b)))
        else:
            s = b
            while b <= last_i and not voiced[b]:
                b += 1
            pauses.append((_t(s), _t(b)))
    return phrases, pauses


def _derive_sections(tension: np.ndarray, active: np.ndarray,
                     struct_bars: int, is_flat: bool = False) -> list | None:
    """Lay out the song from the performance's own energy contour: the loudest /
    most intense sustained region becomes the chorus, the quieter sung region
    before it the verse, the rising bar a pre-chorus, a contrasting dip after the
    first chorus a bridge. Thresholds are relative to THIS take, so every
    performance shapes its own arrangement."""
    sections = ["verse"] * struct_bars
    content = [b for b in range(struct_bars) if active[b] > _ACTIVE_BAR]
    if not content:
        return None

    first, last = content[0], content[-1]
    for b in range(first):
        sections[b] = "intro"
    for b in range(last + 1, struct_bars):
        sections[b] = "outro"

    tc = np.array([tension[b] for b in content], dtype=float)

    if is_flat or float(tc.max() - tc.min()) < 0.12:
        # Flat performance: no clear climax. Split the sung span verse→chorus at
        # ~60% so the beat still lifts, anchored to the real vocal span.
        split = content[min(len(content) - 1, int(len(content) * 0.6))]
        for b in content:
            sections[b] = "chorus" if b >= split else "verse"
    else:
        hi = float(np.percentile(tc, 60))
        lo = float(np.percentile(tc, 38))

        def level(b: int) -> str:
            t = tension[b]
            return "high" if t >= hi else ("low" if t <= lo else "mid")

        # Group consecutive sung bars by intensity level
        groups: list[tuple[str, list]] = []
        for b in content:
            if groups and groups[-1][0] == level(b) and groups[-1][1][-1] == b - 1:
                groups[-1][1].append(b)
            else:
                groups.append((level(b), [b]))

        seen_chorus = False
        for lv, bs in groups:
            if lv == "high":
                name = "chorus2" if seen_chorus else "chorus"
                for b in bs:
                    sections[b] = name
                seen_chorus = True
            elif lv == "low" and seen_chorus and len(bs) >= 2:
                for b in bs:
                    sections[b] = "bridge"
            else:
                for b in bs:
                    sections[b] = "verse"

        # The sung bar immediately before a chorus is the build → pre-chorus
        for b in range(1, struct_bars):
            if sections[b] in ("chorus", "chorus2") and sections[b - 1] == "verse":
                sections[b - 1] = "pre"

    # Guarantee the beat has dynamics: if nothing read as a chorus, crown the
    # single most intense sung bar.
    if not any(s in ("chorus", "chorus2") for s in sections):
        peak = max(content, key=lambda b: tension[b])
        sections[peak] = "chorus"

    return sections


def _classify_pause(sb: float, eb: float, beat_dur: float, sections: list,
                    tension: np.ndarray, struct_bars: int) -> str:
    """Decide how the band answers a breath:
      fill        — leading into a chorus lift → a drum fill builds back in
      bass_move   — a longer mid-energy gap → the bass walks an answer
      silence     — a longer low-energy gap → open up, let the void breathe
      chord_change— a short gap → the harmony just re-articulates into the space
    """
    # +0.1-beat epsilon so a breath that ends on a downbeat (eb≈19.96 for a bar-5
    # entry) resolves to the bar the singer resumes INTO, not the one just below it.
    start_bar = int(sb // 4)
    resume_bar = int((eb + 0.1) // 4)
    length = eb - sb
    # A pause that runs to the end of the structural span has no real bar to
    # resume INTO — the singer doesn't come back, so a fill/walk would build into
    # emptiness. The honest answer is silence.
    if resume_bar >= struct_bars:
        return "silence"
    nxt = sections[min(resume_bar, struct_bars - 1)]
    prv = sections[min(start_bar, struct_bars - 1)]

    if nxt in ("chorus", "chorus2") and prv not in ("chorus", "chorus2"):
        return "fill"
    if length >= 1.5:
        lo_b = max(0, min(struct_bars - 1, start_bar))
        hi_b = max(0, min(struct_bars - 1, resume_bar))
        around = float(np.mean(tension[lo_b:hi_b + 1])) if hi_b >= lo_b else 0.0
        return "silence" if around < 0.40 else "bass_move"
    return "chord_change"


def _energy_arc(energy: np.ndarray, content: list) -> str:
    if len(content) < 3:
        return "steady"
    e = np.array([energy[b] for b in content], dtype=float)
    x = np.arange(len(e), dtype=float)
    slope = float(np.polyfit(x, e, 1)[0]) * len(e)   # total rise across the span
    peak_pos = int(np.argmax(e)) / max(1, len(e) - 1)
    if 0.30 <= peak_pos <= 0.70 and e.max() - min(e[0], e[-1]) > 0.25:
        return "peaks_middle"
    if slope > 0.18:
        return "builds"
    if slope < -0.18:
        return "fades"
    rng = float(e.max() - e.min())
    return "dynamic" if rng > 0.35 else "steady"


def _seed_from(tension: np.ndarray, density: np.ndarray, struct_bars: int) -> int:
    """Deterministic seed from the performance's own contour, so two different
    takes get different structure (no reusable fingerprint) but the SAME take is
    reproducible. FNV-1a over the quantized shape — stable across processes
    (unlike hash() on tuples under PYTHONHASHSEED)."""
    h = 2166136261
    for arr in (tension, density):
        for v in arr:
            q = int(round(float(v) * 9)) & 0xFFFFFFFF
            h = ((h ^ q) * 16777619) & 0xFFFFFFFF
    h = ((h ^ (struct_bars & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    return int(h & 0x7FFFFFFF)


def build_performance_map(audio_bytes: bytes, tempo: float, bars: int = 16) -> dict | None:
    """Build the performance map for the recorded vocal. See module docstring.
    Returns None on too-short/too-quiet input (caller falls back to the template)."""
    try:
        y, sr = load_audio(audio_bytes)            # mono, 22050 Hz
        if y is None or len(y) < sr // 2:
            return None
        # Clamp to the SAME range the synthesizer renders at (beat_synthesizer
        # clips tempo to 60..190). If they disagree the per-bar grids drift apart
        # and the reactions land on the wrong bar — e.g. a ballad detected at 52.
        tempo = float(np.clip(tempo, 60.0, 190.0)) if tempo and tempo > 1 else 90.0
        beat_dur = 60.0 / tempo
        bar_dur = 4.0 * beat_dur

        # ── Frame-level features (cheap) ──────────────────────────────────────
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=_HOP)[0]
        if len(rms) < 4:
            return None
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=_HOP)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=_HOP)
        try:
            onset_times = librosa.onset.onset_detect(
                y=y, sr=sr, hop_length=_HOP, units="time")
        except Exception:
            onset_times = np.array([])

        # ── Voiced gate → phrases + pauses ────────────────────────────────────
        peak = float(np.max(rms))
        if peak < 1e-5:
            return None
        thr = max(peak * 0.12, float(np.percentile(rms, 45)) * 0.6, 1e-4)
        voiced = _smooth_bool(rms > thr, k=5)
        phrases_sec, pauses_sec = _runs(times, voiced)
        if not phrases_sec:
            return None

        # ── Per-bar arrays over the vocal's own bars (mirror harmony tiling) ──
        vocal_dur = len(y) / sr
        vocal_bars = max(1, int(np.ceil(vocal_dur / bar_dur)))
        struct_bars = int(min(vocal_bars, bars))

        energy = np.zeros(struct_bars)
        active = np.zeros(struct_bars)
        bright = np.zeros(struct_bars)
        density = np.zeros(struct_bars)
        for b in range(struct_bars):
            t0, t1 = b * bar_dur, (b + 1) * bar_dur
            sel = (times >= t0) & (times < t1)
            if not np.any(sel):
                continue
            active[b] = float(np.mean(voiced[sel]))
            energy[b] = float(np.mean(rms[sel]))
            bright[b] = float(np.mean(cent[sel]))
            density[b] = float(np.sum((onset_times >= t0) & (onset_times < t1))) / bar_dur

        # Energy on a FIXED dB scale (not per-take min/max) so absolute dynamics
        # survive: a flat take stays flat (→ steady beat) instead of having its
        # noise floor stretched into a fake verse/chorus swing. 18 dB window maps
        # the loudest bar→1.0, ~18 dB below→0.0.
        e_peak = float(np.max(energy)) or 1e-9
        e_db = 20.0 * np.log10(np.maximum(energy, e_peak * 1e-3) / e_peak)
        energy_n = np.clip((e_db + 18.0) / 18.0, 0.0, 1.0)
        bright_n = _norm01(bright)
        density_n = _norm01(density)
        tension = (_W_ENERGY * energy_n + _W_BRIGHT * bright_n + _W_DENSITY * density_n)
        tension = _moving_avg(tension, 3)
        # A silent bar carries no tension regardless of residual noise energy.
        tension = tension * (active > 0.05)

        # How dynamic is this take, really? Measured on raw energy in dB over the
        # sung bars — drives both the flat/structured decision and how much the
        # per-bar tension is allowed to swing (flat take → compressed → steady).
        content0 = [b for b in range(struct_bars) if active[b] > _ACTIVE_BAR]
        if content0:
            ec = np.maximum(energy[content0], e_peak * 1e-3)
            ec_db = 20.0 * np.log10(ec / e_peak)
            dyn_db = float(np.percentile(ec_db, 90) - np.percentile(ec_db, 10))
        else:
            dyn_db = 0.0
        is_flat = dyn_db < 3.0
        dyn = float(np.clip(dyn_db / 12.0, 0.0, 1.0))   # 0 = flat, 1 = very dynamic
        # Average over the SAME bars the compression is applied to (keep), not the
        # tension>0 subset, so a voiced-but-zero-tension bar doesn't skew the mean.
        keep = active > 0.05
        mean_t = float(np.mean(tension[keep])) if np.any(keep) else 0.0
        # Compress the swing toward the mean when the take is flat; keep full
        # contrast when it's dynamic. Silent bars (tension==0) stay at 0.
        tension = np.where(keep, mean_t + (tension - mean_t) * (0.35 + 0.65 * dyn), 0.0)
        if not is_flat and float(np.ptp(tension[keep])) > 1e-6:
            # Stretch a genuinely dynamic take to use the full 0..1 range so the
            # section thresholds and busyness have headroom to react.
            tmin = float(np.min(tension[keep]))
            tmax = float(np.max(tension[keep]))
            tension = np.where(keep, (tension - tmin) / (tmax - tmin), 0.0)

        # ── Sections from the performance ─────────────────────────────────────
        sections = _derive_sections(tension, active, struct_bars, is_flat=is_flat)
        if sections is None:
            return None
        content = [b for b in range(struct_bars) if active[b] > _ACTIVE_BAR]

        # ── Pauses (within the structural span), with reaction kinds ──────────
        struct_beats = struct_bars * 4.0
        phrase_bars = sorted({int((s / beat_dur) // 4) for (s, _e) in phrases_sec
                              if (s / beat_dur) < struct_beats})
        pauses_struct: list[tuple[float, float, str]] = []
        for (s, e) in pauses_sec:
            sb, eb = s / beat_dur, e / beat_dur
            if sb >= struct_beats:
                continue
            eb = min(eb, struct_beats)
            if eb - sb < _MIN_PAUSE_BEATS:
                continue
            kind = _classify_pause(sb, eb, beat_dur, sections, tension, struct_bars)
            pauses_struct.append((round(sb, 3), round(eb, 3), kind))

        # ── Tile structure across the full beat (phase-locked to harmony) ─────
        def tile_list(src: list) -> list:
            return [src[b % struct_bars] for b in range(bars)]

        sec_full = tile_list(sections)
        tension_full = [float(tension[b % struct_bars]) for b in range(bars)]
        active_full = [float(active[b % struct_bars]) for b in range(bars)]
        energy_full = [float(energy_n[b % struct_bars]) for b in range(bars)]

        # Tiling can drop an "intro"/"outro" into the middle — keep only the
        # leading intro block and the trailing outro block; interior → verse.
        i = 0
        while i < bars and sec_full[i] == "intro":
            i += 1
        for k in range(i, bars):
            if sec_full[k] == "intro":
                sec_full[k] = "verse"
        j = bars - 1
        while j >= 0 and sec_full[j] == "outro":
            j -= 1
        for k in range(0, j + 1):
            if sec_full[k] == "outro":
                sec_full[k] = "verse"

        # Tile pauses + phrase bars across the beat
        pauses_full: list[tuple[float, float, str]] = []
        phrase_bars_full: list[int] = []
        reps = int(np.ceil(bars * 4.0 / struct_beats)) if struct_beats > 0 else 1
        for rep in range(max(1, reps)):
            off = rep * struct_beats
            for (sb, eb, kind) in pauses_struct:
                s2 = sb + off
                if s2 >= bars * 4.0:
                    break
                pauses_full.append((round(s2, 3), round(min(eb + off, bars * 4.0), 3), kind))
            for pb in phrase_bars:
                bb = int(pb + rep * struct_bars)
                if bb < bars:
                    phrase_bars_full.append(bb)

        peaks_full = [b for b in range(bars) if sec_full[b] in ("chorus", "chorus2")]

        # ── Confidence: enough voiced content + a readable dynamic shape ──────
        # A flat take still yields a usable (steady) map, so the floor stays well
        # above the caller's ~0.35 gate; dynamics and voiced coverage lift it.
        voiced_frac = float(np.mean(voiced))
        confidence = float(np.clip(0.40 + 0.30 * min(1.0, voiced_frac * 2.2)
                                   + 0.30 * dyn, 0, 1))

        return {
            "sections": sec_full,
            "bar_tension": [round(t, 3) for t in tension_full],
            "bar_active": [round(a, 3) for a in active_full],
            "bar_energy": [round(e, 3) for e in energy_full],
            "pauses": pauses_full,
            "phrase_bars": sorted(set(phrase_bars_full)),
            "peaks": peaks_full,
            "energy_arc": _energy_arc(energy_n, content),
            "seed": _seed_from(tension, density_n, struct_bars),
            "struct_bars": struct_bars,
            "confidence": round(confidence, 3),
        }
    except Exception:
        return None
