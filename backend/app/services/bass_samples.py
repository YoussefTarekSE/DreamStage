"""
Real tuned 808 bass, built from the owner's Bass Shots.

An 808 IS a single sustained tonal sample played back at different speeds to hit
different notes — so a real recorded 808 one-shot, retuned per bassline note, is
both authentic and far better than the GM-soundfont "Synth Bass" or the numpy
sine, which are the most prominent "fake/cheap"-sounding part of a beat.

``tools/build_808.py`` curates the instruments (``app/assets/bass/808*.wav`` +
``manifest.json``, each micro-tuned so its fundamental sits exactly on a
semitone). This module retunes them per note by resampling (with proper 808
portamento glide between notes) and shapes the amplitude envelope.

Used as the FIRST choice in ``beat_synthesizer.bass_note`` for sub-style basses
(808 / sub / deep); the SoundFont and numpy synth remain the fallbacks, and
pluck-style basses keep the synth (a different instrument character).
"""
import os
import json
import threading

import numpy as np

try:
    import soundfile as sf
    _HAS_SF = True
except Exception:
    _HAS_SF = False

SR = 44_100
_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "assets", "bass"))

# Output level per variant, calibrated so the real bass sits where the synth it
# replaces did (so the mixer's low-end balance stays valid). The 808 is matched
# on RMS (a sustained sub); the pluck is matched on PEAK (a transient — same
# punch, but its faster natural decay keeps it from booming). See tests.
_LEVEL = 0.62
_LEVEL_VARIANT = {"pluck": 1.11}


def _level_for(variant: str) -> float:
    return _LEVEL_VARIANT.get(variant, _LEVEL)

_manifest: dict | None = None
_cache: dict = {}          # name -> (y float32, f0_hz)
_lock = threading.Lock()
_available: bool | None = None


def _load_manifest() -> dict:
    global _manifest
    if _manifest is None:
        path = os.path.join(_DIR, "manifest.json")
        try:
            with open(path) as fh:
                _manifest = json.load(fh)
        except Exception:
            _manifest = {}
    return _manifest


def available() -> bool:
    """True when soundfile is present and at least the primary 808 exists."""
    global _available
    if _available is None:
        man = _load_manifest()
        _available = bool(
            _HAS_SF and man.get("808")
            and os.path.exists(os.path.join(_DIR, man["808"]["file"]))
        )
    return bool(_available)


def _get_source(name: str):
    """Return (y, f0_hz) for an instrument, cached. None if missing."""
    man = _load_manifest()
    if name not in man:
        return None
    cached = _cache.get(name)
    if cached is not None:
        return cached
    path = os.path.join(_DIR, man[name]["file"])
    if not os.path.exists(path):
        return None
    try:
        with _lock:
            y, sr = sf.read(path)
            if getattr(y, "ndim", 1) > 1:
                y = y.mean(axis=1)
            y = np.ascontiguousarray(y, dtype=np.float32)
            entry = (y, float(man[name]["f0_hz"]))
            _cache[name] = entry
        return entry
    except Exception:
        return None


def _render_with_ratio_curve(y: np.ndarray, ratio_curve: np.ndarray) -> np.ndarray:
    """Read `y` at a per-output-sample speed (ratio_curve), linearly interpolated.

    Constant ratio → pure pitch shift; a ramp at the start → portamento glide.
    Read positions past the end of the source clamp to the last sample (the 808
    has already decayed to ~silence by then)."""
    read_pos = np.empty(len(ratio_curve), dtype=np.float64)
    read_pos[0] = 0.0
    np.cumsum(ratio_curve[:-1], out=read_pos[1:])
    n = len(y)
    idx = read_pos.astype(np.int64)
    np.clip(idx, 0, n - 2, out=idx)
    frac = (read_pos - idx).astype(np.float32)
    return (y[idx] * (1.0 - frac) + y[idx + 1] * frac).astype(np.float32)


def render_hz(target_hz: float, dur: float, vel: float = 1.0,
              prev_freq: float = None, variant: str = "808") -> np.ndarray | None:
    """Render a real 808 note at `target_hz` for `dur` seconds.

    `prev_freq` (Hz) adds 808 portamento glide from the previous note. Returns a
    mono float32 array at SR, or None if the instrument is unavailable.
    """
    if not available() or dur <= 0:
        return None
    src = _get_source(variant) or _get_source("808")
    if src is None:
        return None
    y, f0 = src
    if f0 <= 0 or len(y) < 8:
        return None

    target_hz = float(np.clip(target_hz, 20.0, 400.0))
    release = 0.045
    out_n = int((dur + release) * SR)
    if out_n < 8:
        return None

    target_ratio = target_hz / f0
    ratio_curve = np.full(out_n, target_ratio, dtype=np.float64)

    # ── 808 portamento: glide the read speed from prev pitch to target ────────
    if prev_freq is not None and prev_freq > 20.0 and abs(prev_freq - target_hz) > 0.5:
        semis = abs(12.0 * np.log2(target_hz / prev_freq))
        glide_s = float(np.clip(semis * 0.008, 0.025, 0.12))
        g = min(int(glide_s * SR), out_n)
        if g > 1:
            freq_ramp = np.exp(np.linspace(np.log(max(prev_freq, 20.0)),
                                           np.log(target_hz), g))
            ratio_curve[:g] = freq_ramp / f0

    body = _render_with_ratio_curve(y, ratio_curve)

    # ── Amplitude envelope: tiny attack smooth + note-end release. The source's
    # own natural decay carries the body (real 808 character). ────────────────
    env = np.ones(out_n, dtype=np.float32)
    attack = min(int(0.004 * SR), out_n)
    rel = min(int(release * SR), out_n)
    if attack > 0:
        env[:attack] = np.linspace(0.0, 1.0, attack)
    if rel > 0:
        env[-rel:] *= np.linspace(1.0, 0.0, rel)

    return body * env * float(vel) * _level_for(variant)
