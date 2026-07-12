"""
Real chord/melody loops (owner's library) as the beat's melodic FOREGROUND.

The synth pad/lead are the "cheap"-sounding layers. This plays the owner's real
recorded loops where they add the most character with the least harmonic clash
against the vocal-derived progression (bar_degrees):

  feature : full-range loop in the INTRO (no bass/pads there -> zero clash) — the
            song opens with a real instrument, previewing the chorus hook.
  topline : the same loop, high-passed, as the chorus HOOK — sits ABOVE the pads
            in its own register and REPLACES the synth lead in those bars, so the
            audible melodic foreground is a real instrument, not a synth.
  bed     : a low-passed, low-level atmospheric pad under the choruses (warmth).

Loops are curated + canonicalised (to 120 BPM, whole bars) by tools/build_textures.py
into app/assets/textures/ + manifest.json. Here we pitch-shift to the song KEY,
time-stretch to the song TEMPO, filter per role, re-normalise, and tile. Results
are cached per song (key+tempo are fixed) so the phase-vocoder runs ~once, not per
candidate. Mirrors bass_samples.py: available() gate, cache, graceful fallback.
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

try:
    import librosa
    _HAS_LIBROSA = True
except Exception:
    _HAS_LIBROSA = False

try:
    from scipy.signal import butter, sosfilt
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

SR = 44_100
_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "assets", "textures"))

# Per-role output level, calibrated against the synth pad (see tests). The topline
# is the foreground hook (loudest); the bed is a quiet wash; the feature owns the
# empty intro.
_LEVEL = {"feature": 0.34, "topline": 0.30, "bed": 0.13}

_manifest: dict | None = None
_cache: dict = {}          # (name, semis, tempo_i, mode) -> processed mono buffer
_src_cache: dict = {}      # name -> (y, entry)
_lock = threading.Lock()
_available: bool | None = None
# Bound the processed-buffer cache (each entry is ~1-2 MB) so a long-running
# server processing many songs (varied key/tempo) can't grow it without limit.
# Within one song only ~3 keys are used, so this still gives a full per-song hit.
_MAX_CACHE = 16


def _load_manifest() -> dict:
    global _manifest
    if _manifest is None:
        try:
            with open(os.path.join(_DIR, "manifest.json")) as fh:
                _manifest = json.load(fh)
        except Exception:
            _manifest = {}
    return _manifest


def available() -> bool:
    global _available
    if _available is None:
        man = _load_manifest()
        ok = _HAS_SF and _HAS_LIBROSA and bool(man)
        if ok:
            ok = any(os.path.exists(os.path.join(_DIR, e["file"])) for e in man.values())
        _available = bool(ok)
    return bool(_available)


def list_candidates(role: str) -> list:
    return [dict(name=k, **v) for k, v in _load_manifest().items() if v.get("role") == role]


def _nearest_semitones(src_pc: int, dst_pc: int) -> int:
    """Shift (in [-6, 6]) to move src pitch-class onto dst, nearest direction."""
    d = (int(dst_pc) - int(src_pc)) % 12
    return d if d <= 6 else d - 12


def pick(role: str, song_key_pc: int, song_mode: str, valence: float = 0.5, rng=None):
    """Deterministically choose the best loop of `role` for this song (stable across
    candidates so the cache hits and the scored beat == the winner). Prefers small
    pitch-shift, matching mode (avoids major/minor clash), and brightness≈valence.
    Returns the manifest entry (with 'name') or None."""
    cands = list_candidates(role)
    if not cands:
        return None

    def score(e):
        shift = abs(_nearest_semitones(e["root_pc"], song_key_pc))
        mode_pen = 0.0 if e.get("mode") == song_mode else 1.0
        bright_pen = abs(float(e.get("brightness", 0.5)) - float(valence))
        return -(0.5 * shift + 3.0 * mode_pen + 1.2 * bright_pen)

    ranked = sorted(cands, key=score, reverse=True)
    best = score(ranked[0])
    # Choose among close high-quality matches so hooks vary across seeded cuts
    # without sacrificing key/mode fit or breaking same-attempt determinism.
    top = [entry for entry in ranked if best - score(entry) <= 1.5][:3]
    if rng is not None and len(top) > 1:
        return rng.choice(top)
    return top[0]


def _get_source(name: str):
    man = _load_manifest()
    if name not in man:
        return None
    cached = _src_cache.get(name)
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
            if sr != SR:
                y = librosa.resample(y, orig_sr=sr, target_sr=SR).astype(np.float32)
            entry = (y, man[name])
            _src_cache[name] = entry
        return entry
    except Exception:
        return None


def _filt(y: np.ndarray, kind: str, cut: float) -> np.ndarray:
    if not _HAS_SCIPY:
        return y
    nyq = SR / 2.0
    sos = butter(3, np.clip(cut / nyq, 1e-4, 0.99), btype=kind, output="sos")
    return sosfilt(sos, y).astype(np.float32)


def _process(name: str, semis: int, song_tempo: float, mode: str, valence: float):
    """Stretch + shift + filter the source for `mode` (feature/topline/bed); cached."""
    # valence only changes the bed's low-pass cutoff — bucket it into the key for
    # beds so two songs (same key/tempo/mode, different valence) don't collide.
    vbucket = round(float(valence), 1) if mode == "bed" else 0.0
    key = (name, int(semis), int(round(song_tempo)), mode, vbucket)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    src = _get_source(name)
    if src is None:
        return None
    y, entry = src
    y = y.copy()
    try:
        # Time-stretch from the canonical ref tempo to the song tempo (only for
        # tempo-locked loops; beds/non-lockable stay as-is — they're sustained).
        ref_bpm = float(entry.get("ref_bpm", 0.0))
        if mode in ("feature", "topline") and ref_bpm > 0:
            rate = float(np.clip(song_tempo / ref_bpm, 0.5, 2.0))   # song faster => speed up
            if abs(rate - 1.0) > 0.02:
                y = librosa.effects.time_stretch(y, rate=rate)
        # Pitch-shift onto the song key (small, <= a tritone).
        if abs(semis) >= 1:
            y = librosa.effects.pitch_shift(y, sr=SR, n_steps=float(semis))
        # Role tone-shaping.
        if mode == "topline":
            y = _filt(y, "highpass", 380.0)          # sit above the pads as a hook
        elif mode == "bed":
            y = _filt(y, "highpass", 80.0)           # never fight the 808/kick sub
            y = _filt(y, "lowpass", float(np.clip(700 + valence * 700, 700, 2200)))
        else:  # feature
            y = _filt(y, "highpass", 55.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        peak = float(np.max(np.abs(y))) if len(y) else 0.0
        if peak > 1e-6:
            y = y / peak * 0.9                        # re-normalise (DSP can clip)
    except Exception:
        return None
    with _lock:
        if len(_cache) >= _MAX_CACHE:
            _cache.pop(next(iter(_cache)), None)   # FIFO evict oldest
        _cache[key] = y
    return y


def _tile(buf: np.ndarray, n: int, xfade: int = 1024) -> np.ndarray:
    """Loop `buf` to length n by windowed overlap-add: each copy fades in/out over
    `xfade` samples and copies hop by (len-xfade), so the seams crossfade (linear
    weights sum to 1 -> no level bump) and the build-time end fades are masked."""
    L = len(buf)
    if L == 0 or n <= 0:
        return np.zeros(max(0, n), dtype=np.float32)
    if L >= n:
        return buf[:n].copy()
    xfade = int(min(xfade, L // 4))
    if xfade < 1:
        reps = int(np.ceil(n / L))
        return np.tile(buf, reps)[:n].astype(np.float32)
    win = buf.copy()
    win[:xfade] *= np.linspace(0.0, 1.0, xfade, dtype=np.float32)
    win[-xfade:] *= np.linspace(1.0, 0.0, xfade, dtype=np.float32)
    hop = L - xfade
    out = np.zeros(n + L, dtype=np.float32)
    pos = 0
    while pos < n:
        out[pos:pos + L] += win
        pos += hop
    return out[:n].astype(np.float32)


def render(entry: dict, song_key_pc: int, song_tempo: float, n_samples: int,
           mode: str, valence: float = 0.5) -> np.ndarray | None:
    """Render a chosen loop (manifest entry from pick()) for `mode`, tiled to
    `n_samples`, at SR mono float32. Returns None if unavailable.

    The result is edge-faded (quick in, longer out) so a loop placed for one
    arrangement section never cuts off abruptly at the section boundary — the
    hard stop into the next section (e.g. the chorus loop dropping into the
    sparse outro) was the audible "weird" transition."""
    if not available() or not entry or n_samples <= 0:
        return None
    semis = _nearest_semitones(entry["root_pc"], song_key_pc)
    buf = _process(entry["name"], semis, song_tempo, mode, valence)
    if buf is None or len(buf) < 8:
        return None
    tiled = _tile(buf, n_samples) * _LEVEL.get(mode, 0.2)
    # Edge fades: ~12 ms in, ~220 ms out (so it breathes out of the section).
    fin = min(int(0.012 * SR), len(tiled) // 8)
    fout = min(int(0.22 * SR), len(tiled) // 3)
    if fin > 0:
        tiled[:fin] *= np.linspace(0.0, 1.0, fin, dtype=np.float32)
    if fout > 0:
        tiled[-fout:] *= np.linspace(1.0, 0.0, fout, dtype=np.float32)
    return tiled.astype(np.float32)
