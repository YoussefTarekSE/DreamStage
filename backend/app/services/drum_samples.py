"""
Real recorded drum one-shots (Alesis SR16, MIT-licensed) for the core kit.

Synthesized / General-MIDI drums are the most "fake"-sounding part of a beat.
These are real hardware drum-machine samples — punchy kick, snare, hats, clap —
used as the first choice in beat_synthesizer's drum functions, falling back to
the SoundFont, then numpy synthesis, if the files are missing.

Source: github.com/avarasp/SR16-drum-samples-free-pack (MIT License).
"""
import os
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

SR = 44_100
_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "assets", "drums"))
_KIT_DIR = os.path.join(_DIR, "kits")   # shippable genre kits (built from owned samples)

# Shippable default kit (MIT SR16 one-shots), keyed by role
_FILES = {
    "kick":   "Kick.wav",
    "snare":  "Snare.wav",
    "closed": "ClosedHat.wav",
    "open":   "OpenHat.wav",
    "clap":   "Claps.wav",
}
# Role → filename inside an FL kit folder (when imported locally)
_FL_FILES = {
    "kick": "kick.wav", "snare": "snare.wav", "closed": "closedhat.wav",
    "open": "openhat.wav", "clap": "clap.wav",
}

# Genre → FL kit folder. Trap/hip-hop get the 808, pop/dance the 909, and
# soul/R&B/lo-fi an acoustic-feel kit. Used only when the local FL kits exist.
GENRE_KIT = {
    "trap_dark": "trap", "trap_melodic": "trap", "drill": "trap", "uk_drill": "trap",
    "phonk": "trap", "hiphop_modern": "trap",
    "hiphop_boom_bap": "acoustic", "jazz_hop": "acoustic", "rnb_smooth": "acoustic",
    "rnb_neo_soul": "acoustic", "soul_ballad": "acoustic", "lofi_chill": "acoustic",
    "pop_bright": "dance", "afrobeats": "dance", "dancehall": "dance",
    "reggaeton": "dance", "amapiano": "dance", "club_house": "dance",
}

_cache: dict = {}
_lock = threading.Lock()
_available: bool = None
_active_kit: str = None        # FL kit folder selected for the current beat


def available() -> bool:
    """True when soundfile is present and the default (SR16) kit exists."""
    global _available
    if _available is None:
        _available = _HAS_SF and all(
            os.path.exists(os.path.join(_DIR, f)) for f in _FILES.values()
        )
    return bool(_available)


def _fl_kit_available(kit: str) -> bool:
    return bool(kit) and all(
        os.path.exists(os.path.join(_KIT_DIR, kit, f)) for f in _FL_FILES.values()
    )


def select_kit(genre_name: str) -> None:
    """Pick the FL kit for this genre (if the local FL kits were imported)."""
    global _active_kit
    kit = GENRE_KIT.get(genre_name)
    _active_kit = kit if _fl_kit_available(kit) else None


def _load(path: str):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if sr != SR and _HAS_LIBROSA:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR).astype(np.float32)
    peak = float(np.max(np.abs(y)))
    if peak > 1e-6:
        y = y / peak * 0.9
    return y


def get(name: str) -> np.ndarray:
    """
    Return a drum one-shot (mono float32 at SR, peak 0.9) for the active kit:
    the genre's FL kit if imported locally, otherwise the shippable SR16 kit.
    Caller scales by per-hit velocity. Returns None if unavailable.
    """
    if not _HAS_SF or name not in _FILES:
        return None
    # Prefer the genre-appropriate FL kit (local only)
    if _active_kit and name in _FL_FILES:
        key = ("fl", _active_kit, name)
        cached = _cache.get(key)
        if cached is not None:
            return cached.copy()
        fl_path = os.path.join(_KIT_DIR, _active_kit, _FL_FILES[name])
        if os.path.exists(fl_path):
            try:
                with _lock:
                    y = _load(fl_path)
                    _cache[key] = y
                return y.copy()
            except Exception:
                pass
    # Fall back to the shippable SR16 kit
    if not available():
        return None
    cached = _cache.get(name)
    if cached is not None:
        return cached.copy()
    try:
        with _lock:
            y = _load(os.path.join(_DIR, _FILES[name]))
            _cache[name] = y
        return y.copy()
    except Exception:
        return None
