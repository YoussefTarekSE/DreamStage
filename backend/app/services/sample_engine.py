"""
Sample engine: renders every instrument from a real SoundFont (.sf2) so the beat
is made of recorded instruments — real kicks, snares, 808s, pianos, basses —
instead of synthesized sine/noise waves.

Backend: tinysoundfont (pip-installable, bundles its own native binary — no
system FluidSynth/DLL needed), driving a General-MIDI SoundFont. This works
identically on Windows (local dev) and Linux (Render), which the previous
pyfluidsynth path did NOT (it needed an apt/DLL install that was absent locally,
so every beat silently fell back to numpy synthesis = the "noise").

SoundFont search order (first found wins):
    1. backend/app/assets/soundfont.sf2     (GeneralUser GS — bundled/downloaded)
    2. /usr/share/sounds/sf2/FluidR3_GM.sf2  (apt fluid-soundfont-gm, on Render)
    3. /usr/share/sounds/sf2/FluidR3_GS.sf2
    4. /usr/share/soundfonts/default.sf2

If neither tinysoundfont nor a SoundFont is present, is_available() returns
False and beat_synthesizer.py falls back to its numpy synthesis.

Licensing:
    GeneralUser GS — free for any use including commercial (S. Christian Collins).
    FluidR3_GM.sf2 — MIT License.
"""
import os
import math
import logging
import threading
import numpy as np

logger = logging.getLogger(__name__)

SR = 44_100

# ── SoundFont paths ───────────────────────────────────────────────────────────
_ASSETS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "assets"))
_SF2_CANDIDATES = [
    os.path.join(_ASSETS, "soundfont.sf2"),        # bundled / downloaded GeneralUser GS
    os.path.join(_ASSETS, "GeneralUser-GS.sf2"),
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/FluidR3_GS.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    "/usr/share/soundfonts/default.sf2",
]

# ── GM drum note numbers (percussion bank 128) ───────────────────────────────
NOTE_KICK       = 36   # Bass Drum 1
NOTE_KICK_SOFT  = 35   # Acoustic Bass Drum
NOTE_SNARE      = 38   # Acoustic Snare
NOTE_SNARE_ELEC = 40   # Electric Snare
NOTE_RIM        = 37   # Side Stick / Rimshot
NOTE_CLAP       = 39   # Hand Clap
NOTE_HH_CLOSED  = 42   # Closed Hi-Hat
NOTE_HH_PEDAL   = 44   # Pedal Hi-Hat
NOTE_HH_OPEN    = 46   # Open Hi-Hat
NOTE_CRASH      = 49   # Crash Cymbal 1
NOTE_RIDE       = 51   # Ride Cymbal 1
NOTE_COWBELL    = 56   # Cowbell
NOTE_AGOGO_LO   = 68   # Low Agogo  (log-drum approximation)
NOTE_AGOGO_HI   = 67   # High Agogo
NOTE_MARACAS    = 70   # Maracas    (shaker approximation)

# GM percussion lives on bank 128 in a GM SoundFont.
DRUM_BANK = 128

# ── Instrument channels (kept distinct for clarity; renders are isolated) ─────
DRUM_CH  = 9
BASS_CH  = 0
CHORD_CH = 1
PAD_CH   = 2
LEAD_CH  = 3
ARP_CH   = 4

# ── GM program numbers (bank 0) ───────────────────────────────────────────────
PROG_BASS_FINGERED = 33
PROG_BASS_PICKED   = 34
PROG_BASS_FRETLESS = 35
PROG_BASS_SYNTH1   = 38   # Synth Bass 1 (808-like)
PROG_BASS_SYNTH2   = 39   # Synth Bass 2 (deep sub)
PROG_PIANO         = 0
PROG_EPIANO        = 4
PROG_PAD_WARM      = 89    # Pad 2 (warm)
PROG_PAD_CHOIR     = 91    # Pad 4 (choir)
PROG_LEAD_SAW      = 81    # Lead 2 (sawtooth)
PROG_LEAD_SQUARE   = 80    # Lead 1 (square)
PROG_VIBES         = 11    # Vibraphone — warm bell, musical melodic lead
PROG_HARP          = 46    # Orchestral Harp — clean pluck for arpeggios
PROG_EPIANO2       = 5     # Electric Piano 2 (FM/Rhodes-ish)

_BASS_PROGS = {
    "808":   PROG_BASS_SYNTH1,
    "deep":  PROG_BASS_SYNTH2,
    "pluck": PROG_BASS_FINGERED,
    "sub":   PROG_BASS_SYNTH1,
}

# ── Engine singleton ──────────────────────────────────────────────────────────
_synth:     object = None
_sfid:      int    = -1
_available: bool   = None    # None = not yet probed
_lock = threading.Lock()
_cache: dict = {}


def is_available() -> bool:
    """True when tinysoundfont + a SoundFont are both loaded."""
    global _available
    if _available is None:
        _init()
    return bool(_available)


def _init() -> None:
    global _synth, _sfid, _available
    with _lock:
        if _available is not None:
            return
        try:
            import tinysoundfont  # pip install tinysoundfont (bundled binary)

            sf2 = next((p for p in _SF2_CANDIDATES if os.path.exists(p)), None)
            if sf2 is None:
                logger.warning(
                    "[sample_engine] No SoundFont found (checked %s). "
                    "Beats will use numpy synthesis. Drop a GM .sf2 at %s.",
                    _SF2_CANDIDATES, os.path.join(_ASSETS, "soundfont.sf2"),
                )
                _available = False
                return

            # samplerate MUST be int (the native set_output rejects a float);
            # gain is in dB — individual notes are peak-normalised afterwards, so
            # a small positive value just keeps the raw render healthy.
            synth = tinysoundfont.Synth(samplerate=int(SR), gain=3.0)
            sfid = synth.sfload(sf2)
            _synth, _sfid, _available = synth, sfid, True
            logger.info("[sample_engine] SoundFont ready (tinysoundfont): %s", sf2)
        except Exception as exc:
            logger.info(
                "[sample_engine] SoundFont engine unavailable (%s: %s) — "
                "beat_synthesizer will use numpy synthesis",
                type(exc).__name__, exc,
            )
            _available = False


def _hz_to_midi(freq: float) -> int:
    return int(round(69.0 + 12.0 * math.log2(max(freq, 8.0) / 440.0)))


def _buf_to_mono(buf, n_frames: int) -> np.ndarray:
    """tinysoundfont.generate_simple → memoryview of interleaved stereo float32."""
    if buf is None:
        return np.zeros(n_frames, dtype=np.float32)
    arr = np.frombuffer(buf, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(n_frames, dtype=np.float32)
    stereo = arr.reshape(-1, 2)
    mono = stereo.mean(axis=1).astype(np.float32)
    if len(mono) < n_frames:
        mono = np.append(mono, np.zeros(n_frames - len(mono), dtype=np.float32))
    return mono[:n_frames]


def _render(channel: int, bank: int, preset: int, key: int, velocity: int,
            hold_sec: float, release_sec: float, norm_peak: float = 0.9) -> np.ndarray:
    """
    Render a single isolated note/hit to mono float32 and peak-normalise it so
    it drops into the beat mixer at the same reference level the numpy waves used
    (the caller then scales by per-hit velocity).
    """
    synth = _synth
    n_hold = max(1, int(SR * hold_sec))
    n_rel  = max(1, int(SR * release_sec))
    if synth is None:
        return np.zeros(n_hold + n_rel, dtype=np.float32)

    with _lock:
        # Silence + drain the channel so no previous note bleeds into this render.
        try:
            synth.sounds_off(channel)
        except Exception:
            pass
        synth.generate_simple(256)
        synth.program_select(channel, _sfid, bank, preset)
        synth.noteon(channel, int(np.clip(key, 0, 127)), int(np.clip(velocity, 1, 127)))
        raw_hold = synth.generate_simple(n_hold)
        synth.noteoff(channel, int(np.clip(key, 0, 127)))
        raw_rel = synth.generate_simple(n_rel)

    hold = _buf_to_mono(raw_hold, n_hold)
    rel  = _buf_to_mono(raw_rel, n_rel)
    out  = np.concatenate([hold, rel]).astype(np.float32)

    peak = float(np.max(np.abs(out)))
    if peak > 1e-4:
        out = out / peak * norm_peak
    return out


# ── Public rendering API (matches the old pyfluidsynth engine) ────────────────

def drum_hit(note: int, velocity: int = 100, duration_sec: float = 0.5) -> np.ndarray:
    key = ("d", note, velocity, round(duration_sec, 3))
    cached = _cache.get(key)
    if cached is not None:
        return cached.copy()
    if not is_available():
        result = np.zeros(int(SR * duration_sec), dtype=np.float32)
    else:
        result = _render(DRUM_CH, DRUM_BANK, 0, int(np.clip(note, 27, 87)),
                         velocity, duration_sec, 0.05)
    _cache[key] = result
    return result.copy()


def melodic_note(channel: int, program: int, midi_note: int,
                 velocity: int, duration_sec: float,
                 release_sec: float = 0.12) -> np.ndarray:
    key = ("n", channel, program, midi_note, velocity,
           round(duration_sec, 3), round(release_sec, 3))
    cached = _cache.get(key)
    if cached is not None:
        return cached.copy()
    if not is_available():
        result = np.zeros(int(SR * (duration_sec + release_sec)), dtype=np.float32)
    else:
        result = _render(channel, 0, int(program), int(midi_note),
                         velocity, duration_sec, release_sec)
    _cache[key] = result
    return result.copy()


def bass(freq: float, duration_sec: float, style: str = "808",
         velocity: int = 90) -> np.ndarray:
    prog = _BASS_PROGS.get(style, PROG_BASS_SYNTH1)
    return melodic_note(BASS_CH, prog, _hz_to_midi(freq), velocity,
                        duration_sec, release_sec=0.06)


def chord_note(freq: float, duration_sec: float, velocity: int = 52,
               valence: float = 0.5) -> np.ndarray:
    prog = PROG_EPIANO if valence > 0.6 else PROG_PIANO
    return melodic_note(CHORD_CH, prog, _hz_to_midi(freq), velocity,
                        duration_sec, release_sec=0.22)


def pad_note(freq: float, duration_sec: float, velocity: int = 45) -> np.ndarray:
    return melodic_note(PAD_CH, PROG_PAD_WARM, _hz_to_midi(freq), velocity,
                        duration_sec, release_sec=0.35)


def lead_note(freq: float, duration_sec: float, velocity: int = 68) -> np.ndarray:
    # Vibraphone instead of a buzzy saw — a warm, bell-like melodic lead that
    # sits musically over the vocal (and reads well an octave up).
    return melodic_note(LEAD_CH, PROG_VIBES, _hz_to_midi(freq), velocity,
                        duration_sec, release_sec=0.25)


def arp_note(freq: float, duration_sec: float, velocity: int = 62) -> np.ndarray:
    # Harp pluck instead of a square wave — clean and pleasant for arpeggios.
    return melodic_note(ARP_CH, PROG_HARP, _hz_to_midi(freq), velocity,
                        duration_sec, release_sec=0.10)


def clear_cache() -> None:
    """Release cached rendered samples (call between beats on low-memory hosts)."""
    _cache.clear()
