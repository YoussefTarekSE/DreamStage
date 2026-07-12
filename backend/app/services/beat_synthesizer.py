"""
Advanced beat synthesizer.
- 18 genre families, each with 3-5 unique pattern variations
- Song arrangement: intro → verse → pre-chorus → chorus/drop → outro
- Deep vocal analysis drives genre, key, mode, swing, energy
- Musical key awareness (bass + chords + lead in the detected key)
- Arpeggiated synth layer, counter-melody, humanization
- Anti-repetition: exclude_genres + attempt-seeded diversity
- Every generation is unique and musically coherent
- Reverb/delay on melodic layers for professional depth
"""
import io
import zlib
import random
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from . import sample_engine as _se
from . import drum_samples as _ds
from . import bass_samples as _bs
from . import texture_loops as _tx

SR    = 44_100
RNG   = random.Random()

try:
    import pyloudnorm as pyln
    _HAS_PYLOUDNORM = True
    def _PYLN_METER(sr):
        return pyln.Meter(sr)
except ImportError:
    _HAS_PYLOUDNORM = False
    def _PYLN_METER(sr):
        raise RuntimeError("pyloudnorm not available")


# ── Instrument synthesis ──────────────────────────────────────────────────────

def kick(style: str = "808", vel: float = 1.0) -> np.ndarray:
    s = _ds.get("kick")
    if s is not None:
        return (s * vel).astype(np.float32)
    if _se.is_available():
        _note = _se.NOTE_KICK_SOFT if style == "snap" else _se.NOTE_KICK
        _dur  = {"808": 1.10, "punchy": 0.40, "snap": 0.22, "thud": 0.55}.get(style, 0.50)
        return _se.drum_hit(_note, int(np.clip(vel * 112, 45, 127)), _dur)

    # ── numpy fallback ────────────────────────────────────────────────────────
    if style == "808":
        dur = 1.10; n = int(SR * dur); t = np.linspace(0, dur, n)
        freq  = 180 * np.exp(-5 * t) + 38
        phase = 2 * np.pi * np.cumsum(freq) / SR
        body  = np.sin(phase)
        body  = np.tanh(body * 3.2) / np.tanh(3.2)
        body *= np.exp(-2.8 * t) * vel
        click1 = np.sin(2 * np.pi * 2800 * t) * np.exp(-200 * t) * 0.32
        click2 = np.sin(2 * np.pi * 1100 * t) * np.exp(-110 * t) * 0.18
        noise_n = min(n, int(SR * 0.012))
        noise_burst = np.zeros(n, dtype=np.float32)
        raw_noise = np.random.randn(noise_n).astype(np.float32)
        sos = butter(3, [600, 7000], btype='band', fs=SR, output='sos')
        noise_burst[:noise_n] = sosfilt(sos, raw_noise) * np.exp(-400 * t[:noise_n]) * 0.22
        return np.clip((body + click1 + click2 + noise_burst), -1, 1).astype(np.float32) * 0.90
    elif style == "punchy":
        dur = 0.40; n = int(SR * dur); t = np.linspace(0, dur, n)
        freq  = 130 * np.exp(-12 * t) + 52
        phase = 2 * np.pi * np.cumsum(freq) / SR
        body  = np.tanh(np.sin(phase) * 2.5) / np.tanh(2.5)
        body *= np.exp(-8 * t) * vel
        click = np.sin(2 * np.pi * 3400 * t) * np.exp(-140 * t) * 0.36
        return np.clip((body + click), -1, 1).astype(np.float32) * 0.88
    elif style == "snap":
        dur = 0.22; n = int(SR * dur); t = np.linspace(0, dur, n)
        freq  = 90 * np.exp(-18 * t) + 48
        phase = 2 * np.pi * np.cumsum(freq) / SR
        body  = np.tanh(np.sin(phase) * 2.0) / np.tanh(2.0)
        return (body * np.exp(-14 * t) * vel * 0.82).astype(np.float32)
    else:  # thud
        dur = 0.55; n = int(SR * dur); t = np.linspace(0, dur, n)
        freq  = 105 * np.exp(-4.5 * t) + 40
        phase = 2 * np.pi * np.cumsum(freq) / SR
        body  = np.tanh(np.sin(phase) * 2.8) / np.tanh(2.8)
        return (body * np.exp(-5.5 * t) * vel * 0.88).astype(np.float32)


def snare(style: str = "crack", vel: float = 1.0) -> np.ndarray:
    if style != "rimshot":          # rimshot has no real sample → use SF/numpy
        s = _ds.get("clap" if style == "clap_snare" else "snare")
        if s is not None:
            return (s * vel).astype(np.float32)
    if _se.is_available():
        _note = {"crack": _se.NOTE_SNARE_ELEC, "fat": _se.NOTE_SNARE,
                 "rimshot": _se.NOTE_RIM, "clap_snare": _se.NOTE_CLAP}.get(style, _se.NOTE_SNARE)
        _dur  = {"crack": 0.28, "fat": 0.38, "rimshot": 0.14, "clap_snare": 0.30}.get(style, 0.28)
        return _se.drum_hit(_note, int(np.clip(vel * 108, 45, 127)), _dur)

    # ── numpy fallback ────────────────────────────────────────────────────────
    if style == "crack":
        dur = 0.28; n = int(SR * dur); t = np.linspace(0, dur, n)
        noise = np.random.randn(n).astype(np.float32)
        sos   = butter(3, [200, 9000], btype='band', fs=SR, output='sos')
        noise = sosfilt(sos, noise).astype(np.float32)
        # Body tone: adds thickness so it isn't just noise
        tone  = np.sin(2 * np.pi * 230 * t) * np.exp(-38 * t) * 0.40
        # Short room tail for snare to sit in the mix
        tail  = np.sin(2 * np.pi * 185 * t) * np.exp(-18 * t) * 0.18
        env   = np.exp(-13 * t) * vel
        return np.clip((noise * env * 0.55 + tone + tail), -1, 1).astype(np.float32) * 0.82

    elif style == "fat":
        dur = 0.38; n = int(SR * dur); t = np.linspace(0, dur, n)
        noise = np.random.randn(n).astype(np.float32)
        sos   = butter(3, [100, 7500], btype='band', fs=SR, output='sos')
        noise = sosfilt(sos, noise).astype(np.float32)
        tone  = np.sin(2 * np.pi * 185 * t) * np.exp(-22 * t) * 0.50
        tail  = np.sin(2 * np.pi * 150 * t) * np.exp(-12 * t) * 0.20
        env   = np.exp(-9 * t) * vel
        return np.clip((noise * env * 0.65 + tone + tail), -1, 1).astype(np.float32) * 0.85

    elif style == "rimshot":
        dur = 0.14; n = int(SR * dur); t = np.linspace(0, dur, n)
        noise = np.random.randn(n).astype(np.float32)
        sos   = butter(4, [1200, 10000], btype='band', fs=SR, output='sos')
        noise = sosfilt(sos, noise).astype(np.float32)
        tone  = np.sin(2 * np.pi * 400 * t) * np.exp(-60 * t) * 0.30
        return np.clip((noise * np.exp(-40 * t) * vel * 0.55 + tone), -1, 1).astype(np.float32)

    else:  # clap_snare
        dur = 0.30; n = int(SR * dur); t = np.linspace(0, dur, n)
        noise = np.random.randn(n).astype(np.float32)
        sos   = butter(3, [400, 10000], btype='band', fs=SR, output='sos')
        noise = sosfilt(sos, noise).astype(np.float32)
        # Double-hit clap transient
        dbl   = np.exp(-30 * t) + 0.55 * np.exp(-30 * np.maximum(t - 0.010, 0))
        tone  = np.sin(2 * np.pi * 250 * t) * np.exp(-32 * t) * 0.25
        return np.clip((noise * dbl * vel * 0.68 + tone), -1, 1).astype(np.float32)


def hihat(style: str = "closed", vel: float = 1.0) -> np.ndarray:
    if style in ("closed", "open"):
        s = _ds.get(style)
        if s is not None:
            return (s * vel).astype(np.float32)
    if _se.is_available():
        _note = _se.NOTE_HH_OPEN if style in ("open", "ride") else _se.NOTE_HH_CLOSED
        _dur  = {"closed": 0.06, "pedal": 0.10, "open": 0.45, "ride": 0.60}.get(style, 0.06)
        return _se.drum_hit(_note, int(np.clip(vel * 88, 28, 112)), _dur)

    # ── numpy fallback ────────────────────────────────────────────────────────
    decay = {"closed": 0.045, "pedal": 0.075, "open": 0.38, "ride": 0.55}.get(style, 0.045)
    dur = decay + 0.02; n = int(SR * dur); t = np.linspace(0, dur, n)

    # Band-limited noise (main body)
    noise    = np.random.randn(n).astype(np.float32)
    sos_band = butter(5, [6000, 16000], btype="band", fs=SR, output="sos")
    filtered = sosfilt(sos_band, noise).astype(np.float32)

    # Metallic partials (inharmonic, typical cymbal frequencies)
    metal_freqs = [10250, 8660, 12500, 14200, 7330, 16800]
    metal_amps  = [0.14,  0.12, 0.10,  0.08,  0.06,  0.05]
    metallic = sum(
        amp * np.sin(2 * np.pi * f * t)
        for f, amp in zip(metal_freqs, metal_amps)
    ).astype(np.float32)

    rate = 9 if style == "closed" else 4
    env  = (np.exp(-rate / decay * t) * vel).astype(np.float32)

    return ((filtered * 0.30 + metallic * 0.08) * env).astype(np.float32)


def clap(vel: float = 1.0) -> np.ndarray:
    s = _ds.get("clap")
    if s is not None:
        return (s * vel).astype(np.float32)
    if _se.is_available():
        return _se.drum_hit(_se.NOTE_CLAP, int(np.clip(vel * 100, 40, 120)), 0.18)
    dur = 0.18; n = int(SR * dur); t = np.linspace(0, dur, n)
    noise = np.random.randn(n).astype(np.float32)
    sos   = butter(3, [500, 10000], btype='band', fs=SR, output='sos')
    filt  = sosfilt(sos, noise).astype(np.float32)
    env   = (np.exp(-30 * t) + 0.5 * np.exp(-30 * (t - 0.012))) * vel
    return (filt * np.clip(env, 0, 1) * 0.65).astype(np.float32)


def shaker(vel: float = 1.0) -> np.ndarray:
    if _se.is_available():
        return _se.drum_hit(_se.NOTE_MARACAS, int(np.clip(vel * 78, 28, 100)), 0.09)
    dur = 0.08; n = int(SR * dur); t = np.linspace(0, dur, n)
    noise = np.random.randn(n).astype(np.float32)
    sos   = butter(3, [3200, 12000], btype='band', fs=SR, output='sos')
    filt  = sosfilt(sos, noise).astype(np.float32)
    return (filt * np.exp(-26 * t) * vel * 0.24).astype(np.float32)


def cowbell(vel: float = 1.0) -> np.ndarray:
    if _se.is_available():
        return _se.drum_hit(_se.NOTE_COWBELL, int(np.clip(vel * 102, 45, 122)), 0.20)
    dur = 0.16; n = int(SR * dur); t = np.linspace(0, dur, n)
    wave = (np.sin(2 * np.pi * 562 * t) * 0.55
            + np.sin(2 * np.pi * 845 * t) * 0.45)
    return (wave * np.exp(-22 * t) * vel * 0.52).astype(np.float32)


def log_drum(vel: float = 1.0) -> np.ndarray:
    if _se.is_available():
        return _se.drum_hit(_se.NOTE_AGOGO_LO, int(np.clip(vel * 108, 50, 127)), 0.48)
    dur = 0.48; n = int(SR * dur); t = np.linspace(0, dur, n)
    freq  = 98 * np.exp(-3 * t) + 44
    phase = 2 * np.pi * np.cumsum(freq) / SR
    wave  = np.sin(phase) * 0.78 + np.sin(phase * 1.5) * 0.22
    noise = np.random.randn(n).astype(np.float32)
    sos   = butter(3, [200, 4000], btype='band', fs=SR, output='sos')
    noise = sosfilt(sos, noise).astype(np.float32)
    return ((wave * np.exp(-6 * t) + noise * np.exp(-28 * t) * 0.12) * vel * 0.75).astype(np.float32)


def bass_note(freq: float, dur: float, style: str = "808", vel: float = 1.0,
              prev_freq: float = None) -> np.ndarray:
    # Real recorded bass (owner's Bass Shots), retuned per note — the single most
    # prominent "is this a real beat?" element. Sub-style basses use a tuned 808;
    # pluck-style basses use a real plucked/finger bass one-shot (no portamento,
    # since a plucked note is articulated, not slid like an 808).
    if _bs.available():
        if style in ("808", "sub", "deep"):
            # Punchy decaying 808 for trap; the sustained drone for smooth subs.
            variant = "808" if style == "808" else "808_alt"
            real = _bs.render_hz(freq, dur, vel=vel, prev_freq=prev_freq, variant=variant)
            if real is not None and len(real):
                return real
        elif style == "pluck":
            real = _bs.render_hz(freq, dur, vel=vel, prev_freq=None, variant="pluck")
            if real is not None and len(real):
                return real

    if _se.is_available():
        return _se.bass(freq, dur, style=style, velocity=int(np.clip(vel * 100, 42, 118)))

    # ── numpy fallback ────────────────────────────────────────────────────────
    n = int(SR * dur)
    t = np.linspace(0, dur, n)

    if style == "808":
        # Portamento: exponential glide from previous note frequency
        if prev_freq is not None and abs(prev_freq - freq) > 0.5:
            # Glide length proportional to the pitch distance (larger jumps glide longer)
            semitone_dist = abs(12.0 * np.log2(max(freq, 1) / max(prev_freq, 1)))
            glide_s = float(np.clip(semitone_dist * 0.008, 0.025, 0.12))
            g = int(glide_s * SR)
            freq_arr = np.ones(n) * freq
            if g > 0:
                freq_arr[:g] = np.exp(
                    np.linspace(np.log(max(prev_freq, 20)), np.log(max(freq, 20)), g)
                )
            phase = 2 * np.pi * np.cumsum(freq_arr) / SR
        else:
            phase = 2 * np.pi * freq * t

        # 808 body: fundamental + 2nd harmonic for body + 3rd for character
        wave = (np.sin(phase) * 0.72
                + np.sin(phase * 2) * 0.18
                + np.sin(phase * 3) * 0.10)
        # Tape saturation: the essential 808 character
        wave = np.tanh(wave * 2.8) / np.tanh(2.8)

        # Sub oscillator: pure sine one octave below for extra sub thickness
        # This is how trap producers layer a second 808 tuned one octave down
        sub_freq = freq * 0.5
        sub_phase = 2 * np.pi * sub_freq * t
        sub_wave  = np.sin(sub_phase) * 0.22
        # Sub fades out faster (avoids muddy buildup)
        sub_env   = np.exp(-1.5 * t / max(dur, 0.1))
        wave      = wave + sub_wave * sub_env

    elif style == "deep":
        wave = np.sin(2 * np.pi * freq * t)

    elif style == "pluck":
        # Pluck: sharp attack, fast decay of upper harmonics
        phase = 2 * np.pi * freq * t
        wave  = (np.sin(phase)
                 + np.sin(phase * 2) * 0.32 * np.exp(-9 * t)
                 + np.sin(phase * 3) * 0.12 * np.exp(-15 * t))

    else:  # sub
        wave = (np.sin(2 * np.pi * freq * t) * 0.85
                + np.sin(2 * np.pi * freq * 0.5 * t) * 0.15)

    attack  = min(int(SR * 0.006), n)
    release = min(int(SR * 0.04), n)
    env     = np.ones(n, dtype=np.float32)
    if attack  > 0: env[:attack]   = np.linspace(0, 1, attack)
    if release > 0: env[-release:] = np.linspace(1, 0, release)
    return (wave * env * vel * 0.64).astype(np.float32)


def _bl_saw(phase: np.ndarray, freq: float, max_harm: int = 20) -> np.ndarray:
    """Band-limited sawtooth via additive synthesis (no aliasing). `phase` is
    the running 2π·f·t phase of the fundamental; harmonics ride on multiples."""
    w = np.zeros_like(phase)
    k = 1
    while k <= max_harm and freq * k < SR * 0.45:
        w += np.sin(k * phase) / k
        k += 1
    return w


def _bl_square(phase: np.ndarray, freq: float, max_harm: int = 15) -> np.ndarray:
    """Band-limited square (odd harmonics only) — hollow, plucky character."""
    w = np.zeros_like(phase)
    k = 1
    while k <= max_harm and freq * k < SR * 0.45:
        w += np.sin(k * phase) / k
        k += 2
    return w


def synth_pad(freq: float, dur: float, vel: float = 0.18,
              valence: float = 0.5) -> np.ndarray:
    if _se.is_available():
        return _se.chord_note(freq, dur, velocity=int(np.clip(vel * 280, 22, 78)),
                              valence=valence)

    # ── numpy fallback: warm detuned-saw pad (analog-style) ───────────────────
    # Five lightly detuned saws stacked (a "supersaw") give a lush, moving pad
    # instead of a thin sine; a lowpass keeps it warm. Cutoff opens with valence
    # so brighter songs get an airier pad and darker songs a mellower one.
    n = int(SR * dur); t = np.linspace(0, dur, n)
    ph = 2 * np.pi * freq * t
    detune = [1.0, 2 ** (7 / 1200), 2 ** (-7 / 1200), 2 ** (4 / 1200), 2 ** (-4 / 1200)]
    amp    = [0.34, 0.22, 0.22, 0.12, 0.12]
    wave = sum(a * _bl_saw(ph * d, freq * d, 16) for d, a in zip(detune, amp))
    wave += np.sin(ph * 0.5) * 0.06   # sub octave for body
    cutoff = float(np.clip(900.0 + valence * 1500.0, 500.0, SR * 0.45))
    sos = butter(3, cutoff / (SR / 2.0), btype="low", output="sos")
    wave = sosfilt(sos, wave).astype(np.float32)

    attack  = int(SR * 0.12); release = int(SR * 0.22)
    env = np.ones(n)
    if attack  < n: env[:attack]   = np.linspace(0, 1, min(attack, n))
    if release < n: env[-release:] = np.linspace(1, 0, min(release, n))
    return (wave * env * vel * 0.85).astype(np.float32)


def synth_lead(freq: float, dur: float, vel: float = 0.22) -> np.ndarray:
    if _se.is_available():
        return _se.lead_note(freq, dur, velocity=int(np.clip(vel * 295, 32, 92)))

    # ── numpy fallback: singing saw lead with vibrato + tilted lowpass ────────
    n = int(SR * dur); t = np.linspace(0, dur, n)
    vib   = 1.0 + 0.004 * np.sin(2 * np.pi * 5.5 * t)   # subtle, musical vibrato
    phase = 2 * np.pi * freq * np.cumsum(vib) / SR
    wave  = _bl_saw(phase, freq, 18)
    # Cutoff tracks pitch so high notes stay present and low notes stay warm
    cutoff = float(np.clip(freq * 6.0 + 1600.0, 1600.0, SR * 0.45))
    sos = butter(3, cutoff / (SR / 2.0), btype="low", output="sos")
    wave = sosfilt(sos, wave).astype(np.float32)

    attack = int(SR * 0.012); release = int(SR * 0.08)
    env = np.ones(n)
    if attack  < n: env[:attack]   = np.linspace(0, 1, min(attack, n))
    if release < n: env[-release:] = np.linspace(1, 0, min(release, n))
    return (wave * env * vel * 0.72).astype(np.float32)


def synth_arp(freq: float, dur: float, vel: float = 0.18) -> np.ndarray:
    if _se.is_available():
        return _se.arp_note(freq, dur, velocity=int(np.clip(vel * 320, 28, 88)))

    # ── numpy fallback: plucky filtered square ────────────────────────────────
    n = int(SR * dur); t = np.linspace(0, dur, n)
    ph = 2 * np.pi * freq * t
    wave = _bl_square(ph, freq, 13)
    cutoff = float(np.clip(freq * 8.0 + 2200.0, 2200.0, SR * 0.45))
    sos = butter(3, cutoff / (SR / 2.0), btype="low", output="sos")
    wave = sosfilt(sos, wave).astype(np.float32)
    # Plucky exponential decay (percussive attack, quick fade)
    env = np.exp(-3.0 * t / max(dur, 0.05))
    a = int(SR * 0.003)
    if a < n: env[:a] *= np.linspace(0, 1, a)
    return (wave * env * vel * 0.6).astype(np.float32)


# ── Music theory ──────────────────────────────────────────────────────────────

NOTE_MIDI = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
             'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]

PENTA_MAJOR = [0, 2, 4, 7, 9]      # scale indices for major pentatonic
PENTA_MINOR = [0, 3, 5, 7, 10]     # scale indices for minor pentatonic


def midi_to_hz(midi: int) -> float:
    return 440.0 * (2 ** ((midi - 69) / 12))


def key_notes(key: str, mode: str, octave: int = 2) -> list:
    root      = NOTE_MIDI.get(key, 0) + 12 * octave
    intervals = MAJOR_SCALE if mode == "major" else MINOR_SCALE
    return [root + i for i in intervals]


# ── Genre definitions ─────────────────────────────────────────────────────────
# Each genre has 3-5 unique drum patterns on a 32-step grid.
# Steps 0-7 = beat 1,  8-15 = beat 2,  16-23 = beat 3,  24-31 = beat 4.

GENRES = {
    "trap_dark": {
        "kick_style": "808", "snare_style": "crack", "bass_style": "808", "swing": 0.52,
        "patterns": [
            dict(kick= [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1],
                 hh_o= [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]),
            dict(kick= [1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0],
                 hh_o= [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
            dict(kick= [1,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1],
                 hh_c= [1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1],
                 hh_o= [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
            dict(kick= [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,0,1,0,0,1,0,1,0,0,1,0,0,1,0,1,0,0,1,0,0,1,0,1,0,0,1,0,0,1,0],
                 hh_o= [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]),
        ],
    },

    "trap_melodic": {
        "kick_style": "808", "snare_style": "crack", "bass_style": "808", "swing": 0.51,
        "patterns": [
            dict(kick= [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                 hh_o= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]),
            dict(kick= [1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
                 hh_o= [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
            dict(kick= [1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0],
                 snare=[0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1,1],
                 hh_o= [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
        ],
    },

    "hiphop_boom_bap": {
        "kick_style": "punchy", "snare_style": "fat", "bass_style": "deep", "swing": 0.58,
        "patterns": [
            dict(kick= [1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
                 hh_o= [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
            dict(kick= [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0,1,0],
                 hh_o= [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
            dict(kick= [1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0],
                 hh_o= [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
            dict(kick= [1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0],
                 hh_o= [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]),
        ],
    },

    "hiphop_modern": {
        "kick_style": "punchy", "snare_style": "crack", "bass_style": "pluck", "swing": 0.54,
        "patterns": [
            dict(kick= [1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
                 hh_o= [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
            dict(kick= [1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1],
                 hh_o= [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
            dict(kick= [1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0],
                 hh_o= [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]),
        ],
    },

    "rnb_smooth": {
        "kick_style": "punchy", "snare_style": "fat", "bass_style": "808", "swing": 0.56,
        "patterns": [
            dict(kick= [1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
                 hh_o= [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]),
            dict(kick= [1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
                 hh_o= [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
            dict(kick= [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1],
                 hh_o= [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
        ],
    },

    "rnb_neo_soul": {
        "kick_style": "punchy", "snare_style": "rimshot", "bass_style": "pluck", "swing": 0.60,
        "patterns": [
            dict(kick= [1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0],
                 snare=[0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1],
                 hh_o= [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
            dict(kick= [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                 snare=[0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
                 hh_o= [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
            dict(kick= [1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0],
                 hh_o= [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]),
        ],
    },

    "pop_bright": {
        "kick_style": "punchy", "snare_style": "clap_snare", "bass_style": "pluck", "swing": 0.50,
        "patterns": [
            dict(kick= [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
                 hh_o= [0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0]),
            dict(kick= [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1],
                 hh_o= [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
            dict(kick= [1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0],
                 hh_o= [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
            dict(kick= [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
                 hh_o= [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
        ],
    },

    "afrobeats": {
        "kick_style": "punchy", "snare_style": "crack", "bass_style": "pluck", "swing": 0.53,
        "patterns": [
            dict(kick= [1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
                 hh_o= [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
            dict(kick= [1,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,0,1,0,1,0,0,1,0,0,1,0,1,0,0,1,0,0,1,0,1,0,0,1,0,0,1,0,1,0,0],
                 hh_o= [0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0]),
            dict(kick= [1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0],
                 hh_o= [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
        ],
    },

    "dancehall": {
        "kick_style": "snap", "snare_style": "crack", "bass_style": "808", "swing": 0.52,
        "patterns": [
            dict(kick= [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0],
                 snare=[0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
                 hh_o= [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1]),
            dict(kick= [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0],
                 snare=[0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,1,0,0,1,0,0,1,0,1,0,0,1,0,0,1,0,1,0,0,1,0,0,1,0,1,0,0,1,0,0],
                 hh_o= [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
        ],
    },

    "lofi_chill": {
        "kick_style": "punchy", "snare_style": "rimshot", "bass_style": "deep", "swing": 0.62,
        "patterns": [
            dict(kick= [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,1,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0],
                 hh_o= [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
            dict(kick= [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0],
                 hh_o= [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
            dict(kick= [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0],
                 hh_o= [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
        ],
    },

    "soul_ballad": {
        "kick_style": "punchy", "snare_style": "fat", "bass_style": "deep", "swing": 0.55,
        "patterns": [
            dict(kick= [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
                 hh_o= [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
            dict(kick= [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0],
                 hh_o= [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
        ],
    },

    "drill": {
        "kick_style": "808", "snare_style": "crack", "bass_style": "808", "swing": 0.50,
        "patterns": [
            dict(kick= [1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1],
                 hh_c= [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                 hh_o= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]),
            dict(kick= [1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                 hh_o= [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
            dict(kick= [1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0],
                 hh_c= [1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1],
                 hh_o= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ],
    },

    # ── New genres ────────────────────────────────────────────────────────────

    "uk_drill": {
        "kick_style": "thud", "snare_style": "crack", "bass_style": "808", "swing": 0.50,
        "patterns": [
            dict(kick= [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                 snare=[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                 hh_c= [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                 hh_o= [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
            dict(kick= [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                 snare=[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                 hh_c= [1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1],
                 hh_o= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]),
            dict(kick= [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                 snare=[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                 hh_c= [1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1],
                 hh_o= [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
        ],
    },

    "phonk": {
        "kick_style": "808", "snare_style": "crack", "bass_style": "808", "swing": 0.53,
        "patterns": [
            dict(kick= [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
                 hh_o= [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1]),
            dict(kick= [1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                 hh_o= [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
            dict(kick= [1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0],
                 hh_o= [0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0]),
        ],
    },

    "reggaeton": {
        "kick_style": "punchy", "snare_style": "crack", "bass_style": "sub", "swing": 0.50,
        "patterns": [
            # Classic dembow rhythm
            dict(kick= [1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
                 hh_o= [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
            dict(kick= [1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0],
                 hh_o= [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
        ],
    },

    "amapiano": {
        "kick_style": "punchy", "snare_style": "clap_snare", "bass_style": "deep", "swing": 0.54,
        "patterns": [
            dict(kick= [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
                 hh_o= [0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0]),
            dict(kick= [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0],
                 hh_o= [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
        ],
    },

    # Four-on-the-floor kick, offbeat open hats — unmistakably different from all trap/hiphop
    "club_house": {
        "kick_style": "punchy", "snare_style": "clap_snare", "bass_style": "deep", "swing": 0.50,
        "patterns": [
            dict(kick= [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_o= [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]),
            dict(kick= [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
                 hh_o= [0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0]),
            dict(kick= [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0],
                 hh_o= [0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1]),
        ],
    },

    # Heavy jazz swing (0.68), rimshot-led, syncopated kick — sounds nothing like trap
    "jazz_hop": {
        "kick_style": "punchy", "snare_style": "rimshot", "bass_style": "pluck", "swing": 0.68,
        "patterns": [
            dict(kick= [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                 snare=[0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0],
                 hh_o= [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
            dict(kick= [1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                 snare=[0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,0,1,0,0,1,0,1,0,0,1,0,0,1,0,1,0,0,1,0,0,1,0,1,0,0,1,0,0,1,0],
                 hh_o= [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
            dict(kick= [1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0],
                 snare=[0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0],
                 hh_c= [1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0],
                 hh_o= [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
        ],
    },
}


# ── Production family clustering ──────────────────────────────────────────────
# When a user clicks "Generate Another Beat", the new beat must come from a
# DIFFERENT production family — not just a different name in the same cluster.
GENRE_FAMILIES = {
    "trap_drill": {"trap_dark", "trap_melodic", "drill", "uk_drill", "phonk"},
    "hiphop":     {"hiphop_boom_bap", "hiphop_modern", "jazz_hop"},
    "rnb_soul":   {"rnb_smooth", "rnb_neo_soul", "soul_ballad", "lofi_chill"},
    "pop_world":  {"pop_bright", "afrobeats", "dancehall", "reggaeton", "amapiano", "club_house"},
}
GENRE_TO_FAMILY = {
    g: fam for fam, genres in GENRE_FAMILIES.items() for g in genres
}

# ── Vocal-style → genre pool ──────────────────────────────────────────────────
# When the user picks a vocal style, the beat is constrained to genres that suit
# that style — so a Rap take never lands on a slow ballad and an R&B take stays
# smooth. Within the pool the tempo/emotion of the actual vocal still chooses the
# specific genre (and family-blocking still drives "Generate Another Beat").
STYLE_GENRE_POOLS = {
    "rap":      ["hiphop_modern", "hiphop_boom_bap", "trap_dark", "trap_melodic",
                 "drill", "uk_drill", "phonk", "jazz_hop"],
    "rnb_soul": ["rnb_smooth", "rnb_neo_soul", "soul_ballad", "lofi_chill"],
    "pop":      ["pop_bright", "afrobeats", "dancehall", "reggaeton", "amapiano",
                 "club_house"],
    "melodic":  ["trap_melodic", "rnb_smooth", "rnb_neo_soul", "pop_bright",
                 "afrobeats", "lofi_chill"],
}

# Natural tempo centre of each genre — lets a constrained pool still pick the
# tempo-appropriate option (a fast rap → drill/trap; a slow rap → boom-bap/jazz).
_GENRE_TEMPO_CENTER = {
    "drill": 142, "uk_drill": 144, "trap_dark": 140, "phonk": 138, "trap_melodic": 140,
    "hiphop_modern": 96, "hiphop_boom_bap": 90, "jazz_hop": 84,
    "rnb_smooth": 92, "rnb_neo_soul": 80, "soul_ballad": 68, "lofi_chill": 78,
    "pop_bright": 118, "afrobeats": 108, "dancehall": 100, "reggaeton": 95,
    "amapiano": 112, "club_house": 124,
}


def _pick_from_pool(pool: list, analysis: dict, full_exclude: set,
                    exclude: set) -> str:
    """Choose the tempo-appropriate genre from a style-constrained pool,
    honouring family-blocking; pick among the closest-fitting few for variety."""
    tempo = float(analysis.get("tempo", 95) or 95)
    ranked = sorted(pool, key=lambda g: abs(_GENRE_TEMPO_CENTER.get(g, 100) - tempo))
    avail = [g for g in ranked if g not in full_exclude]
    if not avail:
        avail = [g for g in ranked if g not in exclude]   # relax family block
    if not avail:
        avail = list(pool)
    # Pick among the best-fitting half so the result tracks tempo but still varies
    top = avail[: max(2, len(avail) // 2)]
    return RNG.choice(top)


# ── Genre selection logic ─────────────────────────────────────────────────────

def select_genre(analysis: dict, exclude: list = None, previous_genre: str = None,
                 style_bias: str = None, force_genre: str = None) -> str:
    """
    Select the best genre based on deep vocal analysis.
    - exclude: specific genre names to skip (last 2 used)
    - previous_genre: when set, blocks the entire production family of that genre,
      forcing a cross-family pick so the user hears an obviously different style.
    - style_bias: a STYLE_GENRE_POOLS key (rap/rnb_soul/pop/melodic). When set,
      the genre is chosen from that pool so the beat matches the chosen vocal
      style; the vocal's tempo still selects the specific genre within it.
    - force_genre: when a valid genre name, use it directly. Lets the AI present
      several DISTINCT producer interpretations of the same vocal (the harmony
      still follows the singer regardless of genre).
    """
    if force_genre and force_genre in GENRES:
        return force_genre
    exclude = set(exclude or [])

    # Block the entire production family of the previous beat
    _family_blocked: set = set()
    if previous_genre:
        _prev_family = GENRE_TO_FAMILY.get(previous_genre)
        if _prev_family:
            _family_blocked = GENRE_FAMILIES[_prev_family]

    # Style bias short-circuits to a curated, tempo-ranked pool
    if style_bias and style_bias in STYLE_GENRE_POOLS:
        return _pick_from_pool(STYLE_GENRE_POOLS[style_bias], analysis,
                               exclude | _family_blocked, exclude)

    # Block the entire production family of the previous beat
    family_blocked: set = set()
    if previous_genre:
        prev_family = GENRE_TO_FAMILY.get(previous_genre)
        if prev_family:
            family_blocked = GENRE_FAMILIES[prev_family]
    full_exclude = exclude | family_blocked

    tempo       = analysis.get("tempo", 90)
    mode        = analysis.get("mode", "major")
    valence     = analysis.get("valence", 0.5)
    emotion     = analysis.get("emotion", "smooth")
    vocal_style = analysis.get("vocal_style", "rhythmic")
    energy      = analysis.get("overall_rms", 0.15)
    swing       = analysis.get("swing_ratio", 0.5)

    def pick(*options) -> str | None:
        # Return the best available genre from options, or None if all are
        # family-blocked (caller falls through to the next branch).
        available = [g for g in options if g not in full_exclude]
        if available:
            return RNG.choice(available)
        return None  # all options blocked — let caller try next condition

    def pick_relaxed(*options) -> str:
        # Last-resort pick: honor only explicit exclude, not family block.
        available = [g for g in options if g not in exclude]
        if not available:
            available = list(options)
        return RNG.choice(available)

    # Emotion-first selection (most specific).
    # Each branch uses `pick()` which returns None when family-blocked,
    # allowing fall-through to the next condition.
    g = None
    if emotion == "dark" and tempo >= 130:
        g = pick("trap_dark", "uk_drill", "drill")
    if g is None and emotion == "dark" and tempo >= 110:
        g = pick("trap_dark", "drill", "phonk")
    if g is None and emotion == "dark":
        g = pick("trap_dark", "lofi_chill", "phonk")

    if g is None and emotion == "euphoric" and tempo >= 125:
        g = pick("trap_melodic", "phonk", "pop_bright", "club_house")
    if g is None and emotion == "euphoric":
        g = pick("pop_bright", "afrobeats", "reggaeton", "club_house")

    if g is None and emotion == "uplifting" and vocal_style == "melodic":
        g = pick("pop_bright", "afrobeats", "rnb_smooth", "club_house")
    if g is None and emotion == "uplifting":
        g = pick("pop_bright", "hiphop_modern", "afrobeats", "jazz_hop")

    if g is None and emotion in ("melancholic", "intimate"):
        g = pick("rnb_neo_soul", "soul_ballad", "lofi_chill", "jazz_hop")

    if g is None and emotion == "energetic" and tempo >= 140:
        g = pick("drill", "uk_drill", "trap_dark")
    if g is None and emotion == "energetic" and tempo >= 120:
        g = pick("hiphop_modern", "trap_melodic", "drill", "jazz_hop")
    if g is None and emotion == "energetic":
        g = pick("hiphop_boom_bap", "hiphop_modern", "afrobeats", "jazz_hop")

    if g is not None:
        return g

    # Tempo + swing fallback
    if tempo >= 145:
        g = pick("trap_dark", "uk_drill", "drill", "phonk")
    if g is None and tempo >= 130:
        g = pick("trap_melodic", "drill", "hiphop_modern", "trap_dark", "club_house")
    if g is None and tempo >= 118 and swing > 0.55:
        g = pick("afrobeats", "dancehall", "amapiano", "club_house")
    if g is None and tempo >= 118:
        g = pick("hiphop_modern", "hiphop_boom_bap", "pop_bright", "jazz_hop")
    if g is None and tempo >= 108 and swing > 0.55:
        g = pick("afrobeats", "dancehall", "amapiano", "reggaeton")
    if g is None and tempo >= 108:
        g = pick("hiphop_boom_bap", "hiphop_modern", "pop_bright", "jazz_hop")
    if g is None and tempo >= 98 and mode == "major":
        g = pick("pop_bright", "rnb_smooth", "afrobeats", "reggaeton", "club_house")
    if g is None and tempo >= 98:
        g = pick("rnb_smooth", "hiphop_modern", "rnb_neo_soul", "jazz_hop")
    if g is None and tempo >= 85:
        g = pick("rnb_smooth", "rnb_neo_soul", "soul_ballad", "jazz_hop")
    if g is None and tempo >= 72:
        g = pick("soul_ballad", "lofi_chill", "rnb_neo_soul", "jazz_hop")
    if g is None:
        g = pick("soul_ballad", "lofi_chill", "jazz_hop")

    if g is not None:
        return g

    # All preferred options were family-blocked. Pick any genre outside the
    # blocked family (still honoring explicit exclude).
    cross_family = [genre for genre in GENRES if genre not in full_exclude]
    if cross_family:
        return RNG.choice(cross_family)
    # Absolute last resort: ignore family block entirely
    return pick_relaxed(*list(GENRES.keys()))


# ── Chord progressions ────────────────────────────────────────────────────────

def build_bass_line(key: str, mode: str, bars: int, bar_degrees: list = None) -> list:
    """Key-aware bass line — one MIDI note per beat.

    When `bar_degrees` is given (one 0-indexed scale degree per bar, derived from
    the singer's actual notes), the bass follows THAT harmony so the low end moves
    with the vocal. Otherwise it falls back to a stock progression in the key.
    """
    scale = key_notes(key, mode, octave=2)
    total_beats = bars * 4

    if bar_degrees:
        notes = []
        for beat_idx in range(total_beats):
            degree    = bar_degrees[(beat_idx // 4) % len(bar_degrees)]
            root_midi = scale[degree % len(scale)]
            if beat_idx % 4 == 0:
                note = root_midi
            elif beat_idx % 4 == 2 and RNG.random() < 0.38:
                note = root_midi + 7                      # fifth
            elif RNG.random() < 0.18:
                note = scale[(degree + 2) % len(scale)]   # third
            else:
                note = root_midi
            notes.append(note)
        return notes

    if mode == "major":
        progressions = [
            [1, 1, 5, 5, 6, 6, 4, 4],
            [1, 4, 5, 4],
            [1, 6, 4, 5],
            [1, 5, 6, 3],
            [2, 5, 1, 6],
            [1, 1, 4, 4],
        ]
    else:
        progressions = [
            [1, 7, 6, 7],
            [1, 6, 3, 7],
            [1, 1, 4, 5],
            [1, 6, 7, 1],
            [1, 3, 4, 5],
            [1, 4, 6, 7],
        ]

    prog = RNG.choice(progressions)
    beats_per_chord = 4 if len(prog) <= 4 else 2
    notes = []

    for beat_idx in range(total_beats):
        chord_idx = (beat_idx // beats_per_chord) % len(prog)
        degree    = prog[chord_idx] - 1
        root_midi = scale[degree % len(scale)]

        if beat_idx % 4 == 0:
            note = root_midi
        elif beat_idx % 4 == 2 and RNG.random() < 0.38:
            note = root_midi + 7  # fifth
        elif RNG.random() < 0.18:
            note = scale[(degree + 2) % len(scale)]  # third
        else:
            note = root_midi
        notes.append(note)

    return notes


def build_chord_progression(key: str, mode: str, bars: int, bar_degrees: list = None) -> list:
    """
    Returns list of (beat_start, [midi_notes]) chord events.

    When `bar_degrees` is given (a chord per bar derived from the singer's notes)
    the chords follow the vocal's harmony; otherwise a stock progression is used.

    Uses inversions and drop voicings for harmonic interest.
    - Root position:   [root, 3rd, 5th, 7th]
    - First inversion: [3rd, 5th, root+12, 7th]  (root moved up an octave)
    - Drop 2 voicing:  [root, 5th, 7th, 3rd+12]  (wider spread, more open sound)

    The chord type (inversion) changes every 2 bars to avoid monotony.
    """
    scale_3 = key_notes(key, mode, octave=3)

    if bar_degrees:
        # 0-indexed degrees straight from the vocal harmony
        degrees_per_bar = [bar_degrees[i % len(bar_degrees)] for i in range(bars)]
    else:
        if mode == "major":
            progressions = [[1, 5, 6, 4], [1, 6, 4, 5], [1, 4, 5, 4], [2, 5, 1, 6], [1, 3, 4, 5]]
        else:
            progressions = [[1, 7, 6, 7], [1, 6, 3, 7], [1, 6, 7, 1], [1, 4, 5, 5], [1, 3, 6, 7]]
        prog = RNG.choice(progressions)
        degrees_per_bar = [prog[i % len(prog)] - 1 for i in range(bars)]

    # Voicing type alternates every 2 bars for variety
    voicing_types = ["root", "drop2", "first_inv", "root"]
    RNG.shuffle(voicing_types)

    chords = []
    for i in range(bars):
        degree  = degrees_per_bar[i]
        root    = scale_3[degree % len(scale_3)]
        third   = scale_3[(degree + 2) % len(scale_3)]
        fifth   = scale_3[(degree + 4) % len(scale_3)]
        seventh = scale_3[(degree + 6) % len(scale_3)]

        # Select voicing type for this bar
        vtype = voicing_types[i % len(voicing_types)]

        if vtype == "first_inv":
            # First inversion: bass on 3rd, root moves up
            notes = [third, fifth, root + 12, seventh]
        elif vtype == "drop2":
            # Drop-2: wide open voicing, more spatial feel
            notes = [root, fifth, seventh, third + 12]
        else:
            # Root position with sub-root option for low end
            if RNG.random() < 0.35:
                notes = [root - 12, third, fifth, seventh]  # sub-root for depth
            else:
                notes = [root, third, fifth, seventh]

        chords.append((i * 4, notes))
    return chords


def build_lead_melody(key: str, mode: str, bars: int, energy_arc: str = "steady") -> list:
    """
    Pentatonic lead melody with musical contour driven by energy_arc.
    Contour maps the phrase register (low=more bass octave, high=brighter register)
    so the melody has direction — tension and release — not a random walk.
    energy_arc: "builds" | "fades" | "peaks_middle" | "steady" | "dynamic"
    """
    penta_idx = PENTA_MAJOR if mode == "major" else PENTA_MINOR
    scale_4   = key_notes(key, mode, octave=4)
    penta_hi  = [scale_4[i % len(scale_4)] for i in penta_idx]
    penta_lo  = [n - 12 for n in penta_hi]

    # Build height contour: 0.0 = prefer low register, 1.0 = prefer high
    n_phrases = max(1, bars // 2)
    if energy_arc == "builds":
        contour = np.linspace(0.15, 0.90, n_phrases)
    elif energy_arc == "fades":
        contour = np.linspace(0.85, 0.20, n_phrases)
    elif energy_arc == "peaks_middle":
        x = np.linspace(0, 1, n_phrases)
        contour = np.sin(np.pi * x) * 0.75 + 0.15
    else:  # steady / dynamic — gentle arch per phrase
        x = np.linspace(0, 1, n_phrases)
        contour = 0.40 + 0.30 * np.sin(2 * np.pi * x / max(n_phrases, 2))

    notes       = []
    total_beats = bars * 4

    for phrase_idx in range(n_phrases):
        phrase_start = phrase_idx * 8.0
        if phrase_start >= total_beats:
            break

        # Height value for this phrase drives register selection
        height = float(contour[phrase_idx % len(contour)])
        penta_primary   = penta_hi if height > 0.5 else penta_lo
        penta_secondary = penta_lo if height > 0.5 else penta_hi

        # Build 2-bar phrase: notes with rhythmic diversity
        b = 0.0
        phrase_notes = []
        while b < 8.0:
            if RNG.random() < 0.18:  # occasional rest
                b += RNG.choice([0.5, 1.0])
                continue
            pool = penta_primary if RNG.random() < 0.78 else penta_secondary
            note = RNG.choice(pool)
            # Rhythmic variety weighted toward 0.5 and 1.0 beats
            dur = RNG.choices(
                [0.5, 0.5, 1.0, 1.0, 1.0, 1.5, 2.0],
                weights=[15, 15, 25, 25, 20, 10, 5],
            )[0]
            if b + dur > 8.0:
                dur = 8.0 - b
            phrase_notes.append((b, note, dur))
            b += dur

        for (off, note, dur) in phrase_notes:
            t = phrase_start + off
            if t >= total_beats:
                break
            # Subtle variation on repeat phrases: occasional neighbor-note shift
            if phrase_idx > 0 and RNG.random() < 0.18:
                pool = penta_primary
                try:
                    idx = pool.index(note)
                    note = pool[(idx + RNG.choice([-1, 1])) % len(pool)]
                except (ValueError, IndexError):
                    pass
            notes.append((t, note, min(dur, total_beats - t)))

    return notes


def build_arp(key: str, mode: str, bars: int, beat_dur: float,
              bar_degrees: list = None) -> list:
    """16th-note arpeggio on chord tones — (time_sec, midi_note, dur_sec).

    Follows `bar_degrees` (the SAME chords as the bass/pads/melody) when given, so
    the arp is consonant with the rest of the beat instead of arpeggiating an
    unrelated progression (which makes the layers clash).
    """
    scale_4 = key_notes(key, mode, octave=4)
    if bar_degrees:
        degrees_per_bar = [bar_degrees[b % len(bar_degrees)] for b in range(bars)]
    else:
        if mode == "major":
            prog = RNG.choice([[1, 5, 6, 4], [1, 4, 5, 4], [1, 6, 4, 5]])
        else:
            prog = RNG.choice([[1, 7, 6, 7], [1, 6, 3, 7], [1, 4, 5, 5]])
        degrees_per_bar = [prog[b % len(prog)] - 1 for b in range(bars)]

    arp_dur  = beat_dur * 0.5  # 8th-note arps
    notes    = []
    step_sec = arp_dur

    for bar in range(bars):
        degree     = degrees_per_bar[bar]
        chord_root = scale_4[degree % len(scale_4)]
        chord_3rd  = scale_4[(degree + 2) % len(scale_4)]
        chord_5th  = scale_4[(degree + 4) % len(scale_4)]
        patterns   = [
            [chord_root, chord_3rd, chord_5th, chord_3rd],  # up-down
            [chord_root, chord_5th, chord_3rd, chord_root + 12],  # skip
            [chord_3rd, chord_root, chord_5th, chord_3rd],  # varied
        ]
        arp_pattern = RNG.choice(patterns)
        beat_start  = bar * 4 * beat_dur

        for beat in range(4):
            note = arp_pattern[beat % len(arp_pattern)]
            t    = beat_start + beat * beat_dur
            notes.append((t, note, step_sec * 0.88))

    return notes


# ── Melodic layer effects (reverb + presence) ─────────────────────────────────

def _apply_melodic_fx(lead: np.ndarray, arp: np.ndarray, pad: np.ndarray,
                      sr: int, genre_name: str, valence: float = 0.5,
                      texture: np.ndarray = None) -> np.ndarray:
    """
    Add genre-appropriate reverb to lead melody, arp, and chord pads.

    Without this, dry sine waves sound like a demo. With it, melodic layers
    sit in an acoustic space that matches the genre's production style:
      - Trap/drill: short, tight room (low reverb)
      - Soul/R&B: medium plate reverb
      - Lo-fi: longer vintage room reverb
    """
    reverb_table = {
        "trap_dark":       (0.08, 0.14, 0.06, 0.10),
        "trap_melodic":    (0.12, 0.18, 0.08, 0.12),
        "drill":           (0.05, 0.10, 0.05, 0.08),
        "uk_drill":        (0.05, 0.10, 0.05, 0.08),
        "phonk":           (0.10, 0.16, 0.08, 0.10),
        "hiphop_boom_bap": (0.16, 0.20, 0.12, 0.14),
        "hiphop_modern":   (0.12, 0.18, 0.10, 0.12),
        "jazz_hop":        (0.20, 0.22, 0.16, 0.18),
        "rnb_smooth":      (0.18, 0.22, 0.14, 0.16),
        "rnb_neo_soul":    (0.22, 0.25, 0.16, 0.18),
        "soul_ballad":     (0.28, 0.30, 0.20, 0.22),
        "lofi_chill":      (0.26, 0.28, 0.18, 0.20),
        "pop_bright":      (0.14, 0.20, 0.10, 0.14),
        "afrobeats":       (0.12, 0.18, 0.10, 0.12),
        "club_house":      (0.14, 0.20, 0.10, 0.14),
        "dancehall":       (0.14, 0.18, 0.10, 0.12),
        "reggaeton":       (0.12, 0.16, 0.08, 0.12),
        "amapiano":        (0.18, 0.22, 0.14, 0.16),
    }
    # (lead_room, lead_wet, arp_room, pad_room_factor)
    entry = reverb_table.get(genre_name, (0.14, 0.18, 0.10, 0.14))
    lead_room, lead_wet, arp_room, pad_room = entry

    try:
        from pedalboard import Pedalboard, Reverb

        lead_board = Pedalboard([
            Reverb(room_size=lead_room, damping=0.78,
                   wet_level=lead_wet, dry_level=1.0 - lead_wet * 0.5),
        ])
        lead_out = lead_board(lead, sr)

        arp_board = Pedalboard([
            Reverb(room_size=arp_room, damping=0.82,
                   wet_level=0.18, dry_level=0.82),
        ])
        arp_out = arp_board(arp, sr)

        pad_board = Pedalboard([
            Reverb(room_size=pad_room, damping=0.72,
                   wet_level=0.22, dry_level=0.78),
        ])
        pad_out = pad_board(pad, sr)

        out = lead_out + arp_out + pad_out
        if texture is not None:
            # Real loops carry their own recorded space; just a touch of glue.
            tex_board = Pedalboard([
                Reverb(room_size=max(pad_room, 0.18), damping=0.7,
                       wet_level=0.16, dry_level=0.92),
            ])
            out = out + tex_board(texture, sr)
        return out.astype(np.float32)

    except Exception:
        base = lead + arp + pad
        if texture is not None:
            base = base + texture
        return base.astype(np.float32)


def _numpy_crash(vel: float = 0.75) -> np.ndarray:
    """Crash cymbal synthesized without FluidSynth."""
    dur = 2.0; n = int(SR * dur); t = np.linspace(0, dur, n)
    noise = np.random.randn(n).astype(np.float32)
    sos   = butter(4, [4000, min(18000, SR // 2 - 100)], btype='band', fs=SR, output='sos')
    filt  = sosfilt(sos, noise).astype(np.float32)
    # Metallic partials for cymbal character
    metal = sum(
        a * np.sin(2 * np.pi * f * t)
        for f, a in [(9800, 0.08), (7200, 0.06), (11500, 0.05), (13200, 0.04)]
    ).astype(np.float32)
    env   = np.exp(-2.2 * t)
    return np.clip((filt * 0.22 + metal) * env * vel, -1.0, 1.0).astype(np.float32)


# ── Per-instrument effects ────────────────────────────────────────────────────

def apply_drum_fx(kick_mix: np.ndarray, snare_mix: np.ndarray,
                  hat_mix: np.ndarray, sr: int = SR) -> tuple:
    """
    Process each drum layer independently.
    Kick: transient-preserving compression + sub boost.
    Snare: compression + short room reverb.
    Hat:  compression + high-shelf air.
    """
    try:
        from pedalboard import Pedalboard, Compressor, Reverb, Gain, HighShelfFilter

        kick_board = Pedalboard([
            # Fast attack punch + slow release lets sub breathe
            Compressor(threshold_db=-8, ratio=4.0, attack_ms=0.8, release_ms=80.0),
            Gain(gain_db=1.5),
        ])
        kick_out = kick_board(kick_mix, sr)

        snare_board = Pedalboard([
            Compressor(threshold_db=-12, ratio=3.0, attack_ms=1.5, release_ms=120.0),
            # Short reverb (room_size=0.08) gives snare body without washing out
            Reverb(room_size=0.08, damping=0.88, wet_level=0.16, dry_level=0.84),
        ])
        snare_out = snare_board(snare_mix, sr)

        hat_board = Pedalboard([
            Compressor(threshold_db=-18, ratio=2.0, attack_ms=1.0, release_ms=60.0),
            # Air shelf: hi-hats need to sparkle above the vocal
            HighShelfFilter(cutoff_frequency_hz=10000, gain_db=1.5),
        ])
        hat_out = hat_board(hat_mix, sr)

        return kick_out, snare_out, hat_out
    except Exception:
        return kick_mix, snare_mix, hat_mix


# ── Ghost note synthesis ──────────────────────────────────────────────────────

def ghost_snare(vel: float = 0.22) -> np.ndarray:
    if _se.is_available():
        # Ghost notes are side-stick hits at very low velocity (felt, not heard)
        return _se.drum_hit(_se.NOTE_RIM, int(np.clip(vel * 127, 12, 38)), 0.20)

    # ── numpy fallback ────────────────────────────────────────────────────────
    dur = 0.20; n = int(SR * dur); t = np.linspace(0, dur, n)
    noise = np.random.randn(n).astype(np.float32)
    sos   = butter(3, [250, 8000], btype="band", fs=SR, output="sos")
    noise = sosfilt(sos, noise).astype(np.float32)
    tone  = np.sin(2 * np.pi * 210 * t) * np.exp(-30 * t) * 0.25
    return np.clip((noise * np.exp(-18 * t) * vel * 0.45 + tone), -1, 1).astype(np.float32)


# ── Humanization ──────────────────────────────────────────────────────────────

def humanize_velocity(base: float, variance: float = 0.11) -> float:
    return float(np.clip(base + RNG.gauss(0, variance), 0.28, 1.0))


def humanize_timing(t_sec: float, swing_ratio: float = 0.5,
                     step_dur: float = 0.1) -> float:
    """
    Apply groove-aware timing humanization.

    Straight timing (swing_ratio=0.5): small random offset only.
    Swing (swing_ratio>0.55): odd 8th notes pushed back by swing amount.
    This creates the "laid back" feel of swing grooves.
    """
    # Very small jitter (±0.8 ms) — keep the drums essentially grid-tight so they
    # LOCK with the bass/chords/melody (which sit exactly on the grid). Heavy
    # jitter made the drums float against the rigid harmony = "loose/off" groove.
    jitter = RNG.gauss(0, 0.0008)

    # Swing only for clearly-swung material, and gently. The melodic layers are
    # rigid, so a big drum swing would un-lock the groove; keep it subtle.
    beat_8th = round(t_sec / (step_dur * 4))  # position in 8th notes
    if swing_ratio > 0.56 and beat_8th % 2 == 1:
        swing_push = min((swing_ratio - 0.5), 0.12) * step_dur * 1.0
        return t_sec + swing_push + jitter
    return t_sec + jitter


# ── Master bus ────────────────────────────────────────────────────────────────

def _stereo_widen(core: np.ndarray, sr: int, width: float = 0.55) -> tuple:
    """
    Mono-safe stereo widener (Hilbert mid/side).

    A real beat is stereo; a mono preview sounds small. We synthesize a side
    channel from the 90°-phase-shifted (Hilbert) version of the mid/high content
    and form L = M + S, R = M - S.

    Because S cancels when summed to mono (L + R = 2·M exactly), this adds a
    genuinely wide image with ZERO mono-compatibility coloration — the sub stays
    centered and the final mixdown is unaffected. `width` scales the side level.
    """
    try:
        from scipy.signal import hilbert
        # Side comes only from >180 Hz content — keep bass mono and tight.
        sos_hi = butter(4, 180.0 / (sr / 2.0), btype="high", output="sos")
        upper  = sosfilt(sos_hi, core).astype(np.float32)
        # 90° phase shift → decorrelated from the original → real width
        side   = np.imag(hilbert(upper)).astype(np.float32) * float(width)
        L = (core + side).astype(np.float32)
        R = (core - side).astype(np.float32)
        return L, R
    except Exception:
        return core.astype(np.float32), core.astype(np.float32)


def _master_beat(mono: np.ndarray, sr: int, genre_name: str = "") -> np.ndarray:
    """
    Master the mono beat into a polished, competitive STEREO file.

    The beat preview a user auditions before accepting IS this output (the mixer
    only runs later, at final-mix time), so it must already sound produced:
    bus glue, tonal EQ, harmonic saturation, mono-safe stereo width and a
    transparent brick-wall limiter to a streaming-loud ceiling.

    Returns float32 stereo of shape (N, 2).
    """
    try:
        from pedalboard import (
            Pedalboard, Compressor, Limiter,
            HighShelfFilter, LowShelfFilter, PeakFilter, HighpassFilter,
        )
        x = mono.astype(np.float32)

        # 1 — Clean, glue, and tonal balance on the mono core
        core = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=28),                       # kill DC/rumble
            Compressor(threshold_db=-15, ratio=2.2, attack_ms=18.0, release_ms=180.0),  # bus glue
            LowShelfFilter(cutoff_frequency_hz=90,  gain_db=2.0),         # low-end weight
            PeakFilter(cutoff_frequency_hz=300, gain_db=-1.6, q=0.9),     # clear the mud
            PeakFilter(cutoff_frequency_hz=3000, gain_db=1.2, q=0.6),     # presence
            PeakFilter(cutoff_frequency_hz=7500, gain_db=-1.2, q=0.8),    # tame harsh upper-mids
            HighShelfFilter(cutoff_frequency_hz=12000, gain_db=1.4),      # air (gentle, above harsh band)
        ])(x, sr)

        # 2 — Soft tape-style saturation: glue + warmth, sub-audible THD (light)
        drive = 1.25
        core = (np.tanh(core * drive) / np.tanh(drive)).astype(np.float32)

        # 3 — Mono-safe stereo image (width nudged per genre for variety)
        wide_genres = {"pop_bright", "afrobeats", "amapiano", "dancehall",
                       "rnb_smooth", "rnb_neo_soul", "lofi_chill", "soul_ballad"}
        width = 0.62 if genre_name in wide_genres else 0.48
        L, R = _stereo_widen(core, sr, width=width)

        # 4 — Transparent limiter per channel to catch transient peaks
        lim = Pedalboard([Limiter(threshold_db=-1.0, release_ms=90.0)])
        L = lim(L, sr).astype(np.float32)
        R = lim(R, sr).astype(np.float32)
        stereo = np.stack([L, R], axis=1).astype(np.float32)

        # 5 — Loudness: target -14 LUFS (streaming standard). This keeps the
        #     preview competitive without crushing dynamics, and keeps the beat
        #     near the vocal level the mixer/scorer expect. The final mix
        #     re-normalizes anyway, so this only shapes the preview.
        target_lufs = -14.0
        if _HAS_PYLOUDNORM:
            try:
                cur = _PYLN_METER(sr).integrated_loudness(stereo)
                if np.isfinite(cur) and cur > -60:
                    stereo = (stereo * (10.0 ** ((target_lufs - cur) / 20.0))).astype(np.float32)
            except Exception:
                pass
        # Safety ceiling at -0.7 dBFS true peak
        peak = float(np.max(np.abs(stereo)))
        ceil = 10.0 ** (-0.7 / 20.0)
        if peak > ceil:
            stereo *= ceil / peak
        return stereo
    except Exception:
        # Fallback: dual-mono, peak-normalized — never fail the render
        peak = float(np.max(np.abs(mono))) or 1.0
        m = (mono / peak * 0.89).astype(np.float32)
        return np.stack([m, m], axis=1)


# ── Song arrangement ──────────────────────────────────────────────────────────

# Per-section energy multiplier (drives velocity so the song breathes).
_SECTION_VEL = {
    "intro": 0.70, "verse": 1.00, "pre": 1.06, "chorus": 1.16,
    "bridge": 0.82, "chorus2": 1.20, "outro": 0.66,
}


def _build_arrangement(bars: int) -> list:
    """
    Per-bar section labels for a full, professionally-shaped song:
        intro → verse → pre-chorus → chorus → bridge → final chorus → outro
    Scales to any length and always includes a bridge for mid-song contrast, so
    the beat evolves instead of looping one pattern.
    """
    if bars < 8:
        return (["intro"] + ["verse"] * max(1, bars - 2) + ["outro"])[:bars]
    blocks = [("intro", 0.09), ("verse", 0.30), ("pre", 0.10), ("chorus", 0.18),
              ("bridge", 0.13), ("chorus2", 0.14), ("outro", 0.06)]
    secs: list = []
    for name, frac in blocks:
        secs += [name] * max(1, round(bars * frac))
    if len(secs) > bars:                      # trim, but keep the outro last
        secs = secs[:bars - 1] + ["outro"]
    while len(secs) < bars:                    # pad the final chorus
        secs.insert(len(secs) - 1, "chorus2")
    return secs


def _choose_section_patterns(patterns: list) -> tuple:
    """Pick distinct drum patterns for verse / chorus / bridge where the genre
    offers more than one, so sections contrast instead of repeating."""
    n = len(patterns)
    if n == 1:
        return patterns[0], patterns[0], patterns[0]
    idx = list(range(n))
    RNG.shuffle(idx)
    verse  = patterns[idx[0]]
    chorus = patterns[idx[1]]
    bridge = patterns[idx[2 % n]]
    return verse, chorus, bridge


def _resample_pitch(wave: np.ndarray, semitones: float) -> np.ndarray:
    """Cheap pitch shift by resampling (also changes length slightly) — used to
    make per-hit hi-hat variants so repeated hats don't sound machine-gunned."""
    if abs(semitones) < 1e-3 or len(wave) < 4:
        return wave
    ratio = 2.0 ** (semitones / 12.0)
    new_n = max(2, int(len(wave) / ratio))
    idx = np.linspace(0, len(wave) - 1, new_n)
    return np.interp(idx, np.arange(len(wave)), wave).astype(np.float32)


# ── Main generator ────────────────────────────────────────────────────────────

def generate_beat(
    analysis: dict = None,
    tempo: float = 90.0,
    bars: int = 16,
    seed: int = None,
    exclude_genres: list = None,
    attempt: int = 1,
    previous_genre: str = None,
    style_bias: str = None,
    master: bool = True,
    bar_degrees: list = None,
    melody: list = None,
    melody_loop_beats: float = 16.0,
    force_genre: str = None,
    performance: dict = None,
) -> tuple:
    """
    Generate a unique beat from deep vocal analysis.
    Returns (wav_bytes, genre_name).

    `performance` (optional) is a performance map from performance_map.py: the
    arrangement, per-bar tension and pause reactions are taken from what the
    singer actually did, so the beat is accompaniment shaped by THIS take rather
    than a bar-count template the vocal is dropped onto. When it's absent or
    low-confidence the generator falls back to the template, byte-for-byte as
    before.

    master=False returns the raw mono mix (fast) for candidate scoring; the
    chosen winner is then polished once via master_beat_bytes(). This avoids
    running the full master chain on every throwaway candidate.
    """
    # Seed: deterministic per-analysis-per-attempt for reproducibility,
    # but different on every attempt so the user never gets the same beat twice.
    if seed is not None:
        RNG.seed(seed)
    else:
        # Process-STABLE hash of the analysis (crc32, NOT builtin hash() — that is
        # salted per-process by PYTHONHASHSEED, so the "same take → same beat"
        # guarantee would break across server restarts / workers).
        base   = zlib.crc32(repr(sorted((analysis or {}).items())).encode()) & 0xFFFFFFFF
        # Tie the seed to THIS performance's contour so two different takes get
        # different structure/motifs (anti-fingerprint) while the same take stays
        # reproducible; the attempt offset still varies "Generate Another".
        # Gate on the SAME confidence used to consume the map (below) so an
        # ignored/low-confidence map is byte-identical to no map at all.
        if (performance and performance.get("seed") is not None
                and float(performance.get("confidence", 0) or 0) >= 0.35):
            base ^= (int(performance["seed"]) & 0xFFFFFFFF)
        offset = (attempt - 1) * 999_983  # large prime offset per attempt
        RNG.seed((base + offset) % (2 ** 31))

    # Extract analysis fields
    if analysis:
        tempo       = float(analysis.get("tempo", tempo))
        key         = analysis.get("key", "C")
        mode        = analysis.get("mode", "major")
        energy      = float(analysis.get("overall_rms", 0.15))
        valence     = float(analysis.get("valence", 0.5))
        energy_arc  = analysis.get("energy_arc", "steady")
    else:
        key, mode, energy, valence, energy_arc = "C", "major", 0.15, 0.5, "steady"

    tempo = float(np.clip(tempo, 60, 190))

    genre_name = select_genre(analysis or {}, exclude=exclude_genres or [],
                              previous_genre=previous_genre, style_bias=style_bias,
                              force_genre=force_genre)
    genre      = GENRES[genre_name]
    # Pick the genre-appropriate real drum kit (808 for trap, 909 for pop, etc.)
    _ds.select_kit(genre_name)
    # Distinct patterns per section so the arrangement evolves (anti-loop)
    pat_verse, pat_chorus, pat_bridge = _choose_section_patterns(genre["patterns"])
    pattern    = pat_verse   # default for intro/verse/pre/outro

    # ── Timing setup ─────────────────────────────────────────────────────────
    beat_dur    = 60.0 / tempo
    step_dur    = beat_dur / 8          # 32nd-note grid
    total_steps = 32 * bars
    total_samp  = int(SR * step_dur * total_steps) + int(SR * 2)
    mix         = np.zeros(total_samp, dtype=np.float32)

    # ── Pre-render instruments ────────────────────────────────────────────────
    kick_wave  = kick(style=genre["kick_style"])
    snare_wave = snare(style=genre["snare_style"])
    hh_c_wave  = hihat("closed")
    hh_o_wave  = hihat("open")
    # Per-hit pitch variants so repeated hats breathe instead of machine-gunning
    hh_c_variants = [hh_c_wave] + [_resample_pitch(hh_c_wave, s) for s in (0.9, -0.7, 1.6, -1.3)]
    def _hat_c():
        return RNG.choice(hh_c_variants)
    clap_wave  = clap()
    shk_wave   = shaker()
    cow_wave   = cowbell() if genre_name in ("phonk",) else None
    log_wave   = log_drum() if genre_name in ("amapiano",) else None

    # Vocal density (onsets/sec) → percussion busyness. A dense, rapid-fire vocal
    # gets busier hats/shakers/ghosts; a sparse, spacious vocal gets an open beat.
    vocal_density = float((analysis or {}).get("density", 3.0) or 3.0)
    density_factor = float(np.clip((vocal_density - 2.0) / 4.0, 0.0, 1.0))  # 0=sparse, 1=dense

    use_clap   = RNG.random() < 0.42
    use_shaker = RNG.random() < (0.22 + 0.40 * density_factor)
    ghost_prob = 0.16 + 0.34 * density_factor      # subtle fills scale with density
    busy_hats  = density_factor > 0.55             # add offbeat hats for dense vocals
    thin_hats  = density_factor < 0.30             # open, spacious hats for sparse vocals

    # Separate buses for per-instrument FX processing
    kick_bus     = np.zeros(total_samp, dtype=np.float32)
    snare_bus    = np.zeros(total_samp, dtype=np.float32)
    hat_bus      = np.zeros(total_samp, dtype=np.float32)
    lead_bus     = np.zeros(total_samp, dtype=np.float32)
    arp_bus      = np.zeros(total_samp, dtype=np.float32)
    pad_bus      = np.zeros(total_samp, dtype=np.float32)
    texture_bus  = np.zeros(total_samp, dtype=np.float32)   # real owner loops

    def place(wave, t_sec, vel=1.0):
        start = max(0, int(t_sec * SR))
        end   = min(start + len(wave), total_samp)
        if end > start:
            mix[start:end] += wave[:end - start] * float(vel)

    def place_kick(wave, t_sec, vel=1.0):
        start = max(0, int(t_sec * SR))
        end   = min(start + len(wave), total_samp)
        if end > start:
            kick_bus[start:end] += wave[:end - start] * float(vel)

    def place_snare(wave, t_sec, vel=1.0):
        start = max(0, int(t_sec * SR))
        end   = min(start + len(wave), total_samp)
        if end > start:
            snare_bus[start:end] += wave[:end - start] * float(vel)

    def place_hat(wave, t_sec, vel=1.0):
        start = max(0, int(t_sec * SR))
        end   = min(start + len(wave), total_samp)
        if end > start:
            hat_bus[start:end] += wave[:end - start] * float(vel)

    def place_lead(wave, t_sec, vel=1.0):
        start = max(0, int(t_sec * SR))
        end   = min(start + len(wave), total_samp)
        if end > start:
            lead_bus[start:end] += wave[:end - start] * float(vel)

    def place_arp(wave, t_sec, vel=1.0):
        start = max(0, int(t_sec * SR))
        end   = min(start + len(wave), total_samp)
        if end > start:
            arp_bus[start:end] += wave[:end - start] * float(vel)

    def place_pad(wave, t_sec, vel=1.0):
        start = max(0, int(t_sec * SR))
        end   = min(start + len(wave), total_samp)
        if end > start:
            pad_bus[start:end] += wave[:end - start] * float(vel)

    def place_texture(wave, t_sec, vel=1.0):
        start = max(0, int(t_sec * SR))
        end   = min(start + len(wave), total_samp)
        if end > start:
            texture_bus[start:end] += wave[:end - start] * float(vel)

    # ── Arrangement sections ──────────────────────────────────────────────────
    # Full structure: intro → verse → pre → chorus → bridge → final chorus → outro
    # When a reliable performance map is present the sections come FROM the singer
    # (chorus on the real peak, sections breaking where they actually breathe);
    # otherwise the bar-count template is used and everything below is unchanged.
    perf_sections = None
    perf_tension = None
    perf_pauses: list = []
    if (performance and isinstance(performance.get("sections"), list)
            and len(performance["sections"]) == bars
            and float(performance.get("confidence", 0) or 0) >= 0.35):
        perf_sections = list(performance["sections"])
        # Sanitize externally-supplied tension to finite 0..1 floats so a stray
        # NaN/inf can never propagate into a velocity → int() conversion and crash
        # the render (generation must never crash). Wrong length → ignore tension.
        _bt = performance.get("bar_tension")
        if isinstance(_bt, (list, tuple)) and len(_bt) == bars:
            perf_tension = [float(t) if np.isfinite(t) else 0.0 for t in _bt]
        perf_pauses = performance.get("pauses") or []
        _arc = performance.get("energy_arc")
        if _arc:
            energy_arc = _arc
    arrangement = perf_sections if perf_sections else _build_arrangement(bars)

    def section_of(b: int) -> str:
        return arrangement[b] if 0 <= b < len(arrangement) else "verse"

    def bar_drive(b: int) -> float:
        """How hard the singer is going in this bar (0..1) — replaces the single
        global density knob so busyness builds and relaxes WITH the performance.
        Falls back to the global density when there's no map."""
        if perf_tension is not None and 0 <= b < len(perf_tension):
            return float(np.clip(perf_tension[b], 0.0, 1.0))
        return density_factor

    # ── Pause reactions: how the band ANSWERS the singer's breaths ─────────────
    # silence → open up: drop hats/ghosts/arp/lead AND duck the bass; let it breathe
    # fill    → a drum fill in the last beat builds back into the next phrase
    # bass_move → the bass walks an 8th-note answer through the gap
    # chord_change → the chord re-articulates softly INTO the breath (pad swell)
    silence_spans: list = []
    fill_spans: list = []
    bassmove_spans: list = []
    chordchg_spans: list = []
    for (sb, eb, kind) in perf_pauses:
        s_sec, e_sec = float(sb) * beat_dur, float(eb) * beat_dur
        if kind == "silence":
            silence_spans.append((s_sec, e_sec))
        elif kind == "fill":
            cut = max(s_sec, e_sec - beat_dur)
            silence_spans.append((s_sec, cut))     # open up, THEN fill back in
            fill_spans.append((cut, e_sec))
        elif kind == "bass_move":
            bassmove_spans.append((s_sec, e_sec))
        elif kind == "chord_change":
            chordchg_spans.append((s_sec, e_sec))

    def _in_spans(t: float, spans: list) -> bool:
        for (s, e) in spans:
            if s <= t < e:
                return True
        return False

    # Bars where a chorus begins (drop a crash + lift)
    chorus_entry_bars = {
        b for b in range(bars)
        if section_of(b) in ("chorus", "chorus2") and section_of(b - 1) not in ("chorus", "chorus2")
    }

    ghost_wav = ghost_snare()
    swing     = float(analysis.get("swing_ratio", genre.get("swing", 0.5)) if analysis else genre.get("swing", 0.5))

    # Hi-hat rolls on the last bar of each 4-bar phrase (tension into transitions)
    roll_bars = {p + 3 for p in range(0, bars, 4)}

    for step in range(total_steps):
        bar    = step // 32
        t_raw  = step * step_dur
        # Apply groove-aware humanization (swing + jitter)
        t_now  = humanize_timing(t_raw, swing_ratio=swing, step_dur=step_dur)

        section = section_of(bar)
        is_chorus = section in ("chorus", "chorus2")
        is_bridge = section == "bridge"

        # Per-bar busyness AND loudness track how hard the singer is going
        # (build/relax WITH them). With no performance map these equal the original
        # global values, so the non-mapped render is byte-identical.
        if perf_tension is not None:
            drive = bar_drive(bar)
            ghost_prob_b = 0.10 + 0.40 * drive
            busy_hats_b  = drive > 0.58
            thin_hats_b  = drive < 0.32
            drive_vel    = 0.85 + 0.30 * drive   # 0.85..1.15 loudness with the take
        else:
            ghost_prob_b, busy_hats_b, thin_hats_b = ghost_prob, busy_hats, thin_hats
            drive_vel = 1.0

        # When the singer pauses, open up: drop hats/ghosts/shaker in the gap.
        in_silence = _in_spans(t_raw, silence_spans)
        in_fill    = _in_spans(t_raw, fill_spans)

        # Active pattern for this section — verse/chorus/bridge differ
        active = pat_chorus if is_chorus else (pat_bridge if is_bridge else pat_verse)
        p = step % len(active["kick"])

        vel_mod    = _SECTION_VEL.get(section, 1.0) * drive_vel
        skip_snare = section == "intro"
        skip_hh    = section in ("intro", "outro") and step % 4 != 0
        # Bridge breakdown: thin the hats to downbeats so the section feels open
        if is_bridge and step % 8 != 0:
            skip_hh = True

        # Velocity accent: beats 1 and 3 hit harder (downbeat emphasis)
        beat_in_bar = (step % 32) // 8      # 0=beat1, 1=beat2, 2=beat3, 3=beat4
        accent = 1.08 if beat_in_bar in (0, 2) else 0.95
        vel_base = 0.82 * accent

        if active["kick"][p]:
            kick_v = vel_base * 0.95 * vel_mod * (0.85 if is_bridge else 1.0)
            place_kick(kick_wave, t_now, humanize_velocity(kick_v))

        if active["snare"][p] and not skip_snare:
            if use_clap and RNG.random() < 0.28:
                place_snare(clap_wave, t_now, humanize_velocity(0.80 * vel_mod))
            else:
                place_snare(snare_wave, t_now, humanize_velocity(vel_base * 0.85 * vel_mod))

        # Ghost notes: subtle snare hits on 16th-note offbeats (soul/groove fill).
        # Probability scales with how hard the singer is going right now.
        if (section in ("verse", "chorus", "chorus2") and not active["snare"][p]
                and step % 2 == 1 and not in_silence and RNG.random() < ghost_prob_b):
            place_snare(ghost_wav, t_now, humanize_velocity(0.22, variance=0.06))

        # Busy-hat fill for dense vocals: extra closed hat on the 8th-note offbeat
        if (busy_hats_b and not skip_hh and not in_silence and step % 4 == 2
                and section in ("verse", "chorus", "chorus2")):
            place_hat(_hat_c(), t_now, humanize_velocity(0.30 * vel_mod))

        if active["hh_c"][p] and not skip_hh and not in_silence:
            # Sparse vocal → thin closed hats toward 8th-notes for an open feel
            if not (thin_hats_b and step % 4 != 0):
                place_hat(_hat_c(), t_now, humanize_velocity(0.44 * vel_mod))
        if active["hh_o"][p] and not skip_hh and not in_silence:
            place_hat(hh_o_wave, t_now, humanize_velocity(0.54 * vel_mod))

        # Pause fill: when the singer breathes before a lift, the band answers —
        # a clean 16th-note hat roll that crescendos back into the phrase, with a
        # snare push just before the singer returns. (16ths, not 32nds — a roll,
        # not a machine-gun.)
        if in_fill and section not in ("intro", "outro") and step % 2 == 0:
            prog = 0.5
            for (_fs, _fe) in fill_spans:
                if _fs <= t_raw < _fe:
                    prog = (t_raw - _fs) / max(_fe - _fs, 1e-6)
                    break
            place_hat(_hat_c(), t_now, humanize_velocity(0.26 + 0.30 * prog))
            if step % 8 == 6:                       # one snare push near the lift
                place_snare(snare_wave, t_now, humanize_velocity(0.45 * vel_mod))

        # Hi-hat rolls: last bar of each phrase gets a triplet roll build
        if bar in roll_bars and section not in ("intro", "outro", "bridge"):
            bar_step = step % 32
            if bar_step >= 28:  # last 4 steps of the bar
                for roll_offset in [0.0, step_dur * 0.333, step_dur * 0.667]:
                    t_roll = t_now + roll_offset
                    roll_vel = humanize_velocity(0.30 + (bar_step - 28) * 0.05)
                    place_hat(_hat_c(), t_roll, roll_vel)

        if use_shaker and step % 4 == 0 and not in_silence and section not in ("intro", "outro"):
            place(shk_wave, t_now, humanize_velocity(0.28 * vel_mod))

        # Genre-specific extra percussion
        if cow_wave is not None and is_chorus and step % 8 == 4:
            place(cow_wave, t_now, humanize_velocity(0.55))
        if log_wave is not None and section in ("verse", "pre", "chorus", "chorus2") and step % 16 == 8:
            place(log_wave, t_now, humanize_velocity(0.70))

        # Pre-chorus build: 8th-note hi-hat doubling (energy ramp)
        if section == "pre" and step % 4 == 2:
            place_hat(_hat_c(), t_now + step_dur * 0.5, humanize_velocity(0.35))

    # ── Apply per-instrument FX (compress + reverb snare) ────────────────────
    kick_bus, snare_bus, hat_bus = apply_drum_fx(kick_bus, snare_bus, hat_bus, SR)
    mix += kick_bus + snare_bus + hat_bus

    # ── Bass line (with portamento between notes for 808) ─────────────────────
    bass_style = genre.get("bass_style", "808")
    bass_vol   = 0.90 if "808" in bass_style else 0.72
    bass_notes = build_bass_line(key, mode, bars, bar_degrees=bar_degrees)
    prev_bass_freq = None

    for beat_i, midi in enumerate(bass_notes):
        bar = beat_i // 4
        sec = section_of(bar)
        if sec in ("intro", "outro"):   # no bass in intro/outro
            prev_bass_freq = None
            continue
        t_start  = beat_i * beat_dur
        freq     = midi_to_hz(midi)
        note_dur = beat_dur * RNG.uniform(0.74, 0.93)
        vel      = humanize_velocity(bass_vol, 0.07)
        if sec in ("chorus", "chorus2"):
            vel *= 1.08
        elif sec == "bridge":
            vel *= 0.82          # pull bass back so the bridge breathes
        if perf_tension is not None:
            vel *= (0.85 + 0.30 * bar_drive(bar))   # low end leans in with the take
        if silence_spans and _in_spans(t_start, silence_spans):
            vel *= 0.45          # singer's breath — open the low end up too
        place(bass_note(freq, note_dur, style=bass_style, vel=vel,
                        prev_freq=prev_bass_freq), t_start)
        prev_bass_freq = freq

        # bass_move: the singer left a gap — the bass walks an 8th-note answer
        # into it (a pickup to the next chord) instead of holding through silence.
        if bassmove_spans and _in_spans(t_start + beat_dur * 0.5, bassmove_spans):
            nxt = bass_notes[beat_i + 1] if beat_i + 1 < len(bass_notes) else midi
            pf = midi_to_hz(nxt)
            place(bass_note(pf, beat_dur * 0.45, style=bass_style,
                            vel=vel * 0.88, prev_freq=freq),
                  t_start + beat_dur * 0.5)
            prev_bass_freq = pf

    # ── Crash cymbal at every chorus entry (lift into the hook) ───────────────
    # NOTE: added straight to `mix` — the drum buses were already summed above,
    # so placing it on kick_bus here would be silent.
    crash_w = (_se.drum_hit(_se.NOTE_CRASH, 88, 2.0)
               if _se.is_available() else _numpy_crash(0.70))
    for cb in chorus_entry_bars:
        place(crash_w, cb * 4 * beat_dur, 0.55)

    # ── Chord pads (on separate bus for reverb treatment) ─────────────────────
    for (beat_start, midi_notes) in build_chord_progression(key, mode, bars, bar_degrees=bar_degrees):
        bar = beat_start // 4
        sec = section_of(bar)
        if sec == "intro":
            continue
        t_start     = beat_start * beat_dur
        chord_dur_s = 4.0 * beat_dur * 0.88
        pad_vel     = 0.22 + valence * 0.10
        if sec in ("chorus", "chorus2"):
            pad_vel *= 1.18
        elif sec == "bridge":
            pad_vel *= 1.25      # pads carry the stripped bridge
        if perf_tension is not None:
            pad_vel *= (0.90 + 0.20 * bar_drive(bar))
        for midi in midi_notes:
            place_pad(synth_pad(midi_to_hz(midi), chord_dur_s, vel=pad_vel, valence=valence), t_start)

        # chord_change reaction: when the singer breathes inside this bar, the
        # chord re-articulates softly INTO that space (a short pad swell) so the
        # harmony moves to meet the singer instead of just holding.
        if chordchg_spans:
            bar_t0, bar_t1 = t_start, t_start + 4.0 * beat_dur
            for (cs, ce) in chordchg_spans:
                if bar_t0 <= cs < bar_t1:
                    for midi in midi_notes:
                        place_pad(synth_pad(midi_to_hz(midi), max(ce - cs, beat_dur),
                                            vel=pad_vel * 0.55, valence=valence), cs)
                    break

    # ── Real owner loops (texture) — decide before the lead so the synth lead can
    # step aside where a real loop becomes the hook ───────────────────────────
    # The synth pad/lead are the "cheap"-sounding layers. A real recorded loop,
    # pitched to the song key and locked to tempo, becomes the chorus hook
    # (topline) and opens the song (intro feature). It never touches the
    # vocal-derived bass/pad backbone — purely additive on texture_bus.
    # Vocal-first: when the singer's own transcribed melody exists, THAT is the
    # chorus hook — a canned loop reused across songs must never displace it.
    # The topline loop is only the hook when no reliable vocal melody exists.
    tx_topline_entry = None
    topline_secs: set = set()
    if _tx.available() and not melody and RNG.random() < 0.9:
        tx_topline_entry = _tx.pick("topline", NOTE_MIDI.get(key, 0), mode, valence, rng=RNG)
        # Only let the synth lead step aside if the loop ACTUALLY renders (this
        # probe also warms the per-song cache for the real placement below); else
        # the chorus would be left with no melodic hook at all.
        if tx_topline_entry is not None:
            _probe = _tx.render(tx_topline_entry, NOTE_MIDI.get(key, 0), tempo,
                                int(0.5 * SR), "topline", valence)
            if _probe is not None and len(_probe):
                topline_secs = {"chorus", "chorus2"}
            else:
                tx_topline_entry = None

    # ── Arpeggiated synth (verse/pre/chorus, not intro/bridge/outro) ──────────
    # Arp only when the lead is NOT playing the vocal melody — two busy melodic
    # lines at once muddy the beat. With a melody, the lead is the single hook.
    if not melody and RNG.random() < 0.6:
        # Arp gives the VERSES light movement; the chorus is left for the lead
        # hook so the two don't pile up.
        arp_vel = 0.20 + valence * 0.07
        for (t_sec, midi, dur_s) in build_arp(key, mode, bars, beat_dur, bar_degrees=bar_degrees):
            bar = int(t_sec / (4 * beat_dur))
            sec = section_of(bar)
            if sec not in ("verse", "pre"):
                continue
            if _in_spans(t_sec, silence_spans):   # let the singer's pause breathe
                continue
            place_arp(synth_arp(midi_to_hz(midi), dur_s, vel=arp_vel), t_sec)

    # ── Lead melody ───────────────────────────────────────────────────────────
    # When the singer's actual transcribed melody is supplied, the lead PLAYS
    # THAT TUNE (looped, an octave up so it sparkles over the vocal instead of
    # masking it). Otherwise it falls back to a generated pentatonic line.
    if melody:
        lead_vel    = 0.22 + valence * 0.06
        total_beats = bars * 4
        loop        = float(melody_loop_beats) if melody_loop_beats else 16.0
        n_loops     = int(np.ceil(total_beats / loop))
        # Reserve the melodic hook for the build + chorus (and bridge); leave the
        # verses sparse (drums + bass + pad) so the song has dynamics and space
        # instead of every layer sounding at once. Where a real loop is the hook
        # (topline_secs), the synth lead steps aside so they don't pile up.
        lead_sections = {"pre", "chorus", "chorus2", "bridge"} - topline_secs
        for rep in range(n_loops):
            for (beat_pos, m_midi, dur_beats) in melody:
                t_beat = rep * loop + beat_pos
                if t_beat >= total_beats:
                    continue
                sec = section_of(int(t_beat // 4))
                if sec not in lead_sections:
                    continue
                if _in_spans(t_beat * beat_dur, silence_spans):  # honour the singer's space
                    continue
                v = lead_vel * (1.14 if sec in ("chorus", "chorus2")
                                else (0.80 if sec == "bridge" else 1.0))
                place_lead(synth_lead(midi_to_hz(m_midi + 12),
                                      dur_beats * beat_dur * 0.92, vel=v),
                           t_beat * beat_dur)
    elif RNG.random() < 0.72:
        # Start the lead at the first non-intro bar so it threads the whole song
        intro_bars  = sum(1 for s in arrangement if s == "intro")
        lead_start_bar = max(intro_bars, RNG.randint(2, 4))
        lead_offset    = lead_start_bar * 4 * beat_dur
        lead_vel       = 0.24 + valence * 0.08

        # Lead hook lives in the build + chorus + bridge (not the verses, which the
        # arp covers) — keeps each section's role distinct. A real loop hook
        # (topline_secs) takes over those sections from the synth lead.
        lead_sections = {"pre", "chorus", "chorus2", "bridge"} - topline_secs
        for (beat_pos, midi, dur_beats) in build_lead_melody(key, mode, bars - lead_start_bar, energy_arc):
            t_start = lead_offset + beat_pos * beat_dur
            bar     = int(t_start / (4 * beat_dur))
            sec     = section_of(bar)
            if sec not in lead_sections:
                continue
            if _in_spans(t_start, silence_spans):   # don't sing over the singer's silence
                continue
            v = lead_vel * (1.14 if sec in ("chorus", "chorus2", "bridge") else 1.0)
            place_lead(synth_lead(midi_to_hz(midi), dur_beats * beat_dur * 0.84, vel=v), t_start)

    # ── Render the real owner loops onto the texture bus ──────────────────────
    if _tx.available():
        song_key_pc = NOTE_MIDI.get(key, 0)
        # Contiguous section runs (sec, start_bar, end_bar)
        runs, b = [], 0
        while b < bars:
            sec = section_of(b)
            e = b
            while e < bars and section_of(e) == sec:
                e += 1
            runs.append((sec, b, e))
            b = e

        # 1) INTRO feature — the song opens with a real instrument (intro has no
        # bass/pads, so zero harmonic clash). Use the chorus hook loop so the
        # intro previews it; let it ring ~1 bar into the first section.
        feat_entry = (tx_topline_entry
                      or _tx.pick("topline", song_key_pc, mode, valence, rng=RNG)
                      or _tx.pick("bed", song_key_pc, mode, valence, rng=RNG))
        if feat_entry is not None:
            for sec, b0, b1 in runs:
                if sec != "intro":
                    continue
                # half-bar ring-out into the next section (smooths the transition
                # without the full-range loop muddying the verse low end)
                n = int(((b1 - b0) * 4 + 2) * beat_dur * SR)
                wav = _tx.render(feat_entry, song_key_pc, tempo, n, "feature", valence)
                if wav is not None:
                    place_texture(wav, b0 * 4 * beat_dur, 1.0)

        # 2) CHORUS topline hook — a real melodic loop, high-passed to sit above
        # the pads, replacing the synth lead in those bars.
        if tx_topline_entry is not None:
            for sec, b0, b1 in runs:
                if sec not in topline_secs:
                    continue
                n = int((b1 - b0) * 4 * beat_dur * SR)
                wav = _tx.render(tx_topline_entry, song_key_pc, tempo, n, "topline", valence)
                if wav is not None:
                    place_texture(wav, b0 * 4 * beat_dur, 1.0)

        # 3) CHORUS bed — quiet, low-passed warmth under the hook. The hook is
        # either the real-loop topline or the singer's own melody lead; the bed
        # is background texture (not a hook), so it supports both.
        bed_secs = topline_secs or ({"chorus", "chorus2"} if melody else set())
        if bed_secs:
            bed_entry = _tx.pick("bed", song_key_pc, mode, valence, rng=RNG)
            if bed_entry is not None:
                for sec, b0, b1 in runs:
                    if sec not in bed_secs:
                        continue
                    n = int((b1 - b0) * 4 * beat_dur * SR)
                    wav = _tx.render(bed_entry, song_key_pc, tempo, n, "bed", valence)
                    if wav is not None:
                        place_texture(wav, b0 * 4 * beat_dur, 1.0)

    # ── Apply reverb/depth to melodic layers ─────────────────────────────────
    # Pass texture only when real loops were actually placed, so the no-loop path
    # is unchanged from before this feature.
    _tex = texture_bus if (_tx.available() and np.any(texture_bus)) else None
    melodic_combined = _apply_melodic_fx(lead_bus, arp_bus, pad_bus,
                                         SR, genre_name, valence, texture=_tex)
    mix += melodic_combined

    # ── Trim ──────────────────────────────────────────────────────────────────
    exact_len = int(SR * step_dur * total_steps)
    mix = mix[:exact_len]

    # Pre-master headroom so the limiter has something clean to work with
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix / peak * 0.70

    if not master:
        # Fast path: raw mono mix for candidate scoring (mastered later)
        fade = int(SR * 0.06)
        if fade > 0 and len(mix) > 2 * fade:
            mix[:fade]  *= np.linspace(0, 1, fade)
            mix[-fade:] *= np.linspace(1, 0, fade)
        buf = io.BytesIO()
        sf.write(buf, mix, SR, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read(), genre_name

    # ── Professional master bus → polished stereo (this IS the preview) ────────
    buf = io.BytesIO()
    sf.write(buf, _finalize_master(mix, genre_name), SR, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read(), genre_name


def _finalize_master(mono: np.ndarray, genre_name: str) -> np.ndarray:
    """Master a mono mix into polished, edge-faded stereo."""
    stereo = _master_beat(mono.astype(np.float32), SR, genre_name)
    fade = int(SR * 0.06)
    if fade > 0 and len(stereo) > 2 * fade:
        stereo[:fade]  *= np.linspace(0, 1, fade)[:, None]
        stereo[-fade:] *= np.linspace(1, 0, fade)[:, None]
    return stereo


def master_beat_bytes(mono_wav_bytes: bytes, genre_name: str) -> bytes:
    """Apply the master chain to an unmastered mono beat (the scored winner)."""
    y, _ = sf.read(io.BytesIO(mono_wav_bytes))
    if y.ndim > 1:
        y = y.mean(axis=1)
    buf = io.BytesIO()
    sf.write(buf, _finalize_master(y, genre_name), SR, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()
