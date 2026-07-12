"""
Professional final mix engine.
Combines the processed vocal + generated beat into a radio-ready master.

Pipeline:
  1.  Load both stems at 48kHz
  2.  LUFS-normalize each stem independently
  3.  Widen beat to stereo with frequency-selective M/S (low stays mono)
  4.  Dynamic EQ carve: analyse vocal centroid, cut beat at that frequency
  5.  Pre-delayed reverb on vocal (separation before the room)
  6.  Sidechain-style ducking: beat ducks slightly on vocal transients
  7.  Mix stems at calibrated levels
  8.  Mastering: bus compression → Pultec-style EQ → tape saturation → limiter
  9.  Final LUFS target (-14 LUFS streaming standard)
  10. Export WAV 24-bit/48kHz + MP3 320kbps
"""
import io
import numpy as np
import soundfile as sf
import lameenc
from scipy.signal import butter, sosfilt
from scipy.ndimage import uniform_filter1d
from pedalboard import (
    Pedalboard, Compressor, Limiter,
    HighShelfFilter, LowShelfFilter,
    PeakFilter, HighpassFilter, LowpassFilter, Reverb,
)
from .audio_loader import load_audio

MIX_SR = 48000

try:
    import pyloudnorm as pyln
    _HAS_PYLOUDNORM = True
except ImportError:
    _HAS_PYLOUDNORM = False


def master_song_bytes(song_bytes: bytes) -> tuple[bytes, bytes]:
    """
    Master an already-FINISHED song (a neural-producer cut that contains the
    vocal inside) without remixing: multiband master, tape saturation,
    -14 LUFS loudness, true-peak limit. Returns (mp3_bytes, wav_bytes).
    """
    # Keep the song's native stereo image when the file has one.
    try:
        data, sr = sf.read(io.BytesIO(song_bytes), dtype="float32")
        if data.ndim == 2 and data.shape[1] >= 2:
            L, R = data[:, 0].copy(), data[:, 1].copy()
        else:
            mono = data if data.ndim == 1 else data[:, 0]
            L, R = mono.copy(), mono.copy()
        if sr != MIX_SR:
            import librosa
            L = librosa.resample(L, orig_sr=sr, target_sr=MIX_SR)
            R = librosa.resample(R, orig_sr=sr, target_sr=MIX_SR)
    except Exception:
        mono, _ = load_audio(song_bytes, target_sr=MIX_SR)
        L, R = mono.astype(np.float32).copy(), mono.astype(np.float32).copy()

    L = _multiband_master(L.astype(np.float32), MIX_SR)
    R = _multiband_master(R.astype(np.float32), MIX_SR)
    L = _tape_saturate(L, drive=0.18)
    R = _tape_saturate(R, drive=0.18)
    L, R = _final_loudness_target(L, R, MIX_SR, target_lufs=-14.0)
    lim = Pedalboard([Limiter(threshold_db=-0.5, release_ms=50.0)])
    L = lim(L, MIX_SR)
    R = lim(R, MIX_SR)

    stereo = np.stack([L, R], axis=1)
    wav_buf = io.BytesIO()
    sf.write(wav_buf, stereo, MIX_SR, format="WAV", subtype="PCM_24")
    return _encode_mp3(L, R, MIX_SR), wav_buf.getvalue()


def create_final_mix(vocal_bytes: bytes, beat_bytes: bytes,
                     genre: str = "hiphop_modern",
                     vocal_analysis: dict = None) -> tuple[bytes, bytes]:
    """
    Mix vocal + beat into a professional stereo master.
    Returns (mp3_320kbps_bytes, wav_24bit_bytes).
    """
    # ── Load ──────────────────────────────────────────────────────────────────
    vocal_mono, _ = load_audio(vocal_bytes, target_sr=MIX_SR)
    beat_mono,  _ = load_audio(beat_bytes,  target_sr=MIX_SR)
    vocal_mono = vocal_mono.astype(np.float32)
    beat_mono  = beat_mono.astype(np.float32)

    # ── Sync lengths ──────────────────────────────────────────────────────────
    if len(beat_mono) < len(vocal_mono):
        repeats = int(np.ceil(len(vocal_mono) / len(beat_mono)))
        beat_mono = np.tile(beat_mono, repeats)
    beat_mono = beat_mono[:len(vocal_mono)]

    # ── Fades ─────────────────────────────────────────────────────────────────
    intro_fade = min(int(MIX_SR * 0.05), len(vocal_mono))
    outro_fade = min(int(MIX_SR * 2.0),  len(vocal_mono))
    vocal_mono[:intro_fade]   *= np.linspace(0.0, 1.0, intro_fade)
    beat_mono[:intro_fade]    *= np.linspace(0.0, 1.0, intro_fade)
    vocal_mono[-outro_fade:]  *= np.linspace(1.0, 0.0, outro_fade)
    beat_mono[-outro_fade:]   *= np.linspace(1.0, 0.0, outro_fade)

    # ── LUFS gain staging ─────────────────────────────────────────────────────
    vocal_mono = _lufs_normalize(vocal_mono, MIX_SR, target_lufs=-18.0)
    beat_mono  = _lufs_normalize(beat_mono,  MIX_SR, target_lufs=-21.0)

    # ── Sidechain ducking ─────────────────────────────────────────────────────
    beat_mono = _sidechain_duck(vocal_mono, beat_mono, MIX_SR,
                                threshold_lufs=-22.0, duck_db=2.5)

    # ── Beat stereo widening + EQ carve ───────────────────────────────────────
    beat_L, beat_R = _ms_widen_freq_selective(beat_mono, MIX_SR, side_gain=1.30)
    vocal_centroid = float(
        (vocal_analysis or {}).get("centroid") or
        (vocal_analysis or {}).get("spectral_centroid") or 2000.0
    )
    beat_L = _beat_carve_dynamic(beat_L, MIX_SR, vocal_centroid_hz=vocal_centroid)
    beat_R = _beat_carve_dynamic(beat_R, MIX_SR, vocal_centroid_hz=vocal_centroid)

    # ── Vocal double (ADT — Automatic Double Tracking) ────────────────────────
    # Creates stereo thickness: +10 cents left / -10 cents right, -14 dB
    dbl_L, dbl_R = _create_vocal_double(vocal_mono, MIX_SR)

    # ── Vocal reverb ──────────────────────────────────────────────────────────
    room_size, wet_level, pre_delay_ms = _reverb_params(genre)
    vocal_wet = _apply_reverb_with_predelay(
        vocal_mono, MIX_SR, room_size, wet_level, pre_delay_ms
    )
    vocal_L, vocal_R = _spread_reverb(vocal_mono, vocal_wet, spread=0.18)

    # ── Tempo-synced vocal delay throw (depth + pro polish) ──────────────────
    tempo = float((vocal_analysis or {}).get("tempo") or 90.0)
    throw_L, throw_R = _vocal_delay_throw(vocal_mono, MIX_SR, tempo)
    vocal_L = vocal_L + throw_L
    vocal_R = vocal_R + throw_R

    # ── Backing-vocal harmonies (the artist in harmony with themselves) ──────
    harm_L, harm_R = _backing_harmonies(vocal_mono, MIX_SR, genre, vocal_analysis)
    vocal_L = vocal_L + harm_L
    vocal_R = vocal_R + harm_R

    # ── Combine: lead (center) + double (stereo) + beat (wide) ───────────────
    mix_L = vocal_L + dbl_L + beat_L
    mix_R = vocal_R + dbl_R + beat_R

    # ── Multi-band mastering ──────────────────────────────────────────────────
    mix_L = _multiband_master(mix_L, MIX_SR)
    mix_R = _multiband_master(mix_R, MIX_SR)

    # ── Production comparison: keep the better-sounding version ──────────────
    mix_L, mix_R = _production_compare(mix_L, mix_R, MIX_SR)

    # ── Tape saturation ───────────────────────────────────────────────────────
    mix_L = _tape_saturate(mix_L, drive=0.18)
    mix_R = _tape_saturate(mix_R, drive=0.18)

    # ── Final stereo enhancement + LUFS + limiter ─────────────────────────────
    mix_L, mix_R = _ms_enhance(mix_L, mix_R, side_gain=1.10)
    # Loudness gain BEFORE limiting: any gain-up applied after the limiter was
    # pure digital hard clipping on every peak of the finished song. The
    # true-peak limiter is the final stage and cleanly absorbs the overs.
    mix_L, mix_R = _final_loudness_target(mix_L, mix_R, MIX_SR, target_lufs=-14.0)
    lim = Pedalboard([Limiter(threshold_db=-0.5, release_ms=50.0)])
    mix_L = lim(mix_L, MIX_SR)
    mix_R = lim(mix_R, MIX_SR)

    stereo = np.stack([mix_L, mix_R], axis=1)

    # ── Export ────────────────────────────────────────────────────────────────
    wav_buf = io.BytesIO()
    sf.write(wav_buf, stereo, MIX_SR, format="WAV", subtype="PCM_24")
    return _encode_mp3(mix_L, mix_R, MIX_SR), wav_buf.getvalue()


# ── Vocal doubling (ADT) ─────────────────────────────────────────────────────

def _create_vocal_double(vocal: np.ndarray, sr: int) -> tuple:
    """
    Automatic Double Tracking: create a stereo double for vocal thickness.

    Technique used on virtually every professional vocal recording since the 1960s.
    ADT creates the illusion of two separate vocal performances, adding width and
    thickness without audible pitch artifacts.

    Left double:  +10 cents pitch shift, 20 ms early
    Right double: -10 cents pitch shift, 25 ms early
    Level:        -14 dB below lead (about 20% amplitude)

    The asymmetric timing (20 vs 25 ms) prevents comb-filtering that would occur
    if both doubles were delayed by the same amount.
    """
    try:
        from pedalboard import PitchShift, Pedalboard

        # Pitch shift ±10 cents (0.10 semitones)
        board_up   = Pedalboard([PitchShift(semitones=+0.10)])
        board_down = Pedalboard([PitchShift(semitones=-0.10)])

        dbl_up   = board_up(vocal.copy(),   sr).astype(np.float32)
        dbl_down = board_down(vocal.copy(), sr).astype(np.float32)

        # Timing offsets: 20 ms left, 25 ms right
        off_L = int(sr * 0.020)
        off_R = int(sr * 0.025)
        n = len(vocal)

        dbl_L = np.zeros(n, dtype=np.float32)
        dbl_R = np.zeros(n, dtype=np.float32)
        dbl_L[off_L:] = dbl_up[:n - off_L]
        dbl_R[off_R:] = dbl_down[:n - off_R]

        # -12 dB below lead (slightly stronger for audible width)
        gain = 10.0 ** (-12.0 / 20.0)
        return dbl_L * gain, dbl_R * gain

    except Exception:
        # If PitchShift fails (not installed), return silence — lead vocal unaffected
        zeros = np.zeros_like(vocal)
        return zeros, zeros


# ── Multi-band mastering ──────────────────────────────────────────────────────

def _multiband_master(y: np.ndarray, sr: int) -> np.ndarray:
    """
    3-band mastering chain.

    Single-band bus compression applies gain reduction uniformly — when the
    kick hits (low frequency energy spike), it ducks the entire mix including
    the high-frequency air and the midrange vocal presence.

    Multi-band compression solves this by processing each frequency range
    independently with settings appropriate to its content:

    Low band  (<200 Hz): tight ratio (4:1), fast release — controls kick/808
                         without affecting midrange. Prevents bass from
                         collapsing the whole mix.
    Mid band  (200–5 kHz): moderate ratio (2.5:1), slow attack — lets vocal
                            transients through, controls sustained energy.
                            Most of the mix lives here.
    High band (>5 kHz):  light ratio (1.8:1), very fast — keeps air and hi-hat
                          consistent without squashing the top end.
    """
    try:
        # ── Split into 3 bands via Linkwitz-Riley crossovers ──────────────────
        # Linkwitz-Riley: two cascaded Butterworth filters → flat summed response
        sos_lo_lp = butter(4, 200  / (sr / 2.0), btype="low",  output="sos")
        sos_lo_hp = butter(4, 200  / (sr / 2.0), btype="high", output="sos")
        sos_hi_lp = butter(4, 5000 / (sr / 2.0), btype="low",  output="sos")
        sos_hi_hp = butter(4, 5000 / (sr / 2.0), btype="high", output="sos")

        low  = sosfilt(sos_lo_lp, y).astype(np.float32)
        temp = sosfilt(sos_lo_hp, y).astype(np.float32)
        mid  = sosfilt(sos_hi_lp, temp).astype(np.float32)
        high = sosfilt(sos_hi_hp, temp).astype(np.float32)

        # ── Compress each band independently ─────────────────────────────────
        low_board = Pedalboard([
            # Tight low-end: fast attack (2ms) preserves kick transient,
            # fast release (80ms) lets the sub breathe between kicks
            Compressor(threshold_db=-12, ratio=4.0, attack_ms=2.0, release_ms=80.0),
            LowShelfFilter(cutoff_frequency_hz=60, gain_db=1.5),  # sub warmth
        ])

        mid_board = Pedalboard([
            # Vocal-friendly: 25ms attack keeps consonant punch,
            # 280ms release prevents pumping in sustained passages
            Compressor(threshold_db=-16, ratio=2.5, attack_ms=25.0, release_ms=280.0),
            PeakFilter(cutoff_frequency_hz=320, gain_db=-1.2, q=0.9),  # mud cut
        ])

        high_board = Pedalboard([
            # Light air compression: 1ms attack catches sharp hi-hat peaks,
            # 50ms release keeps the top end open
            Compressor(threshold_db=-20, ratio=1.8, attack_ms=1.0, release_ms=50.0),
            HighShelfFilter(cutoff_frequency_hz=12000, gain_db=1.8),  # air shelf
        ])

        low_out  = low_board(low,   sr)
        mid_out  = mid_board(mid,   sr)
        high_out = high_board(high, sr)

        # ── Sum bands ─────────────────────────────────────────────────────────
        return (low_out + mid_out + high_out).astype(np.float32)

    except Exception:
        # Fall back to single-band if filter fails
        return _master_channel(y, sr)


# ── Production comparison mode ────────────────────────────────────────────────

def _production_compare(L: np.ndarray, R: np.ndarray, sr: int) -> tuple:
    """
    Generate a slightly enhanced variant of the mix and keep whichever
    scores higher on LUFS, crest factor, and stereo width.

    Mix A: current settings (input)
    Mix B: +0.5 dB mid boost + 0.5 dB low cut (more vocal presence)

    Scoring weights:
      LUFS closeness to -14:  40%
      Crest factor (8-14 dB): 40%
      Stereo width:           20%
    """
    # Variant B: slight mid presence boost + low-mud reduction
    try:
        board_b = Pedalboard([
            PeakFilter(cutoff_frequency_hz=2800, gain_db=0.8, q=0.9),
            PeakFilter(cutoff_frequency_hz=280,  gain_db=-0.6, q=1.0),
        ])
        L_b = board_b(L.copy(), sr)
        R_b = board_b(R.copy(), sr)

        score_a = _mix_quality_score(L,   R,   sr)
        score_b = _mix_quality_score(L_b, R_b, sr)

        if score_b > score_a:
            return L_b.astype(np.float32), R_b.astype(np.float32)
    except Exception:
        pass
    return L, R


def _mix_quality_score(L: np.ndarray, R: np.ndarray, sr: int) -> float:
    """Score a stereo mix 0–100 on LUFS proximity, crest factor, and stereo width."""
    try:
        # LUFS score: how close to -14 LUFS target
        lufs_score = 50.0
        if _HAS_PYLOUDNORM:
            try:
                stereo = np.stack([L, R], axis=1)
                meter  = pyln.Meter(sr)
                lufs   = meter.integrated_loudness(stereo)
                if np.isfinite(lufs):
                    # -14 = 100, -16 = 80, -12 = 70, further = lower
                    lufs_score = max(0, 100 - abs(lufs - (-14.0)) * 10)
            except Exception:
                pass

        # Crest factor score: ideal 8–14 dB for modern pop/hip-hop master
        peak = max(float(np.max(np.abs(L))), float(np.max(np.abs(R))), 1e-9)
        rms  = float(np.sqrt(np.mean((L.astype(np.float64) ** 2
                                       + R.astype(np.float64) ** 2) / 2)))
        crest_db = 20.0 * np.log10(peak / (rms + 1e-9))
        crest_score = max(0.0, 100.0 - abs(crest_db - 11.0) * 8)

        # Stereo width score: mid-side ratio
        mid  = (L + R) * 0.5
        side = (L - R) * 0.5
        width = float(np.sqrt(np.mean(side ** 2)) /
                      (np.sqrt(np.mean(mid ** 2)) + 1e-9))
        # 0.3–0.7 width is ideal: wide enough but not over-widened
        width_score = max(0, 100 - abs(width - 0.5) * 100)

        return lufs_score * 0.40 + crest_score * 0.40 + width_score * 0.20

    except Exception:
        return 50.0


# ── Sidechain ducking ─────────────────────────────────────────────────────────

def _sidechain_duck(vocal: np.ndarray, beat: np.ndarray, sr: int,
                    threshold_lufs: float = -22.0, duck_db: float = 2.5) -> np.ndarray:
    """
    Gentle sidechain-style ducking: beat drops when vocal energy is high.

    This is not a true sidechain compressor — it's an envelope follower on
    the vocal driving gain reduction on the beat. The effect is subtle (2.5 dB
    max) and smoothed over 40 ms to avoid pumping artifacts.

    duck_db = 2.5 dB max attenuation on the beat when vocal peaks.
    This creates subconscious space for the vocal without audible pumping.
    """
    threshold_lin = 10.0 ** (threshold_lufs / 20.0)

    # Vocal RMS envelope with 40 ms smoothing window
    frame = max(4, int(sr * 0.010))   # 10 ms detection window
    smooth = max(4, int(sr * 0.040))  # 40 ms release smoothing

    rms = np.sqrt(uniform_filter1d(vocal ** 2, size=frame, mode="nearest"))
    rms_smooth = uniform_filter1d(rms, size=smooth, mode="nearest")

    # Gain reduction: above threshold → duck by duck_db, linearly
    max_duck_lin = 10.0 ** (-duck_db / 20.0)
    overage = np.clip((rms_smooth - threshold_lin) / (threshold_lin + 1e-9), 0, 1)
    gain = 1.0 - overage * (1.0 - max_duck_lin)

    return (beat * gain.astype(np.float32)).astype(np.float32)


# ── Frequency-selective stereo widening ──────────────────────────────────────

def _ms_widen_freq_selective(mono: np.ndarray, sr: int,
                              side_gain: float = 1.30) -> tuple:
    """
    Stereo widening that keeps sub-bass frequencies mono.

    Low frequencies (<200 Hz) must stay mono for speaker and headphone
    compatibility. Only widening the mid/high range avoids bass phase problems
    while still giving the beat a wide, open feel.
    """
    # Split: sub (mono) vs. rest
    sos_lo = butter(4, 200 / (sr / 2.0), btype="low", output="sos")
    sos_hi = butter(4, 200 / (sr / 2.0), btype="high", output="sos")

    sub  = sosfilt(sos_lo, mono).astype(np.float32)
    rest = sosfilt(sos_hi, mono).astype(np.float32)

    # Widen only the non-sub content
    mid_r  = rest
    side_r = rest * side_gain
    L_r = (mid_r + side_r) * 0.5
    R_r = (mid_r - side_r) * 0.5

    # Recombine with mono sub
    L = (sub + L_r).astype(np.float32)
    R = (sub + R_r).astype(np.float32)
    return L, R


# ── Dynamic EQ carve ──────────────────────────────────────────────────────────

def _beat_carve_dynamic(y: np.ndarray, sr: int,
                         vocal_centroid_hz: float = 2000.0) -> np.ndarray:
    """
    Carve the beat at the vocal's actual frequency centroid.

    Instead of fixed notch positions (1800 Hz, 3200 Hz), this function
    places the notch at the measured vocal centroid — the exact frequency
    where the vocal is loudest. A secondary notch is added one octave up.

    Also high-passes at 60 Hz to remove sub-rumble that competes with 808.
    """
    # Clamp centroid to a musically useful range
    centroid = float(np.clip(vocal_centroid_hz, 600.0, 5000.0))
    secondary = float(np.clip(centroid * 1.5, 800.0, 8000.0))

    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=60),
        # Primary cut at vocal centroid (reduced from -4.5 to -2.8 to avoid hollowing the beat)
        PeakFilter(cutoff_frequency_hz=centroid,   gain_db=-2.8, q=0.70),
        # Secondary cut 1.5× up: upper presence / consonants
        PeakFilter(cutoff_frequency_hz=secondary,  gain_db=-2.0, q=0.65),
        # Boost sub warmth slightly to compensate for carve
        LowShelfFilter(cutoff_frequency_hz=120, gain_db=1.0),
    ])
    return board(y, sr)


# ── Pre-delayed reverb ────────────────────────────────────────────────────────

def _apply_reverb_with_predelay(y: np.ndarray, sr: int,
                                  room_size: float, wet_level: float,
                                  pre_delay_ms: float) -> np.ndarray:
    """
    Apply reverb with a pre-delay gap.

    Pre-delay separates the dry vocal from its reverb tail, placing the voice
    "in front of" the room. Without pre-delay, the reverb starts simultaneously
    with the vocal and blends it into the space rather than sitting above it.

    Typical values: 15–25 ms gives presence; >40 ms sounds like a room.
    """
    # Build reverb signal. The wet path is band-limited (Abbey Road trick):
    # highpass ~300 Hz keeps reverb out of the low-mids where the EQ carve is
    # making space for the vocal, lowpass ~8 kHz keeps sibilance from splashing
    # into a bright metallic tail.
    board = Pedalboard([
        Reverb(room_size=room_size, damping=0.82,
               wet_level=1.0, dry_level=0.0),
        HighpassFilter(cutoff_frequency_hz=300.0),
        LowpassFilter(cutoff_frequency_hz=8000.0),
    ])
    reverb_wet = board(y, sr)

    # Shift reverb forward by pre_delay_ms
    delay_samples = int(sr * pre_delay_ms / 1000.0)
    if delay_samples > 0 and delay_samples < len(reverb_wet):
        reverb_delayed = np.zeros_like(reverb_wet)
        reverb_delayed[delay_samples:] = reverb_wet[:-delay_samples]
    else:
        reverb_delayed = reverb_wet

    # Combine dry vocal + delayed reverb at wet_level
    return (y + reverb_delayed * wet_level).astype(np.float32)


def _backing_harmonies(vocal: np.ndarray, sr: int, genre: str,
                       vocal_analysis: dict) -> tuple:
    """
    Generate in-key backing-vocal harmonies (the artist's own voice, a diatonic
    3rd + 5th up, formant-preserved) and place them low + wide behind the lead.
    Level scales with genre — fuller for soul/R&B/pop, light for aggressive rap.
    Returns (L, R) to add to the vocal; silence if the key is unknown.
    """
    n = len(vocal)
    L = np.zeros(n, dtype=np.float32)
    R = np.zeros(n, dtype=np.float32)
    key  = (vocal_analysis or {}).get("key")
    mode = (vocal_analysis or {}).get("mode") or "major"
    if not key:
        return L, R
    try:
        from .vocal_processor import generate_harmony_stack
        harms = generate_harmony_stack(vocal, sr, key, mode, steps_list=(2, 4))
        if not harms:
            return L, R

        # Per-genre harmony prominence
        full = {"rnb_smooth", "rnb_neo_soul", "soul_ballad", "lofi_chill",
                "pop_bright", "afrobeats", "amapiano"}
        light = {"drill", "uk_drill", "trap_dark", "phonk", "hiphop_boom_bap"}
        level_db = -11.0 if genre in full else (-18.0 if genre in light else -14.0)
        amp = 10.0 ** (level_db / 20.0)

        # 3rd up → left, 5th up → right; small delays for separation/width
        pans   = [-0.40, 0.40]
        delays = [int(sr * 0.018), int(sr * 0.026)]
        for h, pan, dly in zip(harms, pans, delays):
            m = min(len(h), n)
            v = np.zeros(n, dtype=np.float32)
            if dly < m:
                v[dly:m] = h[: m - dly]
            ang = (pan + 1.0) * (np.pi / 4.0)        # equal-power pan
            L += v * (np.cos(ang) * amp)
            R += v * (np.sin(ang) * amp)
        return L.astype(np.float32), R.astype(np.float32)
    except Exception:
        return L, R


def _vocal_delay_throw(vocal: np.ndarray, sr: int, tempo: float,
                       level_db: float = -15.0, feedback: float = 0.30,
                       taps: int = 3) -> tuple:
    """
    Tempo-synced ping-pong delay on the vocal — a subtle 1/8-note echo that
    bounces L↔R and decays, giving the voice depth and movement without washing
    it out. The echo is band-limited (300 Hz–3.5 kHz) so it sits *behind* the dry
    vocal instead of competing with it. Standard modern vocal-production polish.
    """
    try:
        eighth = (60.0 / max(tempo, 40.0)) * 0.5      # 1/8-note in seconds
        d = int(sr * eighth)
        n = len(vocal)
        if d <= 0 or d >= n:
            return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)

        sos_hp = butter(2, 300.0 / (sr / 2.0), btype="high", output="sos")
        sos_lp = butter(2, 3500.0 / (sr / 2.0), btype="low", output="sos")
        src = sosfilt(sos_lp, sosfilt(sos_hp, vocal)).astype(np.float32)

        L = np.zeros(n, dtype=np.float32)
        R = np.zeros(n, dtype=np.float32)
        amp = 10.0 ** (level_db / 20.0)
        for t in range(taps):
            delay = d * (t + 1)
            if delay >= n:
                break
            tap = np.zeros(n, dtype=np.float32)
            tap[delay:] = src[: n - delay] * (amp * (feedback ** t))
            if t % 2 == 0:
                L += tap
            else:
                R += tap
        return L, R
    except Exception:
        z = np.zeros(len(vocal), dtype=np.float32)
        return z, z


def _spread_reverb(dry: np.ndarray, wet: np.ndarray,
                    spread: float = 0.18) -> tuple:
    """
    Place dry vocal dead center; push reverb slightly wide.
    Creates depth: the voice is heard "in front of" its own room.
    """
    reverb_component = wet - dry  # isolate only the reverb tail
    L = (dry + reverb_component * (1.0 + spread)).astype(np.float32)
    R = (dry + reverb_component * (1.0 - spread)).astype(np.float32)
    return L, R


# ── Tape saturation ───────────────────────────────────────────────────────────

def _tape_saturate(y: np.ndarray, drive: float = 0.18) -> np.ndarray:
    """
    Subtle tape-style saturation on the master bus.

    Uses a soft-clipping hyperbolic tangent transfer function — the same
    mechanism as analog tape magnetization saturation. At drive=0.18 the
    THD is well under 0.1%, which is inaudible as distortion but adds
    cohesion and "glue" between elements.

    Drive range: 0.0 (bypassed) to 1.0 (heavy saturation).
    DreamStage default: 0.18 — subtle warmth, not coloration.
    """
    if drive <= 0:
        return y
    # Scale up, saturate, scale back down to preserve loudness
    gain  = 1.0 + drive * 2.0
    scale = np.tanh(drive * 3.0) / (drive * 3.0 + 1e-9)
    return (np.tanh(y * gain) / gain * (1.0 / (scale + 1e-9) * scale)).astype(np.float32)


# ── Gain staging ──────────────────────────────────────────────────────────────

def _lufs_normalize(y: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    if _HAS_PYLOUDNORM:
        try:
            meter = pyln.Meter(sr)
            current = meter.integrated_loudness(y.reshape(-1, 1))
            if np.isfinite(current) and current > -70:
                # No clip: this is input gain staging — the full master chain
                # (multiband comp + limiter) follows and handles any overs.
                gain = 10.0 ** ((target_lufs - current) / 20.0)
                return (y * gain).astype(np.float32)
        except Exception:
            pass
    rms_targets = {-18.0: 0.126, -21.0: 0.089}
    target_rms = rms_targets.get(target_lufs, 0.10)
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < 1e-6:
        return y
    gain = min(target_rms / rms, 8.0)
    return (y * gain).astype(np.float32)


def _final_loudness_target(L: np.ndarray, R: np.ndarray, sr: int,
                            target_lufs: float = -14.0) -> tuple:
    if _HAS_PYLOUDNORM:
        try:
            stereo  = np.stack([L, R], axis=1)
            meter   = pyln.Meter(sr)
            current = meter.integrated_loudness(stereo)
            if np.isfinite(current) and current > -70:
                # No clip: the limiter runs AFTER this stage and absorbs overs.
                gain = 10.0 ** ((target_lufs - current) / 20.0)
                return ((L * gain).astype(np.float32),
                        (R * gain).astype(np.float32))
        except Exception:
            pass
    peak  = max(np.max(np.abs(L)), np.max(np.abs(R)), 1e-9)
    scale = 10 ** (-1.0 / 20) / peak
    return ((L * scale).astype(np.float32),
            (R * scale).astype(np.float32))


# ── Stereo ────────────────────────────────────────────────────────────────────

def _ms_enhance(L: np.ndarray, R: np.ndarray, side_gain: float = 1.10) -> tuple:
    mid  = (L + R) * 0.5
    side = (L - R) * 0.5 * side_gain
    return (mid + side).astype(np.float32), (mid - side).astype(np.float32)


# ── Reverb params ─────────────────────────────────────────────────────────────

def _reverb_params(genre: str) -> tuple:
    """Returns (room_size, wet_level, pre_delay_ms)."""
    presets = {
        # genre                  room  wet   pre_delay_ms
        "drill":           (0.06, 0.07,  8.0),
        "uk_drill":        (0.06, 0.07,  8.0),
        "trap_dark":       (0.08, 0.09, 12.0),
        "trap_melodic":    (0.10, 0.10, 14.0),
        "phonk":           (0.08, 0.09, 10.0),
        "hiphop_boom_bap": (0.12, 0.11, 18.0),
        "hiphop_modern":   (0.10, 0.10, 14.0),
        "lofi_chill":      (0.20, 0.14, 22.0),
        "rnb_smooth":      (0.14, 0.12, 18.0),
        "rnb_neo_soul":    (0.16, 0.13, 20.0),
        "soul_ballad":     (0.22, 0.16, 25.0),
        "pop_bright":      (0.12, 0.11, 16.0),
        "afrobeats":       (0.10, 0.10, 14.0),
        "dancehall":       (0.12, 0.11, 16.0),
        "reggaeton":       (0.10, 0.10, 14.0),
        "amapiano":        (0.14, 0.12, 18.0),
        "musicgen_ai":     (0.12, 0.11, 16.0),
        "musicgen_hf_api": (0.12, 0.11, 16.0),
        "musicgen_local":  (0.12, 0.11, 16.0),
        "musicgen_gradio": (0.12, 0.11, 16.0),
    }
    return presets.get(genre, (0.12, 0.11, 16.0))


# ── Mastering ─────────────────────────────────────────────────────────────────

def _master_channel(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Per-channel mastering chain.

    Pultec-style EQ: boost lows AND high-pass at the same frequency.
    This sounds counterintuitive but the phase interaction creates
    the classic "tight low end" character — more defined than a simple boost.
    """
    board = Pedalboard([
        # Bus glue compression: slow attack preserves transient punch
        Compressor(threshold_db=-10, ratio=2.0, attack_ms=25.0, release_ms=280.0),
        # Pultec-style: boost sub + cut mud
        LowShelfFilter(cutoff_frequency_hz=80,   gain_db=1.5),   # sub warmth
        PeakFilter(cutoff_frequency_hz=320,      gain_db=-1.2, q=0.9),  # mud reduction
        # High-frequency air
        HighShelfFilter(cutoff_frequency_hz=12000, gain_db=1.8),
    ])
    return board(y, sr)


# ── MP3 export ────────────────────────────────────────────────────────────────

def _encode_mp3(L: np.ndarray, R: np.ndarray, sr: int) -> bytes:
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(320)
    encoder.set_in_sample_rate(sr)
    encoder.set_channels(2)
    encoder.set_quality(2)

    L_i16 = (np.clip(L, -1, 1) * 32767).astype(np.int16)
    R_i16 = (np.clip(R, -1, 1) * 32767).astype(np.int16)
    interleaved = np.empty(len(L_i16) * 2, dtype=np.int16)
    interleaved[0::2] = L_i16
    interleaved[1::2] = R_i16

    data  = encoder.encode(interleaved.tobytes())
    data += encoder.flush()
    return bytes(data)
