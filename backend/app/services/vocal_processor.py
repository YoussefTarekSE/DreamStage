"""
Professional vocal processing pipeline.

Signal-flow order:
  1. Highpass filter      — remove sub-rumble
  2. Noise reduction      — VAD-guided, conservative
  3. Adaptive noise gate  — threshold from signal's own noise floor
  4. Compression          — dynamics control, preserve expression
  5. Selective pitch correction — WORLD vocoder, quality-first
  6. EQ shaping           — presence, warmth, air
  7. Harmonic exciter     — subtle 2nd/3rd harmonic
  8. De-esser             — frequency-selective dynamic
  9. LUFS normalization
 10. True-peak limiter

Pitch correction philosophy (Quality First):
  A slightly out-of-tune vocal is always preferable to a robotic one.
  The system only corrects notes that clearly need it, preserves vibrato
  completely, skips frames where correction would introduce artifacts,
  and runs a bypass test to confirm correction actually improves quality.
"""
import io
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from dataclasses import dataclass, field
from scipy.signal import butter, sosfilt
from scipy.ndimage import gaussian_filter1d
from pedalboard import (
    Pedalboard, NoiseGate, Compressor,
    HighShelfFilter, LowShelfFilter, Limiter,
    Gain, HighpassFilter, PeakFilter,
)
from .audio_loader import load_audio

PROCESS_SR = 44100

# ── Pitch correction configuration ───────────────────────────────────────────
#
# TUNE_THRESHOLD_CENTS: minimum pitch error before any correction is applied.
# Notes closer than this threshold to the nearest semitone are left untouched.
#   natural → 20 cents (clearly audible errors only)
#   subtle  → 15 cents
#   modern  →  8 cents
#   heavy   →  5 cents
#
# MAX_CORRECTION_CENTS: never apply more than this much correction in a single
# frame regardless of strength. Large corrections always introduce artifacts.
#
# CORRECTION_STRENGTH: how far to push eligible frames toward the target.
# Applied ONLY after threshold and artifact-risk checks pass.

TUNE_THRESHOLD_CENTS = {
    "natural": 20.0,
    "subtle":  15.0,
    "modern":   8.0,
    "heavy":    5.0,
}

MAX_CORRECTION_CENTS = {
    "natural":  40.0,
    "subtle":   80.0,
    "modern":  150.0,
    "heavy":   200.0,
}

CORRECTION_STRENGTH = {
    "natural": 0.35,
    "subtle":  0.55,
    "modern":  0.80,
    "heavy":   1.00,
}

# Artifact risk: max broadband aperiodicity (0=harmonic, 1=noise) before we
# skip correction. Breathy/noisy frames don't survive pitch shifting cleanly.
MAX_APERIODICITY = {
    "natural": 0.35,
    "subtle":  0.45,
    "modern":  0.60,
    "heavy":   0.75,
}

# Apply criteria per level: (min in-tuneness gain in points, min timbre fidelity).
#   - in-tuneness gain: how many points (≈ 2 × cents) closer to the chromatic
#     grid the corrected take is. A bigger bar means "only fix clearly off pitch."
#   - min timbre fidelity: log-spectrum correlation between corrected and raw;
#     a floor here vetoes correction if WORLD resynthesis would alter the voice
#     (protects identity on difficult signals).
# natural/subtle protect the performance; modern/heavy are an explicit effect
# choice, so they apply far more readily (lower gain bar).
_APPLY_CRITERIA = {
    "natural": (6.0, 0.96),
    "subtle":  (4.0, 0.95),
    "modern":  (2.0, 0.93),
    "heavy":   (0.5, 0.90),
}

# ── Vocal styles (the user-facing selector) ───────────────────────────────────
# Each named style resolves to a pitch-correction profile, a tonal EQ flavour, a
# compression character, a harmonic-exciter amount, a stereo-width amount, and a
# beat genre bias. This is what makes the style selection audibly change the
# processed vocal AND steer the beat (e.g. a Rap take never gets a piano ballad).
#   pitch:      None = no pitch correction (cleanup only); else a level key above
#   eq/comp:    flavour keys consumed by _vocal_eq / _comp_settings
#   genre_bias: pool name read by the beat generator (None = let analysis decide)
VOCAL_STYLES = {
    "none":       dict(pitch=None,      eq="natural", comp="natural", excite=0.04, width=0.012, genre_bias=None,       label="No Autotune"),
    "natural":    dict(pitch="natural", eq="natural", comp="natural", excite=0.05, width=0.018, genre_bias=None,       label="Natural"),
    "subtle":     dict(pitch="subtle",  eq="subtle",  comp="subtle",  excite=0.09, width=0.025, genre_bias=None,       label="Subtle"),
    "modern_pop": dict(pitch="modern",  eq="modern",  comp="modern",  excite=0.14, width=0.032, genre_bias="pop",      label="Modern Pop"),
    "rnb":        dict(pitch="subtle",  eq="rnb",     comp="subtle",  excite=0.10, width=0.034, genre_bias="rnb_soul", label="R&B"),
    "rap":        dict(pitch="natural", eq="rap",     comp="modern",  excite=0.08, width=0.010, genre_bias="rap",      label="Rap"),
    "melodic":    dict(pitch="modern",  eq="modern",  comp="subtle",  excite=0.12, width=0.036, genre_bias="melodic",  label="Melodic"),
    "heavy":      dict(pitch="heavy",   eq="heavy",   comp="heavy",   excite=0.18, width=0.030, genre_bias=None,       label="Heavy"),
    # legacy alias (older projects stored "modern")
    "modern":     dict(pitch="modern",  eq="modern",  comp="modern",  excite=0.14, width=0.030, genre_bias="pop",      label="Modern"),
}


def resolve_style(style: str) -> dict:
    """Return the processing profile for a style name, defaulting to 'subtle'."""
    return VOCAL_STYLES.get((style or "subtle").lower(), VOCAL_STYLES["subtle"])


def style_genre_bias(style: str) -> str | None:
    """Genre pool a vocal style steers the beat toward (None = analysis decides)."""
    return resolve_style(style).get("genre_bias")

try:
    import pyworld as pw
    _HAS_PYWORLD = True
except ImportError:
    _HAS_PYWORLD = False

try:
    import pyloudnorm as pyln
    _HAS_PYLOUDNORM = True
except ImportError:
    _HAS_PYLOUDNORM = False


# ── Correction statistics (returned for logging / diagnostics) ────────────────

@dataclass
class CorrectionStats:
    total_voiced_frames:   int   = 0
    corrected_frames:      int   = 0
    skipped_in_tune:       int   = 0
    skipped_vibrato:       int   = 0
    skipped_artifact_risk: int   = 0
    avg_correction_cents:  float = 0.0
    max_correction_cents:  float = 0.0
    bypass_used:           bool  = False
    bypass_score_raw:      float = 0.0
    bypass_score_corrected: float = 0.0
    # In-tuneness (0-100, higher = closer to the chromatic grid) measured on the
    # F0 contour the corrector actually operated on. Drives the bypass test.
    intune_raw:            float = 0.0
    intune_corrected:      float = 0.0

    @property
    def pct_left_untouched(self) -> float:
        if self.total_voiced_frames == 0:
            return 100.0
        return 100.0 * (self.total_voiced_frames - self.corrected_frames) / self.total_voiced_frames


# ── Main entry point ──────────────────────────────────────────────────────────

def process_vocal(audio_bytes: bytes,
                  autotune_level: str = "subtle") -> bytes:
    """
    Process a vocal recording according to the chosen vocal STYLE.

    `autotune_level` carries the style name (one of VOCAL_STYLES: natural, subtle,
    modern_pop, rnb, rap, melodic, heavy, none — legacy 'modern' still accepted).
    The style drives compression character, pitch-correction strength, tonal EQ,
    harmonic excitement and stereo width, so the selection audibly changes the
    result. Cleanup (noise reduction, gating, de-essing, loudness) always runs —
    even 'No Autotune' returns a clean, present vocal; it just skips pitch tuning.
    """
    sty = resolve_style(autotune_level)

    y, sr = load_audio(audio_bytes, target_sr=PROCESS_SR)
    y = y.astype(np.float32)

    # 1 — Highpass
    y = Pedalboard([HighpassFilter(cutoff_frequency_hz=80)])(y, sr)

    # 2 — Noise reduction
    y = _denoise(y, sr)

    # 3 — Adaptive gate
    noise_floor_db = _estimate_noise_floor_db(y, sr)
    gate_threshold = float(np.clip(noise_floor_db + 6.0, -54.0, -30.0))
    y = Pedalboard([
        NoiseGate(threshold_db=gate_threshold, ratio=2.0,
                  attack_ms=5.0, release_ms=200.0),
    ])(y, sr)

    # 4 — Compression (character per style)
    y = Pedalboard([Compressor(**_comp_settings(sty["comp"]))])(y, sr)

    # 5 — Selective pitch correction (skipped entirely for 'No Autotune')
    if sty["pitch"] is not None:
        y = _pitch_correct_selective(y, sr, sty["pitch"])

    # 6 — Tonal EQ (flavour per style)
    y = _vocal_eq(y, sr, sty["eq"])

    # 7 — Harmonic exciter (presence/air per style)
    y = _harmonic_excite(y, sr, amount=sty["excite"])

    # 7.5 — Micro-chorus width (tasteful stereo thickness per style)
    y = _micro_chorus(y, sr, amount=sty["width"])

    # 8 — De-essing
    y = _deess(y, sr, threshold_db=_estimate_sibilance_threshold(y, sr))

    # 9 — LUFS normalize
    y = _normalize(y, sr, target_lufs=-16.0)

    # 10 — True-peak limiter
    y = Pedalboard([Limiter(threshold_db=-1.0, release_ms=60.0)])(y, sr)

    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_24")
    buf.seek(0)
    return buf.read()


# ── Selective pitch correction — the complete new system ─────────────────────

def _pitch_correct_selective(y: np.ndarray, sr: int,
                              level: str = "natural") -> np.ndarray:
    """
    Quality-first pitch correction dispatcher.

    Runs the WORLD-based selective corrector, then compares corrected vs.
    raw vocal on naturalness and identity metrics. Returns whichever scores
    higher.
    """
    if level == "natural" and CORRECTION_STRENGTH.get(level, 0) == 0:
        return y

    if _HAS_PYWORLD:
        corrected, stats = _world_selective(y, sr, level)
    else:
        corrected, stats = _legacy_selective(y, sr, level)

    # No frames were corrected → nothing to gain, and resynthesis only risks
    # artifacts. Keep the raw take untouched.
    if stats.corrected_frames == 0:
        return y

    # ── Bypass test ───────────────────────────────────────────────────────────
    # The old implementation compared the raw vocal against ITSELF, which always
    # scored a perfect 100 — so correction was ALWAYS discarded and the autotune
    # selector did nothing. This version weighs the real trade-off:
    #
    #   benefit = in-tuneness gain  (how much closer to the chromatic grid the
    #             corrected take sits — measured on the F0 the corrector touched)
    #   guard   = timbre fidelity   (log-spectrum correlation of corrected vs raw;
    #             WORLD changes only F0 so formants are preserved by construction,
    #             but this still vetoes resynthesis on signals it can't render
    #             faithfully, protecting the artist's identity)
    #
    # Apply when both clear the level's bar; otherwise keep the natural take.
    gain = stats.intune_corrected - stats.intune_raw
    sim  = _timbre_similarity(corrected, y, sr)
    min_gain, min_sim = _APPLY_CRITERIA.get(level, (4.0, 0.95))

    stats.bypass_score_raw       = sim     # store for diagnostics/telemetry
    stats.bypass_score_corrected = gain

    if sim >= min_sim and gain >= min_gain:
        return corrected
    stats.bypass_used = True
    return y


def _world_selective(y: np.ndarray, sr: int,
                     level: str) -> tuple[np.ndarray, CorrectionStats]:
    """
    Per-frame selective pitch correction using the WORLD vocoder.

    WORLD decomposes audio into three components:
      F0  — fundamental frequency contour (what we modify)
      sp  — spectral envelope / formants (we leave this unchanged — voice identity)
      ap  — aperiodicity (used for artifact risk scoring)

    For each voiced frame:
      1. Measure pitch error (cents from nearest semitone)
      2. If error < TUNE_THRESHOLD → skip (already in tune)
      3. If frame is in a vibrato region → skip (preserve natural modulation)
      4. If aperiodicity > MAX_APERIODICITY → skip (correction would add noise)
      5. If correction > MAX_CORRECTION_CENTS → clamp (limit artifact size)
      6. Otherwise: correct toward nearest semitone by CORRECTION_STRENGTH
    """
    stats = CorrectionStats()
    try:
        y_d = y.astype(np.float64)
        f0, sp, ap = pw.wav2world(y_d, sr)

        threshold_cents = TUNE_THRESHOLD_CENTS.get(level, 20.0)
        max_corr_cents  = MAX_CORRECTION_CENTS.get(level, 80.0)
        strength        = CORRECTION_STRENGTH.get(level, 0.55)
        max_ap          = MAX_APERIODICITY.get(level, 0.45)

        # WORLD frame rate: default hop is 5ms
        frame_rate = sr / 80.0   # WORLD uses 5ms = 80 samples at 16kHz equiv
        # Actually at process_sr=44100, wav2world uses ~5ms frames = 220 samples
        frame_rate_actual = 1000.0 / 5.0  # 200 frames/sec

        vibrato_mask = _detect_vibrato_mask(f0, frame_rate_actual)

        # ── Scale-aware tuning ───────────────────────────────────────────────
        # Detect the song's key from the notes actually sung, then snap toward
        # the nearest note IN THAT KEY (not just the nearest chromatic semitone).
        # This is how key-locked autotune works — the vocal lands on musically
        # correct notes and sits in the same key the beat is built in. If the key
        # is unclear (too little pitched content), fall back to chromatic snapping.
        _key = _mode = None
        try:
            from .vocal_harmony import detect_key_from_histogram, snap_midi_to_scale
            voiced_f0 = f0[f0 > 0]
            if len(voiced_f0) >= 20:
                vmidi = 69.0 + 12.0 * np.log2(voiced_f0 / 440.0)
                pc_hist = np.bincount(np.round(vmidi).astype(int) % 12, minlength=12).astype(float)
                _key, _mode = detect_key_from_histogram(pc_hist)
        except Exception:
            _key = _mode = None

        # Artifact-risk is judged on LOW-band aperiodicity (≤ ~4 kHz), where the
        # fundamental and lower harmonics that pitch-shifting actually moves
        # live. The broadband mean is dominated by naturally noise-like high
        # frequencies (breath, sibilance) and would wrongly veto clean,
        # correctable notes — leaving the gentler autotune modes doing nothing.
        n_low_ap = (max(1, int(ap.shape[1] * 4000.0 / (sr / 2.0)))
                    if ap.ndim > 1 else 1)

        f0_corrected = f0.copy()
        corr_cents   = np.zeros(len(f0), dtype=np.float64)
        corrections  = []

        for i, f0_val in enumerate(f0):
            if f0_val <= 0:
                continue   # unvoiced
            stats.total_voiced_frames += 1

            # Convert to MIDI for cent calculation. Target = nearest note in the
            # detected key's scale (key-locked); chromatic semitone if no key.
            midi   = 69.0 + 12.0 * np.log2(f0_val / 440.0)
            if _key is not None:
                nearest = snap_midi_to_scale(midi, _key, _mode)
            else:
                nearest = round(midi)
            error_cents = (midi - nearest) * 100.0   # positive = sharp, negative = flat

            # Rule 1: skip if already in tune
            if abs(error_cents) < threshold_cents:
                stats.skipped_in_tune += 1
                continue

            # Rule 2: skip vibrato regions
            if i < len(vibrato_mask) and vibrato_mask[i]:
                stats.skipped_vibrato += 1
                continue

            # Rule 3: skip high-artifact-risk frames (breathy, noisy in the
            # pitched low band — where shifting would smear into noise)
            ap_low = float(np.mean(ap[i, :n_low_ap])) if ap.ndim > 1 else float(ap[i])
            if ap_low > max_ap:
                stats.skipped_artifact_risk += 1
                continue

            # Rule 4: limit correction amount
            correction_cents = -error_cents * strength
            if abs(correction_cents) > max_corr_cents:
                correction_cents = np.sign(correction_cents) * max_corr_cents

            corr_cents[i] = correction_cents
            stats.corrected_frames  += 1
            corrections.append(abs(correction_cents))

        # ── Retune GLIDE, not per-frame snap ────────────────────────────────
        # Independent per-frame snapping turns natural slides/scoops into
        # stair-step warble (the "robot" sound). Instead, smooth the CORRECTION
        # curve within each voiced phrase so corrections glide between notes
        # like a singer. The glide time IS the style's retune speed: gentle
        # styles drift slowly and stay human; "heavy" snaps fast on purpose
        # (that stepped sound is the hard-autotune effect people choose it for).
        glide_ms = {"natural": 90.0, "subtle": 65.0,
                    "modern": 30.0, "heavy": 12.0}.get(level, 45.0)
        corr_cents = _glide_corrections(corr_cents, f0, frame_rate_actual,
                                        glide_ms=glide_ms)
        voiced = f0 > 0
        f0_corrected[voiced] = f0[voiced] * (2.0 ** (corr_cents[voiced] / 1200.0))

        if corrections:
            stats.avg_correction_cents = float(np.mean(corrections))
            stats.max_correction_cents = float(np.max(corrections))

        # In-tuneness before vs after, measured ONLY over the frames eligible for
        # correction (voiced, non-vibrato). Including vibrato/in-tune frames —
        # which never change — would dilute the gain and hide a real correction.
        elig = (f0 > 0)
        if len(vibrato_mask) == len(f0):
            elig = elig & (~vibrato_mask)
        if np.sum(elig) >= 5:
            stats.intune_raw       = _intuneness_from_hz(f0[elig], _key, _mode)
            stats.intune_corrected = _intuneness_from_hz(f0_corrected[elig], _key, _mode)
        else:
            stats.intune_raw       = _intuneness_from_hz(f0[f0 > 0], _key, _mode)
            stats.intune_corrected = _intuneness_from_hz(f0_corrected[f0_corrected > 0], _key, _mode)

        # ── Splice-back: the artist's ORIGINAL audio everywhere possible ────
        # Resynthesizing the whole take through WORLD gives every second a
        # subtle vocoder texture ("alien voice") even where nothing changed.
        # Instead, resynthesize once, then splice ONLY the corrected regions
        # back into the untouched original with short crossfades. Frames that
        # needed no correction stay bit-exact original audio.
        changed = np.abs(corr_cents) > 3.0
        # Ignore corrections shorter than ~30ms: isolated blips at note onsets
        # are pitch-tracker noise, not singing — correcting them buys nothing
        # and costs a splice. (Real autotune excludes onset transients too.)
        changed = _drop_short_runs(changed, min_frames=6)
        corr_cents[~changed] = 0.0
        if not np.any(changed):
            stats.corrected_frames = 0
            return y.astype(np.float32), stats
        f0_corrected = f0.copy()
        voiced = f0 > 0
        f0_corrected[voiced] = f0[voiced] * (2.0 ** (corr_cents[voiced] / 1200.0))

        corrected = pw.synthesize(f0_corrected, sp, ap, sr).astype(np.float32)
        out = _splice_corrected_regions(y.astype(np.float32), corrected,
                                        changed, sr)
        return out, stats

    except Exception:
        return y, CorrectionStats()


def _drop_short_runs(mask: np.ndarray, min_frames: int) -> np.ndarray:
    """Zero out True-runs shorter than min_frames in a boolean mask."""
    out = mask.copy()
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return out
    start = prev = idx[0]
    runs = []
    for k in idx[1:]:
        if k - prev > 1:
            runs.append((start, prev))
            start = k
        prev = k
    runs.append((start, prev))
    for (a, b) in runs:
        if b - a + 1 < min_frames:
            out[a:b + 1] = False
    return out


def _glide_corrections(corr_cents: np.ndarray, f0: np.ndarray,
                       frame_rate: float, glide_ms: float = 40.0) -> np.ndarray:
    """Smooth the correction curve within each voiced phrase so retuning
    GLIDES between notes (like a singer, or autotune's retune-speed knob)
    instead of snapping every 5ms frame independently (stair-step warble).
    Smoothing never crosses unvoiced gaps — phrases stay independent."""
    sigma = max(1.0, (glide_ms / 1000.0) * frame_rate / 2.0)
    out = corr_cents.copy()
    voiced = f0 > 0
    i, n = 0, len(f0)
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        while j < n and voiced[j]:
            j += 1
        if j - i >= 3:
            out[i:j] = gaussian_filter1d(corr_cents[i:j], sigma=sigma,
                                         mode="nearest")
        i = j
    return out


def _splice_corrected_regions(y: np.ndarray, corrected: np.ndarray,
                              changed_frames: np.ndarray, sr: int,
                              hop_s: float = 0.005) -> np.ndarray:
    """Replace only the corrected regions of `y` with the resynthesized audio,
    using ~10ms raised-cosine crossfades at each boundary. WORLD synthesis is
    time-aligned with its input (5ms hop), so frame i maps to sample i*hop."""
    hop = max(1, int(sr * hop_s))
    n = min(len(y), len(corrected))
    out = y[:n].copy()

    # Group changed frames into runs. Merge runs separated by <150ms: every
    # splice boundary is a potential audible seam, so one clean region beats
    # thirty micro-splices (a slide correction is a sawtooth of tiny runs).
    # Pad by 3 frames so a correction never starts mid-phoneme.
    merge_gap = max(4, int(0.150 / hop_s))
    idx = np.flatnonzero(changed_frames)
    if len(idx) == 0:
        return out
    runs, start, prev = [], idx[0], idx[0]
    for k in idx[1:]:
        if k - prev > merge_gap:
            runs.append((start, prev))
            start = k
        prev = k
    runs.append((start, prev))

    xf = max(8, int(0.010 * sr))
    fade_in  = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, xf))
    fade_out = fade_in[::-1]

    for (fa, fb) in runs:
        a = max(0, (fa - 3) * hop)
        b = min(n, (fb + 4) * hop)
        if b - a < 2 * xf:
            continue
        out[a:b] = corrected[a:b]
        # crossfade the edges back into the original
        out[a:a + xf] = y[a:a + xf] * fade_out + corrected[a:a + xf] * fade_in
        out[b - xf:b] = corrected[b - xf:b] * fade_out + y[b - xf:b] * fade_in

    return out


def _detect_vibrato_mask(f0: np.ndarray, frame_rate: float) -> np.ndarray:
    """
    Detect vibrato regions in the F0 contour.

    Vibrato is a periodic modulation of pitch at 4–7 Hz with amplitude
    > 25 cents. We detect it by:
      1. Computing the deviation of F0 from a long-time smooth (200ms)
      2. Measuring the sign-change rate of this deviation (= oscillation rate)
      3. Measuring the local peak-to-peak range in cents
      4. Frames where rate ∈ [4–7 Hz] AND range > 25 cents are marked as vibrato

    The mask is dilated by ±2 frames to avoid correcting the transition
    into and out of vibrato.
    """
    n = len(f0)
    mask = np.zeros(n, dtype=bool)
    voiced = f0 > 80

    if np.sum(voiced) < 10:
        return mask

    # Convert to semitones for scale-invariant measurement
    f0_semitones = np.zeros(n, dtype=np.float64)
    f0_semitones[voiced] = 12.0 * np.log2(np.maximum(f0[voiced], 1.0) / 440.0)

    # Long-time smooth: 200ms gaussian to get the "base" pitch
    sigma_frames = max(1.0, frame_rate * 0.20)
    f0_smooth = gaussian_filter1d(f0_semitones, sigma=sigma_frames)

    # Deviation from smooth = modulation signal
    deviation = np.zeros(n, dtype=np.float64)
    deviation[voiced] = f0_semitones[voiced] - f0_smooth[voiced]

    # Window: 250ms
    win = max(6, int(frame_rate * 0.25))

    for i in range(n):
        if not voiced[i]:
            continue
        s = max(0, i - win)
        e = min(n, i + win)

        local_voiced  = voiced[s:e]
        if np.sum(local_voiced) < win // 2:
            continue

        local_dev = deviation[s:e][local_voiced]

        # Range in cents
        range_semitones = float(
            np.max(f0_semitones[s:e][local_voiced]) -
            np.min(f0_semitones[s:e][local_voiced])
        )
        range_cents = range_semitones * 100.0

        if range_cents < 25.0:
            continue   # too small to be vibrato

        # Oscillation rate: count sign changes in the deviation signal
        sign_changes = int(np.sum(np.diff(np.sign(local_dev)) != 0))
        duration_sec = float(np.sum(local_voiced)) / frame_rate
        if duration_sec < 1e-3:
            continue
        rate_hz = sign_changes / (2.0 * duration_sec)  # sign changes → half-cycles

        if 4.0 <= rate_hz <= 7.5:
            mask[i] = True

    # Dilate mask by ±2 frames so we don't correct the vibrato onset/offset
    dilated = mask.copy()
    for i in range(n):
        if mask[i]:
            dilated[max(0, i-2):min(n, i+3)] = True

    return dilated


def _smooth_f0_at_correction_boundaries(
    f0_orig: np.ndarray,
    f0_corr: np.ndarray,
    voiced: np.ndarray,
    sigma: float = 1.5,
) -> np.ndarray:
    """
    Apply Gaussian smoothing ONLY to frames where correction was applied,
    blending into adjacent uncorrected frames to prevent glitches.

    Frames that were NOT corrected keep their exact original F0 value.
    """
    result = f0_corr.copy()
    changed = voiced & (np.abs(f0_orig - f0_corr) > 1e-6)

    if not np.any(changed):
        return result

    # Smooth the corrected signal
    smoothed = gaussian_filter1d(f0_corr, sigma=sigma)

    # Only apply the smoothed value where correction actually occurred
    result[changed] = smoothed[changed]
    return result


def _legacy_selective(y: np.ndarray, sr: int,
                      level: str) -> tuple[np.ndarray, CorrectionStats]:
    """
    Selective correction without pyworld (pYIN + PitchShift fallback).

    Instead of the original approach (shift entire clip by median offset),
    this version:
      1. Measures per-frame pitch errors
      2. Only corrects if median error exceeds threshold
      3. Uses a single global shift ONLY if the majority of frames need it
         (> 40% of frames outside the threshold)
    """
    stats = CorrectionStats()
    try:
        from pedalboard import PitchShift

        f0, voiced_flag, prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sr,
            frame_length=2048,
        )

        threshold_cents = TUNE_THRESHOLD_CENTS.get(level, 20.0)
        strength        = CORRECTION_STRENGTH.get(level, 0.55)

        voiced_f0 = f0[voiced_flag & ~np.isnan(f0) & (prob > 0.5)]
        if len(voiced_f0) < 10:
            return y, stats

        midi_vals  = librosa.hz_to_midi(voiced_f0)
        errors     = (midi_vals - np.round(midi_vals)) * 100.0  # cents

        stats.total_voiced_frames = len(voiced_f0)
        stats.intune_raw = float(100.0 - 2.0 * np.mean(np.clip(np.abs(errors), 0.0, 50.0)))
        stats.intune_corrected = stats.intune_raw   # default: unchanged

        # Only correct if enough frames are genuinely out of tune
        needs_correction = np.abs(errors) >= threshold_cents
        stats.skipped_in_tune = int(np.sum(~needs_correction))
        pct_needing_correction = float(np.mean(needs_correction))

        if pct_needing_correction < 0.40:
            # Less than 40% of frames need correction — leave it alone
            return y, stats

        # Apply global correction only to the eligible frames' median
        eligible_errors = errors[needs_correction]
        correction_semitones = -float(np.median(eligible_errors)) * strength / 100.0
        max_corr = MAX_CORRECTION_CENTS.get(level, 80.0) / 100.0
        correction_semitones = float(np.clip(correction_semitones, -max_corr, max_corr))

        if abs(correction_semitones) < 0.01:
            return y, stats

        corrected = Pedalboard([PitchShift(semitones=correction_semitones)])(y, sr)
        stats.corrected_frames      = int(np.sum(needs_correction))
        stats.avg_correction_cents  = float(np.mean(np.abs(eligible_errors)) * strength)
        stats.max_correction_cents  = float(np.max(np.abs(eligible_errors)) * strength)
        # Resulting in-tuneness after the global shift (errors shift uniformly,
        # then wrap to the nearest semitone)
        shift_cents = correction_semitones * 100.0
        corr_err    = ((errors + shift_cents + 50.0) % 100.0) - 50.0
        stats.intune_corrected = float(100.0 - 2.0 * np.mean(np.clip(np.abs(corr_err), 0.0, 50.0)))
        return corrected, stats

    except Exception:
        return y, CorrectionStats()


# ── Quality scoring for bypass test ───────────────────────────────────────────

def generate_harmony_stack(y: np.ndarray, sr: int, key: str, mode: str,
                           steps_list=(2, 4)) -> list:
    """
    Build in-key backing-vocal harmonies from a lead vocal — the artist singing
    in harmony with themselves (real R&B/pop/soul production).

    For each interval in `steps_list` (2 = diatonic 3rd, 4 = 5th), every voiced
    note is shifted up by that many SCALE degrees. The shift is applied to the
    original F0 (so the singer's vibrato/expression is preserved) and resynthesised
    with the UNCHANGED spectral envelope (so it keeps the artist's exact timbre).
    One WORLD analysis feeds all harmonies. Returns a list of mono float32 arrays
    (empty if pyworld is unavailable or the key is unknown).
    """
    if not _HAS_PYWORLD or not key:
        return []
    try:
        from .vocal_harmony import snap_midi_to_scale, scale_step_up
        y_d = y.astype(np.float64)
        f0, sp, ap = pw.wav2world(y_d, sr)

        voiced = f0 > 0
        if int(np.sum(voiced)) < 10:
            return []

        # Per-voiced-frame semitone shift to the target diatonic interval
        midi = np.zeros_like(f0)
        midi[voiced] = 69.0 + 12.0 * np.log2(f0[voiced] / 440.0)

        out = []
        for steps in steps_list:
            shift = np.zeros_like(f0)
            for i in np.nonzero(voiced)[0]:
                snapped = snap_midi_to_scale(midi[i], key, mode)
                target  = scale_step_up(snapped, key, mode, steps)
                shift[i] = target - snapped
            f0_h = f0.copy()
            f0_h[voiced] = f0[voiced] * (2.0 ** (shift[voiced] / 12.0))
            harm = pw.synthesize(f0_h, sp, ap, sr).astype(np.float32)
            out.append(harm)
        return out
    except Exception:
        return []


def _intuneness_from_hz(f0_hz: np.ndarray, key: str = None, mode: str = None) -> float:
    """
    In-tuneness of an F0 contour, 0–100. 100 = every voiced frame sits exactly
    on target; 0 = every frame is a quarter-tone (50 cents) off.

    Target is the nearest note IN THE KEY's scale when (key, mode) are given —
    so snapping an in-tune-but-out-of-scale note to the scale registers as a
    real improvement — otherwise the nearest chromatic semitone.
    """
    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    f0_hz = f0_hz[f0_hz > 0]
    if len(f0_hz) < 5:
        return 50.0
    midi = 69.0 + 12.0 * np.log2(f0_hz / 440.0)
    if key is not None:
        from .vocal_harmony import snap_midi_to_scale
        targets = np.array([snap_midi_to_scale(m, key, mode) for m in midi], dtype=np.float64)
    else:
        targets = np.round(midi)
    cents = np.clip(np.abs((midi - targets) * 100.0), 0.0, 50.0)
    return float(100.0 - 2.0 * np.mean(cents))


def _timbre_similarity(y_proc: np.ndarray, y_ref: np.ndarray, sr: int) -> float:
    """
    Timbre-fidelity guard: correlation between the time-averaged log magnitude
    spectra of the processed and raw vocal (0–1, 1 = identical timbre).

    This is alignment-invariant (it averages over time, so the small frame
    delay WORLD synthesis introduces does not matter) and directly interpretable
    — a faithful WORLD round-trip measures ~0.98, a genuinely altered timbre
    (e.g. shifted formants) drops below ~0.92. It is far more reliable than a
    frame-aligned mel-cepstral distance, which mis-aligns and reports a large
    false change.
    """
    try:
        n = min(len(y_proc), len(y_ref))
        Sa = np.log(np.abs(librosa.stft(y_ref[:n], n_fft=2048)).mean(axis=1) + 1e-6)
        Sb = np.log(np.abs(librosa.stft(y_proc[:n], n_fft=2048)).mean(axis=1) + 1e-6)
        c = float(np.corrcoef(Sa, Sb)[0, 1])
        return c if np.isfinite(c) else 0.0
    except Exception:
        return 0.0


# ── Noise reduction ───────────────────────────────────────────────────────────

def _denoise(y: np.ndarray, sr: int) -> np.ndarray:
    try:
        frame_len = 2048
        hop = 512
        rms_frames = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
        noise_floor_threshold = np.percentile(rms_frames, 15)
        quiet_indices = np.where(rms_frames < noise_floor_threshold)[0]

        if len(quiet_indices) >= 8:
            chunks = [y[idx * hop: min(idx * hop + frame_len, len(y))]
                      for idx in quiet_indices[:30]]
            noise_sample = np.concatenate(chunks)
        else:
            noise_sample = y[:int(sr * 0.15)]

        return nr.reduce_noise(
            y=y, y_noise=noise_sample, sr=sr,
            stationary=False, prop_decrease=0.70,
            time_constant_s=2.0, freq_mask_smooth_hz=200,
        ).astype(np.float32)
    except Exception:
        return nr.reduce_noise(y=y, sr=sr, prop_decrease=0.60).astype(np.float32)


# ── Compression ───────────────────────────────────────────────────────────────

def _comp_settings(level: str) -> dict:
    return {
        "natural": dict(threshold_db=-22, ratio=2.5, attack_ms=15.0, release_ms=180.0),
        "subtle":  dict(threshold_db=-20, ratio=3.5, attack_ms=10.0, release_ms=120.0),
        "modern":  dict(threshold_db=-18, ratio=5.0, attack_ms=7.0,  release_ms=80.0),
        "heavy":   dict(threshold_db=-16, ratio=6.0, attack_ms=5.0,  release_ms=60.0),
    }.get(level, dict(threshold_db=-20, ratio=3.5, attack_ms=10.0, release_ms=120.0))


# ── EQ ────────────────────────────────────────────────────────────────────────

def _vocal_eq(y: np.ndarray, sr: int, autotune_level: str) -> np.ndarray:
    if autotune_level == "natural":
        board = Pedalboard([
            LowShelfFilter(cutoff_frequency_hz=120,  gain_db=1.0),
            PeakFilter(cutoff_frequency_hz=250,      gain_db=-1.2, q=1.2),  # mud
            PeakFilter(cutoff_frequency_hz=450,      gain_db=-1.0, q=1.0),  # boxiness
            PeakFilter(cutoff_frequency_hz=1500,     gain_db=1.0,  q=1.0),  # vowel clarity
            PeakFilter(cutoff_frequency_hz=3000,     gain_db=1.8,  q=0.8),  # presence
            PeakFilter(cutoff_frequency_hz=5500,     gain_db=1.2,  q=1.2),  # consonants
            HighShelfFilter(cutoff_frequency_hz=12000, gain_db=1.5),
        ])
    elif autotune_level == "subtle":
        board = Pedalboard([
            LowShelfFilter(cutoff_frequency_hz=150,  gain_db=1.5),
            PeakFilter(cutoff_frequency_hz=250,      gain_db=-1.5, q=1.2),
            PeakFilter(cutoff_frequency_hz=400,      gain_db=-1.8, q=1.0),
            PeakFilter(cutoff_frequency_hz=1500,     gain_db=1.2,  q=1.0),
            PeakFilter(cutoff_frequency_hz=2800,     gain_db=2.8,  q=0.9),
            PeakFilter(cutoff_frequency_hz=5000,     gain_db=1.5,  q=1.2),
            HighShelfFilter(cutoff_frequency_hz=10000, gain_db=2.0),
        ])
    elif autotune_level == "modern":
        board = Pedalboard([
            LowShelfFilter(cutoff_frequency_hz=180,  gain_db=2.0),
            PeakFilter(cutoff_frequency_hz=250,      gain_db=-2.0, q=1.2),
            PeakFilter(cutoff_frequency_hz=350,      gain_db=-2.0, q=1.2),
            PeakFilter(cutoff_frequency_hz=1500,     gain_db=1.5,  q=1.0),
            PeakFilter(cutoff_frequency_hz=2500,     gain_db=3.8,  q=1.0),
            PeakFilter(cutoff_frequency_hz=5000,     gain_db=2.2,  q=1.5),
            HighShelfFilter(cutoff_frequency_hz=10000, gain_db=2.5),
        ])
    elif autotune_level == "rnb":
        # Warm, smooth, silky — soulful body, gentle (non-harsh) presence, airy top
        board = Pedalboard([
            LowShelfFilter(cutoff_frequency_hz=200,  gain_db=2.2),   # rich warmth
            PeakFilter(cutoff_frequency_hz=300,      gain_db=-1.8, q=1.1),  # clean mud
            PeakFilter(cutoff_frequency_hz=900,      gain_db=-1.0, q=1.2),  # ease nasal
            PeakFilter(cutoff_frequency_hz=2200,     gain_db=1.6,  q=0.8),  # smooth presence
            PeakFilter(cutoff_frequency_hz=4500,     gain_db=-1.0, q=2.0),  # tame harsh edge
            HighShelfFilter(cutoff_frequency_hz=11000, gain_db=2.2),  # silk/air
        ])
    elif autotune_level == "rap":
        # Present, punchy, intelligible — controlled lows, strong upper-mid bite,
        # crisp consonants so every word cuts. Dry and in-your-face.
        board = Pedalboard([
            HighShelfFilter(cutoff_frequency_hz=80,  gain_db=-1.0),  # trim boom under the beat
            LowShelfFilter(cutoff_frequency_hz=140,  gain_db=0.8),
            PeakFilter(cutoff_frequency_hz=400,      gain_db=-2.2, q=1.1),  # clear low-mud
            PeakFilter(cutoff_frequency_hz=1800,     gain_db=2.0,  q=0.9),  # body/forward
            PeakFilter(cutoff_frequency_hz=3500,     gain_db=3.5,  q=1.0),  # bite/presence
            PeakFilter(cutoff_frequency_hz=6500,     gain_db=2.2,  q=1.4),  # consonant clarity
            HighShelfFilter(cutoff_frequency_hz=11000, gain_db=1.4),
        ])
    else:  # heavy
        board = Pedalboard([
            LowShelfFilter(cutoff_frequency_hz=200,  gain_db=2.5),
            PeakFilter(cutoff_frequency_hz=250,      gain_db=-2.5, q=1.2),
            PeakFilter(cutoff_frequency_hz=300,      gain_db=-2.5, q=1.0),
            PeakFilter(cutoff_frequency_hz=1500,     gain_db=1.8,  q=1.0),
            PeakFilter(cutoff_frequency_hz=2200,     gain_db=4.2,  q=1.0),
            PeakFilter(cutoff_frequency_hz=4500,     gain_db=2.8,  q=1.5),
            HighShelfFilter(cutoff_frequency_hz=9000,  gain_db=3.0),
        ])
    return board(y, sr)


# ── De-esser ──────────────────────────────────────────────────────────────────

def _deess(y: np.ndarray, sr: int, threshold_db: float = -24.0) -> np.ndarray:
    try:
        sos_hi = butter(4, [5000, 9500], btype='band', fs=sr, output='sos')
        y_sib  = sosfilt(sos_hi, y).astype(np.float32)

        frame_len     = max(4, int(sr * 0.005))
        threshold_lin = 10.0 ** (threshold_db / 20.0)
        gain          = np.ones(len(y), dtype=np.float32)
        hop           = max(1, frame_len // 2)

        for i in range(0, len(y) - frame_len, hop):
            rms = float(np.sqrt(np.mean(y_sib[i: i + frame_len] ** 2)))
            if rms > threshold_lin:
                overage_db   = 20.0 * np.log10(rms / threshold_lin + 1e-9)
                reduction_db = min(overage_db * 0.5, 6.0)
                gain[i: i + frame_len] = np.minimum(
                    gain[i: i + frame_len],
                    10.0 ** (-reduction_db / 20.0),
                )

        gain = gaussian_filter1d(gain, sigma=int(sr * 0.003)).astype(np.float32)
        # Wideband de-ess: subtract only the ATTENUATED portion of the sibilant
        # band from the original. Bit-exact passthrough at unity gain, and the
        # >9.5 kHz "air" (which the EQ shelves and exciter just added) survives
        # untouched — reconstructing as lowpass+band here previously deleted the
        # entire top octave from every processed vocal.
        return (y - y_sib * (1.0 - gain)).astype(np.float32)
    except Exception:
        return y


# ── Normalization ─────────────────────────────────────────────────────────────

def _normalize(y: np.ndarray, sr: int, target_lufs: float = -16.0) -> np.ndarray:
    # NO np.clip here: the true-peak limiter is the next stage in process_vocal
    # and handles overs cleanly. Hard-clipping first (quiet phone recordings
    # need +15-25 dB of gain, and vocals have a 12-18 dB crest factor) bakes in
    # square-wave distortion the limiter can never undo.
    if _HAS_PYLOUDNORM:
        try:
            meter = pyln.Meter(sr)
            current = meter.integrated_loudness(y.reshape(-1, 1))
            if np.isfinite(current) and current > -70:
                gain = 10.0 ** ((target_lufs - current) / 20.0)
                return (y * gain).astype(np.float32)
        except Exception:
            pass
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < 1e-6:
        return y
    gain = min(0.18 / rms, 8.0)
    return (y * gain).astype(np.float32)


# ── Adaptive measurements ─────────────────────────────────────────────────────

def _estimate_noise_floor_db(y: np.ndarray, sr: int) -> float:
    try:
        frame_len = max(512, int(sr * 0.020))
        hop = frame_len // 2
        frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop)
        rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=0))
        rms_per_frame = rms_per_frame[rms_per_frame > 1e-9]
        if len(rms_per_frame) == 0:
            return -48.0
        return float(20.0 * np.log10(float(np.percentile(rms_per_frame, 5)) + 1e-9))
    except Exception:
        return -48.0


def _estimate_sibilance_threshold(y: np.ndarray, sr: int) -> float:
    try:
        sos = butter(4, [5000, 9500], btype="band", fs=sr, output="sos")
        y_sib = sosfilt(sos, y).astype(np.float32)
        frame_len = max(4, int(sr * 0.005))
        frames = librosa.util.frame(y_sib, frame_length=frame_len,
                                    hop_length=frame_len // 2)
        rms_frames = np.sqrt(np.mean(frames ** 2, axis=0))
        rms_frames = rms_frames[rms_frames > 1e-9]
        if len(rms_frames) == 0:
            return -24.0
        peak_rms = float(np.percentile(rms_frames, 80))
        return float(np.clip(20.0 * np.log10(peak_rms + 1e-9) - 6.0, -36.0, -18.0))
    except Exception:
        return -24.0


# ── Micro-chorus for vocal thickness ─────────────────────────────────────────

def _micro_chorus(y: np.ndarray, sr: int, amount: float = 0.020) -> np.ndarray:
    """
    Add vocal thickness via a 12 ms Haas delay copy.

    The delay is below the Haas limit (~30 ms) so it fuses with the direct
    signal and is perceived as one wider, thicker vocal — not an echo.
    A mild high-pass on the delayed copy keeps low-end clean.
    """
    if amount <= 0:
        return y
    try:
        delay_samples = int(sr * 0.012)
        if delay_samples >= len(y):
            return y
        sos = butter(3, 200 / (sr / 2.0), btype='high', output='sos')
        delayed = sosfilt(sos, y).astype(np.float32)
        delayed_shifted = np.zeros_like(y)
        delayed_shifted[delay_samples:] = delayed[:-delay_samples]
        return (y + delayed_shifted * amount).astype(np.float32)
    except Exception:
        return y


# ── Harmonic exciter ──────────────────────────────────────────────────────────

def _harmonic_excite(y: np.ndarray, sr: int, amount: float = 0.07) -> np.ndarray:
    if amount <= 0:
        return y
    try:
        sos_hi   = butter(4, 2000 / (sr / 2.0), btype="high", output="sos")
        sos_lo   = butter(4, 2000 / (sr / 2.0), btype="low",  output="sos")
        presence = sosfilt(sos_hi, y).astype(np.float32)
        low_body = sosfilt(sos_lo, y).astype(np.float32)
        excited  = np.tanh(presence * (1.0 + amount * 4.0)) / (1.0 + amount * 4.0)
        return (low_body + presence + (excited - presence) * amount).astype(np.float32)
    except Exception:
        return y
