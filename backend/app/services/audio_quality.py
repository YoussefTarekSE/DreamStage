"""
Multi-dimensional audio quality gate.
Checks signal level, clipping, duration, SNR, and tonal content.
Returns {"ok": True} or {"ok": False, "reason": str, "message_en": str, "message_ar": str}.
"""
import numpy as np
import librosa
from .audio_loader import load_audio

MIN_RMS           = 0.008
CLIPPING_THRESHOLD = 0.97
MIN_DURATION_SEC  = 1.5
MIN_SNR_DB        = 6.0     # voice must be at least 6dB above noise floor
MIN_VOICED_PCT    = 0.20    # at least 20% of the recording must contain pitched voice


def check_quality(audio_bytes: bytes) -> dict:
    try:
        y, sr = load_audio(audio_bytes)
    except Exception:
        return _fail("corrupt",
                     "We couldn't read your recording. Please try again.",
                     "لم نتمكن من قراءة تسجيلك. يرجى المحاولة مرة أخرى.")

    duration = len(y) / sr

    # ── Duration ──────────────────────────────────────────────────────────────
    if duration < MIN_DURATION_SEC:
        return _fail("too_short",
                     "That recording is too short. Please sing the full phrase.",
                     "التسجيل قصير جداً. يرجى غناء العبارة كاملة.")

    # ── Level ─────────────────────────────────────────────────────────────────
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < MIN_RMS:
        return _fail("too_quiet",
                     "We can barely hear you. Move closer to your mic and try again.",
                     "لا نكاد نسمعك. اقترب من الميكروفون وحاول مرة أخرى.")

    # ── Clipping ──────────────────────────────────────────────────────────────
    if bool(np.mean(np.abs(y) > CLIPPING_THRESHOLD) > 0.01):
        return _fail("clipping",
                     "Your recording is too loud and distorted. Lower your mic volume and try again.",
                     "تسجيلك عالٍ جداً ومشوّه. خفّض مستوى الميكروفون وحاول مرة أخرى.")

    # ── SNR estimate ──────────────────────────────────────────────────────────
    # Estimate noise floor from the quietest 15% of frames
    try:
        frame_len = 2048
        hop = 512
        rms_frames = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
        noise_floor = np.percentile(rms_frames, 15)
        signal_level = np.percentile(rms_frames, 85)
        if noise_floor > 1e-9:
            snr_db = 20.0 * np.log10(signal_level / noise_floor)
            if snr_db < MIN_SNR_DB:
                return _fail("too_noisy",
                             "There's too much background noise. Find a quieter space and try again.",
                             "هناك ضوضاء كثيرة في الخلفية. ابحث عن مكان أهدأ وحاول مرة أخرى.")
    except Exception:
        pass

    # ── Pitched voice detection ───────────────────────────────────────────────
    # Verify the recording actually contains singing/melody (not just talking or noise)
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sr,
            frame_length=2048,
        )
        valid_f0 = f0[~np.isnan(f0)]
        voiced_pct = float(len(valid_f0[valid_f0 > 0]) / max(len(f0), 1))
        if voiced_pct < MIN_VOICED_PCT:
            return _fail("no_pitch",
                         "We couldn't detect a melody. Make sure you're singing, not just speaking.",
                         "لم نتمكن من اكتشاف لحن. تأكد من أنك تغني وليس مجرد التحدث.")
    except Exception:
        pass  # pyin can fail on very short clips — don't block on it

    return {"ok": True}


def _fail(reason: str, msg_en: str, msg_ar: str) -> dict:
    return {"ok": False, "reason": reason, "message_en": msg_en, "message_ar": msg_ar}
