"""
Beat generator — vocal-first generation pipeline.

Current production path: the programmatic synthesizer (beat_synthesizer.py),
fed by deep vocal analysis (ml_analyzer.py), vocal-key harmony transcription
(vocal_harmony.py), and the performance map (performance_map.py) so the
arrangement reacts to what the singer actually did.

MusicGen tiers were REMOVED 2026-07-11: facebook/musicgen weights are
CC-BY-NC (non-commercial only) — legally unusable in a subscription product —
and none of them ever produced a beat in practice. The neural path forward is
ACE-Step 1.5 Vocal2BGM (MIT code + weights) served from the GPU worker.
"""
import os
import time
import tempfile
import asyncio
import logging
import numpy as np
import librosa
from .audio_loader import load_audio
from .beat_synthesizer import generate_beat as synth_beat
from .metrics import time_ms
from .audio_analysis import (
    detect_key_and_mode,
    detect_swing,
    estimate_valence,
    classify_emotion,
    classify_vocal_style,
)

logger = logging.getLogger(__name__)


def analyze_vocal_mood(processed_bytes: bytes) -> dict:
    """
    Deep analysis of the processed vocal recording.
    Returns full feature dict consumed by both prompt builder and synthesizer.
    """
    try:
        y, sr = load_audio(processed_bytes)

        # Core features
        rms = float(np.sqrt(np.mean(y ** 2)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.atleast_1d(tempo_arr)[0])

        # Onset density
        onsets = librosa.onset.onset_detect(y=y, sr=sr)
        density = len(onsets) / max(len(y) / sr, 1.0)

        # Key and mode
        key, mode = detect_key_and_mode(y, sr)

        # Swing
        swing_ratio = detect_swing(y, sr, tempo)

        # High-level features
        valence = estimate_valence(tempo, mode, centroid, rms)
        emotion = classify_emotion(mode, valence, rms, tempo)

        # F0 for vocal style
        try:
            f0 = librosa.yin(y, fmin=librosa.note_to_hz("C2"),
                             fmax=librosa.note_to_hz("C6"), sr=sr)
            f0_voiced = f0[f0 > 80]
        except Exception:
            f0_voiced = np.array([])
        vocal_style = classify_vocal_style(f0_voiced, density)

        return {
            "tempo":       round(tempo, 1),
            "rms":         round(rms, 4),
            "centroid":    round(centroid, 1),
            "density":     round(density, 2),
            "key":         key,
            "mode":        mode,
            "valence":     round(valence, 3),
            "emotion":     emotion,
            "vocal_style": vocal_style,
            "swing_ratio": round(swing_ratio, 3),
            "overall_rms": round(rms, 4),
        }
    except Exception:
        return {
            "tempo": 90.0, "rms": 0.18, "centroid": 1500.0, "density": 2.0,
            "key": "C", "mode": "major", "valence": 0.5, "emotion": "smooth",
            "vocal_style": "rhythmic", "swing_ratio": 0.5, "overall_rms": 0.18,
        }


# Genre reference strings for the AI prompt
_GENRE_REFS = {
    "euphoric":   ("melodic trap", "Travis Scott, Playboi Carti, Kid Cudi"),
    "uplifting":  ("pop hip-hop", "Pharrell Williams, Kanye West, Tyler the Creator"),
    "dark":       ("dark trap / drill", "Metro Boomin, 808 Mafia, Pi'erre Bourne"),
    "melancholic": ("neo-soul R&B", "Frank Ocean, Daniel Caesar, SZA"),
    "energetic":  ("hard-hitting hip-hop", "Kendrick Lamar, J. Cole, Pusha T"),
    "intimate":   ("soul ballad", "Sam Smith, Adele, H.E.R."),
    "smooth":     ("modern R&B", "The Weeknd, Drake, Bryson Tiller"),
}


def build_prompt(voice_profile: dict, vocal_mood: dict, style_hint: str = "") -> str:
    tempo    = vocal_mood.get("tempo")    or voice_profile.get("tempo_bpm") or 90
    rms      = vocal_mood.get("rms", 0.18)
    centroid = vocal_mood.get("centroid", 1500.0)
    key      = vocal_mood.get("key", voice_profile.get("key", "C"))
    mode     = vocal_mood.get("mode", voice_profile.get("mode", "major"))
    emotion  = vocal_mood.get("emotion", "smooth")
    valence  = vocal_mood.get("valence", 0.5)
    tone     = voice_profile.get("tone_type", "balanced")

    if style_hint.strip():
        return (
            f"{style_hint.strip()}, professional studio instrumental, no vocals, "
            f"{int(tempo)} BPM, key of {key} {mode}, 808 bass, crisp hi-hats, "
            f"punchy snare, layered melody, mixed and mastered, major label quality"
        )

    genre_label, ref = _GENRE_REFS.get(emotion, ("modern hip-hop", "Drake, The Weeknd"))
    energy_desc  = "high-energy dynamic" if rms > 0.22 else ("mid-tempo groove" if rms > 0.10 else "chill mellow")
    bright_desc  = "bright airy open" if centroid > 2100 else ("warm rich dark" if centroid < 1300 else "balanced clear")
    tone_desc    = {"warm": "warm golden analog", "bright": "crisp bright digital", "balanced": "balanced studio"}.get(tone, "studio")
    mood_adj     = "uplifting euphoric" if valence > 0.65 else ("dark brooding" if valence < 0.35 else "soulful emotional")

    return (
        f"{energy_desc} {genre_label} instrumental beat, {mood_adj} mood, "
        f"{bright_desc} sound, {tone_desc} production, {int(tempo)} BPM, key of {key} {mode}, "
        f"punchy kick drum, crisp snare, layered hi-hats, deep 808 bass in key, "
        f"melodic top-line, professional mixing, absolutely no vocals, "
        f"inspired by {ref}, major label quality, radio ready"
    )


async def generate_beat_from_vocal(
    processed_bytes: bytes,
    voice_profile: dict,
    vocal_mood: dict,
    style_hint: str = "",
    hf_api_key: str = "",
    exclude_genres: list = None,
    attempt: int = 1,
    previous_genre: str = None,
    style_bias: str = None,
    force_genre: str = None,
) -> tuple:
    """
    Generate a beat conditioned on the artist's vocal.
    Returns (beat_bytes, genre_used, analysis_used).

    analysis_used includes "_tier_attempts": [{tier, name, duration_ms, success, reason?}]
    for use by the telemetry layer.
    """
    # Use ML-enriched analysis when available
    # (ml_analyzer adds genre_hint, energy_label, vocal_range, tone)
    try:
        from .ml_analyzer import analyze_full_ml
        with time_ms("ml_analysis_duration_ms"):
            ml_analysis = await asyncio.to_thread(analyze_full_ml, processed_bytes)
        merged_mood = {**vocal_mood, **{
            k: v for k, v in ml_analysis.items()
            if k not in ("tempo", "key", "mode") or vocal_mood.get(k) is None
        }}
    except Exception as exc:
        logger.info("[beat_generator] ml_analyzer skipped (%s: %s) — using vocal_mood", type(exc).__name__, exc)
        merged_mood = vocal_mood

    # ── Vocal harmony: make the beat follow the singer's actual tune ──────────
    # Transcribe the vocal's notes, detect the real key, and pick the chord per
    # bar that supports what the artist sang. These chord degrees drive the bass
    # + chords; the detected key overrides the rougher chroma estimate.
    bar_degrees = None
    vocal_melody = None
    melody_loop_beats = 16.0
    # What the AI actually heard and decided — powers the honest per-cut
    # producer's note (Three Laws #2: the AI explains every decision).
    decisions: dict = {}
    try:
        from .vocal_harmony import transcribe_harmony
        _tempo = float(merged_mood.get("tempo", 90) or 90)
        with time_ms("transcription_duration_ms"):
            harmony = await asyncio.to_thread(
                transcribe_harmony, processed_bytes, tempo=_tempo, bars=16
            )
        if harmony and harmony.get("confidence", 0) >= 0.45:
            merged_mood["key"]  = harmony["key"]
            merged_mood["mode"] = harmony["mode"]
            bar_degrees = harmony["bar_degrees"]
            vocal_melody = harmony.get("melody") or None
            melody_loop_beats = harmony.get("melody_loop_beats", 16.0)
            decisions["harmony"] = {
                "key": harmony["key"],
                "mode": harmony["mode"],
                "confidence": round(float(harmony.get("confidence", 0)), 2),
                "melody_notes": len(vocal_melody or []),
            }
            logger.info("[beat_generator] vocal harmony: key=%s %s degrees=%s melody_notes=%d conf=%.2f",
                        harmony["key"], harmony["mode"], bar_degrees[:8],
                        len(vocal_melody or []), harmony["confidence"])
    except Exception as exc:
        logger.info("[beat_generator] harmony transcription skipped (%s: %s)",
                    type(exc).__name__, exc)

    # ── Performance map: shape the arrangement around what the singer DID ──────
    # Reads the take's energy/phrasing/pauses so the synth tier lays out sections
    # (chorus on the real peak), drives busyness bar-by-bar, and answers the
    # singer's breaths with fills/bass-moves/silence — accompaniment, not a
    # template. None/low-confidence → the synth falls back to its bar template.
    performance = None
    try:
        from .performance_map import build_performance_map
        _ptempo = float(merged_mood.get("tempo", 90) or 90)
        performance = await asyncio.to_thread(
            build_performance_map, processed_bytes, tempo=_ptempo, bars=16
        )
        if performance:
            if performance.get("confidence", 0) >= 0.35:
                decisions["performance"] = {
                    "arc": performance.get("energy_arc"),
                    "pauses": len(performance.get("pauses", [])),
                    "confidence": round(float(performance.get("confidence", 0)), 2),
                }
            logger.info("[beat_generator] performance map: arc=%s conf=%.2f "
                        "sections=%s pauses=%d",
                        performance.get("energy_arc"), performance.get("confidence", 0),
                        performance.get("sections"), len(performance.get("pauses", [])))
    except Exception as exc:
        logger.info("[beat_generator] performance map skipped (%s: %s)",
                    type(exc).__name__, exc)

    merged_mood["_decisions"] = decisions

    prompt = build_prompt(voice_profile, merged_mood, style_hint)

    logger.info(
        "[beat_generator] merged_mood: tempo=%.1f key=%s %s emotion=%s "
        "valence=%.3f rms=%.4f vocal_style=%s",
        merged_mood.get("tempo"), merged_mood.get("key"), merged_mood.get("mode"),
        merged_mood.get("emotion"), merged_mood.get("valence"),
        merged_mood.get("rms"), merged_mood.get("vocal_style"),
    )
    logger.info("[beat_generator] prompt: %s", prompt)

    tier_attempts: list[dict] = []

    def _attempt(tier: int, name: str) -> dict:
        return {"tier": tier, "name": name, "started_at": time.time()}

    def _finish(entry: dict, success: bool, reason: str | None = None) -> dict:
        entry["duration_ms"] = int((time.time() - entry.pop("started_at")) * 1000)
        entry["success"] = success
        if reason:
            entry["reason"] = reason
        return entry

    # MusicGen tiers 1-3 REMOVED (2026-07-11): facebook/musicgen weights are
    # CC-BY-NC — non-commercial only, legally unusable in a subscription
    # product — and none of the three tiers ever produced a beat in practice.

    # ── Neural producer: ACE-Step 1.5 Vocal2BGM (MIT) ────────────────────────
    # A real music model that LISTENS to this exact performance and composes a
    # full production around it — the vocal-first doctrine as a model. Engages
    # only when an ACE-Step server is reachable (the founder's GPU laptop /
    # the beta GPU worker); otherwise the synthesizer below always delivers.
    # NOTE: its output already CONTAINS the vocal — genre "acestep_v2bgm"
    # tells the mixer to master it as-is instead of layering the vocal again.
    from . import acestep_client as _ace
    if len(processed_bytes or b"") > 10_000 and _ace.is_available():
        t0 = _attempt(0, "acestep_accomp")
        # force_genre carries the Producer Cuts direction (taste-led spread,
        # branch lane, or exploration) into the neural band's brief, so each
        # cut takes a genuinely different direction — same as the synth path.
        neural = await _ace.generate_accompaniment(
            processed_bytes, merged_mood, style_hint=style_hint or "",
            genre_hint=force_genre)
        if neural and len(neural) > 10_000:
            tier_attempts.append(_finish(t0, True))
            logger.info("Beat generated via ACE-Step Vocal2BGM (neural tier)")
            merged_mood["_tier_attempts"] = tier_attempts
            return neural, _ace.ENGINE_GENRE, merged_mood
        tier_attempts.append(_finish(t0, False, "acestep_failed_or_timeout"))

    # ── Synthesizer — 5 candidates, pick best ────────────────────────────────
    logger.info("Synthesizer: generating 5 candidates for scoring")
    t4 = _attempt(4, "synthesizer")
    with time_ms("beat_candidate_generation_duration_ms"):
        best_bytes, best_genre, best_score = await asyncio.to_thread(
            _generate_best_synth_candidate,
            analysis=merged_mood,
            exclude_genres=exclude_genres or [],
            base_attempt=attempt,
            previous_genre=previous_genre,
            style_bias=style_bias,
            bar_degrees=bar_degrees,
            melody=vocal_melody,
            melody_loop_beats=melody_loop_beats,
            force_genre=force_genre,
            performance=performance,
        )
    tier_attempts.append(_finish(t4, True))
    merged_mood["beat_score"] = best_score
    merged_mood["_tier_attempts"] = tier_attempts
    return best_bytes, best_genre, merged_mood


def _generate_best_synth_candidate(
    analysis: dict,
    exclude_genres: list,
    base_attempt: int = 1,
    n_candidates: int = 5,
    max_regen: int = 2,
    previous_genre: str = None,
    style_bias: str = None,
    bar_degrees: list = None,
    melody: list = None,
    melody_loop_beats: float = 16.0,
    force_genre: str = None,
    performance: dict = None,
) -> tuple[bytes, str, dict]:
    """
    Generate n_candidates beats, score each, return the best.
    If the best candidate scores below REJECTION_THRESHOLD, regenerate
    up to max_regen times with fresh seeds before accepting.
    """
    from .beat_scorer import score_beat, should_reject

    best_bytes = b""
    best_genre = ""
    best_score: dict = {}
    best_total = -1.0

    for regen in range(max_regen + 1):
        candidates = []
        used_in_this_round: list[str] = []

        for i in range(n_candidates):
            attempt_n = base_attempt * 100 + regen * 10 + i
            # Render UNMASTERED for scoring — the full master chain runs only
            # once, on the winner (master_beat_bytes below). Saves ~N× mastering.
            wav, genre = synth_beat(
                analysis=analysis,
                bars=16,
                exclude_genres=exclude_genres + used_in_this_round,
                attempt=attempt_n,
                previous_genre=previous_genre,
                style_bias=style_bias,
                master=False,
                bar_degrees=bar_degrees,
                melody=melody,
                melody_loop_beats=melody_loop_beats,
                force_genre=force_genre,
                performance=performance,
            )
            score = score_beat(wav, analysis, genre_used=genre)
            candidates.append((wav, genre, score))
            used_in_this_round.append(genre)
            logger.debug("Candidate %d: genre=%s score=%.1f", i + 1,
                         genre, score.get("total", 0))

        # Pick best in this round
        round_best = max(candidates, key=lambda c: c[2].get("total", 0))
        wav, genre, score = round_best

        if score.get("total", 0) > best_total:
            best_bytes = wav
            best_genre = genre
            best_score = score
            best_total = score.get("total", 0)

        if not should_reject(score):
            logger.info("Accepted candidate: genre=%s score=%.1f", genre, best_total)
            break
        if regen < max_regen:
            logger.info("Best candidate score %.1f below threshold, regenerating (%d/%d)...",
                        best_total, regen + 1, max_regen)

    logger.info("Final beat: genre=%s score=%.1f", best_genre, best_total)

    # Master the winning candidate once (candidates were rendered unmastered)
    if best_bytes:
        try:
            from .beat_synthesizer import master_beat_bytes
            best_bytes = master_beat_bytes(best_bytes, best_genre)
        except Exception as exc:
            logger.error("[beat_generator] mastering winner failed (%s) — using unmastered", exc)

    return best_bytes, best_genre, best_score


# MusicGen helper functions removed 2026-07-11: facebook/musicgen weights are
# CC-BY-NC (non-commercial), so those tiers could never legally ship in the
# subscription product. See backend/ai/inference/ for the archived clients.
