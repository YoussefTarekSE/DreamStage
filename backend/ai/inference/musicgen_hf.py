"""
MusicGen via Hugging Face Inference API.

Production beat generation path: no GPU required, always available on Render.
Uses huggingface_hub InferenceClient as primary method (handles both the legacy
api-inference.huggingface.co endpoint and the newer Inference Providers router).
Falls back to raw httpx if the package is unavailable.

HF Inference API notes:
  - Model:        facebook/musicgen-small (text-to-music)
  - Free tier:    rate limited but functional for DreamStage's volume
  - Timeout:      Models cold-start in 30-90 s on free tier
  - Output:       audio bytes (WAV or FLAC depending on provider)

Priority chain in beat_generator.py:
  1. Local audiocraft (melody conditioning, best quality, needs GPU/patience)
  2. HF Inference API [THIS MODULE] (text-only, reliable, no GPU)
  3. Gradio Space (melody conditioning, unreliable)
  4. Programmatic synthesizer (rule-based fallback, always works)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Try both the new Inference Providers router and the legacy endpoint.
_HF_URLS = [
    "https://router.huggingface.co/hf-inference/models/facebook/musicgen-small",
    "https://api-inference.huggingface.co/models/facebook/musicgen-small",
]
_DEFAULT_TIMEOUT = 90.0


async def generate_beat_hf_api(
    prompt: str,
    hf_token: str,
    duration_hint_seconds: int = 30,
    timeout: float = _DEFAULT_TIMEOUT,
) -> bytes:
    """
    Generate a beat via HF Inference API (text-to-music).

    Tries InferenceClient first (handles API versioning automatically), then
    falls back to direct HTTP against both the new router and the legacy URL.

    Returns:
        Audio bytes (WAV or FLAC, >10 KB)

    Raises:
        RuntimeError: If all methods fail
    """
    # ── Direct HTTP (minimal payload, two URL candidates) ────────────────────
    # NOTE: huggingface_hub InferenceClient is intentionally skipped here.
    # facebook/musicgen-small has no inference providers in the HF registry
    # (as of 2026-06), so the InferenceClient raises StopIteration in its
    # provider-selection loop — confusing and not recoverable.
    # The httpx path tries both the new router and the legacy endpoint directly.
    return await _generate_via_http(prompt, hf_token, timeout)


async def _generate_via_http(prompt: str, hf_token: str, timeout: float) -> bytes:
    """
    Raw HTTP fallback.  Tries the new Inference Providers router first, then
    the legacy api-inference.huggingface.co endpoint.

    Minimal payload — no extra parameters that might cause a 400 on older models.
    """
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type":  "application/json",
        "Accept":        "audio/wav",
    }
    # Minimal payload — avoid sending unsupported parameters (guidance_scale,
    # do_sample) that caused silent 400 failures on some model versions.
    payload = {"inputs": prompt}

    last_error = "no attempt made"

    for url in _HF_URLS:
        logger.info("[hf] HTTP POST %s", url)
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
                response = await client.post(url, headers=headers, json=payload)

            logger.info(
                "[hf] HTTP response: url=%s status=%d content_type=%s bytes=%d",
                url,
                response.status_code,
                response.headers.get("content-type", "?"),
                len(response.content),
            )

            if response.status_code == 200:
                if len(response.content) > 10_000:
                    logger.info("[hf] HTTP OK: %s (%d bytes)", url, len(response.content))
                    return response.content
                # 200 but tiny — probably a JSON error body
                logger.warning("[hf] HTTP 200 but tiny body: %s", response.text[:300])
                last_error = f"200 but tiny: {response.text[:200]}"
                continue

            elif response.status_code == 503:
                try:
                    wait = min(float(response.json().get("estimated_time", 30)), 60.0)
                except Exception:
                    wait = 30.0
                logger.info("[hf] model loading at %s, waiting %.0fs", url, wait)
                await asyncio.sleep(wait)
                # One retry after the cold-start wait
                async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as c2:
                    r2 = await c2.post(url, headers=headers, json=payload)
                    logger.info("[hf] retry status=%d bytes=%d", r2.status_code, len(r2.content))
                    if r2.status_code == 200 and len(r2.content) > 10_000:
                        return r2.content
                    last_error = f"503 retry status={r2.status_code}"
                continue

            elif response.status_code == 401:
                raise RuntimeError(
                    "HF API: 401 Unauthorized — check HF_API_KEY. "
                    f"Body: {response.text[:200]}"
                )

            elif response.status_code == 400:
                # 400 can mean "model not supported by this provider" — continue
                # to next URL rather than raising, so we try the legacy endpoint.
                logger.warning("[hf] HTTP 400 at %s: %s", url, response.text[:300])
                last_error = f"400 at {url}: {response.text[:200]}"
                continue

            else:
                last_error = (
                    f"status={response.status_code} at {url}: {response.text[:200]}"
                )
                logger.warning("[hf] unexpected status: %s", last_error)
                continue

        except (RuntimeError, asyncio.TimeoutError):
            raise
        except httpx.TimeoutException as exc:
            logger.warning("[hf] HTTP timeout on %s: %s", url, exc)
            last_error = f"timeout on {url}"
        except Exception as exc:
            logger.warning("[hf] HTTP error on %s: %s: %s", url, type(exc).__name__, exc)
            last_error = f"{type(exc).__name__}: {exc}"

    raise RuntimeError(
        f"HF API: all methods failed. Last error: {last_error}. "
        "Check logs above for per-attempt status codes."
    )


def build_rich_prompt(
    analysis: dict,
    style_hint: str = "",
) -> str:
    """
    Build a maximally specific text prompt from vocal analysis.

    Since the HF API has no melody conditioning, the prompt must encode
    as much musical context as possible.
    """
    if style_hint.strip():
        tempo = analysis.get("tempo", 90)
        key   = analysis.get("key", "C")
        mode  = analysis.get("mode", "major")
        return (
            f"{style_hint.strip()}, professional studio instrumental, no vocals, "
            f"{int(tempo)} BPM, key of {key} {mode}, "
            f"808 bass, crisp hi-hats, punchy snare, "
            f"layered melody, mixed and mastered, major label quality"
        )

    tempo        = analysis.get("tempo", 90)
    key          = analysis.get("key", "C")
    mode         = analysis.get("mode", "major")
    emotion      = analysis.get("emotion", "smooth")
    valence      = analysis.get("valence", 0.5)
    rms          = analysis.get("rms", 0.18)
    centroid     = analysis.get("centroid", 1500.0)
    vocal_range  = analysis.get("vocal_range", {})
    tone         = analysis.get("tone", {})
    energy_label = analysis.get("energy_label", "medium")

    _EMOTION_GENRES = {
        "euphoric":    ("melodic trap", "Travis Scott, Kid Cudi, Playboi Carti"),
        "uplifting":   ("pop hip-hop",  "Pharrell Williams, Tyler the Creator"),
        "dark":        ("dark trap",    "Metro Boomin, 808 Mafia, Pi'erre Bourne"),
        "melancholic": ("neo-soul R&B", "Frank Ocean, Daniel Caesar, SZA"),
        "energetic":   ("hard hip-hop", "Kendrick Lamar, J. Cole, Pusha T"),
        "intimate":    ("soul ballad",  "Sam Smith, H.E.R., Giveon"),
        "smooth":      ("modern R&B",   "The Weeknd, Drake, Bryson Tiller"),
    }
    genre_label, refs = _EMOTION_GENRES.get(emotion, ("modern hip-hop", "Drake, The Weeknd"))

    energy_desc = {
        "high":   "high-energy dynamic",
        "medium": "mid-tempo groove",
        "low":    "chill mellow ambient",
    }.get(energy_label, "mid-tempo")

    brightness  = tone.get("brightness", 0.5) if isinstance(tone, dict) else 0.5
    bright_desc = ("bright airy open" if brightness > 0.60
                   else ("warm rich dark" if brightness < 0.35 else "balanced clear"))

    mood_adj = ("uplifting euphoric" if valence > 0.65
                else ("dark brooding" if valence < 0.35 else "soulful emotional"))

    range_label   = (vocal_range.get("range_label", "")
                     if isinstance(vocal_range, dict) else "")
    register_hint = ""
    if range_label in ("soprano", "alto"):
        register_hint = "high-register melody, "
    elif range_label in ("tenor", "baritone"):
        register_hint = "mid-register melody, "

    return (
        f"{energy_desc} {genre_label} instrumental beat, {mood_adj} mood, "
        f"{bright_desc} sound, {int(tempo)} BPM, key of {key} {mode}, "
        f"punchy 808 kick drum, crisp snare, layered hi-hats, "
        f"deep 808 bass in key of {key}, "
        f"{register_hint}melodic top-line synth, "
        f"professional mixing and mastering, absolutely no vocals, "
        f"inspired by {refs}, major label quality, radio ready"
    )


async def generate_from_analysis(
    analysis: dict,
    hf_token: str,
    style_hint: str = "",
    duration_seconds: int = 30,
) -> tuple[bytes, str]:
    """
    Top-level function: build prompt from analysis, call HF API, return beat.

    Returns:
        (wav_bytes, genre_used)
    """
    prompt = build_rich_prompt(analysis, style_hint)
    logger.info("[hf] generating: %s", prompt[:100])

    wav_bytes = await generate_beat_hf_api(
        prompt=prompt,
        hf_token=hf_token,
        duration_hint_seconds=duration_seconds,
    )
    return wav_bytes, "musicgen_hf_api"
