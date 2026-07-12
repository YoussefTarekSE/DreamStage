"""
Client for an ACE-Step 1.5 server — DreamStage's neural producer tier.

ACE-Step's "complete" task is Vocal2BGM: it LISTENS to the artist's actual
performance and composes a full accompaniment for it (drums/bass/keys...),
returning a finished song with the vocal already inside. MIT-licensed code
and weights — legally clean for the paid product.

Enabled whenever an ACE-Step server is reachable at ACESTEP_URL
(default http://localhost:8001 — the founder's GPU laptop running
`uv run acestep-api`; later, the beta GPU worker). When the server is
offline every call cheaply no-ops and the programmatic synthesizer takes
over — the app never depends on the GPU being up.

Task choice (2026-07-12, measured): the "complete" task REGENERATES the
whole song and only sometimes preserves the source vocal — a lottery that
produced the "vocal is gone" complaint. The "lego" task instead generates
ONLY the accompaniment tracks over the vocal as context (same duration,
no vocal copy inside). DreamStage then mixes the artist's REAL vocal on
top with its own polished vocal chain — the voice is guaranteed present.
Cuts carry genre "acestep_accomp" and flow through the NORMAL mix path.
(Legacy "acestep_v2bgm" cuts were full songs; mix.py masters those as-is.)
"""
import asyncio
import json
import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

ACESTEP_URL = os.environ.get("ACESTEP_URL", "http://localhost:8001").rstrip("/")
# Full-song diffusion takes 30-200s on an RTX 3070 for a typical take.
TOTAL_TIMEOUT_S = float(os.environ.get("ACESTEP_TIMEOUT", "300"))
POLL_INTERVAL_S = 3.0

ENGINE_GENRE = "acestep_accomp"


def is_available(timeout: float = 2.0) -> bool:
    """Cheap reachability probe — the tier engages only when this passes."""
    try:
        r = httpx.get(f"{ACESTEP_URL}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


# Production briefs per genre lane — this is how the Producer Cuts system's
# taste learning and five-interpretations spread reach the neural band: the
# chosen force_genre becomes the brief, so each cut takes a distinct direction.
_GENRE_BRIEFS = {
    "rnb_neo_soul":    "warm neo-soul, rhodes keys, soft pocket drums, round bass",
    "rnb_smooth":      "smooth R&B, silky keys, laid-back drums, deep bass",
    "soul_ballad":     "soulful ballad, piano, gentle strings, sparse drums",
    "lofi_chill":      "lo-fi chill, dusty drums, mellow keys, tape warmth",
    "jazz_hop":        "jazzy hip-hop, upright bass, brushed drums, warm keys",
    "afrobeats":       "afrobeats groove, log drums, syncopated percussion, warm bass",
    "amapiano":        "amapiano, log drum bass, airy pads, shakers",
    "dancehall":       "dancehall riddim, punchy drums, deep bass",
    "reggaeton":       "reggaeton, dembow rhythm, deep bass, crisp percussion",
    "club_house":      "house groove, four-on-the-floor drums, warm bass, keys",
    "pop_bright":      "bright modern pop, punchy drums, airy synths, melodic bass",
    "trap_melodic":    "melodic trap, deep 808 bass, tight hi-hats, ambient keys",
    "hiphop_modern":   "modern hip-hop, hard drums, deep 808, melodic keys",
    "hiphop_boom_bap": "boom bap hip-hop, dusty drums, deep bass, soulful keys",
    "trap_dark":       "dark brooding trap, deep 808s, sparse keys, cold atmosphere",
    "drill":           "drill, sliding 808 bass, aggressive hi-hats, dark keys",
    "uk_drill":        "UK drill, gliding 808s, sparse dark keys",
    "phonk":           "memphis phonk, distorted 808 bass, dark keys, cowbell",
}


def _build_caption(analysis: dict, style_hint: str = "",
                   genre_hint: str | None = None) -> str:
    """Production brief for the neural band, from what we heard and the
    direction the Producer Cuts system chose (taste/archetype/branch)."""
    emotion = str(analysis.get("emotion", "") or "")
    key = analysis.get("key")
    mode = analysis.get("mode")
    parts = []
    if style_hint.strip():
        parts.append(style_hint.strip())
    if genre_hint and genre_hint in _GENRE_BRIEFS:
        parts.append(_GENRE_BRIEFS[genre_hint])
    elif not style_hint.strip():
        parts.append("warm accompaniment")
    if emotion:
        parts.append(f"{emotion} mood")
    parts.append("following the vocal's key, tempo and phrasing")
    if key and mode:
        parts.append(f"in {key} {mode}")
    return ", ".join(parts)


async def generate_accompaniment(vocal_bytes: bytes, analysis: dict,
                                 style_hint: str = "",
                                 genre_hint: str | None = None) -> bytes | None:
    """Generate an accompaniment stem FOR the processed vocal (lego task —
    no vocal inside the output). Returns WAV bytes or None on any failure —
    callers always have the synthesizer fallback."""
    caption = _build_caption(analysis, style_hint, genre_hint)
    try:
        # Generous read timeout: the server lazy-loads the model on its very
        # first request (~60-90s); subsequent submits return in milliseconds.
        timeout = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            files = {"src_audio": ("vocal.wav", vocal_bytes, "audio/wav")}
            data = {
                "task_type": "lego",
                "instruction": "Generate the drums, bass, keyboard track based on the audio context:",
                "prompt": caption,
                # WAV: saved via soundfile (no ffmpeg dependency on the GPU
                # host), and the mixer re-encodes the final MP3 itself anyway.
                "audio_format": "wav",
                # EXACTLY the founder-approved eval conditions (the python
                # API's dataclass defaults, read from acestep/inference.py):
                # steps=8, guidance=7.0, shift=1.0, no LM codes, one candidate.
                # Empirical note 2026-07-12: the docs recommend 32-64 steps for
                # the base model, but 32 steps produced unusable noise on the
                # complete task while 8 produced the renders the founder liked.
                # Trust ears over docs; change only with A/B listening.
                "inference_steps": 8,
                "guidance_scale": 7.0,
                "shift": 1.0,
                "batch_size": 1,
                "thinking": False,
            }
            r = await client.post(f"{ACESTEP_URL}/release_task", data=data, files=files)
            r.raise_for_status()
            task_id = r.json()["data"]["task_id"]
            logger.info("[acestep] task %s submitted (caption: %s)", task_id, caption)

            deadline = time.time() + TOTAL_TIMEOUT_S
            while time.time() < deadline:
                await asyncio.sleep(POLL_INTERVAL_S)
                q = await client.post(f"{ACESTEP_URL}/query_result",
                                      json={"task_id_list": [task_id]})
                q.raise_for_status()
                items = q.json().get("data") or []
                if not items:
                    continue
                status = items[0].get("status")
                if status == 2:
                    logger.warning("[acestep] task %s failed on server", task_id)
                    return None
                if status == 1:
                    result = items[0].get("result")
                    if isinstance(result, str):
                        result = json.loads(result)
                    file_url = (result or [{}])[0].get("file")
                    if not file_url:
                        return None
                    if file_url.startswith("/"):
                        file_url = f"{ACESTEP_URL}{file_url}"
                    a = await client.get(file_url)
                    a.raise_for_status()
                    audio = a.content
                    if len(audio) < 10_000:
                        return None
                    logger.info("[acestep] task %s done: %d bytes", task_id, len(audio))
                    return audio
            logger.warning("[acestep] task %s timed out after %.0fs", task_id, TOTAL_TIMEOUT_S)
            return None
    except Exception as exc:
        logger.warning("[acestep] unavailable/failed: %s: %s", type(exc).__name__, exc)
        return None
