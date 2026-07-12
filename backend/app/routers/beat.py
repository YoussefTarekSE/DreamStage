import asyncio
import json
import time
import logging
import traceback
from datetime import datetime, timezone

import boto3
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from supabase import create_client
from ..auth import get_current_user
from ..config import settings
from ..errors import safe_error_detail
from ..storage import upload_file, delete_file, generate_signed_url
from ..services.beat_generator import generate_beat_from_vocal
from ..services.vocal_processor import style_genre_bias
from ..services.telemetry import record_beat_event
from ..services.metrics import increment, time_ms
from ..services import producer_cuts as pc
from ..services import jobs

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/studio", tags=["beat"])

# DreamStage is an AI producer: the creative session is UNLIMITED. Every
# generation is a Producer Cut, kept forever (see services/producer_cuts.py).
BEAT_URL_TTL = 3600

_GENRE_LABELS = {
    "trap_dark":       "Dark Trap",
    "trap_melodic":    "Melodic Trap",
    "hiphop_boom_bap": "Boom Bap",
    "hiphop_modern":   "Modern Hip-Hop",
    "rnb_smooth":      "Smooth R&B",
    "rnb_neo_soul":    "Neo-Soul",
    "pop_bright":      "Pop",
    "afrobeats":       "Afrobeats",
    "dancehall":       "Dancehall",
    "lofi_chill":      "Lo-Fi Chill",
    "soul_ballad":     "Soul Ballad",
    "drill":           "Drill",
    "uk_drill":        "UK Drill",
    "phonk":           "Phonk",
    "reggaeton":       "Reggaeton",
    "amapiano":        "Amapiano",
    "musicgen_ai":     "AI-Generated",
    "musicgen_hf_api": "AI-Generated",
    "musicgen_local":  "AI-Generated",
    "musicgen_gradio": "AI-Generated",
    "acestep_v2bgm":   "Neural Producer",
    "acestep_accomp":  "Neural Producer",
}

_VIBE_MESSAGES = {
    "trap_dark":       ("Dark & brooding — pure midnight energy.", "مظلم وآسر — طاقة منتصف الليل."),
    "trap_melodic":    ("Melodic trap with emotional depth.", "تراب لحني بعمق عاطفي."),
    "hiphop_boom_bap": ("Classic boom bap — raw and authentic.", "بوم باب كلاسيكي — أصيل وقوي."),
    "hiphop_modern":   ("Modern hip-hop with punch and melody.", "هيب هوب عصري بقوة ولحن."),
    "rnb_smooth":      ("Smooth R&B vibes — silky and soulful.", "أجواء R&B ناعمة وروحانية."),
    "rnb_neo_soul":    ("Neo-soul grooves — warm and intimate.", "نيو سول دافئ وحميمي."),
    "pop_bright":      ("Bright pop energy — radio-ready.", "طاقة بوب مشرقة — جاهزة للراديو."),
    "afrobeats":       ("Afrobeats heat — infectious and vibrant.", "أفروبيتس — حيوي ومعدٍ."),
    "dancehall":       ("Dancehall riddim — island energy.", "ريدم دانسهول — طاقة الجزر."),
    "lofi_chill":      ("Lo-fi chill — relax and flow.", "لو-فاي هادئ — استرخِ وانسَب."),
    "soul_ballad":     ("Soulful ballad — emotional and timeless.", "بالاد روحاني — عاطفي وخالد."),
    "drill":           ("Hard-hitting drill — aggressive and cold.", "دريل صلب — قوي وبارد."),
    "uk_drill":        ("UK drill — dark, sparse, and gritty.", "دريل بريطاني — مظلم وعميق."),
    "phonk":           ("Memphis phonk — gritty and hypnotic.", "فونك ممفيس — نشوة غامضة."),
    "reggaeton":       ("Reggaeton heat — unstoppable rhythm.", "ريغاتون — إيقاع لا يوقف."),
    "amapiano":        ("Amapiano log drums — deep and moving.", "أماپيانو — أعمق وأكثر تحريكاً."),
    "musicgen_ai":     ("AI beat generated from your voice.", "إيقاع AI مُولَّد من صوتك."),
    "musicgen_hf_api": ("AI beat generated from your voice.", "إيقاع AI مُولَّد من صوتك."),
    "musicgen_local":  ("AI beat conditioned on your vocal melody.", "إيقاع AI مُصمَّم على لحن صوتك."),
    "musicgen_gradio": ("AI beat conditioned on your vocal melody.", "إيقاع AI مُصمَّم على لحن صوتك."),
    "acestep_v2bgm":   ("A full production composed around your exact performance.",
                        "إنتاج كامل مُؤلَّف حول أدائك بالضبط."),
    "acestep_accomp":  ("A neural band composed this accompaniment for your exact performance.",
                        "فرقة عصبية ألّفت هذا المصاحب لأدائك بالضبط."),
}


# Keys match performance_map's energy_arc vocabulary:
# builds | fades | peaks_middle | steady | dynamic
_ARC_EN = {
    "builds":       "Your energy built toward the end, so the arrangement builds with you",
    "fades":        "You opened strong, so the beat leads with its peak and eases out with you",
    "peaks_middle": "The chorus lands right where your performance peaked",
    "steady":       "Your delivery stayed steady, so the groove stays locked",
    "dynamic":      "Your dynamics move a lot, so the arrangement rises and falls with you",
}
_ARC_AR = {
    "builds":       "طاقتك تصاعدت نحو النهاية، فالتوزيع يتصاعد معك",
    "fades":        "بدأت بقوة، فالإيقاع يفتتح بذروته ويهدأ معك",
    "peaks_middle": "الكورس يقع تماماً حيث بلغ أداؤك ذروته",
    "steady":       "أداؤك كان ثابتاً، فالإيقاع يبقى متماسكاً",
    "dynamic":      "ديناميكيتك متحركة، فالتوزيع يعلو ويهدأ معك",
}


def _producer_note(analysis: dict, genre_label: str, tempo: int,
                   neural: bool = False) -> tuple[str, str]:
    """The honest producer's note (Three Laws #2): a short explanation composed
    from what the AI actually measured in THIS take — never a canned slogan.
    Only states what really happened; sections without data are omitted.
    `neural` cuts are composed by the neural band listening to the take, so
    the mechanical follows-your-chords claim (true for the synth) is not made."""
    dec = analysis.get("_decisions") or {}
    en_parts: list[str] = []
    ar_parts: list[str] = []

    h = dec.get("harmony")
    if h and neural:
        en_parts.append(f"I heard you in {h['key']} {h['mode']} at {tempo} BPM "
                        f"and the band composed its accompaniment around your take.")
        ar_parts.append(f"سمعتك في مقام {h['key']} {h['mode']} على {tempo} BPM "
                        f"والفرقة ألّفت مصاحبتها حول أدائك.")
    elif h:
        en = (f"I heard you in {h['key']} {h['mode']} at {tempo} BPM — "
              f"the bass and chords follow the progression you sang")
        ar = (f"سمعتك في مقام {h['key']} {h['mode']} على {tempo} BPM — "
              f"الباص والكوردات تتبع تسلسلك الغنائي")
        if h.get("melody_notes"):
            en += f", and the lead echoes {h['melody_notes']} notes of your own melody"
            ar += f"، واللحن الرئيسي يردّد {h['melody_notes']} نغمة من لحنك"
        en_parts.append(en + ".")
        ar_parts.append(ar + ".")
    else:
        en_parts.append(f"Built at {tempo} BPM around the feel of your voice.")
        ar_parts.append(f"مبني على {tempo} BPM حول إحساس صوتك.")

    # Arc/pause reactions are mechanical claims about the SYNTH arrangement —
    # only state them when the synth actually did those things.
    p = dec.get("performance")
    if p and not neural:
        arc = str(p.get("arc") or "")
        if arc in _ARC_EN:
            en_parts.append(_ARC_EN[arc] + ".")
            ar_parts.append(_ARC_AR[arc] + ".")
        if p.get("pauses"):
            en_parts.append(f"I answered {p['pauses']} of your breaths with fills and drops.")
            ar_parts.append(f"جاوبت على {p['pauses']} من وقفاتك بفلترات وسكتات.")

    return " ".join(en_parts), " ".join(ar_parts)


def get_supabase():
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


def get_r2():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{settings.r2_account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        region_name="auto",
    )


class GenerateBeatRequest(BaseModel):
    style_hint: str = ""
    # Branch a new exploration from a past favourite cut (its lineage parent).
    branch_from: int | None = None


def _load_cuts(project: dict) -> list:
    """Read the project's Producer Cuts. Falls back to reconstructing a minimal
    history from beat_genre_history so AI memory still works before migration 007
    is applied (the producer_cuts column then simply isn't present)."""
    cuts = project.get("producer_cuts")
    if isinstance(cuts, str):
        try:
            cuts = json.loads(cuts)
        except Exception:
            cuts = None
    if isinstance(cuts, list):
        return cuts
    history = project.get("beat_genre_history")
    if isinstance(history, str):
        try:
            history = json.loads(history)
        except Exception:
            history = []
    return [{"cut": i + 1, "genre": g, "parent_cut": None}
            for i, g in enumerate(history or [])]


def _artist_taste(supabase, user_id: str) -> list:
    """The artist's learned taste, ranked strongest-first, from every project:
    accepted ≫ favourited ≈ branched-from, eroded by cuts passed over (skips).
    Per-project lists are kept separate so branch links and each project's
    newest cut are visible to the skip signal. Best-effort (returns [] on any
    failure, e.g. before migration 008)."""
    try:
        res = (supabase.table("projects").select("producer_cuts")
               .eq("user_id", user_id).execute())
    except Exception:
        return []
    project_lists: list = []
    for row in (res.data or []):
        pcs = row.get("producer_cuts")
        if isinstance(pcs, str):
            try:
                pcs = json.loads(pcs)
            except Exception:
                pcs = []
        if pcs:
            project_lists.append(pcs)
    return pc.ranked_taste(pc.taste_weights(project_lists))


@router.post("/projects/{project_id}/generate-beat")
async def generate_beat_endpoint(
    project_id: str,
    body: GenerateBeatRequest,
    user: dict = Depends(get_current_user),
):
    """Submit a beat-generation JOB and return immediately.

    A neural cut takes 1-2 minutes; a single long HTTP request dies to
    browser timeouts and refreshes (it did — repeatedly). The client polls
    GET /projects/{id}/jobs/{job_id} until done; an active job can also be
    re-attached after a page refresh via GET /projects/{id}/jobs/active.
    """
    supabase = get_supabase()
    user_id  = user["user_id"]

    # Validate the project up-front so a bad request fails fast, not in-job.
    proj_result = (
        supabase.table("projects").select("id")
        .eq("id", project_id).eq("user_id", user_id).single().execute()
    )
    if not proj_result.data:
        raise HTTPException(status_code=404, detail="Project not found")

    # One generation at a time per project — double-clicks and impatient
    # retries attach to the job already running instead of stacking GPU work.
    existing = jobs.latest_active_job(supabase, project_id, user_id, "beat")
    if existing:
        return {"job_id": existing["id"], "status": existing["status"], "existing": True}

    job_id = jobs.create_job(supabase, user_id, project_id, "beat",
                             payload={"style_hint": body.style_hint,
                                      "branch_from": body.branch_from})
    asyncio.create_task(_run_beat_job(job_id, project_id, user_id, body))
    return {"job_id": job_id, "status": "queued", "existing": False}


async def _run_beat_job(job_id: str, project_id: str, user_id: str,
                        body: GenerateBeatRequest) -> None:
    """In-process job executor. Later, a remote GPU worker can claim queued
    rows instead — the client contract (poll the job) stays identical."""
    supabase = get_supabase()
    try:
        jobs.mark_running(supabase, job_id)
        result = await _execute_generate(project_id, user_id, body)
        jobs.mark_done(supabase, job_id, result)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {"message_en": str(exc.detail)}
        jobs.mark_failed(supabase, job_id, detail)
    except Exception:
        logger.exception("beat job crashed job_id=%s project_id=%s", job_id, project_id)
        jobs.mark_failed(supabase, job_id, {
            "reason": "generation_failed",
            "message_en": "Beat generation failed. Please try again.",
            "message_ar": "فشل توليد الإيقاع. يرجى المحاولة مرة أخرى.",
        })


@router.get("/projects/{project_id}/jobs/active")
def get_active_job(project_id: str, kind: str = "beat",
                   user: dict = Depends(get_current_user)):
    """Newest queued/running job for this project — lets the UI resume
    polling after a refresh instead of losing a generation in flight."""
    supabase = get_supabase()
    job = jobs.latest_active_job(supabase, project_id, user["user_id"], kind)
    if not job:
        return {"job_id": None}
    return {"job_id": job["id"], "status": job["status"]}


@router.get("/projects/{project_id}/jobs/{job_id}")
def get_job_status(project_id: str, job_id: str,
                   user: dict = Depends(get_current_user)):
    supabase = get_supabase()
    job = jobs.get_job(supabase, job_id, user["user_id"])
    if not job or job.get("project_id") != project_id:
        raise HTTPException(status_code=404, detail="Job not found")
    out = {"job_id": job["id"], "status": job["status"]}
    if job["status"] == "done":
        out["result"] = job.get("result")
    elif job["status"] == "failed":
        out["error"] = job.get("error")
    return out


async def _execute_generate(project_id: str, user_id: str,
                            body: GenerateBeatRequest) -> dict:
    supabase = get_supabase()

    proj_result = (
        supabase.table("projects")
        .select("*")
        .eq("id", project_id)
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    if not proj_result.data:
        raise HTTPException(status_code=404, detail="Project not found")

    project = proj_result.data
    # Unlimited creative session — the artist keeps creating Producer Cuts and
    # decides when the song is finished. No attempt cap.
    cuts = _load_cuts(project)

    # ── Deep vocal analysis ───────────────────────────────────────────────────
    _FALLBACK_MOOD = {
        "tempo": 90.0, "rms": 0.18, "centroid": 1500.0, "density": 2.0,
        "key": "C", "mode": "major", "valence": 0.5, "emotion": "smooth",
        "vocal_style": "rhythmic", "swing_ratio": 0.5, "overall_rms": 0.18,
    }
    vocal_mood      = dict(_FALLBACK_MOOD)
    processed_bytes = b""
    processed_key   = project.get("processed_vocal_key")

    if processed_key:
        try:
            obj             = get_r2().get_object(Bucket=settings.r2_bucket_name, Key=processed_key)
            processed_bytes = obj["Body"].read()
            logger.info("[beat] loaded processed vocal project=%s bytes=%d", project_id, len(processed_bytes))
        except Exception as exc:
            # BUG WAS HERE: bare `pass` silently discarded this exception,
            # leaving vocal_mood = hardcoded defaults and processed_bytes = b"".
            # Every subsequent call with the same attempt number produced an
            # identical beat because hash(str(defaults)) is constant.
            logger.error(
                "[beat] R2 read or audio analysis failed for project=%s "
                "key=%s — falling back to defaults. Error: %s",
                project_id, processed_key, exc,
            )

    # Voice profile supplements analysis (key/tone from onboarding recordings)
    vp            = supabase.table("voice_profiles").select("*").eq("user_id", user_id).maybe_single().execute()
    voice_profile = vp.data or {}

    # BUG WAS HERE: setdefault() only sets a key when it is ABSENT from the
    # dict. analyze_vocal_mood() always returns all these keys, so setdefault
    # never fired — voice_profile data was silently ignored every time.
    # Fix: direct assignment so voice_profile (derived from 3-5 onboarding
    # recordings) actually overrides single-recording estimates for stable features.
    for field in ("key", "mode", "valence", "emotion", "vocal_style", "swing_ratio"):
        if field in voice_profile and voice_profile[field] is not None and voice_profile[field] != "":
            vocal_mood[field] = voice_profile[field]

    # ── AI memory: avoid repeating ideas; branch explores around a favourite ──
    branch_from = body.branch_from if body.branch_from else None
    memory = pc.select_memory(cuts, branch_from=branch_from)
    # Five elite producers, five interpretations: the first five cuts each take a
    # distinct direction (the harmony still follows the vocal); then free explore.
    # The AI learns the artist's taste from accepted + favourited cuts across all
    # projects — leading the spread with their favoured lane and revisiting it.
    taste = _artist_taste(supabase, user_id)
    force_genre = pc.choose_force_genre(cuts, vocal_mood, branch_from=branch_from, taste=taste)

    # The chosen vocal style steers the beat toward a matching genre pool
    # (e.g. Rap → hip-hop/trap, R&B → smooth soul) so the beat fits the performance.
    style_bias = style_genre_bias(project.get("autotune_level"))

    # ── Generate the Producer Cut ─────────────────────────────────────────────
    cut_num    = pc.next_cut_number(cuts)
    cut_label  = pc.build_label(cuts, branch_from)
    _gen_start = time.time()
    try:
        with time_ms("beat_generation_duration_ms"):
            beat_bytes, genre_used, used_analysis = await generate_beat_from_vocal(
                processed_bytes=processed_bytes,
                voice_profile=voice_profile,
                vocal_mood=vocal_mood,
                style_hint=body.style_hint,
                hf_api_key=settings.hf_api_key,
                exclude_genres=memory["exclude_genres"],
                attempt=cut_num,
                previous_genre=memory["previous_genre"],
                style_bias=style_bias,
                force_genre=force_genre,
            )
    except Exception as exc:
        increment("failed_generations")
        duration_ms = int((time.time() - _gen_start) * 1000)
        record_beat_event(
            supabase,
            project_id=project_id,
            tier_used="failed",
            tier_attempts=[],
            duration_ms=duration_ms,
            analysis=vocal_mood,
            genre="failed",
            success=False,
            failure_reason=f"{type(exc).__name__}: {str(exc)[:200]}",
        )
        logger.exception("beat generation failed project_id=%s", project_id)
        raise HTTPException(status_code=500, detail={
            "reason": "generation_failed",
            "message_en": "Beat generation failed. Please try again.",
            "message_ar": "فشل توليد الإيقاع. يرجى المحاولة مرة أخرى.",
            **({"debug": traceback.format_exc()} if settings.environment != "production" else {}),
        })

    # ── Upload the new cut — NEVER delete a past cut (every idea is kept) ──────
    beat_key = f"projects/{project_id}/cut_{cut_num}.wav"
    try:
        upload_file(beat_key, beat_bytes, content_type="audio/wav")
    except Exception:
        raise HTTPException(status_code=500, detail={
            "reason": "upload_failed",
            "message_en": "The cut was created but couldn't be saved. Please try again.",
            "message_ar": "تم إنشاء النسخة لكن تعذّر حفظها. يرجى المحاولة مرة أخرى.",
        })

    # ── Record the Producer Cut ───────────────────────────────────────────────
    beat_score_data = used_analysis.get("beat_score") or {}
    mixer_analysis  = {
        k: used_analysis.get(k)
        for k in (
            "tempo", "centroid", "spectral_centroid", "key", "mode",
            "emotion", "valence", "swing_ratio", "density",
            "vocal_style", "overall_rms",
        )
        if used_analysis.get(k) is not None
    }
    key         = used_analysis.get("key", "C")
    mode        = used_analysis.get("mode", "major")
    tempo       = int(used_analysis.get("tempo", 90))
    genre_label = _GENRE_LABELS.get(genre_used, genre_used.replace("_", " ").title())

    cut_record = pc.make_cut_record(
        cut=cut_num, label=cut_label, beat_key=beat_key, genre=genre_used,
        genre_label=genre_label, key=f"{key} {mode}", tempo=tempo,
        emotion=used_analysis.get("emotion", "smooth"),
        score=beat_score_data.get("total") if beat_score_data else None,
        parent_cut=branch_from, created_at=datetime.now(timezone.utc).isoformat(),
    )
    # The honest producer's note — what the AI actually heard and decided
    # (Three Laws #2). Persisted on the cut so it survives page reloads.
    note_en, note_ar = _producer_note(used_analysis, genre_label, tempo,
                                      neural=genre_used.startswith("acestep"))
    cut_record["note_en"] = note_en
    cut_record["note_ar"] = note_ar
    cuts = cuts + [cut_record]

    # Core fields — these columns always exist, this must not fail. beat_attempts
    # is the total cut count; beat_key points at the latest cut (back-compat).
    supabase.table("projects").update({
        "beat_attempts": cut_num,
        "beat_key":      beat_key,
        "status":        "beat_generation",
        "tempo_bpm":     tempo,
    }).eq("id", project_id).eq("user_id", user_id).execute()

    # rhythmic_metadata is OPTIONAL (migration 009) — writing it must never
    # kill the cut. Before 2026-07-12 it sat in the core update above and a
    # missing column silently 500'd EVERY beat generation.
    try:
        supabase.table("projects").update({
            "rhythmic_metadata": {
                k: used_analysis.get(k)
                for k in ("tempo", "swing_ratio", "density", "vocal_style")
                if used_analysis.get(k) is not None
            },
        }).eq("id", project_id).eq("user_id", user_id).execute()
    except Exception:
        pass

    # Producer Cuts history — a proper JSONB array (round-trip verified). Isolated
    # so a missing column (pre-migration-008) never crashes the request.
    try:
        supabase.table("projects").update(
            {"producer_cuts": cuts}).eq("id", project_id).eq("user_id", user_id).execute()
    except Exception:
        pass

    # Optional analysis columns — stringified, each isolated (may not exist).
    for col, val in [
        ("vocal_analysis", json.dumps(mixer_analysis)),
        ("beat_scores",    json.dumps(beat_score_data)),
    ]:
        try:
            supabase.table("projects").update({col: val}).eq("id", project_id).eq("user_id", user_id).execute()
        except Exception:
            pass

    record_beat_event(
        supabase,
        project_id=project_id,
        tier_used=genre_used,
        tier_attempts=used_analysis.pop("_tier_attempts", []),
        duration_ms=int((time.time() - _gen_start) * 1000),
        analysis=used_analysis,
        genre=genre_used,
        success=True,
    )
    increment("successful_generations")

    beat_url = generate_signed_url(beat_key, expires_in=BEAT_URL_TTL)
    vibe_en, vibe_ar = _VIBE_MESSAGES.get(genre_used, (
        f"{genre_label} — built around your voice.",
        f"{genre_label} — مبني حول صوتك.",
    ))
    # Vibe opener + the real explanation of what the AI heard and did.
    vibe_en = f"{vibe_en} {note_en}".strip()
    vibe_ar = f"{vibe_ar} {note_ar}".strip()

    return {
        "beat_url":     beat_url,
        "cut":          cut_num,
        "cut_label":    cut_label,
        "parent_cut":   branch_from,
        "total_cuts":   len(cuts),
        "unlimited":    True,
        # back-compat with existing frontend keys
        "attempt":      cut_num,
        "genre":        genre_label,
        "key":          f"{key} {mode}",
        "tempo_bpm":    tempo,
        "emotion":      used_analysis.get("emotion", "smooth"),
        "beat_score":   beat_score_data.get("total") if beat_score_data else None,
        "message_en":   vibe_en,
        "message_ar":   vibe_ar,
    }


@router.post("/projects/{project_id}/accept-beat")
def accept_beat(project_id: str, user: dict = Depends(get_current_user)):
    supabase = get_supabase()
    result = (
        supabase.table("projects")
        .select("beat_key, producer_cuts")
        .eq("id", project_id)
        .eq("user_id", user["user_id"])
        .single()
        .execute()
    )
    if not result.data or not result.data.get("beat_key"):
        raise HTTPException(status_code=400, detail="No beat to accept")

    # Mark the accepted cut — finishing a song with a cut is the STRONGEST taste
    # signal the AI learns from (weighted above favourites). The current beat_key
    # is the chosen cut (a restore can have changed it). Isolated/best-effort.
    accepted_label = None
    accepted_key = result.data["beat_key"]
    cuts = _load_cuts(result.data)
    if cuts:
        for c in cuts:
            if c.get("beat_key") == accepted_key:
                c["accepted"] = True
                accepted_label = c.get("label")
            elif c.get("accepted"):
                c["accepted"] = False        # only one accepted cut at a time
        try:
            supabase.table("projects").update(
                {"producer_cuts": cuts}).eq("id", project_id).eq("user_id", user["user_id"]).execute()
        except Exception:
            pass

    supabase.table("projects").update({"status": "coaching"}).eq("id", project_id).eq("user_id", user["user_id"]).execute()
    return {"status": "accepted", "next": "coaching", "accepted_cut": accepted_label}


@router.get("/projects/{project_id}/beat-url")
def get_beat_url(project_id: str, user: dict = Depends(get_current_user)):
    supabase = get_supabase()
    result = (
        supabase.table("projects")
        .select("beat_key, beat_attempts, beat_genre_history")
        .eq("id", project_id)
        .eq("user_id", user["user_id"])
        .single()
        .execute()
    )
    if not result.data or not result.data.get("beat_key"):
        raise HTTPException(status_code=404, detail="No beat found")

    url           = generate_signed_url(result.data["beat_key"], expires_in=BEAT_URL_TTL)
    genre_history = []
    raw           = result.data.get("beat_genre_history")
    if raw:
        try:
            genre_history = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            pass

    last_genre = genre_history[-1] if genre_history else None
    return {
        "beat_url":      url,
        "beat_attempts": result.data["beat_attempts"],
        "last_genre":    _GENRE_LABELS.get(last_genre, last_genre) if last_genre else None,
    }


def _fetch_project_cuts(supabase, project_id: str, user_id: str) -> tuple[dict, list]:
    res = (
        supabase.table("projects").select("*")
        .eq("id", project_id).eq("user_id", user_id).single().execute()
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Project not found")
    return res.data, _load_cuts(res.data)


@router.get("/projects/{project_id}/cuts")
def list_cuts(project_id: str, user: dict = Depends(get_current_user)):
    """The full Producer Cuts history — every cut ever made for this project,
    each with a fresh signed URL so the artist can replay, compare and restore.
    Nothing good is ever lost."""
    supabase = get_supabase()
    _, cuts = _fetch_project_cuts(supabase, project_id, user["user_id"])
    out = []
    for c in cuts:
        url = None
        if c.get("beat_key"):
            try:
                url = generate_signed_url(c["beat_key"], expires_in=BEAT_URL_TTL)
            except Exception:
                url = None
        out.append(pc.public_cut(c, beat_url=url))
    return {"cuts": out, "total": len(out),
            "favorites": [pc.public_cut(c) for c in cuts if c.get("favorite")]}


@router.post("/projects/{project_id}/cuts/{cut}/favorite")
def toggle_favorite(project_id: str, cut: int, user: dict = Depends(get_current_user)):
    """Save (or unsave) a favourite cut. Favourites are what the AI learns the
    artist's taste from over time."""
    supabase = get_supabase()
    _, cuts = _fetch_project_cuts(supabase, project_id, user["user_id"])
    target = next((c for c in cuts if c.get("cut") == cut), None)
    if target is None:
        raise HTTPException(status_code=404, detail="Cut not found")
    target["favorite"] = not target.get("favorite", False)
    try:
        supabase.table("projects").update(
            {"producer_cuts": cuts}).eq("id", project_id).eq("user_id", user["user_id"]).execute()
    except Exception:
        raise HTTPException(status_code=503, detail={
            "reason": "cuts_unavailable",
            "message_en": "Favourites need migration 008 (producer_cuts) applied.",
        })
    return {"cut": cut, "favorite": target["favorite"]}


@router.post("/projects/{project_id}/cuts/{cut}/restore")
def restore_cut(project_id: str, cut: int, user: dict = Depends(get_current_user)):
    """Make a past cut the current one (for accept / coaching / mix) without
    losing any other cut — the artist can branch from or restore any idea."""
    supabase = get_supabase()
    _, cuts = _fetch_project_cuts(supabase, project_id, user["user_id"])
    target = next((c for c in cuts if c.get("cut") == cut), None)
    if target is None or not target.get("beat_key"):
        raise HTTPException(status_code=404, detail="Cut not found")
    supabase.table("projects").update(
        {"beat_key": target["beat_key"]}).eq("id", project_id).eq("user_id", user["user_id"]).execute()
    url = None
    try:
        url = generate_signed_url(target["beat_key"], expires_in=BEAT_URL_TTL)
    except Exception:
        pass
    return {"restored": cut, "label": target.get("label"), "beat_url": url}
