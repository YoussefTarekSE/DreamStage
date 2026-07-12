import asyncio
import json
import logging
import traceback
import boto3
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from supabase import create_client
from ..auth import get_current_user
from ..config import settings
from ..errors import safe_error_detail
from ..storage import upload_file, delete_file, generate_signed_url
from ..services.mixer import create_final_mix
from ..services.metrics import increment, time_ms

router = APIRouter(prefix="/studio", tags=["mix"])
logger = logging.getLogger(__name__)

DOWNLOAD_URL_TTL = 86400  # 24-hour signed URLs


class FeedbackRequest(BaseModel):
    beat_quality: int = Field(..., ge=1, le=5)
    vocal_preservation: int = Field(..., ge=1, le=5)
    overall_satisfaction: int = Field(..., ge=1, le=5)


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


def _object_exists(r2, key: str | None) -> bool:
    if not key:
        return False
    try:
        r2.head_object(Bucket=settings.r2_bucket_name, Key=key)
        return True
    except Exception:
        return False


@router.post("/projects/{project_id}/mix")
async def create_mix(project_id: str, user: dict = Depends(get_current_user)):
    """
    Final step: merge processed vocal + beat into a professional stereo master.
    Exports MP3 320kbps + WAV 24-bit. Deletes all intermediary files.
    """
    supabase = get_supabase()
    user_id = user["user_id"]

    proj = (
        supabase.table("projects")
        .select("*")
        .eq("id", project_id)
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    if not proj.data:
        raise HTTPException(status_code=404, detail="Project not found")

    project = proj.data
    r2 = get_r2()

    # Return cached mix if already done
    if project.get("final_mp3_key") and _object_exists(r2, project.get("final_mp3_key")):
        mp3_url = generate_signed_url(project["final_mp3_key"], expires_in=DOWNLOAD_URL_TTL)
        wav_url = generate_signed_url(project["final_wav_key"], expires_in=DOWNLOAD_URL_TTL) if _object_exists(r2, project.get("final_wav_key")) else None
        return {
            "mp3_url": mp3_url,
            "wav_url": wav_url,
            "message_en": "Your song is ready. Download it below.",
            "message_ar": "أغنيتك جاهزة. حمّلها أدناه.",
        }

    # Download processed vocal
    vocal_key = project.get("processed_vocal_key")
    if not vocal_key:
        raise HTTPException(status_code=400, detail={
            "reason": "no_vocal",
            "message_en": "No processed vocal found. Please record your vocal first.",
            "message_ar": "لم يتم العثور على صوت معالج. يرجى تسجيل صوتك أولاً.",
        })

    try:
        vocal_obj = r2.get_object(Bucket=settings.r2_bucket_name, Key=vocal_key)
        vocal_bytes = vocal_obj["Body"].read()
    except Exception as e:
        logger.exception("processed vocal load failed project_id=%s", project_id)
        raise HTTPException(status_code=500, detail={
            "reason": "vocal_load_failed",
            "message_en": "Could not load your vocal. Please try again.",
            "message_ar": "تعذّر تحميل صوتك. يرجى المحاولة مرة أخرى.",
        })

    # Download beat
    beat_key = project.get("beat_key")
    if not beat_key:
        raise HTTPException(status_code=400, detail={
            "reason": "no_beat",
            "message_en": "No beat found. Please generate a beat first.",
            "message_ar": "لم يتم العثور على إيقاع. يرجى توليد إيقاع أولاً.",
        })

    try:
        beat_obj = r2.get_object(Bucket=settings.r2_bucket_name, Key=beat_key)
        beat_bytes = beat_obj["Body"].read()
    except Exception as e:
        logger.exception("beat load failed project_id=%s", project_id)
        raise HTTPException(status_code=500, detail={
            "reason": "beat_load_failed",
            "message_en": "Could not load your beat. Please try again.",
            "message_ar": "تعذّر تحميل الإيقاع. يرجى المحاولة مرة أخرى.",
        })

    # Create the final mix — pass last-used genre for reverb character
    genre_hint = "hiphop_modern"
    raw_history = project.get("beat_genre_history")
    if raw_history:
        try:
            history = json.loads(raw_history) if isinstance(raw_history, str) else raw_history
            if history:
                genre_hint = history[-1]
        except Exception:
            pass

    # Extract vocal analysis from project metadata for dynamic EQ carve
    vocal_analysis = None
    try:
        analysis_raw = project.get("vocal_analysis")
        if analysis_raw:
            vocal_analysis = json.loads(analysis_raw) if isinstance(analysis_raw, str) else analysis_raw
    except Exception:
        pass
    if not isinstance(vocal_analysis, dict):
        vocal_analysis = {}
    if not vocal_analysis.get("tempo") and project.get("tempo_bpm"):
        vocal_analysis["tempo"] = project.get("tempo_bpm")
    if not vocal_analysis.get("tempo"):
        cuts = project.get("producer_cuts") or []
        try:
            cuts = json.loads(cuts) if isinstance(cuts, str) else cuts
            current = next((c for c in cuts if c.get("beat_key") == beat_key), None)
            if current and current.get("tempo"):
                vocal_analysis["tempo"] = current["tempo"]
        except Exception:
            pass

    # A neural-producer cut (ACE-Step Vocal2BGM) is already a FINISHED SONG
    # with the vocal inside — layering the vocal again would double it.
    # Master the file as-is instead of remixing.
    neural_cut = False
    try:
        _cuts = project.get("producer_cuts") or []
        _cuts = json.loads(_cuts) if isinstance(_cuts, str) else _cuts
        _cur = next((c for c in _cuts if c.get("beat_key") == beat_key), None)
        neural_cut = bool(_cur and _cur.get("genre") == "acestep_v2bgm")
    except Exception:
        pass

    try:
        with time_ms("mixing_duration_ms"):
            if neural_cut:
                from ..services.mixer import master_song_bytes
                mp3_bytes, wav_bytes = await asyncio.to_thread(
                    master_song_bytes, beat_bytes)
            else:
                mp3_bytes, wav_bytes = await asyncio.to_thread(
                    create_final_mix,
                    vocal_bytes,
                    beat_bytes,
                    genre=genre_hint,
                    vocal_analysis=vocal_analysis,
                )
    except Exception as e:
        increment("failed_mixes")
        logger.exception("mix render failed project_id=%s", project_id)
        raise HTTPException(status_code=500, detail={
            "reason": "mix_failed",
            "message_en": "Mixing failed. Please try again.",
            "message_ar": "فشل المزج. يرجى المحاولة مرة أخرى.",
            **({"debug": traceback.format_exc()} if settings.environment != "production" else {}),
        })

    # Upload final files
    mp3_key = f"projects/{project_id}/final.mp3"
    wav_key = f"projects/{project_id}/final.wav"

    try:
        upload_file(mp3_key, mp3_bytes, content_type="audio/mpeg")
        upload_file(wav_key, wav_bytes, content_type="audio/wav")
    except Exception as e:
        logger.exception("final mix upload failed project_id=%s", project_id)
        raise HTTPException(status_code=500, detail={
            "reason": "upload_failed",
            "message_en": "Mix was created but couldn't be saved. Please try again.",
            "message_ar": "تم إنشاء المزج لكن تعذّر حفظه. يرجى المحاولة مرة أخرى.",
        })

    # NOTE: intermediates are KEPT (pre-Producer-Cuts code deleted the vocal
    # and the accepted cut here). Deleting them broke the creative-session
    # doctrine — the cut history lost its audio, re-mixing became impossible,
    # and "no good idea is ever lost" was a lie. Storage cost is pennies.

    # Mark project as completed (vocal/beat keys stay valid for re-mix/branch)
    supabase.table("projects").update({
        "status": "completed",
        "final_mp3_key": mp3_key,
        "final_wav_key": wav_key,
    }).eq("id", project_id).eq("user_id", user_id).execute()
    increment("successful_mixes")

    mp3_url = generate_signed_url(mp3_key, expires_in=DOWNLOAD_URL_TTL)
    wav_url = generate_signed_url(wav_key, expires_in=DOWNLOAD_URL_TTL)

    return {
        "mp3_url": mp3_url,
        "wav_url": wav_url,
        "message_en": "Your song is ready. Download it — this is yours.",
        "message_ar": "أغنيتك جاهزة. حمّلها — هذه أغنيتك.",
    }


@router.get("/projects/{project_id}/download-urls")
def get_download_urls(project_id: str, user: dict = Depends(get_current_user)):
    supabase = get_supabase()
    result = (
        supabase.table("projects")
        .select("final_mp3_key, final_wav_key, status, name")
        .eq("id", project_id)
        .eq("user_id", user["user_id"])
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Project not found")

    project = result.data
    if project["status"] != "completed":
        raise HTTPException(status_code=400, detail="Project not yet completed")

    return {
        "mp3_url": generate_signed_url(project["final_mp3_key"], expires_in=DOWNLOAD_URL_TTL) if project.get("final_mp3_key") else None,
        "wav_url": generate_signed_url(project["final_wav_key"], expires_in=DOWNLOAD_URL_TTL) if project.get("final_wav_key") else None,
        "name": project["name"],
    }


@router.post("/projects/{project_id}/feedback")
def submit_project_feedback(
    project_id: str,
    body: FeedbackRequest,
    user: dict = Depends(get_current_user),
):
    supabase = get_supabase()
    result = (
        supabase.table("projects")
        .select("id, status")
        .eq("id", project_id)
        .eq("user_id", user["user_id"])
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Project not found")
    if result.data.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Project not yet completed")

    try:
        supabase.table("project_feedback").insert({
            "project_id": project_id,
            "user_id": user["user_id"],
            "beat_quality": body.beat_quality,
            "vocal_preservation": body.vocal_preservation,
            "overall_satisfaction": body.overall_satisfaction,
        }).execute()
    except Exception as exc:
        logger.exception("project feedback save failed project_id=%s", project_id)
        raise HTTPException(status_code=503, detail=safe_error_detail(
            reason="feedback_unavailable",
            message_en="Couldn't save feedback right now. Please try again later.",
            message_ar="Couldn't save feedback right now. Please try again later.",
            debug=f"{type(exc).__name__}: {exc}",
        ))

    return {"saved": True}
