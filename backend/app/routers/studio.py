import asyncio
import logging
import traceback
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import Annotated
from supabase import create_client
from ..auth import get_current_user
from ..config import settings
from ..errors import safe_error_detail
from ..storage import upload_file, delete_file, generate_signed_url
from ..services.audio_quality import check_quality
from ..services.vocal_processor import process_vocal
from ..services.metrics import increment, time_ms

router = APIRouter(prefix="/studio", tags=["studio"])
logger = logging.getLogger(__name__)

MAX_VOCAL_SIZE = 50 * 1024 * 1024   # 50 MB
PROCESSED_URL_TTL = 3600             # 1 hour signed URL


def get_supabase():
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


@router.post("/process-vocal")
async def process_vocal_endpoint(
    file: Annotated[UploadFile, File()],
    project_name: Annotated[str, Form()] = "Untitled Song",
    autotune_level: Annotated[str, Form()] = "subtle",
    language: Annotated[str, Form()] = "en",
    user: dict = Depends(get_current_user),
):
    """
    1. Create project in DB
    2. Run vocal processing pipeline (noise reduction, pitch correction, autotune)
    3. Store processed vocal in R2
    4. Return signed URL for playback
    """
    user_id = user["user_id"]

    # Validate vocal style (8 user-facing styles + legacy "modern")
    valid_levels = ("natural", "subtle", "modern_pop", "rnb", "rap",
                    "melodic", "heavy", "none", "modern")
    if autotune_level not in valid_levels:
        autotune_level = "subtle"

    # Read audio
    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")
    if len(audio_bytes) > MAX_VOCAL_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")

    quality = await asyncio.to_thread(check_quality, audio_bytes)
    if not quality["ok"]:
        raise HTTPException(status_code=422, detail={
            "reason": quality["reason"],
            "message_en": quality["message_en"],
            "message_ar": quality["message_ar"],
        })

    supabase = get_supabase()

    # Create project in DB
    try:
        project_result = (
            supabase.table("projects")
            .insert({
                "user_id": user_id,
                "name": project_name.strip() or "Untitled Song",
                "status": "processing_vocal",
                "language": language,
                "autotune_level": autotune_level,
            })
            .execute()
        )
        project = project_result.data[0]
        project_id = project["id"]
    except Exception as e:
        logger.exception("project creation failed user_id=%s", user_id)
        raise HTTPException(status_code=500, detail={
            "reason": "db_error",
            "message_en": "Couldn't create your project. Please try again.",
            "message_ar": "تعذّر إنشاء مشروعك. يرجى المحاولة مرة أخرى.",
            **({"debug": str(e)} if settings.environment != "production" else {}),
        })

    # Upload raw vocal to R2 temporarily
    raw_key = f"projects/{project_id}/raw_vocal.webm"
    try:
        upload_file(raw_key, audio_bytes, content_type="audio/webm")
    except Exception as e:
        logger.exception("raw vocal upload failed project_id=%s", project_id)
        raise HTTPException(status_code=500, detail={
            "reason": "upload_error",
            "message_en": "Couldn't upload your recording. Please try again.",
            "message_ar": "تعذّر رفع تسجيلك. يرجى المحاولة مرة أخرى.",
        })

    # Run processing pipeline
    try:
        with time_ms("vocal_processing_duration_ms"):
            processed_bytes = await asyncio.to_thread(
                process_vocal, audio_bytes, autotune_level=autotune_level
            )
    except Exception as e:
        delete_file(raw_key)
        logger.exception("vocal processing failed project_id=%s", project_id)
        raise HTTPException(status_code=500, detail={
            "reason": "processing_error",
            "message_en": "Vocal processing failed. Please try recording again.",
            "message_ar": "فشلت معالجة الصوت. يرجى إعادة التسجيل.",
            **({"debug": traceback.format_exc()} if settings.environment != "production" else {}),
        })

    # Upload processed vocal to R2
    processed_key = f"projects/{project_id}/processed_vocal.wav"
    try:
        upload_file(processed_key, processed_bytes, content_type="audio/wav")
    except Exception as e:
        delete_file(raw_key)
        logger.exception("processed vocal upload failed project_id=%s", project_id)
        raise HTTPException(status_code=500, detail={
            "reason": "upload_error",
            "message_en": "Couldn't save your processed vocal. Please try again.",
            "message_ar": "تعذّر حفظ الصوت المعالج. يرجى المحاولة مرة أخرى.",
        })

    # Delete raw vocal — no longer needed
    try:
        delete_file(raw_key)
    except Exception:
        pass  # not critical

    # Update project status + store processed key
    try:
        supabase.table("projects").update({
            "status": "recording",
            "processed_vocal_key": processed_key,
        }).eq("id", project_id).eq("user_id", user_id).execute()
    except Exception:
        pass  # not critical — project is created, vocal is processed

    increment("successful_vocal_processes")

    # Generate signed URL for playback
    processed_url = generate_signed_url(processed_key, expires_in=PROCESSED_URL_TTL)

    return {
        "project_id": project_id,
        "processed_url": processed_url,
        "autotune_level": autotune_level,
        "message_en": "Your vocal is clean and ready. How does it sound?",
        "message_ar": "صوتك نظيف وجاهز. كيف يبدو؟",
    }


@router.post("/projects/{project_id}/replace-vocal")
async def replace_project_vocal(
    project_id: str,
    file: Annotated[UploadFile, File()],
    autotune_level: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    user: dict = Depends(get_current_user),
):
    """Replace only the vocal while keeping the project and beat history."""
    user_id = user["user_id"]
    supabase = get_supabase()

    result = (
        supabase.table("projects")
        .select("*")
        .eq("id", project_id)
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Project not found")

    project = result.data
    valid_levels = ("natural", "subtle", "modern_pop", "rnb", "rap",
                    "melodic", "heavy", "none", "modern")
    level = autotune_level or project.get("autotune_level") or "subtle"
    if level not in valid_levels:
        level = "subtle"
    project_language = language or project.get("language") or "en"

    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")
    if len(audio_bytes) > MAX_VOCAL_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")

    quality = await asyncio.to_thread(check_quality, audio_bytes)
    if not quality["ok"]:
        raise HTTPException(status_code=422, detail={
            "reason": quality["reason"],
            "message_en": quality["message_en"],
            "message_ar": quality["message_ar"],
        })

    raw_key = f"projects/{project_id}/raw_vocal_replacement.webm"
    processed_key = f"projects/{project_id}/processed_vocal.wav"
    try:
        upload_file(raw_key, audio_bytes, content_type="audio/webm")
        with time_ms("vocal_processing_duration_ms"):
            processed_bytes = await asyncio.to_thread(
                process_vocal, audio_bytes, autotune_level=level
            )
        upload_file(processed_key, processed_bytes, content_type="audio/wav")
    except Exception as exc:
        logger.exception("replacement vocal failed project_id=%s", project_id)
        try:
            delete_file(raw_key)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=safe_error_detail(
            reason="replace_vocal_failed",
            message_en="Couldn't replace your vocal. Please try recording again.",
            message_ar="Couldn't replace your vocal. Please try recording again.",
            debug=f"{type(exc).__name__}: {exc}",
        ))

    try:
        delete_file(raw_key)
    except Exception:
        pass

    next_status = "beat_generation" if project.get("beat_key") else "recording"
    try:
        supabase.table("projects").update({
            "status": next_status,
            "processed_vocal_key": processed_key,
            "autotune_level": level,
            "language": project_language,
            "coach_feedback": None,
            "final_mp3_key": None,
            "final_wav_key": None,
        }).eq("id", project_id).eq("user_id", user_id).execute()
    except Exception as exc:
        logger.exception("replacement vocal metadata update failed project_id=%s", project_id)
        raise HTTPException(status_code=500, detail=safe_error_detail(
            reason="metadata_update_failed",
            message_en="Your vocal was processed but couldn't be attached to the project. Please try again.",
            message_ar="Your vocal was processed but couldn't be attached to the project. Please try again.",
            debug=f"{type(exc).__name__}: {exc}",
        ))

    increment("successful_vocal_replacements")
    return {
        "project_id": project_id,
        "processed_url": generate_signed_url(processed_key, expires_in=PROCESSED_URL_TTL),
        "autotune_level": level,
        "preserved_beat": bool(project.get("beat_key")),
        "message_en": "Your new vocal is ready, and your beat is still here.",
        "message_ar": "Your new vocal is ready, and your beat is still here.",
    }


@router.get("/projects/{project_id}/processed-url")
def get_processed_vocal_url(project_id: str, user: dict = Depends(get_current_user)):
    """Get a fresh signed URL for the processed vocal."""
    supabase = get_supabase()
    result = (
        supabase.table("projects")
        .select("processed_vocal_key, status")
        .eq("id", project_id)
        .eq("user_id", user["user_id"])
        .single()
        .execute()
    )
    if not result.data or not result.data.get("processed_vocal_key"):
        raise HTTPException(status_code=404, detail="No processed vocal found")

    url = generate_signed_url(result.data["processed_vocal_key"], expires_in=PROCESSED_URL_TTL)
    return {"url": url}
