import asyncio
import logging
import traceback
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import Annotated
from supabase import create_client
from ..auth import get_current_user
from ..config import settings
from ..services.audio_quality import check_quality
from ..services.audio_analysis import extract_voice_profile
from ..services.metrics import time_ms

router = APIRouter(prefix="/voice-training", tags=["voice-training"])
logger = logging.getLogger(__name__)

MAX_RECORDINGS = 5
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB per file


def get_supabase():
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


@router.post("/check-quality")
async def check_recording_quality(
    file: Annotated[UploadFile, File()],
    user: dict = Depends(get_current_user),
):
    audio_bytes = await file.read()
    if len(audio_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File too large")
    return await asyncio.to_thread(check_quality, audio_bytes)


@router.post("/")
async def submit_voice_training(
    files: Annotated[list[UploadFile], File()],
    language: Annotated[str, Form()] = "en",
    user: dict = Depends(get_current_user),
):
    if not files:
        raise HTTPException(status_code=400, detail="No recordings provided")
    if len(files) > MAX_RECORDINGS:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_RECORDINGS} recordings")

    user_id = user["user_id"]
    valid_recordings: list[bytes] = []

    # Step 1 — read + quality check each recording (no R2 upload needed)
    for i, upload in enumerate(files):
        try:
            audio_bytes = await upload.read()
        except Exception:
            logger.exception("voice training upload read failed user_id=%s index=%d", user_id, i)
            raise HTTPException(status_code=400, detail=f"Failed to read file {i}")

        if len(audio_bytes) == 0:
            raise HTTPException(status_code=422, detail={
                "reason": "empty",
                "message_en": "One of your recordings is empty. Please try again.",
                "message_ar": "أحد تسجيلاتك فارغ. يرجى المحاولة مرة أخرى.",
            })

        if len(audio_bytes) > MAX_FILE_SIZE_BYTES:
            continue

        quality = await asyncio.to_thread(check_quality, audio_bytes)
        if not quality["ok"]:
            raise HTTPException(status_code=422, detail={
                "reason": quality["reason"],
                "message_en": quality["message_en"],
                "message_ar": quality["message_ar"],
            })

        valid_recordings.append(audio_bytes)

    if not valid_recordings:
        raise HTTPException(status_code=422, detail={
            "reason": "no_valid",
            "message_en": "No valid recordings received. Please record all phrases.",
            "message_ar": "لم يتم استلام أي تسجيلات صالحة. يرجى تسجيل جميع العبارات.",
        })

    # Step 2 — analyze all recordings in memory
    try:
        with time_ms("transcription_duration_ms"):
            profile_data = await asyncio.to_thread(
                extract_voice_profile, valid_recordings, language
            )
    except Exception as e:
        logger.exception("voice profile analysis failed user_id=%s", user_id)
        raise HTTPException(status_code=500, detail={
            "reason": "analysis_failed",
            "message_en": f"Voice analysis failed. Please try again.",
            "message_ar": "فشل تحليل الصوت. يرجى المحاولة مرة أخرى.",
            **({"debug": traceback.format_exc()} if settings.environment != "production" else {}),
        })

    # Step 3 — save Voice Profile to Supabase
    try:
        supabase = get_supabase()
        result = (
            supabase.table("voice_profiles")
            .upsert({"user_id": user_id, **profile_data}, on_conflict="user_id")
            .execute()
        )
    except Exception as e:
        logger.exception("voice profile save failed user_id=%s", user_id)
        raise HTTPException(status_code=500, detail={
            "reason": "save_failed",
            "message_en": "Couldn't save your Voice Profile. Please try again.",
            "message_ar": "تعذّر حفظ ملفك الصوتي. يرجى المحاولة مرة أخرى.",
            **({"debug": str(e)} if settings.environment != "production" else {}),
        })

    return {
        "status": "ready",
        "voice_profile": result.data[0] if result.data else profile_data,
        "message_en": "Your Voice Profile is ready. I know your range, your tone, and your style.",
        "message_ar": "ملفك الصوتي جاهز. أعرف نطاقك وجرسك وأسلوبك.",
    }


@router.get("/status")
def get_voice_training_status(user: dict = Depends(get_current_user)):
    supabase = get_supabase()
    result = (
        supabase.table("voice_profiles")
        .select("id, tone_type, min_freq_hz, max_freq_hz, created_at")
        .eq("user_id", user["user_id"])
        .maybe_single()
        .execute()
    )
    if result.data:
        return {"has_profile": True, "profile": result.data}
    return {"has_profile": False}
