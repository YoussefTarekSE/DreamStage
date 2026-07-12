import asyncio
import logging
import traceback
from fastapi import APIRouter, Depends, HTTPException
from supabase import create_client
from ..auth import get_current_user
from ..config import settings
from ..services.vocal_analyzer import analyze_for_coaching
from ..services.coach import generate_coaching

router = APIRouter(prefix="/studio", tags=["coach"])
logger = logging.getLogger(__name__)


def get_supabase():
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


@router.post("/projects/{project_id}/coach-feedback")
async def get_coach_feedback(
    project_id: str,
    user: dict = Depends(get_current_user),
):
    supabase = get_supabase()
    user_id = user["user_id"]

    # Load project
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

    # Return cached feedback if already generated
    if project.get("coach_feedback"):
        return project["coach_feedback"]

    # Load voice profile
    vp = (
        supabase.table("voice_profiles")
        .select("*")
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )
    voice_profile = vp.data or {}

    # Download + analyze processed vocal
    analysis = {}
    processed_key = project.get("processed_vocal_key")
    if processed_key:
        try:
            import boto3
            s3 = boto3.client(
                "s3",
                endpoint_url=f"https://{settings.r2_account_id}.r2.cloudflarestorage.com",
                aws_access_key_id=settings.r2_access_key_id,
                aws_secret_access_key=settings.r2_secret_access_key,
                region_name="auto",
            )
            obj = s3.get_object(Bucket=settings.r2_bucket_name, Key=processed_key)
            processed_bytes = obj["Body"].read()
            analysis = await asyncio.to_thread(analyze_for_coaching, processed_bytes)
        except Exception:
            pass  # use defaults

    # Generate coaching feedback via Groq
    try:
        feedback = await generate_coaching(
            analysis=analysis,
            voice_profile=voice_profile,
            autotune_level=project.get("autotune_level", "subtle"),
            language=project.get("language", "en"),
            groq_api_key=settings.groq_api_key,
        )
    except Exception as e:
        logger.exception("coach feedback generation failed project_id=%s", project_id)
        raise HTTPException(status_code=502, detail={
            "reason": "coach_failed",
            "message_en": "Couldn't generate coaching feedback. Please try again.",
            "message_ar": "تعذّر توليد ملاحظات المدرب. يرجى المحاولة مرة أخرى.",
            **({"debug": traceback.format_exc()} if settings.environment != "production" else {}),
        })

    # Cache feedback in DB + advance status
    try:
        supabase.table("projects").update({
            "coach_feedback": feedback,
            "status": "coaching",
        }).eq("id", project_id).eq("user_id", user_id).execute()
    except Exception:
        pass

    return feedback


@router.post("/projects/{project_id}/skip-coaching")
def skip_coaching(project_id: str, user: dict = Depends(get_current_user)):
    """Advance straight to mixing without re-recording."""
    supabase = get_supabase()
    supabase.table("projects").update({"status": "mixing"}).eq(
        "id", project_id
    ).eq("user_id", user["user_id"]).execute()
    return {"status": "mixing"}
