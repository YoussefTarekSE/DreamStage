from fastapi import APIRouter
from supabase import create_client

from ..config import settings
from ..storage import get_r2_client
from ..services.metrics import snapshot

router = APIRouter()


def _check_database() -> dict:
    try:
        client = create_client(settings.supabase_url, settings.supabase_service_role_key)
        client.table("projects").select("id").limit(1).execute()
        return {"status": "ok"}
    except Exception:
        return {"status": "degraded"}


def _check_storage() -> dict:
    try:
        get_r2_client().head_bucket(Bucket=settings.r2_bucket_name)
        return {"status": "ok"}
    except Exception:
        return {"status": "degraded"}


def _check_ai() -> dict:
    return {
        "status": "ok" if settings.groq_api_key and settings.hf_api_key else "degraded",
        "groq_configured": bool(settings.groq_api_key),
        "huggingface_configured": bool(settings.hf_api_key),
    }


@router.get("/health")
def health_check():
    checks = {
        "database": _check_database(),
        "storage": _check_storage(),
        "ai": _check_ai(),
    }
    status = "ok" if all(c["status"] == "ok" for c in checks.values()) else "degraded"
    return {"status": status, "checks": checks}


@router.get("/metrics")
def metrics():
    return snapshot()
