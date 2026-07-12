"""
Async job queue for long-running work (beat generation, later mixing).

Why: a neural cut takes 1-2 minutes; browsers abort a fetch at ~300s and a
dropped connection used to kill the request mid-save. Submit-then-poll makes
generation survive slow renders, page refreshes, and (later) lets a remote
GPU worker claim 'queued' rows — the jobs table IS the worker queue.

Execution today is in-process (an asyncio task inside the backend); the
schema is worker-ready so moving execution out needs no client changes.
"""
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job(supabase, user_id: str, project_id: str, kind: str,
               payload: dict | None = None) -> str:
    res = supabase.table("jobs").insert({
        "user_id": user_id,
        "project_id": project_id,
        "kind": kind,
        "status": "queued",
        "payload": payload or {},
    }).execute()
    return res.data[0]["id"]


def mark_running(supabase, job_id: str) -> None:
    supabase.table("jobs").update(
        {"status": "running", "updated_at": _now()}).eq("id", job_id).execute()


def mark_done(supabase, job_id: str, result: dict) -> None:
    supabase.table("jobs").update(
        {"status": "done", "result": result, "updated_at": _now()}
    ).eq("id", job_id).execute()


def mark_failed(supabase, job_id: str, error: dict) -> None:
    supabase.table("jobs").update(
        {"status": "failed", "error": error, "updated_at": _now()}
    ).eq("id", job_id).execute()


def get_job(supabase, job_id: str, user_id: str) -> dict | None:
    res = (supabase.table("jobs").select("*")
           .eq("id", job_id).eq("user_id", user_id)
           .maybe_single().execute())
    return res.data if res and res.data else None


def latest_active_job(supabase, project_id: str, user_id: str,
                      kind: str) -> dict | None:
    """The newest queued/running job for a project — lets the UI resume
    polling after a page refresh instead of losing the generation."""
    res = (supabase.table("jobs").select("*")
           .eq("project_id", project_id).eq("user_id", user_id)
           .eq("kind", kind).in_("status", ["queued", "running"])
           .order("created_at", desc=True).limit(1).execute())
    return res.data[0] if res.data else None
