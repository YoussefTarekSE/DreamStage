from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from supabase import create_client
from ..auth import get_current_user
from ..config import settings
from ..storage import delete_files, generate_signed_url

router = APIRouter(prefix="/projects", tags=["projects"])


def get_supabase():
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


class CreateProjectRequest(BaseModel):
    name: str


@router.get("/")
def list_projects(user: dict = Depends(get_current_user)):
    supabase = get_supabase()
    result = (
        supabase.table("projects")
        .select("*")
        .eq("user_id", user["user_id"])
        .order("created_at", desc=True)
        .execute()
    )
    return result.data


@router.post("/")
def create_project(body: CreateProjectRequest, user: dict = Depends(get_current_user)):
    supabase = get_supabase()
    result = (
        supabase.table("projects")
        .insert({"user_id": user["user_id"], "name": body.name, "status": "created"})
        .execute()
    )
    return result.data[0]


@router.get("/{project_id}")
def get_project(project_id: str, user: dict = Depends(get_current_user)):
    supabase = get_supabase()
    result = (
        supabase.table("projects")
        .select("*")
        .eq("id", project_id)
        .eq("user_id", user["user_id"])
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Project not found")

    project = result.data
    if project.get("final_mp3_key"):
        project["download_url"] = generate_signed_url(project["final_mp3_key"], expires_in=3600)

    return project


@router.delete("/{project_id}")
def delete_project(project_id: str, user: dict = Depends(get_current_user)):
    supabase = get_supabase()
    result = (
        supabase.table("projects")
        .select("final_mp3_key, final_wav_key")
        .eq("id", project_id)
        .eq("user_id", user["user_id"])
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Project not found")

    project = result.data
    keys_to_delete = [k for k in [project.get("final_mp3_key"), project.get("final_wav_key")] if k]
    delete_files(keys_to_delete)

    supabase.table("projects").delete().eq("id", project_id).eq("user_id", user["user_id"]).execute()
    return {"deleted": True}


@router.get("/{project_id}/download")
def get_download_urls(project_id: str, user: dict = Depends(get_current_user)):
    supabase = get_supabase()
    result = (
        supabase.table("projects")
        .select("final_mp3_key, final_wav_key, status")
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
        "mp3_url": generate_signed_url(project["final_mp3_key"], expires_in=3600) if project.get("final_mp3_key") else None,
        "wav_url": generate_signed_url(project["final_wav_key"], expires_in=3600) if project.get("final_wav_key") else None,
    }
