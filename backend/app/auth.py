import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from supabase import create_client
from .config import settings

security = HTTPBearer()
logger = logging.getLogger(__name__)


def _get_local_user(token: str) -> dict | None:
    if not settings.supabase_jwt_secret:
        return None
    try:
        payload = jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            options={"verify_aud": False},
        )
    except JWTError:
        return None

    user_id = payload.get("sub")
    if not user_id:
        return None
    app_metadata = payload.get("app_metadata") or {}
    user_metadata = payload.get("user_metadata") or {}
    return {
        "user_id": str(user_id),
        "email": payload.get("email"),
        "role": app_metadata.get("role") or user_metadata.get("role"),
    }


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    token = credentials.credentials
    local_user = _get_local_user(token)
    if local_user:
        return local_user

    try:
        supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)
        response = supabase.auth.get_user(token)
        user = response.user
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return {"user_id": str(user.id), "email": user.email, "role": None}
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("remote auth validation failed: %s", type(exc).__name__)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
