from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    supabase_url: str = Field(..., min_length=1)
    supabase_service_role_key: str = Field(..., min_length=1)
    supabase_jwt_secret: str = ""  # kept for reference, auth uses get_user() instead

    r2_account_id: str = Field(..., min_length=1)
    r2_access_key_id: str = Field(..., min_length=1)
    r2_secret_access_key: str = Field(..., min_length=1)
    r2_bucket_name: str = Field("dreamstage-audio", min_length=1)

    groq_api_key: str = Field(..., min_length=1)
    hf_api_key: str = Field(..., min_length=1)

    supabase_management_token: str = ""
    supabase_project_ref: str = ""
    admin_key: str = ""
    admin_user_ids: str = ""
    admin_emails: str = ""

    environment: str = "development"

    @field_validator("supabase_url")
    @classmethod
    def validate_supabase_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("must be a valid HTTP(S) URL")
        return value.rstrip("/")

    @field_validator("environment")
    @classmethod
    def normalize_environment(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"development", "test", "staging", "production"}:
            raise ValueError("must be one of development, test, staging, or production")
        return normalized


_ENV_BY_FIELD = {
    "supabase_url": "SUPABASE_URL",
    "supabase_service_role_key": "SUPABASE_SERVICE_ROLE_KEY",
    "supabase_jwt_secret": "SUPABASE_JWT_SECRET",
    "r2_account_id": "R2_ACCOUNT_ID",
    "r2_access_key_id": "R2_ACCESS_KEY_ID",
    "r2_secret_access_key": "R2_SECRET_ACCESS_KEY",
    "r2_bucket_name": "R2_BUCKET_NAME",
    "groq_api_key": "GROQ_API_KEY",
    "hf_api_key": "HF_API_KEY",
    "supabase_management_token": "SUPABASE_MANAGEMENT_TOKEN",
    "supabase_project_ref": "SUPABASE_PROJECT_REF",
    "admin_key": "ADMIN_KEY",
    "admin_user_ids": "ADMIN_USER_IDS",
    "admin_emails": "ADMIN_EMAILS",
    "environment": "ENVIRONMENT",
}


def load_settings(**kwargs) -> Settings:
    try:
        return Settings(**kwargs)
    except ValidationError as exc:
        missing = []
        invalid = []
        for error in exc.errors():
            field = str(error["loc"][0])
            env_name = _ENV_BY_FIELD.get(field, field.upper())
            if error["type"] == "missing":
                missing.append(env_name)
            else:
                invalid.append(f"{env_name}: {error['msg']}")

        parts = []
        if missing:
            parts.append("missing required environment variables: " + ", ".join(sorted(missing)))
        if invalid:
            parts.append("invalid environment variables: " + "; ".join(invalid))
        detail = "; ".join(parts) or "invalid settings"
        raise RuntimeError(
            "DreamStage configuration error: "
            f"{detail}. Set them in environment variables or backend/.env."
        ) from exc


settings = load_settings()
