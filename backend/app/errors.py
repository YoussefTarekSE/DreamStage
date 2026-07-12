from __future__ import annotations

from .config import settings


def safe_error_detail(
    *,
    reason: str,
    message_en: str,
    message_ar: str | None = None,
    debug: str | None = None,
) -> dict:
    """Build API error details without exposing internals in production."""
    detail = {
        "reason": reason,
        "message_en": message_en,
        "message_ar": message_ar or message_en,
    }
    if debug and settings.environment != "production":
        detail["debug"] = debug
    return detail
