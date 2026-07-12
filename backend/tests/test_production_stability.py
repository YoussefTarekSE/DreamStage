

REQUIRED_ENV = {
    "SUPABASE_URL": "https://example.supabase.co",
    "SUPABASE_SERVICE_ROLE_KEY": "placeholder",
    "R2_ACCOUNT_ID": "placeholder",
    "R2_ACCESS_KEY_ID": "placeholder",
    "R2_SECRET_ACCESS_KEY": "placeholder",
    "GROQ_API_KEY": "placeholder",
    "HF_API_KEY": "placeholder",
}


def _seed_env(monkeypatch):
    for name, value in REQUIRED_ENV.items():
        monkeypatch.setenv(name, value)


def test_safe_error_detail_hides_debug_in_production(monkeypatch):
    _seed_env(monkeypatch)
    from app import errors

    monkeypatch.setattr(errors.settings, "environment", "production")

    detail = errors.safe_error_detail(
        reason="failed",
        message_en="Please try again.",
        message_ar="Please try again.",
        debug="C:/internal/path\nTraceback details",
    )

    assert detail["reason"] == "failed"
    assert "debug" not in detail


def test_safe_error_detail_keeps_debug_in_development(monkeypatch):
    _seed_env(monkeypatch)
    from app import errors

    monkeypatch.setattr(errors.settings, "environment", "development")

    detail = errors.safe_error_detail(
        reason="failed",
        message_en="Please try again.",
        message_ar="Please try again.",
        debug="developer details",
    )

    assert detail["debug"] == "developer details"


