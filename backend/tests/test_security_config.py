import pathlib
import importlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"


REQUIRED_BACKEND_ENV = {
    "SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY",
    "R2_ACCOUNT_ID",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET_NAME",
    "GROQ_API_KEY",
    "HF_API_KEY",
}


def test_load_settings_reports_missing_required_variables(monkeypatch):
    for name in REQUIRED_BACKEND_ENV:
        monkeypatch.setenv(name, "placeholder")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")

    sys.modules.pop("app.config", None)
    config = importlib.import_module("app.config")

    for name in REQUIRED_BACKEND_ENV | {"SUPABASE_JWT_SECRET", "SUPABASE_MANAGEMENT_TOKEN", "ADMIN_KEY"}:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        config.load_settings(_env_file=None)

    message = str(excinfo.value)
    assert "DreamStage configuration error" in message
    assert "SUPABASE_URL" in message
    assert "SUPABASE_SERVICE_ROLE_KEY" in message
    assert "environment variables" in message


def test_env_examples_document_required_variables():
    root_example = ROOT / ".env.example"
    backend_example = BACKEND / ".env.example"
    frontend_example = ROOT / "frontend" / ".env.example"

    assert root_example.exists()
    assert backend_example.exists()
    assert frontend_example.exists()

    root_text = root_example.read_text(encoding="utf-8")
    backend_text = backend_example.read_text(encoding="utf-8")
    frontend_text = frontend_example.read_text(encoding="utf-8")

    for name in REQUIRED_BACKEND_ENV | {"SUPABASE_MANAGEMENT_TOKEN", "SUPABASE_PROJECT_REF", "ADMIN_KEY"}:
        assert f"{name}=" in root_text
        assert f"{name}=" in backend_text

    assert "NEXT_PUBLIC_SUPABASE_URL=" in root_text
    assert "NEXT_PUBLIC_SUPABASE_ANON_KEY=" in root_text
    assert "NEXT_PUBLIC_SUPABASE_URL=" in frontend_text
    assert "NEXT_PUBLIC_SUPABASE_ANON_KEY=" in frontend_text


def test_migration_script_uses_environment_for_credentials():
    script = (ROOT / "scripts" / "migrate.ps1").read_text(encoding="utf-8")

    assert "SUPABASE_MANAGEMENT_TOKEN" in script
    assert "SUPABASE_PROJECT_REF" in script
    assert "sbp_" not in script
    assert 'Bearer $token' in script


def test_gitignore_excludes_local_secrets_and_generated_caches():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")

    for pattern in (
        ".env",
        ".env.*",
        "!.env.example",
        "secrets/",
        "*.secret",
        "*.token",
        "backend/.venv/",
        "frontend/node_modules/",
        "__pycache__/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
        ".npm/",
    ):
        assert pattern in text
