import asyncio
import importlib
import pathlib
import sys

import pytest
from fastapi import HTTPException


ROOT = pathlib.Path(__file__).resolve().parents[2]
REQUIRED_ENV = {
    "SUPABASE_URL": "https://example.supabase.co",
    "SUPABASE_SERVICE_ROLE_KEY": "placeholder",
    "R2_ACCOUNT_ID": "placeholder",
    "R2_ACCESS_KEY_ID": "placeholder",
    "R2_SECRET_ACCESS_KEY": "placeholder",
    "R2_BUCKET_NAME": "dreamstage-audio",
    "GROQ_API_KEY": "placeholder",
    "HF_API_KEY": "placeholder",
}


def _seed_env(monkeypatch):
    for name, value in REQUIRED_ENV.items():
        monkeypatch.setenv(name, value)


def test_ml_models_are_loaded_once(monkeypatch, tmp_path):
    import app.services.ml_analyzer as ml

    model_path = tmp_path / "genre_classifier.joblib"
    model_path.write_bytes(b"model")
    loads = []

    def fake_load(path):
        loads.append(path)
        return {"loaded_from": path}

    if hasattr(ml._load_model, "cache_clear"):
        ml._load_model.cache_clear()
    monkeypatch.setattr(ml, "_HAS_JOBLIB", True)
    monkeypatch.setattr(ml, "_MODELS_DIR", tmp_path)
    monkeypatch.setattr(ml.joblib, "load", fake_load)

    first = ml._load_model("genre_classifier")
    second = ml._load_model("genre_classifier")

    assert first is second
    assert loads == [model_path]


def test_generate_beat_offloads_sync_analysis_and_synthesis(monkeypatch):
    from app.services import beat_generator
    import app.services.ml_analyzer as ml
    import app.services.performance_map as perf
    import app.services.vocal_harmony as harmony

    calls = []

    async def fake_to_thread(fn, /, *args, **kwargs):
        calls.append(fn.__name__)
        return fn(*args, **kwargs)

    def fake_analyze_full_ml(_audio):
        return {"tempo": 92.0, "key": "D", "mode": "minor", "emotion": "smooth"}

    def fake_transcribe_harmony(*_args, **_kwargs):
        return None

    def fake_build_performance_map(*_args, **_kwargs):
        return None

    def fake_best_candidate(**_kwargs):
        return b"beat-bytes", "hiphop_modern", {"total": 95}

    monkeypatch.setattr(beat_generator.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(ml, "analyze_full_ml", fake_analyze_full_ml)
    monkeypatch.setattr(harmony, "transcribe_harmony", fake_transcribe_harmony)
    monkeypatch.setattr(perf, "build_performance_map", fake_build_performance_map)
    monkeypatch.setattr(beat_generator, "_generate_best_synth_candidate", fake_best_candidate)
    # MusicGen tiers removed 2026-07-11 (CC-BY-NC weights) — nothing to stub.

    beat, genre, analysis = asyncio.run(
        beat_generator.generate_beat_from_vocal(
            processed_bytes=b"audio",
            voice_profile={},
            vocal_mood={"tempo": 90.0, "key": "C", "mode": "major"},
            hf_api_key="",
        )
    )

    assert beat == b"beat-bytes"
    assert genre == "hiphop_modern"
    assert analysis["beat_score"]["total"] == 95
    assert "fake_analyze_full_ml" in calls
    assert "fake_transcribe_harmony" in calls
    assert "fake_build_performance_map" in calls
    assert "fake_best_candidate" in calls


def test_process_vocal_rejects_low_quality_before_project_creation(monkeypatch):
    _seed_env(monkeypatch)
    sys.modules.pop("app.config", None)
    sys.modules.pop("app.routers.studio", None)
    studio = importlib.import_module("app.routers.studio")

    class FakeUpload:
        async def read(self):
            return b"audio"

    def fail_if_called():
        raise AssertionError("database should not be touched for rejected audio")

    def low_quality(_audio):
        return {
            "ok": False,
            "reason": "too_noisy",
            "message_en": "Find a quieter space and try again.",
            "message_ar": "Find a quieter space and try again.",
        }

    monkeypatch.setattr(studio, "get_supabase", fail_if_called)
    monkeypatch.setattr(studio, "check_quality", low_quality, raising=False)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            studio.process_vocal_endpoint(
                file=FakeUpload(),
                project_name="Song",
                autotune_level="subtle",
                language="en",
                user={"user_id": "user-1"},
            )
        )

    assert excinfo.value.status_code == 422
    assert excinfo.value.detail["reason"] == "too_noisy"
    assert "quieter" in excinfo.value.detail["message_en"]


def test_generate_beat_endpoint_does_not_duplicate_route_level_analysis(monkeypatch):
    _seed_env(monkeypatch)
    sys.modules.pop("app.config", None)
    sys.modules.pop("app.routers.beat", None)
    beat_router = importlib.import_module("app.routers.beat")

    class Result:
        def __init__(self, data):
            self.data = data

    class Table:
        def __init__(self, name, updates):
            self.name = name
            self.updates = updates
            self.action = "select"

        def select(self, *_args, **_kwargs):
            self.action = "select"
            return self

        def insert(self, data):
            self.action = "insert"
            self.updates.append((self.name, data))
            return self

        def update(self, data):
            self.action = "update"
            self.updates.append((self.name, data))
            return self

        def eq(self, *_args, **_kwargs):
            return self

        def single(self):
            return self

        def maybe_single(self):
            return self

        def execute(self):
            if self.name == "projects" and self.action == "select":
                return Result({
                    "id": "project-1",
                    "user_id": "user-1",
                    "processed_vocal_key": "processed.wav",
                    "producer_cuts": [],
                    "autotune_level": "subtle",
                })
            if self.name == "voice_profiles":
                return Result({})
            return Result([])

    class Supabase:
        def __init__(self):
            self.updates = []

        def table(self, name):
            return Table(name, self.updates)

    class Body:
        def read(self):
            return b"processed-audio"

    class R2:
        def get_object(self, **_kwargs):
            return {"Body": Body()}

    async def fake_generate(**kwargs):
        assert kwargs["processed_bytes"] == b"processed-audio"
        assert kwargs["vocal_mood"]["tempo"] == 90.0
        return b"beat", "hiphop_modern", {
            "tempo": 101.0,
            "key": "D",
            "mode": "minor",
            "emotion": "smooth",
            "valence": 0.5,
            "beat_score": {"total": 88},
        }

    duplicate_calls = []

    def explode(_bytes):
        duplicate_calls.append(True)
        raise AssertionError("route-level duplicate analysis should not run")

    supabase = Supabase()
    monkeypatch.setattr(beat_router, "get_supabase", lambda: supabase)
    monkeypatch.setattr(beat_router, "get_r2", lambda: R2())
    monkeypatch.setattr(beat_router, "analyze_vocal_mood", explode, raising=False)
    monkeypatch.setattr(beat_router, "generate_beat_from_vocal", fake_generate)
    monkeypatch.setattr(beat_router, "upload_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(beat_router, "generate_signed_url", lambda *_args, **_kwargs: "signed-url")
    monkeypatch.setattr(beat_router, "record_beat_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(beat_router, "_artist_taste", lambda *_args, **_kwargs: [])

    # The endpoint is now a thin job submitter; the analysis-duplication
    # concern lives in the executor, so test it directly.
    response = asyncio.run(
        beat_router._execute_generate(
            project_id="project-1",
            user_id="user-1",
            body=beat_router.GenerateBeatRequest(),
        )
    )

    assert response["beat_url"] == "signed-url"
    assert response["tempo_bpm"] == 101
    assert duplicate_calls == []
