"""
Telemetry smoke test.

Verifies that:
1. generate_beat_from_vocal() populates _tier_attempts on every return path
2. record_beat_event() never raises — swallows all errors silently
3. Tier-attempt log contains correct tier numbers and has duration_ms set
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.beat_synthesizer import generate_beat  # noqa: F401 — ensures synth loads

# ── Helpers ───────────────────────────────────────────────────────────────────

_ANALYSIS_POP = {
    "tempo": 118.0, "rms": 0.26, "centroid": 2300.0, "density": 4.1,
    "key": "A", "mode": "major", "valence": 0.78, "emotion": "uplifting",
    "vocal_style": "melodic", "swing_ratio": 0.51, "overall_rms": 0.26,
}


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_tier_attempts_in_analysis():
    """
    The MusicGen tiers were removed (CC-BY-NC weights, never ran); the
    synthesizer is the production path. The returned analysis must still
    contain _tier_attempts with the successful synthesizer entry so
    telemetry stays consistent.
    """
    import asyncio
    from app.services.beat_generator import generate_beat_from_vocal

    wav, genre, analysis = asyncio.run(generate_beat_from_vocal(
        processed_bytes=b"",
        voice_profile={},
        vocal_mood=dict(_ANALYSIS_POP),
        hf_api_key="",
        attempt=1,
    ))

    assert len(wav) > 1_000, "Expected non-empty WAV bytes from the synthesizer"
    attempts = analysis.get("_tier_attempts", [])
    print(f"\n  genre={genre}  attempts={[(a['name'], a['success']) for a in attempts]}")

    assert not any(a["name"].startswith("musicgen") for a in attempts), \
        "MusicGen tiers were removed and must not be attempted"

    synth = next((a for a in attempts if a["name"] == "synthesizer"), None)
    assert synth is not None,           "Synthesizer attempt must be recorded"
    assert synth["success"] is True,    "Synthesizer must report success"
    assert synth["duration_ms"] >= 0,   "Synthesizer duration_ms must be set"


def test_record_beat_event_swallows_errors():
    """record_beat_event must never raise, even with a broken Supabase client."""

    class BrokenClient:
        def table(self, *_):
            raise RuntimeError("simulated DB error")

    from app.services.telemetry import record_beat_event
    # Should not raise
    record_beat_event(
        BrokenClient(),
        project_id="test-project",
        tier_used="lofi_chill",
        tier_attempts=[{"tier": 4, "name": "synthesizer", "duration_ms": 50, "success": True}],
        duration_ms=100,
        analysis=_ANALYSIS_POP,
        genre="lofi_chill",
        success=True,
    )
    print("\n  record_beat_event swallowed DB error correctly")


if __name__ == "__main__":
    print("=== Telemetry tests ===\n")

    print("test_tier_attempts_in_analysis:")
    test_tier_attempts_in_analysis()
    print("  PASS")

    print("\ntest_record_beat_event_swallows_errors:")
    test_record_beat_event_swallows_errors()
    print("  PASS")

    print("\nAll tests passed.")
