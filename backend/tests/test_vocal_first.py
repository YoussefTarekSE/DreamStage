"""Vocal-first doctrine regression tests.

The singer's transcribed melody IS the composition's hook. A canned texture
loop (reused across songs) must never displace it — the topline loop is only
the chorus hook when no reliable vocal melody exists.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import beat_synthesizer as bs      # noqa: E402
from app.services import texture_loops as tx          # noqa: E402


ANALYSIS = {
    "tempo": 92.0, "energy": 0.6, "valence": 0.55, "emotion": "warm",
    "density": 3.0, "pitch_stability": 0.7,
}
MELODY = [(0.0, 60, 1.0), (1.0, 62, 1.0), (2.0, 64, 2.0),
          (4.0, 67, 1.0), (5.0, 64, 1.0), (6.0, 62, 2.0)]


@pytest.fixture
def render_spy(monkeypatch):
    calls = []
    orig = tx.render

    def spy(entry, key_pc, tempo, n, role, valence):
        calls.append(role)
        return orig(entry, key_pc, tempo, n, role, valence)

    monkeypatch.setattr(tx, "render", spy)
    return calls


@pytest.mark.skipif(not tx.available(), reason="texture loops not installed")
def test_vocal_melody_owns_the_chorus_hook(render_spy):
    """With a vocal melody, the canned topline loop must never render."""
    wav, _genre = bs.generate_beat(
        ANALYSIS, seed=1234, melody=MELODY, melody_loop_beats=8.0,
        bar_degrees=[0, 3, 5, 4] * 4, master=False,
    )
    assert isinstance(wav, (bytes, bytearray)) and len(wav) > 1000
    assert "topline" not in set(render_spy), (
        "canned topline loop displaced the singer's own melody (doctrine #3)"
    )


@pytest.mark.skipif(not tx.available(), reason="texture loops not installed")
def test_topline_still_engages_without_melody(render_spy):
    """Without a vocal melody, the real-loop topline remains available
    (picked with 0.9 probability — probe several seeds)."""
    for seed in (1234, 99, 7, 2024, 555):
        render_spy.clear()
        wav, _genre = bs.generate_beat(ANALYSIS, seed=seed, master=False)
        assert isinstance(wav, (bytes, bytearray)) and len(wav) > 1000
        if "topline" in set(render_spy):
            return
    pytest.fail("topline loop never engaged across 5 seeds without a melody")
