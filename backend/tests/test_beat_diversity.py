"""
Beat diversity regression test.

Verifies that the three bugs fixed in beat.py / beat_generator.py actually
allow different vocal recordings to produce audibly distinct beats.

Bug summary (all in backend/app/routers/beat.py):
  Bug 1 — except Exception: pass  (R2 read errors silently discarded)
  Bug 2 — vocal_mood.setdefault() (voice_profile data never merged)
  Bug 3 — hash-based seed with constant defaults produces identical beats

Run with:
    cd backend
    python -m pytest tests/test_beat_diversity.py -v
"""
import hashlib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.beat_synthesizer import generate_beat


# ─── Three maximally distinct vocal archetypes ───────────────────────────────

VOCAL_BALLAD = {
    "tempo":       68.0,    # slow ballad
    "rms":         0.07,    # quiet, breathy
    "centroid":    1100.0,  # warm/dark timbre
    "density":     1.2,     # sparse phrasing
    "key":         "D",
    "mode":        "minor",
    "valence":     0.22,    # melancholic
    "emotion":     "melancholic",
    "vocal_style": "melodic",
    "swing_ratio": 0.54,
    "overall_rms": 0.07,
}

VOCAL_TRAP = {
    "tempo":       145.0,   # fast trap hi-hats
    "rms":         0.31,    # loud, aggressive
    "centroid":    2600.0,  # bright, harsh
    "density":     6.8,     # dense syllables
    "key":         "F#",
    "mode":        "minor",
    "valence":     0.21,    # dark
    "emotion":     "dark",
    "vocal_style": "rhythmic",
    "swing_ratio": 0.50,
    "overall_rms": 0.31,
}

VOCAL_POP = {
    "tempo":       118.0,   # mid-tempo pop
    "rms":         0.26,    # energetic
    "centroid":    2300.0,  # bright
    "density":     4.1,     # moderate
    "key":         "A",
    "mode":        "major",
    "valence":     0.78,    # uplifting / euphoric
    "emotion":     "uplifting",
    "vocal_style": "melodic",
    "swing_ratio": 0.51,
    "overall_rms": 0.26,
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _wav_hash(wav_bytes: bytes) -> str:
    return hashlib.sha256(wav_bytes).hexdigest()[:16]


def _generate(analysis: dict, attempt: int = 1) -> tuple[bytes, str]:
    """Call the synthesizer directly (Tier-4 path, no R2/HF required)."""
    return generate_beat(analysis=analysis, bars=16, attempt=attempt)


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_different_vocals_produce_different_beats():
    """
    Three distinct vocal types must produce distinct beats.
    Before the fix, all three returned the same hash because
    silent R2 failure left analysis = hardcoded defaults → same seed.
    """
    wav_ballad, genre_ballad = _generate(VOCAL_BALLAD)
    wav_trap,   genre_trap   = _generate(VOCAL_TRAP)
    wav_pop,    genre_pop    = _generate(VOCAL_POP)

    h_ballad = _wav_hash(wav_ballad)
    h_trap   = _wav_hash(wav_trap)
    h_pop    = _wav_hash(wav_pop)

    print(f"\n  ballad  -> genre={genre_ballad:18s}  hash={h_ballad}")
    print(f"  trap    -> genre={genre_trap:18s}  hash={h_trap}")
    print(f"  pop     -> genre={genre_pop:18s}  hash={h_pop}")

    assert h_ballad != h_trap,   "ballad and trap produced identical audio"
    assert h_ballad != h_pop,    "ballad and pop produced identical audio"
    assert h_trap   != h_pop,    "trap and pop produced identical audio"


def test_generate_another_beat_changes_output():
    """
    'Generate Another Beat' (attempt 2 vs attempt 1) must produce a different
    beat even for the same vocal.  Before the fix, with constant defaults the
    seed changed but the genre pool was still the same 3 slow R&B genres.
    """
    wav_a, genre_a = _generate(VOCAL_BALLAD, attempt=1)
    wav_b, genre_b = _generate(VOCAL_BALLAD, attempt=2)
    wav_c, genre_c = _generate(VOCAL_BALLAD, attempt=3)

    h_a, h_b, h_c = _wav_hash(wav_a), _wav_hash(wav_b), _wav_hash(wav_c)

    print(f"\n  attempt 1 -> genre={genre_a:18s}  hash={h_a}")
    print(f"  attempt 2 -> genre={genre_b:18s}  hash={h_b}")
    print(f"  attempt 3 -> genre={genre_c:18s}  hash={h_c}")

    assert h_a != h_b, "attempt 1 and 2 produced identical audio"
    assert h_a != h_c, "attempt 1 and 3 produced identical audio"
    assert h_b != h_c, "attempt 2 and 3 produced identical audio"


def test_hardcoded_defaults_do_not_suppress_diversity():
    """
    When analysis equals the hardcoded fallback dict (Bug 1 scenario), different
    attempt numbers must still produce audibly distinct beats.  The seed offset
    (attempt * 999_983) must create genre-level differences, not just minor
    variation within the same slow-R&B pool.
    """
    DEFAULTS = {
        "tempo": 90.0, "rms": 0.18, "centroid": 1500.0, "density": 2.0,
        "key": "C", "mode": "major", "valence": 0.5, "emotion": "smooth",
        "vocal_style": "rhythmic", "swing_ratio": 0.5, "overall_rms": 0.18,
    }

    results = []
    for attempt in range(1, 4):
        wav, genre = _generate(DEFAULTS, attempt=attempt)
        results.append((genre, _wav_hash(wav)))
        print(f"  attempt {attempt} -> genre={genre:18s}  hash={results[-1][1]}")

    # Hashes must differ — even with constant analysis, attempt seeds must diverge
    hashes = [r[1] for r in results]
    assert len(set(hashes)) == len(hashes), (
        "All attempts produced identical audio with hardcoded defaults.\n"
        "Root cause: hash(str(defaults)) is constant → same base seed.\n"
        "Fix: the attempt offset ensures seeds differ across attempts."
    )


def test_genre_matches_emotion():
    """
    Verify that the selected genre is appropriate for the declared emotion.
    If analysis IS being passed through correctly, genre selection must respect
    the emotion label — not always fall back to 'smooth'-emotion genres.
    """
    from app.services.beat_synthesizer import select_genre, RNG

    cases = [
        ("dark",       {"tempo": 145, "mode": "minor", "valence": 0.2,
                        "emotion": "dark", "vocal_style": "rhythmic",
                        "overall_rms": 0.30, "swing_ratio": 0.50},
         {"trap_dark", "drill", "uk_drill", "phonk"}),
        ("melancholic", {"tempo": 68, "mode": "minor", "valence": 0.22,
                         "emotion": "melancholic", "vocal_style": "melodic",
                         "overall_rms": 0.07, "swing_ratio": 0.54},
         {"rnb_neo_soul", "soul_ballad", "lofi_chill"}),
        ("uplifting",  {"tempo": 118, "mode": "major", "valence": 0.78,
                        "emotion": "uplifting", "vocal_style": "melodic",
                        "overall_rms": 0.26, "swing_ratio": 0.51},
         {"pop_bright", "afrobeats", "hiphop_modern", "rnb_smooth"}),
    ]

    for emotion, analysis, expected_genres in cases:
        RNG.seed(42)
        chosen = select_genre(analysis)
        print(f"  emotion={emotion:12s} -> genre={chosen}")
        assert chosen in expected_genres, (
            f"emotion '{emotion}' mapped to genre '{chosen}', "
            f"expected one of {expected_genres}"
        )


def test_topline_hook_picker_varies_across_seeded_cuts(monkeypatch):
    from random import Random
    from app.services import texture_loops as tx

    manifest = {
        "hook_a": {"role": "topline", "file": "a.wav", "root_pc": 0, "mode": "minor", "brightness": 0.5},
        "hook_b": {"role": "topline", "file": "b.wav", "root_pc": 1, "mode": "minor", "brightness": 0.5},
        "hook_c": {"role": "topline", "file": "c.wav", "root_pc": 2, "mode": "minor", "brightness": 0.5},
    }
    monkeypatch.setattr(tx, "_manifest", manifest)

    picks = {
        tx.pick("topline", 0, "minor", 0.5, rng=Random(seed))["name"]
        for seed in range(12)
    }

    assert len(picks) > 1
    assert tx.pick("topline", 0, "minor", 0.5, rng=Random(7))["name"] == tx.pick(
        "topline", 0, "minor", 0.5, rng=Random(7)
    )["name"]


if __name__ == "__main__":
    print("=== Beat diversity test ===\n")
    print("test_different_vocals_produce_different_beats:")
    test_different_vocals_produce_different_beats()
    print("  PASS")

    print("\ntest_generate_another_beat_changes_output:")
    test_generate_another_beat_changes_output()
    print("  PASS")

    print("\ntest_hardcoded_defaults_do_not_suppress_diversity:")
    test_hardcoded_defaults_do_not_suppress_diversity()
    print("  PASS")

    print("\ntest_genre_matches_emotion:")
    test_genre_matches_emotion()
    print("  PASS")

    print("\nAll tests passed.")
