"""Taste learning — positive AND negative signals (producer_cuts pure logic).

accepted ≫ favourite ≈ branched-from; cuts passed over (heard, then followed
by another cut with no endorsement) slowly erode a genre, and enough skips
cancel an old favourite entirely.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import producer_cuts as pc  # noqa: E402


def _cut(n, genre, favorite=False, accepted=False, parent=None):
    return {"cut": n, "genre": genre, "favorite": favorite,
            "accepted": accepted, "parent_cut": parent}


def test_accept_outranks_favorite():
    cuts = [_cut(1, "trap_dark", accepted=True), _cut(2, "pop_bright", favorite=True),
            _cut(3, "afrobeats")]
    taste = pc.ranked_taste(pc.taste_weights([cuts]))
    assert taste[0] == "trap_dark"
    assert taste[1] == "pop_bright"


def test_branching_is_a_positive_signal():
    # Artist branched twice from cut 1 (rnb) but never favorited anything.
    cuts = [_cut(1, "rnb_smooth"), _cut(2, "rnb_neo_soul", parent=1),
            _cut(3, "soul_ballad", parent=1)]
    weights = pc.taste_weights([cuts])
    assert weights["rnb_smooth"] > 0          # branched-from = endorsement
    taste = pc.ranked_taste(weights)
    assert "rnb_smooth" in taste


def test_skips_erode_and_eventually_cancel_a_favorite():
    # Genre favorited once long ago, then passed over 5 times since:
    # 1 + 5 * (-0.25) < a fresh favourite with no skips.
    old_fav = [_cut(1, "drill", favorite=True)] + [
        _cut(i, "drill") for i in range(2, 7)] + [_cut(7, "pop_bright", favorite=True)]
    weights = pc.taste_weights([old_fav])
    assert weights["pop_bright"] > weights["drill"]
    taste = pc.ranked_taste(weights)
    assert taste[0] == "pop_bright"

    # Enough skips cancel the favourite entirely (net <= 0 drops out).
    dead_fav = [_cut(1, "drill", favorite=True)] + [
        _cut(i, "drill") for i in range(2, 8)] + [_cut(8, "pop_bright")]
    # 6 non-newest drill skips: cuts 2..7 -> 1 + 6*(-0.25) = -0.5
    taste = pc.ranked_taste(pc.taste_weights([dead_fav]))
    assert "drill" not in taste


def test_newest_cut_is_never_penalised():
    # Single unreacted cut: the artist may simply not have responded yet.
    cuts = [_cut(1, "amapiano")]
    weights = pc.taste_weights([cuts])
    assert weights.get("amapiano", 0.0) == 0.0


def test_per_project_isolation_of_newest():
    # Each project's newest cut is exempt independently.
    p1 = [_cut(1, "phonk"), _cut(2, "drill")]        # phonk skipped, drill newest
    p2 = [_cut(1, "phonk")]                            # newest -> exempt
    weights = pc.taste_weights([p1, p2])
    assert weights["phonk"] == pc.SKIP_WEIGHT          # only ONE skip counted
    assert weights.get("drill", 0.0) == 0.0


def test_endorsed_cut_is_not_also_skipped():
    cuts = [_cut(1, "afrobeats", favorite=True), _cut(2, "trap_dark")]
    weights = pc.taste_weights([cuts])
    assert weights["afrobeats"] == pc.FAVORITE_WEIGHT  # no skip stacked on top


def test_back_compat_taste_signal_still_works():
    cuts = [_cut(1, "trap_dark", accepted=True), _cut(2, "pop_bright", favorite=True)]
    taste = pc.taste_genres(pc.taste_signal(cuts))
    assert taste == ["trap_dark", "pop_bright"]


def test_taste_feeds_choose_force_genre():
    # A drill-lover gets the trap_dark lane FIRST on a new bright vocal.
    taste = pc.ranked_taste(pc.taste_weights([[
        _cut(1, "drill", accepted=True), _cut(2, "pop_bright")]]))
    bright = {"valence": 0.9, "tempo": 128}
    first = pc.choose_force_genre([], bright, taste=taste)
    assert first == pc.ARCHETYPES[4]                   # trap_dark lane leads
