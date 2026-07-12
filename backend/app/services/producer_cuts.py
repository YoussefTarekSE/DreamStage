"""
Producer Cuts — the DreamStage creative-session model (pure logic).

DreamStage is an AI producer: every generation is a "Producer Cut" kept forever.
The artist can replay, compare, favorite, restore, and BRANCH from any past cut.
These are pure functions (no DB/IO) so the session logic is unit-testable; the
beat router persists the resulting list to projects.producer_cuts (migration 007).

A cut record:
    {cut, label, beat_key, genre, genre_label, key, tempo, emotion, score,
     parent_cut, favorite, created_at}
"""
from __future__ import annotations


def _label_for_parent(cuts: list, parent_cut: int | None) -> str | None:
    for c in cuts:
        if c.get("cut") == parent_cut:
            return c.get("label")
    return None


def build_label(cuts: list, parent_cut: int | None) -> str:
    """Display label for a new cut.

    Root cuts are numbered ("Cut 1", "Cut 2", ...). A cut branched from another
    appends a letter to its parent's label ("Cut 3A", "Cut 3B", ...), so the
    artist can see the lineage of explorations from a favourite idea.
    """
    if parent_cut is None:
        n_roots = sum(1 for c in cuts if c.get("parent_cut") in (None, 0))
        return f"Cut {n_roots + 1}"
    parent_label = _label_for_parent(cuts, parent_cut) or f"Cut {parent_cut}"
    n_children = sum(1 for c in cuts if c.get("parent_cut") == parent_cut)
    # A.. Z then wrap to A2, B2 (rare) so it never collides
    letter = chr(ord("A") + n_children) if n_children < 26 else f"A{n_children}"
    return f"{parent_label}{letter}"


def select_memory(cuts: list, branch_from: int | None = None,
                  recent: int = 3) -> dict:
    """What the AI 'remembers' when creating the next cut.

    Normal cut: avoid repeating — exclude the recent genres and family-block the
    immediately previous one, so each cut explores a new direction.
    Branch: explore *around* the chosen parent — do NOT block its family, and
    surface the parent's genre as a seed hint so the branch stays in that vibe
    while varying the arrangement.
    """
    genres = [c.get("genre") for c in cuts if c.get("genre")]
    if branch_from is not None:
        parent_genre = next(
            (c.get("genre") for c in cuts if c.get("cut") == branch_from), None
        )
        return {"exclude_genres": [], "previous_genre": None,
                "prefer_genre": parent_genre}
    # Deduplicate while keeping order, take the most recent few as exclusions.
    seen, recent_genres = set(), []
    for g in reversed(genres):
        if g not in seen:
            seen.add(g)
            recent_genres.append(g)
        if len(recent_genres) >= recent:
            break
    return {"exclude_genres": recent_genres,
            "previous_genre": genres[-1] if genres else None,
            "prefer_genre": None}


# Five elite producers hearing the same artist — five distinct interpretations.
# The harmony/melody still follow the singer in every one (key/bar_degrees are
# vocal-derived regardless of genre), so each version genuinely fits the artist.
ARCHETYPES = ["rnb_neo_soul", "afrobeats", "pop_bright", "trap_melodic", "trap_dark"]


# Each archetype lane and the related genres that live in it (for branches that
# explore "different directions from a favourite" and for mapping taste → a lane).
_LANE_GENRES = {
    0: ["rnb_neo_soul", "rnb_smooth", "soul_ballad", "lofi_chill", "jazz_hop"],
    1: ["afrobeats", "amapiano", "dancehall", "reggaeton", "club_house"],
    2: ["pop_bright"],
    3: ["trap_melodic", "hiphop_modern", "hiphop_boom_bap"],
    4: ["trap_dark", "drill", "uk_drill", "phonk"],
}
_GENRE_LANE = {g: lane for lane, gs in _LANE_GENRES.items() for g in gs}


def _lane_of(genre: str | None) -> int | None:
    return _GENRE_LANE.get(genre)


def taste_genres(favorite_genres: list) -> list:
    """The artist's favourite genres, most-favourited first (the learned taste).
    Pure: takes the raw (weighted) list of genres with repeats."""
    counts: dict = {}
    order: list = []
    for g in favorite_genres:
        if not g:
            continue
        if g not in counts:
            order.append(g)
        counts[g] = counts.get(g, 0) + 1
    # sort by count desc, stable on first-seen order
    return sorted(order, key=lambda g: -counts[g])


# Relative strength of each feedback signal when learning the artist's taste.
# Accepting a cut (finishing a song with it) is the strongest endorsement;
# favouriting is a positive bookmark; branching from a cut means the artist
# chose to explore around that idea. Passing a cut over (heard it, generated
# another without any of the above) is weak evidence against — several skips
# together erode a genre's standing, and enough of them cancel an old favourite.
ACCEPT_WEIGHT = 3
FAVORITE_WEIGHT = 1
BRANCH_WEIGHT = 1
SKIP_WEIGHT = -0.25


def taste_signal(cuts_across_projects: list) -> list:
    """Build the weighted genre list from feedback across the artist's cuts.
    `cuts_across_projects` is a flat list of cut dicts (any project). Accepted
    cuts count ACCEPT_WEIGHT×, favourited FAVORITE_WEIGHT×. Feed to taste_genres."""
    weighted: list = []
    for c in cuts_across_projects:
        g = c.get("genre")
        if not g:
            continue
        if c.get("accepted"):
            weighted += [g] * ACCEPT_WEIGHT
        if c.get("favorite"):
            weighted += [g] * FAVORITE_WEIGHT
    return weighted


def taste_weights(project_cut_lists: list) -> dict:
    """Net taste weight per genre across the artist's projects.

    `project_cut_lists` is a list of per-project cut lists (NOT flattened —
    the skip signal needs to know each project's newest cut and branch links).

    Positive: accepted (ACCEPT_WEIGHT), favourite (FAVORITE_WEIGHT), and
    branched-from (BRANCH_WEIGHT — the artist chose to explore around it).
    Negative: a cut passed over — heard, then followed by another cut without
    being favourited/accepted/branched — earns SKIP_WEIGHT. The newest cut of a
    project is never penalised (the artist may not have reacted to it yet).
    """
    weights: dict = {}

    def bump(genre: str, w: float) -> None:
        weights[genre] = weights.get(genre, 0.0) + w

    for cuts in project_cut_lists:
        cuts = [c for c in (cuts or []) if isinstance(c, dict)]
        branched_from = {c.get("parent_cut") for c in cuts if c.get("parent_cut")}
        newest = max((c.get("cut", 0) for c in cuts), default=0)
        for c in cuts:
            g = c.get("genre")
            if not g:
                continue
            endorsed = False
            if c.get("accepted"):
                bump(g, ACCEPT_WEIGHT)
                endorsed = True
            if c.get("favorite"):
                bump(g, FAVORITE_WEIGHT)
                endorsed = True
            if c.get("cut") in branched_from:
                bump(g, BRANCH_WEIGHT)
                endorsed = True
            if not endorsed and c.get("cut", 0) != newest:
                bump(g, SKIP_WEIGHT)
    return weights


def ranked_taste(weights: dict) -> list:
    """Genres the artist has NET-positive feeling for, strongest first.
    A genre whose skips have cancelled its endorsements drops out entirely.
    Stable on insertion order for ties (older evidence first)."""
    return [g for g in sorted(weights, key=lambda g: -weights[g])
            if weights[g] > 0]


def _fit_start(analysis: dict, taste: list | None = None) -> int:
    """Index of the interpretation to LEAD with. The artist's learned taste wins
    when present (lead with their favourite lane); else fit it to the vocal."""
    if taste:
        lane = _lane_of(taste[0])
        if lane is not None:
            return lane
    val = float(analysis.get("valence", 0.5) or 0.5)
    tempo = float(analysis.get("tempo", 90) or 90)
    style = str(analysis.get("vocal_style", "") or "")
    if "rap" in style and tempo >= 110:
        return 3                      # rhythmic + fast → melodic rap first
    if val < 0.4:
        return 0                      # sad / intimate → R&B first
    if val > 0.7 and tempo >= 100:
        return 2                      # bright + up → pop first
    if tempo >= 120:
        return 1                      # fast/groovy → Afro first
    return 0                          # default → soulful R&B


def _branch_genre(cuts: list, parent_cut: int) -> str | None:
    """A branch explores a DIFFERENT DIRECTION within the favourite's vibe:
    rotate through the related genres in the parent's lane (3A, 3B, 3C ...)."""
    parent = next((c for c in cuts if c.get("cut") == parent_cut), None)
    if not parent or not parent.get("genre"):
        return None
    lane = _lane_of(parent["genre"])
    members = _LANE_GENRES.get(lane) if lane is not None else None
    if not members:
        return parent["genre"]
    n_children = sum(1 for c in cuts if c.get("parent_cut") == parent_cut)
    return members[n_children % len(members)]


def _explore_genre(cuts: list, taste: list, n_roots: int) -> str | None:
    """Free exploration (cut 6+). On alternating steps, REVISIT a loved direction
    (the artist's top favourite not used in the last couple of cuts); otherwise
    return None so the normal memory-driven selection explores something new."""
    if not taste:
        return None
    step = n_roots - len(ARCHETYPES)
    if step % 2 != 0:
        return None
    recent = {c.get("genre") for c in cuts[-2:]}
    for g in taste:
        if g not in recent:
            return g
    return None


def archetype_for(cuts: list, analysis: dict, branch_from: int | None = None) -> str | None:
    """Back-compat: the taste-free spread (see choose_force_genre)."""
    return choose_force_genre(cuts, analysis, branch_from=branch_from, taste=None)


def choose_force_genre(cuts: list, analysis: dict, branch_from: int | None = None,
                       taste: list | None = None) -> str | None:
    """The forced genre for the next cut, or None for free memory-driven explore.

    - Branch → a related genre in the parent's lane (different direction, same vibe).
    - First five ROOT cuts → the five producers' distinct interpretations, ordered
      to lead with the artist's learned favourite (taste), else the vocal's best fit.
    - Cut 6+ → revisit a loved direction on alternating steps, else free explore.
    The harmony/melody follow the vocal in every case, so each still fits the artist."""
    if branch_from is not None:
        return _branch_genre(cuts, branch_from)
    n_roots = sum(1 for c in cuts if c.get("parent_cut") in (None, 0))
    if n_roots < len(ARCHETYPES):
        start = _fit_start(analysis or {}, taste)
        return ARCHETYPES[(start + n_roots) % len(ARCHETYPES)]
    return _explore_genre(cuts, taste or [], n_roots)


def next_cut_number(cuts: list) -> int:
    return (max((c.get("cut", 0) for c in cuts), default=0)) + 1


def make_cut_record(cut: int, label: str, beat_key: str, genre: str,
                    genre_label: str, key: str, tempo: int, emotion: str,
                    score, parent_cut: int | None, created_at: str) -> dict:
    return {
        "cut": cut,
        "label": label,
        "beat_key": beat_key,
        "genre": genre,
        "genre_label": genre_label,
        "key": key,
        "tempo": tempo,
        "emotion": emotion,
        "score": score,
        "parent_cut": parent_cut,
        "favorite": False,
        "accepted": False,
        "created_at": created_at,
    }


def public_cut(c: dict, beat_url: str | None = None) -> dict:
    """Cut record for API responses (optionally with a fresh signed URL)."""
    out = {k: c.get(k) for k in (
        "cut", "label", "genre", "genre_label", "key", "tempo", "emotion",
        "score", "parent_cut", "favorite", "accepted", "created_at",
        "note_en", "note_ar")}
    if beat_url is not None:
        out["beat_url"] = beat_url
    return out
