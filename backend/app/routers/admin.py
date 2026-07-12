"""
Admin metrics endpoint.

GET /admin/beat-metrics
    Returns real production telemetry aggregated from beat_telemetry.
    Protected by ADMIN_KEY (Bearer token in Authorization header).

Usage:
    curl -H "Authorization: Bearer <ADMIN_KEY>" \
         https://<host>/admin/beat-metrics
"""
import logging
from collections import Counter
from fastapi import APIRouter, HTTPException, Header
from supabase import create_client
from ..config import settings

router = APIRouter(prefix="/admin", tags=["admin"])
logger = logging.getLogger(__name__)

_TIER_NAMES = {
    "musicgen_local":  "Tier 1 (local MusicGen)",
    "musicgen_hf_api": "Tier 2 (HF API)",
    "musicgen_gradio": "Tier 3 (Gradio Space)",
    "failed":          "failed",
}


def _tier_label(genre_used: str) -> str:
    if genre_used in _TIER_NAMES:
        return _TIER_NAMES[genre_used]
    return "Tier 4 (synthesizer)"


@router.get("/beat-metrics")
def beat_metrics(
    authorization: str = Header(default=""),
    limit: int = 500,
):
    """
    Aggregate beat generation telemetry.

    Query params:
        limit   How many most-recent events to include (default 500, max 2000)
    """
    # ── Auth ──────────────────────────────────────────────────────────────────
    admin_key = getattr(settings, "admin_key", "")
    if not admin_key:
        raise HTTPException(
            status_code=503,
            detail="ADMIN_KEY not configured on this server",
        )
    if authorization != f"Bearer {admin_key}":
        raise HTTPException(status_code=403, detail="Forbidden")

    limit = min(max(limit, 1), 2000)

    # ── Fetch telemetry ───────────────────────────────────────────────────────
    supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)
    result = (
        supabase.table("beat_telemetry")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    rows = result.data or []

    if not rows:
        return {
            "message": "No telemetry data yet. Deploy and wait for real generations.",
            "count": 0,
        }

    total = len(rows)
    successes = [r for r in rows if r.get("success")]
    failures  = [r for r in rows if not r.get("success")]

    # ── Tier distribution ─────────────────────────────────────────────────────
    tier_counts: Counter = Counter()
    for r in successes:
        tier_counts[_tier_label(r.get("tier_used", ""))] += 1

    tier_distribution = {
        tier: {
            "count": count,
            "pct":   round(100.0 * count / total, 1),
        }
        for tier, count in sorted(tier_counts.items())
    }

    # ── Tier attempt analysis (which tiers were tried before success) ─────────
    tier_attempt_totals: Counter = Counter()
    for r in rows:
        for attempt in r.get("tier_attempts") or []:
            tier_attempt_totals[attempt.get("name", "?")] += 1

    tier_attempt_failure_rates: dict = {}
    for r in rows:
        for attempt in r.get("tier_attempts") or []:
            name = attempt.get("name", "?")
            if not attempt.get("success"):
                tier_attempt_failure_rates[name] = tier_attempt_failure_rates.get(name, 0) + 1

    tier_attempt_summary = {
        name: {
            "attempts": tier_attempt_totals[name],
            "failures": tier_attempt_failure_rates.get(name, 0),
            "failure_rate_pct": round(
                100.0 * tier_attempt_failure_rates.get(name, 0) / tier_attempt_totals[name], 1
            ) if tier_attempt_totals[name] else 0,
        }
        for name in tier_attempt_totals
    }

    # ── Durations ─────────────────────────────────────────────────────────────
    durations = [r["duration_ms"] for r in rows if r.get("duration_ms") is not None]
    avg_duration_ms = round(sum(durations) / len(durations)) if durations else None

    tier_durations: dict[str, list[int]] = {}
    for r in successes:
        tier = _tier_label(r.get("tier_used", ""))
        if r.get("duration_ms") is not None:
            tier_durations.setdefault(tier, []).append(r["duration_ms"])
    avg_duration_by_tier = {
        tier: round(sum(ds) / len(ds))
        for tier, ds in tier_durations.items()
    }

    # ── Scores ────────────────────────────────────────────────────────────────
    scores = [
        float(r["final_score"])
        for r in rows
        if r.get("final_score") is not None
    ]
    avg_score = round(sum(scores) / len(scores), 1) if scores else None

    # ── Failure analysis ──────────────────────────────────────────────────────
    failure_rate_pct = round(100.0 * len(failures) / total, 1)
    failure_reasons  = Counter(
        r.get("failure_reason") or "unknown" for r in failures
    )

    # ── Regeneration rate ─────────────────────────────────────────────────────
    # A project is "regenerated" if it appears more than once in telemetry.
    project_gen_counts: Counter = Counter(r["project_id"] for r in rows)
    projects_with_regen = sum(1 for v in project_gen_counts.values() if v > 1)
    regen_rate_pct = round(
        100.0 * projects_with_regen / len(project_gen_counts), 1
    ) if project_gen_counts else 0

    # ── Genre distribution (Tier 4 breakdown) ────────────────────────────────
    tier4_genres: Counter = Counter()
    for r in successes:
        if _tier_label(r.get("tier_used", "")) == "Tier 4 (synthesizer)":
            tier4_genres[r.get("selected_genre", "unknown")] += 1

    # ── Analysis summary distribution ────────────────────────────────────────
    emotions: Counter = Counter()
    for r in rows:
        summary = r.get("analysis_summary") or {}
        if summary.get("emotion"):
            emotions[summary["emotion"]] += 1

    # ── Most recent 10 events (for spot-checking) ────────────────────────────
    recent = [
        {
            "project_id":   r["project_id"],
            "created_at":   r["created_at"],
            "tier_used":    _tier_label(r.get("tier_used", "")),
            "duration_ms":  r.get("duration_ms"),
            "final_score":  r.get("final_score"),
            "success":      r.get("success"),
            "failure_reason": r.get("failure_reason"),
        }
        for r in rows[:10]
    ]

    return {
        "window":                   f"last {total} generations",
        "total_generations":        total,
        "success_count":            len(successes),
        "failure_count":            len(failures),

        "tier_distribution":        tier_distribution,
        "tier_attempt_summary":     tier_attempt_summary,

        "avg_duration_ms":          avg_duration_ms,
        "avg_duration_by_tier_ms":  avg_duration_by_tier,

        "avg_final_score":          avg_score,
        "failure_rate_pct":         failure_rate_pct,
        "top_failure_causes":       dict(failure_reasons.most_common(5)),

        "beat_regeneration_rate_pct": regen_rate_pct,

        "tier4_genre_distribution": dict(tier4_genres.most_common(10)),
        "emotion_distribution":     dict(emotions.most_common()),

        "recent_10": recent,
    }
