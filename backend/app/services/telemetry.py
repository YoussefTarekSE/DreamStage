"""
Beat generation telemetry.

Writes one row to beat_telemetry after every generation attempt (success or
failure).  All failures are swallowed so a telemetry bug never breaks the
user-facing request.
"""
import logging

logger = logging.getLogger(__name__)


def record_beat_event(
    supabase,
    *,
    project_id: str,
    tier_used: str,
    tier_attempts: list,
    duration_ms: int,
    analysis: dict,
    genre: str,
    success: bool,
    failure_reason: str | None = None,
) -> None:
    """
    Insert one row into beat_telemetry.  Silently drops on any error.
    Must be called with the Supabase service-role client.
    """
    try:
        beat_score = analysis.get("beat_score") or {}
        final_score = beat_score.get("total") if isinstance(beat_score, dict) else None

        analysis_summary = {
            k: analysis.get(k)
            for k in ("tempo", "key", "mode", "emotion", "valence", "rms")
            if analysis.get(k) is not None
        }

        row = {
            "project_id":      project_id,
            "tier_used":       tier_used,
            "tier_attempts":   tier_attempts,
            "duration_ms":     duration_ms,
            "analysis_summary": analysis_summary,
            "selected_genre":  genre,
            "selected_bpm":    int(analysis.get("tempo") or 0) or None,
            "candidate_scores": beat_score if isinstance(beat_score, dict) else None,
            "final_score":     final_score,
            "success":         success,
            "failure_reason":  failure_reason,
        }

        supabase.table("beat_telemetry").insert(row).execute()

    except Exception as exc:
        logger.warning("[telemetry] failed to record beat event: %s", exc)
