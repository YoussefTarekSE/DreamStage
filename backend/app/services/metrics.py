from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from threading import Lock

_lock = Lock()
_counters: dict[str, int] = defaultdict(int)
_timings: dict[str, list[int]] = defaultdict(list)


def increment(name: str, value: int = 1) -> None:
    with _lock:
        _counters[name] += value


def observe_ms(name: str, duration_ms: int) -> None:
    with _lock:
        values = _timings[name]
        values.append(max(0, int(duration_ms)))
        if len(values) > 1000:
            del values[: len(values) - 1000]


@contextmanager
def time_ms(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        observe_ms(name, int((time.perf_counter() - start) * 1000))


def snapshot() -> dict:
    with _lock:
        timings = {}
        for name, values in _timings.items():
            if not values:
                continue
            sorted_values = sorted(values)
            timings[name] = {
                "count": len(values),
                "avg_ms": round(sum(values) / len(values)),
                "p95_ms": sorted_values[min(len(sorted_values) - 1, int(len(sorted_values) * 0.95))],
                "max_ms": max(values),
            }
        return {
            "counters": dict(_counters),
            "timings": timings,
        }
