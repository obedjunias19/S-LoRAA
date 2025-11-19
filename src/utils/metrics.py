"""Small metrics helpers used by schedulers/tests."""
from __future__ import annotations

from typing import Iterable


def average_wait_time(wait_times: Iterable[float]) -> float:
    data = list(wait_times)
    if not data:
        return 0.0
    return sum(data) / len(data)


def throughput(completed_count: int, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0:
        return float("inf") if completed_count > 0 else 0.0
    return completed_count / elapsed_seconds

