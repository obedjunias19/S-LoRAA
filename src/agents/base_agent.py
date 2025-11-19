"""Lightweight BaseAgent class used by schedulers in examples/tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from time import monotonic

from schedulers.tasks.task1 import Task


class BaseAgent:
    def __init__(self, agent_id: str, capacity: int = 1) -> None:
        self.id = agent_id
        self.capacity = capacity
        self.busy = 0
        self.last_used: Optional[float] = None

    def can_accept(self) -> bool:
        return self.busy < self.capacity

    def assign_task(self, task: Task) -> None:
        if not self.can_accept():
            raise RuntimeError("Agent is at capacity")
        self.busy += 1
        self.last_used = monotonic()

    def complete_task(self) -> None:
        if self.busy > 0:
            self.busy -= 1

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"<BaseAgent id={self.id} busy={self.busy}/{self.capacity}>"

