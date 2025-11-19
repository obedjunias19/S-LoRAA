"""A second example task module (keeps compatibility with tests).

This file defines an alternative Task class to show how multiple task
types might coexist. For now it is a thin wrapper around the primary
`Task` class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass
class TaskTwo:
    id: str
    payload: Any = None
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"<TaskTwo id={self.id} deps={self.dependencies}>"

