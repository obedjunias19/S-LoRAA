"""Simple Task dataclass used by scheduler examples/tests."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class Task:
    id: str
    payload: Any = None
    dependencies: List[str] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"<Task id={self.id} deps={self.dependencies}>"

