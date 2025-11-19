"""A very small DAG-aware scheduler example.

This scheduler expects Task objects to have a `dependencies` attribute
containing task IDs that must complete before scheduling.
"""
from __future__ import annotations

from typing import Optional, Set

from schedulers.base_scheduler import BaseScheduler
from schedulers.tasks.task1 import Task
from schedulers.agents.base_agent import BaseAgent


class DAGScheduler(BaseScheduler):
    def __init__(self) -> None:
        super().__init__()
        self._completed: Set[str] = set()

    def mark_completed(self, task: Task) -> None:
        self._completed.add(task.id)

    def schedule(self) -> Optional[tuple[Task, BaseAgent]]:
        avail = self.available_agents()
        if not avail:
            return None

        # find first task whose dependencies are satisfied
        for idx, task in enumerate(self.tasks):
            deps = getattr(task, "dependencies", []) or []
            if all(d in self._completed for d in deps):
                self.tasks.pop(idx)
                agent = avail[0]
                agent.assign_task(task)
                return task, agent

        return None

