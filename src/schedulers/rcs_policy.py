"""A simple round-robin / resource centered scheduler example.

This scheduler cycles through available agents and assigns tasks in order.
"""
from __future__ import annotations

from typing import Optional

from schedulers.base_scheduler import BaseScheduler
from schedulers.tasks.task1 import Task
from schedulers.agents.base_agent import BaseAgent


class RCSScheduler(BaseScheduler):
    def __init__(self) -> None:
        super().__init__()
        self._idx = 0

    def schedule(self) -> Optional[tuple[Task, BaseAgent]]:
        avail = self.available_agents()
        if not avail or not self.tasks:
            return None

        task = self.tasks.pop(0)

        # simple round-robin over the available agents
        agent = avail[self._idx % len(avail)]
        self._idx += 1
        agent.assign_task(task)
        return task, agent

