"""A simple LRU-style scheduler implementation.

This example scheduler picks the first ready task and assigns it to the
least-recently-used available agent (by an internal `last_used` attribute).
"""
from __future__ import annotations

from typing import Optional

from schedulers.base_scheduler import BaseScheduler
from schedulers.tasks.task1 import Task
from schedulers.agents.base_agent import BaseAgent


class LRUScheduler(BaseScheduler):
    def schedule(self) -> Optional[tuple[Task, BaseAgent]]:
        avail = self.available_agents()
        if not avail or not self.tasks:
            return None

        # pick the oldest-ready task (FIFO)
        task = self.tasks.pop(0)

        # choose agent with smallest last_used (None counts as oldest)
        def key(a: BaseAgent):
            return a.last_used if a.last_used is not None else -1

        agent = sorted(avail, key=key)[0]
        agent.assign_task(task)
        return task, agent

