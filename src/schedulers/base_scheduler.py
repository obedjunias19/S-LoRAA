"""Base scheduler abstractions and helpers.

This module provides a small but practical BaseScheduler class used by
policy implementations in this package. It is intentionally lightweight
so it can be extended in tests or real schedulers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from schedulers.agents.base_agent import BaseAgent
from schedulers.tasks.task1 import Task


class BaseScheduler(ABC):
    """Abstract base scheduler.

    Responsibilities:
    - Maintain a registry of tasks and agents
    - Provide helper utilities for simple scheduling policies
    - Define the abstract `schedule()` method
    """

    def __init__(self) -> None:
        self.tasks: List[Task] = []
        self.agents: List[BaseAgent] = []

    def register_task(self, task: Task) -> None:
        self.tasks.append(task)

    def register_agent(self, agent: BaseAgent) -> None:
        self.agents.append(agent)

    def available_agents(self) -> List[BaseAgent]:
        return [a for a in self.agents if a.can_accept()]

    @abstractmethod
    def schedule(self) -> Optional[tuple[Task, BaseAgent]]:
        """Perform one scheduling decision.

        Returns a (task, agent) pair if a scheduling decision was made,
        otherwise None.
        """

