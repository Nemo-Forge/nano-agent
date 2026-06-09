"""Core data types for the agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Action(str, Enum):
    """What the planner decided to do this step."""

    TOOL = "tool"
    DONE = "done"


@dataclass
class Step:
    """A single planner decision: either call a tool or finish."""

    action: Action
    thought: str = ""
    tool: str | None = None
    args: dict = field(default_factory=dict)
    answer: str | None = None  # set when action == DONE


@dataclass
class Observation:
    """Result of executing a step's tool call."""

    step: Step
    ok: bool
    result: str
    error: str | None = None


@dataclass
class AgentResult:
    """Final outcome of an agent run."""

    answer: str
    steps_taken: int
    history: list[Observation]
    stopped_reason: str  # "done" | "max_steps" | "error"
