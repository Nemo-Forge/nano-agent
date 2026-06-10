"""Tests for v0.5 features: sliding window, offline http_get, failure breaker."""

import json

import pytest

from nano_agent.agent import Agent
from nano_agent.prompt import build_prompt
from nano_agent.tools import ToolError, default_tools
from nano_agent.types import Action, Observation, Step


class FakeBackend:
    def __init__(self, outputs):
        self._outputs = outputs
        self._i = 0

    def generate(self, prompt, max_tokens, stop):
        out = self._outputs[min(self._i, len(self._outputs) - 1)]
        self._i += 1
        return out


def _obs(i):
    return Observation(step=Step(action=Action.TOOL, tool="bash"), ok=True, result=f"r{i}")


def test_sliding_window_keeps_only_recent(tmp_path):
    history = [_obs(i) for i in range(5)]
    prompt = build_prompt("task", "tools", history, window=2)
    assert "r4" in prompt and "r3" in prompt
    assert "r0" not in prompt and "r1" not in prompt
    assert "3 earlier step(s) omitted" in prompt


def test_window_zero_keeps_everything():
    history = [_obs(i) for i in range(3)]
    prompt = build_prompt("task", "tools", history, window=0)
    assert all(f"r{i}" in prompt for i in range(3))
    assert "omitted" not in prompt


def test_http_get_blocked_offline():
    tool = next(t for t in default_tools(offline_only=True) if t.name == "http_get")
    with pytest.raises(ToolError, match="offline"):
        tool.run({"url": "https://example.com"})


def test_http_get_present_when_online():
    tool = next(t for t in default_tools(offline_only=False) if t.name == "http_get")
    # No network call here: a missing url should raise before any request.
    with pytest.raises(ToolError, match="url"):
        tool.run({})


def test_consecutive_errors_circuit_breaker():
    # Always emits unparseable output -> should stop at max_consecutive_errors.
    agent = Agent(FakeBackend(["not json"]), max_consecutive_errors=3)
    result = agent.run("never succeeds")
    assert result.stopped_reason == "too_many_errors"
    assert result.steps_taken == 3


def test_errors_reset_on_success(tmp_path):
    target = tmp_path / "f.txt"
    write_step = json.dumps(
        {"action": "tool", "tool": "write_file", "args": {"path": str(target), "content": "x"}}
    )
    outputs = [
        "garbage",  # error 1
        write_step,  # ok -> resets the counter
        "garbage",  # error 1 again
        '{"action":"done","answer":"done"}',
    ]
    agent = Agent(FakeBackend(outputs), max_consecutive_errors=2)
    result = agent.run("mixed")
    # Without the reset, two errors would trip the breaker; here it finishes.
    assert result.stopped_reason == "done"
