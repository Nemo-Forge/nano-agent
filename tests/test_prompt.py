"""Tests for planner-output parsing — the most failure-prone part with small models."""

import pytest

from nano_agent.prompt import parse_step
from nano_agent.types import Action


def test_parse_tool_step():
    raw = '{"thought": "list files", "action": "tool", "tool": "bash", "args": {"cmd": "ls"}}'
    step = parse_step(raw)
    assert step.action == Action.TOOL
    assert step.tool == "bash"
    assert step.args == {"cmd": "ls"}


def test_parse_done_step():
    raw = '{"thought": "have answer", "action": "done", "answer": "42"}'
    step = parse_step(raw)
    assert step.action == Action.DONE
    assert step.answer == "42"


def test_parse_ignores_prose_around_json():
    raw = 'Sure! Here is my step:\n{"action": "done", "answer": "ok"}\nHope that helps.'
    step = parse_step(raw)
    assert step.action == Action.DONE
    assert step.answer == "ok"


def test_parse_handles_nested_braces_and_strings():
    raw = '{"action": "tool", "tool": "write_file", "args": {"path": "a", "content": "{not json}"}}'
    step = parse_step(raw)
    assert step.args["content"] == "{not json}"


def test_parse_raises_on_no_json():
    with pytest.raises(ValueError):
        parse_step("I have no idea what to do.")


def test_parse_raises_on_unknown_action():
    with pytest.raises(ValueError):
        parse_step('{"action": "explode"}')
