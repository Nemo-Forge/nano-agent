"""Agent loop tests using a scripted fake backend (no model required)."""

from nano_agent.agent import Agent
from nano_agent.tools import BUILTIN_TOOLS, ToolRegistry


class FakeBackend:
    """Returns pre-scripted step outputs, one per generate() call."""

    def __init__(self, outputs: list[str]) -> None:
        self._outputs = outputs
        self._i = 0

    def generate(self, prompt: str, max_tokens: int, stop: list[str]) -> str:
        out = self._outputs[self._i]
        self._i += 1
        return out


def _agent(outputs, **kw):
    return Agent(FakeBackend(outputs), ToolRegistry(BUILTIN_TOOLS), **kw)


def test_immediate_done():
    agent = _agent(['{"action": "done", "answer": "hello"}'])
    result = agent.run("say hello")
    assert result.answer == "hello"
    assert result.stopped_reason == "done"
    assert result.steps_taken == 0


def test_tool_then_done(tmp_path):
    target = tmp_path / "note.txt"
    outputs = [
        f'{{"action":"tool","tool":"write_file","args":{{"path":"{target}","content":"hi"}}}}',
        '{"action":"done","answer":"wrote the file"}',
    ]
    result = _agent(outputs).run("write a file")
    assert result.stopped_reason == "done"
    assert target.read_text() == "hi"
    assert result.steps_taken == 1


def test_max_steps_guard():
    # Always asks for a tool, never finishes -> must stop at max_steps.
    loop = '{"action":"tool","tool":"bash","args":{"cmd":"true"}}'
    result = _agent([loop] * 10, max_steps=3).run("loop forever")
    assert result.stopped_reason == "max_steps"
    assert result.steps_taken == 3


def test_malformed_output_recovers():
    outputs = [
        "I am confused and will not emit JSON.",
        '{"action":"done","answer":"recovered"}',
    ]
    result = _agent(outputs).run("recover")
    assert result.stopped_reason == "done"
    assert result.answer == "recovered"


def test_unknown_tool_becomes_observation_not_crash():
    outputs = [
        '{"action":"tool","tool":"nonexistent","args":{}}',
        '{"action":"done","answer":"handled"}',
    ]
    result = _agent(outputs).run("use a bad tool")
    assert result.stopped_reason == "done"
    assert result.history[0].ok is False
    assert "unknown tool" in (result.history[0].error or "")
