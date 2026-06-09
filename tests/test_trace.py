"""Tests for JSONL tracing."""

import json

from nano_agent.trace import Tracer


def test_tracer_writes_jsonl(tmp_path):
    path = tmp_path / "trace.jsonl"
    with Tracer(path) as t:
        assert t.enabled
        t.event("run_start", task="hello")
        t.event("step", action="tool", tool="bash")
        t.event("run_end", reason="done")

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first["event"] == "run_start"
    assert first["task"] == "hello"
    assert "ts" in first


def test_tracer_disabled_is_noop():
    t = Tracer(None)
    assert not t.enabled
    t.event("step", tool="bash")  # must not raise
    t.close()


def test_tracer_appends(tmp_path):
    path = tmp_path / "trace.jsonl"
    with Tracer(path) as t:
        t.event("a")
    with Tracer(path) as t:
        t.event("b")
    assert len(path.read_text().strip().splitlines()) == 2
