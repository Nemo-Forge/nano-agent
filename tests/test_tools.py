"""Tests for built-in tools and the registry."""

import pytest

from nano_agent.tools import BUILTIN_TOOLS, ToolError, ToolRegistry


def test_bash_runs_and_captures_output():
    reg = ToolRegistry(BUILTIN_TOOLS)
    out = reg.get("bash").run({"cmd": "echo hello"})
    assert "hello" in out


def test_bash_nonzero_exit_is_reported():
    reg = ToolRegistry(BUILTIN_TOOLS)
    out = reg.get("bash").run({"cmd": "exit 3"})
    assert "[exit 3]" in out


def test_bash_missing_cmd_raises():
    reg = ToolRegistry(BUILTIN_TOOLS)
    with pytest.raises(ToolError):
        reg.get("bash").run({})


def test_read_write_roundtrip(tmp_path):
    reg = ToolRegistry(BUILTIN_TOOLS)
    p = tmp_path / "f.txt"
    reg.get("write_file").run({"path": str(p), "content": "data"})
    assert reg.get("read_file").run({"path": str(p)}) == "data"


def test_read_missing_file_raises():
    reg = ToolRegistry(BUILTIN_TOOLS)
    with pytest.raises(ToolError):
        reg.get("read_file").run({"path": "/no/such/file/here"})


def test_allowlist_filters_tools():
    reg = ToolRegistry(BUILTIN_TOOLS, allowed=["read_file"])
    assert reg.names() == ["read_file"]
    with pytest.raises(ToolError):
        reg.get("bash")


def test_describe_lists_tools():
    reg = ToolRegistry(BUILTIN_TOOLS)
    desc = reg.describe()
    assert "bash" in desc and "read_file" in desc and "write_file" in desc
