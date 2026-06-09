"""Tests for the python tool and the persistent memory tool."""

from nano_agent.tools import ToolRegistry, default_tools


def _reg(tmp_path):
    return ToolRegistry(default_tools(memory_path=str(tmp_path / "mem.json")))


def test_python_tool_runs_code(tmp_path):
    out = _reg(tmp_path).get("python").run({"code": "print(6 * 7)"})
    assert out.strip() == "42"


def test_python_tool_reports_error(tmp_path):
    out = _reg(tmp_path).get("python").run({"code": "raise ValueError('boom')"})
    assert "ValueError" in out
    assert "[exit" in out


def test_memory_set_get_list_delete(tmp_path):
    mem = _reg(tmp_path).get("memory")
    assert mem.run({"op": "set", "key": "name", "value": "nemo"}) == "saved name"
    assert mem.run({"op": "get", "key": "name"}) == "nemo"
    assert "name" in mem.run({"op": "list"})
    assert mem.run({"op": "delete", "key": "name"}) == "deleted name"
    assert "[no value" in mem.run({"op": "get", "key": "name"})


def test_memory_persists_across_instances(tmp_path):
    _reg(tmp_path).get("memory").run({"op": "set", "key": "k", "value": "v"})
    # A fresh registry/store reading the same file must see the value.
    assert _reg(tmp_path).get("memory").run({"op": "get", "key": "k"}) == "v"


def test_memory_without_path_is_absent():
    reg = ToolRegistry(default_tools(memory_path=None))
    assert "memory" not in reg.names()
    assert "python" in reg.names()
