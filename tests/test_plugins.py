"""Tests for the subprocess plugin protocol and discovery."""

import json

import pytest

from nano_agent.plugins import discover_plugins, load_manifest
from nano_agent.tools import ToolError

ECHO_PLUGIN = """#!/usr/bin/env python3
import json, sys
req = json.load(sys.stdin)
msg = req.get("args", {}).get("msg", "")
print(json.dumps({"ok": True, "result": "echo: " + msg}))
"""

FAIL_PLUGIN = """#!/usr/bin/env python3
import json, sys
print(json.dumps({"ok": False, "error": "always fails"}))
"""

CRASH_PLUGIN = """#!/usr/bin/env python3
import sys
sys.stderr.write("boom\\n")
sys.exit(2)
"""


def _write_plugin(plugin_dir, name, script, manifest_extra=None):
    plugin_dir.mkdir(parents=True, exist_ok=True)
    script_path = plugin_dir / f"{name}.py"
    script_path.write_text(script)
    manifest = {
        "name": name,
        "description": f"{name} plugin",
        "args_hint": '{"msg": "hi"}',
        "command": ["python3", f"{name}.py"],
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    (plugin_dir / f"{name}.tool.json").write_text(json.dumps(manifest))


def test_discover_and_run_plugin(tmp_path):
    _write_plugin(tmp_path, "echo", ECHO_PLUGIN)
    tools, warnings = discover_plugins(tmp_path)
    assert warnings == []
    assert len(tools) == 1
    assert tools[0].name == "echo"
    assert tools[0].run({"msg": "hello"}) == "echo: hello"


def test_plugin_reporting_failure_raises_toolerror(tmp_path):
    _write_plugin(tmp_path, "fail", FAIL_PLUGIN)
    tool = discover_plugins(tmp_path)[0][0]
    with pytest.raises(ToolError, match="always fails"):
        tool.run({})


def test_plugin_nonzero_exit_raises_toolerror(tmp_path):
    _write_plugin(tmp_path, "crash", CRASH_PLUGIN)
    tool = discover_plugins(tmp_path)[0][0]
    with pytest.raises(ToolError, match="crash"):
        tool.run({})


def test_invalid_manifest_is_skipped_with_warning(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "broken.tool.json").write_text('{"name": "x"}')  # missing fields
    tools, warnings = discover_plugins(tmp_path)
    assert tools == []
    assert len(warnings) == 1
    assert "broken.tool.json" in warnings[0]


def test_missing_plugin_dir_is_empty(tmp_path):
    tools, warnings = discover_plugins(tmp_path / "does-not-exist")
    assert tools == []
    assert warnings == []


def test_command_runs_from_manifest_dir(tmp_path):
    # A relative command works because the plugin runs with cwd = manifest dir.
    _write_plugin(tmp_path, "echo", ECHO_PLUGIN)
    manifest = load_manifest(tmp_path / "echo.tool.json")
    assert manifest.cwd == str(tmp_path.resolve())
    assert manifest.command == ["python3", "echo.py"]
