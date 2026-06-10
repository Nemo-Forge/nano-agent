"""Subprocess plugin protocol and discovery.

A plugin is any executable described by a `<name>.tool.json` manifest in the
plugin directory (default ~/.nano-agent/tools/). nano-agent invokes the
plugin's command, writes a JSON request to its stdin, and reads a JSON
response from stdout:

  request  (stdin):  {"args": {...}}
  response (stdout): {"ok": true, "result": "..."}
                  or {"ok": false, "error": "..."}

This keeps tools language-agnostic (bash, C, Rust, Go, Python) and sandboxable,
without nano-agent having to import anything the plugin needs.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from nano_agent.tools import Tool, ToolError

MANIFEST_GLOB = "*.tool.json"
_REQUIRED_FIELDS = ("name", "description", "command")


@dataclass
class PluginManifest:
    name: str
    description: str
    command: list[str]
    cwd: str  # the manifest's directory; commands run from here so relative paths work
    args_hint: str = "{}"
    timeout: int = 30


def load_manifest(path: Path) -> PluginManifest:
    """Parse and validate a single manifest file. Raises ValueError if invalid."""
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("manifest must be a JSON object")
    for field in _REQUIRED_FIELDS:
        if field not in data:
            raise ValueError(f"missing required field: {field}")

    command = data["command"]
    if not isinstance(command, list) or not command or not all(isinstance(c, str) for c in command):
        raise ValueError("'command' must be a non-empty list of strings")

    return PluginManifest(
        name=str(data["name"]),
        description=str(data["description"]),
        command=command,
        cwd=str(path.parent.resolve()),
        args_hint=str(data.get("args_hint", "{}")),
        timeout=int(data.get("timeout", 30)),
    )


def _make_subprocess_tool(manifest: PluginManifest) -> Tool:
    def run(args: dict) -> str:
        request = json.dumps({"args": args})
        try:
            proc = subprocess.run(
                manifest.command,
                input=request,
                capture_output=True,
                text=True,
                timeout=manifest.timeout,
                cwd=manifest.cwd,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(f"plugin '{manifest.name}' timed out") from None
        except OSError as e:
            raise ToolError(f"plugin '{manifest.name}' failed to launch: {e}") from e

        if proc.returncode != 0:
            err = (proc.stderr or "").strip() or f"exit {proc.returncode}"
            raise ToolError(f"plugin '{manifest.name}' failed: {err}")
        try:
            resp = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise ToolError(f"plugin '{manifest.name}' returned invalid JSON") from None
        if not isinstance(resp, dict):
            raise ToolError(f"plugin '{manifest.name}' returned a non-object response")
        if not resp.get("ok", False):
            raise ToolError(str(resp.get("error", "plugin reported failure")))
        return str(resp.get("result", ""))

    return Tool(
        name=manifest.name,
        description=manifest.description,
        args_hint=manifest.args_hint,
        run=run,
    )


def discover_plugins(plugin_dir: Path) -> tuple[list[Tool], list[str]]:
    """Scan plugin_dir for manifests. Returns (tools, warnings).

    Invalid manifests are skipped with a warning rather than aborting startup.
    """
    tools: list[Tool] = []
    warnings: list[str] = []
    if not plugin_dir.is_dir():
        return tools, warnings

    for manifest_path in sorted(plugin_dir.glob(MANIFEST_GLOB)):
        try:
            manifest = load_manifest(manifest_path)
        except (ValueError, OSError, json.JSONDecodeError) as e:
            warnings.append(f"skipped {manifest_path.name}: {e}")
            continue
        tools.append(_make_subprocess_tool(manifest))
    return tools, warnings
