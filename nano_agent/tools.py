"""Built-in tools and the tool registry.

A tool takes a dict of args and returns a string result. Tools raise
ToolError on failure; the agent loop turns that into an Observation the
model can react to, rather than crashing the run.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


class ToolError(Exception):
    """Raised by a tool when it cannot complete the requested action."""


@dataclass
class Tool:
    name: str
    description: str
    args_hint: str
    run: Callable[[dict], str]


def _bash(args: dict) -> str:
    cmd = args.get("cmd")
    if not cmd:
        raise ToolError("bash requires a 'cmd' argument")
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=args.get("timeout", 60),
        )
    except subprocess.TimeoutExpired:
        raise ToolError(f"command timed out: {cmd}") from None
    out = (proc.stdout or "") + (proc.stderr or "")
    out = out.strip()
    if proc.returncode != 0:
        return f"[exit {proc.returncode}]\n{out}"
    return out or "[no output]"


def _read_file(args: dict) -> str:
    path = args.get("path")
    if not path:
        raise ToolError("read_file requires a 'path' argument")
    p = Path(path).expanduser()
    if not p.is_file():
        raise ToolError(f"not a file: {path}")
    try:
        text = p.read_text(errors="replace")
    except OSError as e:
        raise ToolError(f"could not read {path}: {e}") from e
    max_chars = args.get("max_chars", 8000)
    if len(text) > max_chars:
        return text[:max_chars] + f"\n[truncated, {len(text)} chars total]"
    return text


def _write_file(args: dict) -> str:
    path = args.get("path")
    content = args.get("content")
    if not path or content is None:
        raise ToolError("write_file requires 'path' and 'content' arguments")
    p = Path(path).expanduser()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.get("append") else "w"
        with p.open(mode) as f:
            f.write(content)
    except OSError as e:
        raise ToolError(f"could not write {path}: {e}") from e
    return f"wrote {len(content)} chars to {path}"


BUILTIN_TOOLS: list[Tool] = [
    Tool(
        name="bash",
        description="Run a shell command and return its output.",
        args_hint='{"cmd": "ls -la /tmp"}',
        run=_bash,
    ),
    Tool(
        name="read_file",
        description="Read a text file.",
        args_hint='{"path": "/etc/hostname"}',
        run=_read_file,
    ),
    Tool(
        name="write_file",
        description="Write text to a file (set append=true to append).",
        args_hint='{"path": "out.txt", "content": "hello"}',
        run=_write_file,
    ),
]


class ToolRegistry:
    """Holds the tools available to an agent, filtered by an allowlist."""

    def __init__(self, tools: list[Tool], allowed: list[str] | None = None) -> None:
        self._tools = {t.name: t for t in tools}
        if allowed is not None:
            self._tools = {n: t for n, t in self._tools.items() if n in allowed}

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(f"unknown tool: {name}")
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> str:
        """Render the tool list for the system prompt."""
        lines = []
        for t in self._tools.values():
            lines.append(f"- {t.name}: {t.description} args example: {t.args_hint}")
        return "\n".join(lines)
