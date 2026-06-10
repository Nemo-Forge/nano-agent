"""Built-in tools and the tool registry.

A tool takes a dict of args and returns a string result. Tools raise
ToolError on failure; the agent loop turns that into an Observation the
model can react to, rather than crashing the run.
"""

from __future__ import annotations

import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from nano_agent.memory import MemoryStore


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


def _python(args: dict) -> str:
    code = args.get("code")
    if not code:
        raise ToolError("python requires a 'code' argument")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=args.get("timeout", 30),
        )
    except subprocess.TimeoutExpired:
        raise ToolError("python snippet timed out") from None
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if proc.returncode != 0:
        return f"[exit {proc.returncode}]\n{out}"
    return out or "[no output]"


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
    Tool(
        name="python",
        description="Run a short Python snippet and return its output.",
        args_hint='{"code": "print(2 + 2)"}',
        run=_python,
    ),
]


def make_http_get_tool(offline_only: bool) -> Tool:
    """Build an http_get tool. When offline_only is set, the tool refuses to run.

    The tool stays visible to the model even when blocked, so the model gets a
    clear 'offline mode' error rather than silently missing a capability.
    """

    def run(args: dict) -> str:
        if offline_only:
            raise ToolError("http_get is blocked: nano-agent is running in offline mode")
        url = args.get("url")
        if not url:
            raise ToolError("http_get requires a 'url' argument")
        if not str(url).startswith(("http://", "https://")):
            raise ToolError("http_get only supports http:// and https:// URLs")
        req = urllib.request.Request(url, headers={"User-Agent": "nano-agent"})
        try:
            with urllib.request.urlopen(req, timeout=args.get("timeout", 20)) as resp:
                body = resp.read(args.get("max_bytes", 16384)).decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise ToolError(f"http_get failed: {e}") from e
        return body

    return Tool(
        name="http_get",
        description="Fetch the body of an HTTP(S) URL (disabled in offline mode).",
        args_hint='{"url": "https://example.com"}',
        run=run,
    )


def make_memory_tool(memory_path: str) -> Tool:
    """Build a stateful `memory` tool backed by a JSON store at memory_path."""
    store = MemoryStore(Path(memory_path).expanduser())

    def run(args: dict) -> str:
        op = args.get("op")
        if op == "set":
            key, value = args.get("key"), args.get("value")
            if not key or value is None:
                raise ToolError("memory set requires 'key' and 'value'")
            store.set(key, str(value))
            return f"saved {key}"
        if op == "get":
            key = args.get("key")
            if not key:
                raise ToolError("memory get requires 'key'")
            val = store.get(key)
            return val if val is not None else f"[no value for {key}]"
        if op == "list":
            keys = store.keys()
            return ", ".join(keys) if keys else "[memory empty]"
        if op == "delete":
            key = args.get("key")
            if not key:
                raise ToolError("memory delete requires 'key'")
            return f"deleted {key}" if store.delete(key) else f"[no value for {key}]"
        raise ToolError("memory requires 'op' to be one of: set, get, list, delete")

    return Tool(
        name="memory",
        description="Persistent key-value memory across sessions.",
        args_hint='{"op": "set", "key": "name", "value": "data"}',
        run=run,
    )


def default_tools(memory_path: str | None = None, offline_only: bool = True) -> list[Tool]:
    """Return the built-in tools, plus http_get and (if a path is given) memory."""
    tools = list(BUILTIN_TOOLS)
    tools.append(make_http_get_tool(offline_only))
    if memory_path:
        tools.append(make_memory_tool(memory_path))
    return tools


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
