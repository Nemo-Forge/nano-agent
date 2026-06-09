"""Configuration: defaults, TOML file loading, and CLI-override merging.

Precedence (highest first): CLI argument -> config file -> built-in default.
The Config dataclass is the single source of truth for defaults.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib


@dataclass
class Config:
    model: str | None = None
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    temperature: float = 0.1
    max_steps: int = 20
    max_tokens_per_step: int = 512
    tools: list[str] | None = None  # None means "all available tools"
    offline_only: bool = True
    memory_path: str = "~/.nano-agent/memory.json"
    log_file: str | None = None


def default_config_path() -> Path:
    return Path("~/.nano-agent/config.toml").expanduser()


def load_config(path: str | Path | None = None) -> Config:
    """Load a Config from a TOML file.

    If path is None, use the default location when it exists, otherwise return
    built-in defaults. An explicitly passed path that does not exist is an error.
    """
    if path is None:
        p = default_config_path()
        if not p.is_file():
            return Config()
    else:
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"config file not found: {p}")

    with p.open("rb") as f:
        data = tomllib.load(f)

    model = data.get("model", {})
    agent = data.get("agent", {})
    tools = data.get("tools", {})
    memory = data.get("memory", {})
    logging = data.get("logging", {})

    return Config(
        model=model.get("path"),
        n_ctx=model.get("n_ctx", Config.n_ctx),
        n_gpu_layers=model.get("n_gpu_layers", Config.n_gpu_layers),
        temperature=model.get("temperature", Config.temperature),
        max_steps=agent.get("max_steps", Config.max_steps),
        max_tokens_per_step=agent.get("max_tokens_per_step", Config.max_tokens_per_step),
        offline_only=agent.get("offline_only", Config.offline_only),
        tools=tools.get("allowed"),
        memory_path=memory.get("path", Config.memory_path),
        log_file=logging.get("file"),
    )


def merge_overrides(cfg: Config, **overrides: object) -> Config:
    """Apply non-None overrides (typically from CLI args) onto a Config."""
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(cfg, **clean)  # type: ignore[arg-type]
