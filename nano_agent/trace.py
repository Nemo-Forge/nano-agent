"""Structured JSONL tracing of agent runs.

Each line is one JSON event with a wall-clock timestamp. A run produces:
  run_start -> (step, observation)* -> run_end
This is machine-readable for later analysis (jetson-bench agent suite, debugging
small-model tool-calling reliability) without coupling the agent loop to a logger.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Tracer:
    """Writes JSONL events. A no-op when path is None."""

    def __init__(self, path: str | Path | None) -> None:
        self._fh = None
        if path is not None:
            p = Path(path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            self._fh = p.open("a")

    @property
    def enabled(self) -> bool:
        return self._fh is not None

    def event(self, kind: str, **fields: Any) -> None:
        if self._fh is None:
            return
        record = {"ts": round(time.time(), 3), "event": kind, **fields}
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> Tracer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
