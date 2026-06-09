"""Persistent key-value memory backing the `memory` tool.

A small JSON file is enough for v0.2: edge agent runs are short and the key
space is tiny. SQLite is on the v1.0 roadmap for concurrent/large use.
"""

from __future__ import annotations

import json
from pathlib import Path


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, str] = self._load()

    def _load(self) -> dict[str, str]:
        if not self.path.is_file():
            return {}
        try:
            return json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))

    def set(self, key: str, value: str) -> None:
        self._data[key] = value
        self._save()

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def keys(self) -> list[str]:
        return list(self._data)

    def delete(self, key: str) -> bool:
        existed = key in self._data
        self._data.pop(key, None)
        if existed:
            self._save()
        return existed
