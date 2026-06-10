#!/usr/bin/env python3
"""Example nano-agent plugin.

Protocol: read one JSON request from stdin, write one JSON response to stdout.
  request:  {"args": {"text": "..."}}
  response: {"ok": true, "result": "..."} | {"ok": false, "error": "..."}
"""

import json
import sys


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except json.JSONDecodeError:
        print(json.dumps({"ok": False, "error": "invalid request JSON"}))
        return 0

    text = request.get("args", {}).get("text")
    if not isinstance(text, str):
        print(json.dumps({"ok": False, "error": "missing 'text' argument"}))
        return 0

    result = f"{len(text.split())} words, {len(text)} characters"
    print(json.dumps({"ok": True, "result": result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
