# Writing nano-agent Plugins

Plugins extend nano-agent with new tools **without modifying or recompiling**
the agent. A plugin can be written in any language — it just has to speak a
tiny JSON-over-stdio protocol.

## How it works

1. You put a `<name>.tool.json` manifest in the plugin directory
   (default `~/.nano-agent/tools/`, or set `[tools] plugin_dir` / `--plugin-dir`).
2. At startup nano-agent discovers every manifest and exposes it as a tool the
   model can call.
3. When the model calls the tool, nano-agent runs the manifest's `command`,
   writes a JSON **request** to its stdin, and reads a JSON **response** from
   its stdout.

Invalid manifests are skipped with a warning — they never crash startup.

## The protocol

**Request** (stdin):

```json
{"args": {"text": "hello world"}}
```

**Response** (stdout) — success:

```json
{"ok": true, "result": "2 words, 11 characters"}
```

**Response** (stdout) — failure:

```json
{"ok": false, "error": "missing 'text' argument"}
```

A non-zero exit code, invalid JSON, or `ok: false` all surface to the agent as
a tool error (which the model can see and react to).

## Manifest format

`wordcount.tool.json`:

```json
{
  "name": "wordcount",
  "description": "Count the words and characters in a piece of text.",
  "args_hint": "{\"text\": \"hello world\"}",
  "command": ["python3", "wordcount.py"],
  "timeout": 10
}
```

| Field | Required | Notes |
|-------|----------|-------|
| `name` | yes | Tool name the model uses. |
| `description` | yes | Shown to the model; make it clear and action-oriented. |
| `command` | yes | Argv list. A relative script path (containing `/`) resolves against the manifest's own directory. |
| `args_hint` | no | Example args JSON shown to the model. Default `"{}"`. |
| `timeout` | no | Seconds before the plugin is killed. Default `30`. |

## A complete example (Python)

`wordcount.py`:

```python
#!/usr/bin/env python3
import json, sys

request = json.load(sys.stdin)
text = request.get("args", {}).get("text")
if not isinstance(text, str):
    print(json.dumps({"ok": False, "error": "missing 'text' argument"}))
else:
    result = f"{len(text.split())} words, {len(text)} characters"
    print(json.dumps({"ok": True, "result": result}))
```

Drop both files in `~/.nano-agent/tools/`, then:

```bash
nano-agent tools          # confirm 'wordcount' is listed
nano-agent run --model ~/models/qwen2.5-1.5b-q4_k_m.gguf \
  --task "count the words in: the quick brown fox"
```

A ready-to-copy version lives in [examples/plugins/](../examples/plugins/).

## Tips

- Plugins run as separate processes, so they can be sandboxed (firejail,
  bubblewrap) and written in fast compiled languages.
- Keep results short — every character a plugin returns becomes prompt tokens
  on the next step, which matters on a 4–8K context edge device.
- Restrict which tools the model may use with `--tools name1,name2` if you want
  to expose only a subset.
