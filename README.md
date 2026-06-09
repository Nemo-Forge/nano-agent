# nano-agent

> Offline-first autonomous LLM agent runtime for edge devices. Runs entirely on NVIDIA Jetson, Raspberry Pi 5, and similar hardware with no cloud dependency.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hardware: Jetson](https://img.shields.io/badge/hardware-Jetson%20Orin-green)]()
[![Hardware: Pi5](https://img.shields.io/badge/hardware-Raspberry%20Pi%205-red)]()
[![Backend: llama.cpp](https://img.shields.io/badge/backend-llama.cpp-blue)]()

## Why

Every major agent framework — LangChain, AutoGen, CrewAI, smolagents — assumes a cloud inference endpoint. None are designed for devices with 4–8 GB RAM, no internet, and 10 W power budgets.

`nano-agent` is the first agent runtime built ground-up for edge constraints:

- **Tiny runtime overhead** — the agent layer adds ~20–40 MB on top of the model; a 1.5B Q4 model + runtime + tools fits comfortably in 4 GB RAM (see [footprint](#footprint))
- **Fully offline** — no API keys, no cloud calls, no phone-home
- **llama.cpp native** — GGUF models, all quantizations (Q4 to Q8), all llama.cpp backends
- **Plugin tools via subprocess/WASM** — extend without recompiling
- **Graceful degradation** — works at 5 tok/s or 30 tok/s, adapts step budget accordingly

## Supported Hardware (verified performance)

| Device | Model | Quant | tok/s | RAM used |
|--------|-------|-------|-------|----------|
| Jetson Orin Nano Super | DeepSeek-R1-Distill 1.5B | Q4_K_M | 9.59 | ~1.2 GB |
| Jetson Orin Nano Super | Qwen 2.5 1.5B | Q4_K_M | 9.37 | ~1.1 GB |
| Jetson Orin Nano Super | Llama 3.2 3B | Q4_K_M | 6.31 | ~2.1 GB |
| Raspberry Pi 5 | Qwen 2.5 1.5B | Q4_K_M | ~8–12 | ~1.1 GB |
| Raspberry Pi 5 + Hailo-10H | (via gguf-npu) | INT4 | TBD | ~0.9 GB |

## Quick Start

> **Status: v0.1 (alpha).** Not yet on PyPI — install from source.

```bash
# Install from source
git clone https://github.com/Nemo-Forge/nano-agent
cd nano-agent
pip install -e .

# On Jetson, build llama-cpp-python with CUDA first:
#   CMAKE_ARGS="-DGGML_CUDA=on" pip install -e .

# Run with a local GGUF model
nano-agent run \
  --model ~/models/qwen2.5-1.5b-q4_k_m.gguf \
  --task "List all .py files in /tmp and summarize what each does"

# CPU only (no GPU offload)
nano-agent run --model ~/models/qwen2.5-1.5b-q4_k_m.gguf --n-gpu-layers 0 \
  --task "what is the hostname of this machine?"

# Interactive REPL
nano-agent repl --model ~/models/qwen2.5-1.5b-q4_k_m.gguf
```

### Run the tests (no model or GPU required)

```bash
pip install -e ".[dev]"
PYTHONPATH=. pytest -q
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  nano-agent                      │
│                                                  │
│  ┌──────────┐    ┌──────────┐    ┌───────────┐  │
│  │  Planner │───▶│ Tool     │───▶│ Observer  │  │
│  │  (LLM)   │    │ Executor │    │ (LLM)     │  │
│  └──────────┘    └──────────┘    └───────────┘  │
│        │              │                │         │
│        └──────────────┴────────────────┘         │
│                       │                          │
│              ┌────────▼────────┐                 │
│              │   llama.cpp     │                 │
│              │   (GGUF backend)│                 │
│              └─────────────────┘                 │
│                                                  │
│  Plugin Tools (subprocess / WASM):               │
│  bash_tool  file_tool  python_tool  http_tool    │
└─────────────────────────────────────────────────┘
```

### Core Loop

```python
while not done:
    plan = llm.think(task, history, available_tools)
    if plan.action == "done":
        break
    result = tool_executor.run(plan.tool, plan.args)
    history.append(Observation(plan, result))
    # Budget: abort if steps > max_steps or tokens > budget
```

### Tool Plugin System

Tools are executables or WASM modules in `~/.nano-agent/tools/`. They communicate via stdin/stdout JSON:

```json
// input
{"tool": "bash", "args": {"cmd": "ls -la /tmp"}}

// output  
{"ok": true, "result": "total 8\ndrwxrwxrwt ...", "tokens_used": 0}
```

Write a tool in any language:

```python
#!/usr/bin/env python3
import json, sys
req = json.load(sys.stdin)
result = run_my_tool(req["args"])
print(json.dumps({"ok": True, "result": result}))
```

## Built-in Tools

| Tool | Description | Status |
|------|-------------|--------|
| `bash` | Execute shell commands | available |
| `read_file` | Read text files | available |
| `write_file` | Write/append files | available |
| `python` | Execute Python snippets in a subprocess | available |
| `memory` | Persistent key-value store across sessions | available |
| `http_get` | HTTP requests (offline-mode aware) | planned (v0.5) |

Restrict which tools the agent may use with `--tools` or the `[tools] allowed`
config key, e.g. `--tools bash,read_file`.

## Configuration

nano-agent reads `~/.nano-agent/config.toml` by default, or any file passed with
`--config`. CLI flags always override the config file. A full sample lives in
[examples/config.toml](examples/config.toml).

```toml
# ~/.nano-agent/config.toml

[model]
path = "~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
n_ctx = 4096
n_gpu_layers = -1        # -1 = all layers on GPU, 0 = CPU only
temperature = 0.1

[agent]
max_steps = 20
max_tokens_per_step = 512
offline_only = true      # reserved for network-capable tools (v0.5)

[tools]
allowed = ["bash", "read_file", "write_file", "python", "memory"]

[memory]
path = "~/.nano-agent/memory.json"

[logging]
file = "~/.nano-agent/trace.jsonl"   # JSONL trace of every run; omit to disable
```

### Run tracing

Pass `--log-file run.jsonl` (or set `[logging] file`) to record a structured,
machine-readable trace. Each run emits `run_start`, one `step` + `observation`
per loop iteration, and `run_end` — one JSON object per line:

```json
{"ts": 1749470000.12, "event": "run_start", "task": "...", "model": "..."}
{"ts": 1749470001.44, "event": "step", "action": "tool", "tool": "bash", "args": {"cmd": "ls"}}
{"ts": 1749470001.51, "event": "observation", "ok": true, "error": null, "result_len": 84}
{"ts": 1749470003.02, "event": "run_end", "answer": "...", "steps": 1, "reason": "done"}
```

## Footprint

The model dominates memory; the agent runtime is small on top of it. Honest
breakdown for a 1.5B Q4_K_M model on an 8 GB Jetson Orin Nano:

| Component | Approx. RAM |
|-----------|-------------|
| Model weights (1.5B Q4_K_M) | ~900–1100 MB |
| KV cache (n_ctx 4096) | ~150–300 MB |
| llama.cpp runtime (C++) | ~50–100 MB |
| **Python interpreter + nano-agent** | **~20–40 MB** |

nano-agent's own overhead is the smallest line in the table — the loop builds a
prompt, calls llama.cpp, parses JSON, and runs a tool. The heavy compute lives
in llama.cpp (C++), so the orchestration language is not on the hot path. See
[ANALYSIS.md](ANALYSIS.md) for the latency breakdown, and
[docs/rust-core.md](docs/rust-core.md) for when a compiled core would help.

## Model Recommendations

| Use Case | Model | Quant | Size |
|----------|-------|-------|------|
| Best reasoning (Jetson Orin) | DeepSeek-R1-Distill 1.5B | Q4_K_M | 0.9 GB |
| Best instruction (any Pi) | Qwen 2.5 1.5B | Q4_K_M | 0.9 GB |
| Coding tasks (Jetson Orin) | Qwen 2.5 Coder 1.5B | Q4_K_M | 0.9 GB |
| Low memory (<2GB) | SmolLM2 360M | Q4_K_M | 0.2 GB |

## Comparison with Other Frameworks

| | nano-agent | smolagents | AutoGen | LangChain |
|--|--|--|--|--|
| Edge/offline first | Yes | Partial | No | No |
| Runtime overhead (excl. model) | ~20–40 MB | ~300 MB–2 GB¹ | ~300 MB+¹ | ~300 MB+¹ |
| llama.cpp native | Yes | Via adapter | No | Via adapter |
| No API key required | Yes | Yes | No | No |
| WASM plugins | Yes | No | No | No |
| Jetson tested | Yes | No | No | No |

¹ Other frameworks pull heavy dependency trees (transformers/torch, etc.) on
top of the model. nano-agent's only required dependency is `llama-cpp-python`.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for full milestones.

- **v0.1** — Plan/tool/observe loop, bash + file tools, llama.cpp backend
- **v0.2** — TOML config, python + memory tools, JSONL tracing, CI *(current)*
- **v0.5** — Subprocess plugin system, sliding-window context, Pi 5 / RK3588 support
- **v1.0** — WASM sandbox, multi-model routing, speculative decoding, NPU backends

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Hardware test reports especially welcome.

## Related Projects

See [SURVEY.md](SURVEY.md) for full ecosystem map.

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — inference backend
- [smolagents](https://github.com/huggingface/smolagents) — HuggingFace agent framework
- [jetson-bench](../jetson-bench/) — benchmark your hardware first

## License

MIT
