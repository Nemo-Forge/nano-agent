# nano-agent Roadmap

Milestones are ordered by priority, not by date. Each builds on the previous.

## v0.1 — Proof of Concept

Goal: working agent loop on Jetson Orin Nano, documented and reproducible.

- [ ] Plan/tool/observe loop (Python, ~500 lines core)
- [ ] llama-cpp-python backend integration
- [ ] Built-in tools: bash, read_file, write_file
- [ ] JSON tool-call parsing with retry on malformed output
- [ ] Basic CLI: `nano-agent run --model <path> --task "<text>"`
- [ ] Max-steps and token budget enforcement
- [ ] Benchmark: 10-step file processing task on Jetson Orin Nano
- [ ] README with reproducible setup instructions

## v0.2 — Public Release

Goal: community-ready repo, first external contributors.

- [x] Config file (TOML) for model path, limits, tool allowlist
- [x] Interactive REPL mode
- [x] python_tool and memory_tool
- [x] Logging: structured JSON (JSONL) trace of all agent steps
- [x] GitHub Actions CI: lint, type check, unit tests
- [ ] `pip install nano-agent` packaging (publish to PyPI — packaging ready, not yet released)
- [ ] Raspberry Pi 5 support + documented benchmarks
- [ ] Launch post with demo video

## v0.5 — Plugin System

Goal: community can contribute tools without forking.

- [ ] Subprocess plugin protocol (stdin/stdout JSON)
- [ ] Tool discovery from `~/.nano-agent/tools/`
- [ ] Tool schema validation and documentation generation
- [ ] Sliding window context management (configurable window size)
- [ ] Streaming output during agent execution
- [ ] Orange Pi 5 and Rockchip RK3588 support
- [ ] http_get tool with offline-mode blocking
- [ ] Error recovery: retry failed tool calls with modified args

## v1.0 — Production

Goal: stable, secure, production-deployable.

- [ ] WASM plugin sandbox (wasmtime or wasmer)
- [ ] firejail/bubblewrap sandbox for bash_tool
- [ ] Multi-model routing: small model for tool selection, large for reasoning
- [ ] Speculative decoding integration (PicoSpec pattern for edge+cloud hybrid)
- [ ] REST API server mode (OpenAI-compatible agent endpoint)
- [ ] Persistent session memory with SQLite backend
- [ ] Model auto-recommendation based on device capabilities
- [ ] gguf-npu backend integration (Hailo, RK3588 NPU acceleration)
- [ ] Python SDK for embedding nano-agent in other applications
- [ ] Security audit of tool execution sandbox

## v1.x — Community Roadmap

Items driven by community demand:

- [ ] Multimodal: vision-capable agent (moondream2, SmolVLM on edge)
- [ ] Multi-agent: two nano-agent instances communicating via local socket
- [ ] Voice interface: whisper.cpp + nano-agent + piper TTS
- [ ] ROS2 integration: nano-agent as a ROS2 node for robotics
- [ ] Docker container with pre-loaded models
- [ ] Web UI for task submission and agent trace visualization
- [ ] Federated agent networks (multiple edge devices collaborating)

## Non-Goals (permanent)

- No cloud inference dependency in core
- No telemetry or analytics
- No mandatory model downloads at startup
- No LangChain/LlamaIndex compatibility layer (too heavy)
