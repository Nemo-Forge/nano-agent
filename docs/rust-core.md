# Design Note: A Rust Core for nano-agent

> Status: exploratory. nano-agent v0.x is Python. This note sketches what a
> compiled (Rust) core would look like, and—honestly—when it is and isn't worth it.

## TL;DR

The agent loop is **not** the bottleneck; LLM inference (already C++ in llama.cpp)
is. Rewriting the loop in Rust buys ~nothing on speed. It buys one real thing:
**a single static binary with no Python runtime** — easier deploy, lower cold
start, smaller floor. Worth it only if single-binary distribution becomes a goal.

## Why consider Rust (not C++)

If we ever go compiled, Rust over C++ because:

- **Single static binary** — `cargo build --release`, copy one file to the device, done. No `pip`, no Python-version drift, no `apt install`.
- **Memory safety without GC** — no segfaults, no garbage-collector pauses on a 10 W device.
- **First-class llama.cpp bindings** — the [`llama-cpp-2`](https://crates.io/crates/llama-cpp-2) crate wraps the same `libllama` we already use.
- **Good cross-compile story** — `aarch64-unknown-linux-gnu` for Jetson/Pi from a dev box.

## Architecture (mirrors the Python design)

Same concept, same separation of concerns. The heavy part still calls into
llama.cpp's C++ — Rust only orchestrates.

```
┌──────────────────────────────────────────────┐
│ Agent loop (Rust)                              │
│   build prompt → backend.generate() → parse    │
│   → tool.run() → repeat                         │
└───────────────┬────────────────────────────────┘
                │ trait Backend
        ┌───────▼────────┐
        │ LlamaBackend   │  via llama-cpp-2 crate (FFI → libllama, C++)
        └───────┬────────┘
                ▼
          llama.cpp (C++) — unchanged, the actual compute
```

## Sketch: core types

```rust
// types.rs
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(tag = "action", rename_all = "lowercase")]
pub enum Step {
    Tool { #[serde(default)] thought: String, tool: String,
           #[serde(default)] args: serde_json::Value },
    Done { #[serde(default)] thought: String, answer: String },
}

pub struct Observation {
    pub ok: bool,
    pub result: String,
    pub error: Option<String>,
}

pub enum StopReason { Done, MaxSteps }

pub struct AgentResult {
    pub answer: String,
    pub steps_taken: usize,
    pub stopped_reason: StopReason,
}
```

## Sketch: the Backend trait + llama.cpp binding

```rust
// backend.rs
pub trait Backend {
    fn generate(&mut self, prompt: &str, max_tokens: usize, stop: &[&str]) -> String;
}

// Wraps libllama through the llama-cpp-2 crate. On Jetson, build with the
// crate's `cuda` feature so all layers offload to the GPU — same as
// n_gpu_layers=-1 in the Python version.
pub struct LlamaBackend { /* model, context, sampler */ }

impl Backend for LlamaBackend {
    fn generate(&mut self, prompt: &str, max_tokens: usize, stop: &[&str]) -> String {
        // tokenize → decode loop → detokenize, stopping on `stop` strings.
        // This is the only place that touches the model.
        todo!()
    }
}
```

## Sketch: tools

```rust
// tools.rs
pub trait Tool {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn run(&self, args: &serde_json::Value) -> Result<String, ToolError>;
}

pub struct Bash;
impl Tool for Bash {
    fn name(&self) -> &str { "bash" }
    fn description(&self) -> &str { "Run a shell command and return its output." }
    fn run(&self, args: &serde_json::Value) -> Result<String, ToolError> {
        let cmd = args["cmd"].as_str().ok_or(ToolError::MissingArg("cmd"))?;
        let out = std::process::Command::new("sh").arg("-c").arg(cmd).output()?;
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }
}
```

## Sketch: the loop

```rust
// agent.rs
pub fn run(backend: &mut dyn Backend, tools: &ToolRegistry,
           task: &str, max_steps: usize) -> AgentResult {
    let mut history = Vec::new();
    for _ in 0..max_steps {
        let prompt = build_prompt(task, tools, &history);
        let raw = backend.generate(&prompt, 512, &["\nOBSERVATION:", "\nSTEP:"]);

        match parse_step(&raw) {
            Ok(Step::Done { answer, .. }) =>
                return AgentResult { answer, steps_taken: history.len(),
                                     stopped_reason: StopReason::Done },
            Ok(Step::Tool { tool, args, .. }) => {
                let obs = exec(tools, &tool, &args); // ToolError → Observation, never panics
                history.push(obs);
            }
            Err(e) => history.push(Observation::parse_error(e)), // self-correct, like Python
        }
    }
    AgentResult { answer: "(stopped: reached max steps)".into(),
                  steps_taken: history.len(), stopped_reason: StopReason::MaxSteps }
}
```

Note this is a near 1:1 port of [`nano_agent/agent.py`](../nano_agent/agent.py) —
the design already isolates the model behind a backend boundary, so only this
file and `backend.rs` differ in spirit.

## Honest comparison

| Dimension | Python core (today) | Rust core |
|---|---|---|
| Inference speed | identical (both call llama.cpp) | identical (both call llama.cpp) |
| Loop overhead per step | <1 ms (irrelevant vs ~15 s inference) | <0.01 ms (still irrelevant) |
| Runtime RAM overhead | ~20–40 MB | ~2–5 MB |
| Cold start (excl. model load) | ~100–300 ms | ~5 ms |
| Model load time (mmap ~1 GB) | seconds | seconds (same) |
| **Distribution** | needs Python + pip + native build | **single static binary** |
| Contributor pool | very large | smaller |
| Tool authoring | Python / subprocess / WASM | subprocess / WASM (Rust for built-ins) |
| Iteration speed | fast | slower (compile, stricter) |
| Time to port v0.2 features | — | est. 2–4 weeks |

## Recommendation

Stay on Python for the orchestration brain while the design is still moving
fast. Revisit a Rust core **only when single-binary deployment becomes a real
requirement** (e.g. shipping to fleets of devices without a Python toolchain).

If/when that happens, the migration is low-risk because the boundaries already
exist:

1. Port `types`, `prompt`, `tools`, `agent` to Rust (pure logic, no model).
2. Wrap llama.cpp with `llama-cpp-2` for `backend.rs`.
3. Keep the **trace JSONL format and config TOML schema identical**, so tooling
   (jetson-bench agent suite, existing configs) works unchanged across both.
4. Optionally ship both: Python for development/embedding, Rust binary for deploy.

A reasonable middle ground that needs **zero** rewrite: distribute the current
Python version as a single file with [PyInstaller](https://pyinstaller.org) or
`shiv`. It is heavier than a Rust binary but removes the `pip` step today.
