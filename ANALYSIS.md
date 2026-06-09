# nano-agent: Deep Analysis

> Research-backed market, technical, and competitive analysis.
> Data sources: adversarially verified claims from deep research (June 2026).

---

## 1. Problem Statement

All major LLM agent frameworks were designed for cloud-hosted inference. They assume:
- Unlimited RAM (cloud VMs: 16–256 GB)
- Low-latency inference (cloud GPUs: 100+ tok/s)
- Network connectivity (API calls to OpenAI/Anthropic)
- No power budget

Edge devices violate every assumption:

| Constraint | Cloud assumption | Edge reality |
|---|---|---|
| RAM | 16–256 GB | 4–8 GB |
| Inference speed | 100–500 tok/s | 6–20 tok/s |
| Network | Always-on | Optional/absent |
| Power | Unlimited | 5–25 W |
| Storage | SSD, unlimited | eMMC/SD, 32–128 GB |

No existing framework handles this. `nano-agent` is designed exclusively for this constraint envelope.

---

## 2. Hardware Reality (verified data)

### Jetson Orin Nano Super
| Model | Quant | tok/s (GPU) | tok/s (CPU) | RAM |
|---|---|---|---|---|
| DeepSeek-R1-Distill 1.5B | Q4_K_M | **9.59** | 6.01 | ~1.2 GB |
| Qwen 2.5 1.5B | Q4_K_M | **9.37** | — | ~1.1 GB |
| Llama 3.2 3B | Q4_K_M | **6.31** | — | ~2.1 GB |

Source: arxiv 2604.24785 (adversarially verified 2-1)

### Raspberry Pi 5
| Model | tok/s | Notes |
|---|---|---|
| Sub-360M models | >20 | Comfortable |
| 1.5B models | 5–15 | Usable for agents |
| 1B+ on Pi 4 | <5 | Not recommended |

Source: arxiv 2511.07425 (adversarially verified 2-1)

### Agent Step Viability at 10 tok/s
A typical agent step generates ~100–300 tokens.

```
At 10 tok/s:
- Short step (100 tok):   10 seconds
- Medium step (200 tok):  20 seconds
- Long step (400 tok):    40 seconds

For a 10-step task: 100–400 seconds (1.6–6.7 minutes)
```

This is slow but viable for background/automation tasks (file processing, code review, data extraction). Not viable for interactive chat. `nano-agent` is designed for **autonomous background tasks**, not real-time conversation.

---

## 3. Competitive Analysis

### Primary Competitors

#### smolagents (HuggingFace, Jan 2025)
- ~26,300 stars (non-verified blog source, June 2026)
- Supports local LLMs via transformers
- **Critical weakness:** Requires transformers backend (~2–4 GB), not llama.cpp native
- **Critical weakness:** No GGUF support without adapter layer
- **Critical weakness:** No offline-first design (downloads models, calls HuggingFace APIs)
- HuggingFace backing = brand risk to nano-agent, but also signals market validation

#### AutoGen (Microsoft, Oct 2023)
- ~35K+ stars
- Multi-agent conversation framework
- **Critical weakness:** No edge/offline support. Requires OpenAI or Azure endpoint.
- **Critical weakness:** Heavyweight Python dependency tree
- Not a real competitor for offline edge use

#### CrewAI (~37K stars)
- Role-based multi-agent orchestration
- **Critical weakness:** Cloud-only inference
- **Critical weakness:** Complex setup, not embedded-friendly

#### LangChain (~116K stars)
- Most starred agent framework
- **Critical weakness:** Famously heavy, complex abstractions
- **Critical weakness:** Cloud-first design
- Developers increasingly criticize LangChain for over-engineering; this is an opportunity

### Indirect Competitors

#### Ollama
- Focuses on inference serving, not agents
- No agent loop, no tool use
- Could add agents in future — watch carefully

#### llama.cpp (mainline)
- Adds agent-adjacent features occasionally
- If they add a native agent loop, it would be the most dangerous competitor
- Probability: ~25% within 18 months. Mitigation: build on top, not against.

---

## 4. Technical Architecture Decisions

### Why llama.cpp, not transformers?

| Criterion | llama.cpp | transformers |
|---|---|---|
| RAM efficiency | Excellent (GGUF, 4-bit) | Poor (FP16 by default) |
| Edge hardware support | Native (Metal, CUDA, Vulkan, CPU) | Limited |
| Binary portability | Single binary | Python ecosystem |
| Startup time | <2 seconds | 10–30 seconds |
| GGUF ecosystem | 100K+ models on HuggingFace | FP16/FP32 only |

llama.cpp is the correct choice for every edge metric.

### Why subprocess/WASM plugins, not Python imports?

- **Security:** Subprocess tools can be sandboxed (firejail, bubblewrap, seccomp)
- **Language agnostic:** Tools can be bash scripts, C binaries, Rust, Go
- **No dependency conflicts:** Each tool has its own environment
- **Hot-reload:** Add tools without restarting the agent

WASM is the future path — provides portable sandboxed execution with near-native speed.

### Why single-model, not multi-model routing?

v0.1 uses one model for planning, tool calls, and observation. This is a deliberate simplification:
- Reduces RAM pressure (one model loaded, not two)
- Simpler mental model for users
- 1.5B models are capable enough for basic agent tasks

v1.0 will add optional multi-model routing: small model for tool selection, larger for reasoning.

### Context Management Strategy

At 4096 token context (default), a 10-step agent run uses:
```
System prompt:      ~300 tokens
Per step (avg):     ~400 tokens (plan + tool call + result)
10 steps:           ~4,000 tokens
Safety margin:      ~296 tokens

Total:              ~4,296 tokens → need 8K context for reliable 10-step tasks
```

Strategy: use sliding window truncation on history, keeping only the last N observations. This is a key architectural constraint not addressed by cloud frameworks.

---

## 5. Star Growth Model

### Comparable projects and their trajectories

| Project | Launch | Stars at 6mo | Stars at 12mo | Stars at 24mo | Backing |
|---|---|---|---|---|---|
| smolagents | Jan 2025 | ~15K | ~26K | est. 35K | HuggingFace |
| lm-evaluation-harness | Aug 2020 | ~500 | ~1.5K | ~4K | EleutherAI |
| llama.cpp | Mar 2023 | ~15K | ~40K | ~80K | Community |

### Projected trajectory for nano-agent

**Conservative (no viral moment):**
```
Month 1:   300 stars
Month 3:   1,200 stars
Month 6:   3,500 stars
Month 12:  8,000 stars
Month 24:  15,000 stars
```

**Optimistic (one HN front page + demo video):**
```
Month 1:   800 stars
Month 3:   4,000 stars
Month 6:   10,000 stars
Month 12:  20,000 stars
Month 24:  35,000 stars
```

**Key viral triggers:**
1. A working demo video: "Jetson Orin Nano autonomously fixes a bug in a codebase, offline" — this is the demo that breaks through
2. HN Show post with reproducible hardware results
3. Being featured on NVIDIA Developer Blog or HuggingFace blog

---

## 6. Risk Matrix

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| HuggingFace adds edge mode to smolagents | 40% / 18 months | High | Ship faster; focus on true offline-first |
| llama.cpp adds native agent loop | 25% / 18 months | Medium | Build on top of llama.cpp, not against |
| 10 tok/s proves too slow for real tasks | 30% | High | Focus on async/batch tasks, not interactive |
| Jetson hardware generation changes | 20% / 24 months | Low | Hardware-agnostic design via llama.cpp |
| Memory pressure on 4GB devices | High (known) | Medium | Sliding window context, aggressive quantization |

---

## 7. Success Metrics

### Technical KPIs (v1.0)
- [ ] Complete 10-step agentic task on Jetson Orin Nano in <5 minutes
- [ ] Total process footprint <500 MB RAM with 1.5B Q4_K_M model loaded
- [ ] Zero external network calls in offline mode
- [ ] Tool execution sandbox: no escapes in basic security audit
- [ ] Cold start (model already loaded): <5 seconds to first token

### Community KPIs (year 1)
- [ ] 5,000+ GitHub stars
- [ ] 50+ community-contributed tool plugins
- [ ] 10+ distinct hardware boards tested and documented
- [ ] Referenced in at least one academic paper

---

## 8. Monetization Paths (long-term)

This is open-source first. Potential paths:

1. **Managed cloud runner** — run nano-agent on cloud, pay per task. Ironic but common.
2. **Enterprise support** — security audits, custom tool development, SLA
3. **Hardware bundles** — pre-configured Jetson kits with nano-agent preloaded
4. **Consulting** — edge AI deployment for industrial/robotics customers

None of these should compromise the open-source core.

---

## 9. Key Open Questions

1. What is the minimum model size that can reliably call tools in a JSON format? Sub-360M models may not be capable of structured tool-calling. Needs empirical testing.
2. How does context window size interact with agent reliability at 1.5B scale? Cloud agents use 100K+ context; we have 4–8K.
3. Can speculative decoding (edge + cloud) make interactive-speed agents viable? PicoSpec showed 2.9x speedup on Jetson + A100 pairs.
4. What is the latency floor for a sandboxed WASM tool call? If WASM startup adds 500ms per step, it may be too slow.
