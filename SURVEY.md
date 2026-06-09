# nano-agent: Ecosystem Survey

> Comprehensive map of related repositories, frameworks, and tools.
> Ratings: Edge-Ready (can run offline on 4–8 GB edge device), GGUF-Native (uses llama.cpp/GGUF directly).

---

## Agent Frameworks

### smolagents
- **Repo:** https://github.com/huggingface/smolagents
- **Stars:** ~26,300 (June 2026, blog-sourced)
- **Org:** HuggingFace
- **Description:** Minimal agent framework (~1,000 lines of core code). Supports local LLMs via transformers. "Code agents" that write and execute Python instead of JSON tool calls.
- **Edge-Ready:** Partial — local LLM support exists but requires transformers, not llama.cpp
- **GGUF-Native:** No
- **Gap vs nano-agent:** No offline-first design, no GGUF, no Jetson/Pi testing, heavy transformers dependency
- **Threat level:** HIGH — HuggingFace can add edge support at any time

### AutoGen
- **Repo:** https://github.com/microsoft/autogen
- **Stars:** ~35,000+
- **Org:** Microsoft Research
- **Description:** Multi-agent conversation framework. Agents communicate via message passing. Strong for complex multi-agent pipelines.
- **Edge-Ready:** No
- **GGUF-Native:** No
- **Gap vs nano-agent:** Requires cloud inference, heavy Python dependencies, no embedded support
- **Threat level:** LOW — fundamentally cloud-architected

### CrewAI
- **Repo:** https://github.com/crewAIInc/crewAI
- **Stars:** ~37,000+
- **Org:** CrewAI Inc (startup)
- **Description:** Role-based multi-agent orchestration. Agents have roles, goals, backstories. Good for structured multi-step workflows.
- **Edge-Ready:** No
- **GGUF-Native:** No
- **Gap vs nano-agent:** Cloud-only, heavyweight, requires LiteLLM or OpenAI
- **Threat level:** LOW

### LangChain
- **Repo:** https://github.com/langchain-ai/langchain
- **Stars:** ~116,000+
- **Org:** LangChain Inc (startup)
- **Description:** The original LLM framework. Everything-including-the-kitchen-sink approach. Agents, RAG, chains, tools, memory.
- **Edge-Ready:** No
- **GGUF-Native:** Via LangChain-community adapter (llama-cpp-python)
- **Gap vs nano-agent:** Notoriously heavy abstraction layers, large dependency tree, cloud-first
- **Threat level:** LOW — too heavy for edge

### LlamaIndex
- **Repo:** https://github.com/run-llama/llama_index
- **Stars:** ~38,000+
- **Description:** Data framework for LLMs. Focus on RAG, document Q&A, data connectors.
- **Edge-Ready:** Partial
- **GGUF-Native:** Via adapter
- **Gap vs nano-agent:** Document/RAG focused, not general agent runtime
- **Threat level:** LOW (different use case)

### AgentScope
- **Repo:** https://github.com/modelscope/agentscope
- **Stars:** ~6,000+
- **Org:** Alibaba ModelScope
- **Description:** Multi-agent platform with distributed execution. Good for complex agent networks.
- **Edge-Ready:** No
- **GGUF-Native:** No
- **Threat level:** LOW

### Pydantic AI
- **Repo:** https://github.com/pydantic/pydantic-ai
- **Stars:** ~12,000+
- **Description:** Type-safe agent framework. Strong structured output, validation.
- **Edge-Ready:** No
- **GGUF-Native:** No
- **Gap vs nano-agent:** Cloud-first but interesting structured output approach worth borrowing
- **Threat level:** LOW-MEDIUM

---

## Inference Backends (direct dependencies)

### llama.cpp
- **Repo:** https://github.com/ggml-org/llama.cpp
- **Stars:** 116,000+ (June 2026, GitHub API verified)
- **Description:** The de facto standard for local LLM inference. GGUF format, all quantizations, all major hardware backends.
- **Relevance:** nano-agent's core inference engine
- **Key backends:** CPU (BLAS), CUDA, Metal, Vulkan, OpenBLAS

### ollama
- **Repo:** https://github.com/ollama/ollama
- **Stars:** ~100,000+
- **Description:** Docker-like model management and serving. Built on llama.cpp underneath.
- **Relevance:** nano-agent could optionally use Ollama as inference backend
- **Gap:** No agent loop, inference-only

### llama-cpp-python
- **Repo:** https://github.com/abetlen/llama-cpp-python
- **Stars:** ~9,000+
- **Description:** Python bindings for llama.cpp. OpenAI-compatible server.
- **Relevance:** nano-agent's Python interface to llama.cpp

### llamafile
- **Repo:** https://github.com/Mozilla-Odin/llamafile
- **Stars:** ~23,000+
- **Org:** Mozilla
- **Description:** Single-file LLM binary. 3–4x faster than Ollama on single-core edge (Orange Pi 5 Pro, verified).
- **Relevance:** Alternative packaging approach for nano-agent distribution
- **Note:** Better single-core performance, Ollama scales better with more cores

---

## Edge AI / Embedded AI Projects

### ExecuTorch
- **Repo:** https://github.com/pytorch/executorch
- **Stars:** ~4,000+
- **Org:** Meta / PyTorch
- **Description:** On-device ML framework. 12+ hardware backends. Supports LLMs (Llama 3.2, Qwen 3, Phi-4-mini).
- **Edge-Ready:** Yes (designed for it)
- **GGUF-Native:** No (uses PyTorch format)
- **Gap vs nano-agent:** Inference-only, no agent runtime, CUDA on Linux is experimental
- **Relevance:** Potential alternative inference backend for nano-agent

### MLC LLM
- **Repo:** https://github.com/mlc-ai/mlc-llm
- **Stars:** ~19,000+
- **Description:** Universal LLM deployment using TVM compilation. Targets phones, embedded, WebGPU.
- **Edge-Ready:** Yes
- **Gap vs nano-agent:** No agent runtime, compilation step required per model

### TensorRT-LLM (NVIDIA)
- **Repo:** https://github.com/NVIDIA/TensorRT-LLM
- **Stars:** ~9,000+
- **Description:** NVIDIA's optimized inference engine. Best performance on Jetson with TensorRT.
- **Edge-Ready:** Jetson only
- **Gap vs nano-agent:** NVIDIA-only, no GGUF, no agent runtime

---

## Speculative Decoding (performance acceleration)

### PicoSpec
- **Paper:** arxiv 2603.19133
- **Description:** Edge+cloud speculative decoding. Draft model on Jetson, target model in cloud.
- **Verified result:** 2.9x speedup (6.87 → 19.88 tok/s) on Jetson AGX + A100 with Llama 3.2 1B + 70B
- **Relevance:** nano-agent v1.0 should integrate this pattern for hybrid edge+cloud deployments

---

## Memory / State Management

### Mem0
- **Repo:** https://github.com/mem0ai/mem0
- **Stars:** ~24,000+
- **Description:** Persistent memory for AI agents. Vector store + key-value.
- **Relevance:** nano-agent's memory tool design is inspired by this
- **Gap:** Cloud-first, heavy vector store dependency

### ChromaDB
- **Repo:** https://github.com/chroma-core/chroma
- **Stars:** ~17,000+
- **Description:** Embedded vector database.
- **Relevance:** Candidate for nano-agent's local vector memory backend

---

## Benchmarking

### jetson-bench (sister project)
- **Repo:** ../jetson-bench/
- **Description:** Unified LLM benchmark suite for edge hardware
- **Relevance:** Use jetson-bench to characterize hardware before deploying nano-agent

### lm-evaluation-harness
- **Repo:** https://github.com/EleutherAI/lm-evaluation-harness
- **Stars:** 12,891 (GitHub API, June 2026)
- **Description:** Standard LLM capability benchmark. 212 task directories, 13,839 YAML tasks. Powers HuggingFace Open LLM Leaderboard.
- **Relevance:** Reference for model capability selection (what's the best 1.5B for tool use?)

---

## Hardware Acceleration (for future nano-agent backends)

### gguf-npu (sister project)
- **Repo:** ../gguf-npu/
- **Description:** GGUF inference backends for edge NPUs (Hailo, Rockchip RK3588, Coral)
- **Relevance:** nano-agent will use gguf-npu backends when available for 9–40x energy efficiency gains

---

## Gap Summary

| Category | Best existing | Gap |
|---|---|---|
| Edge agent runtime | smolagents (partial) | Offline-first, GGUF-native, <500MB |
| Inference on edge | llama.cpp | Already solved |
| Agent tool system | LangChain tools | Edge-safe, sandboxed, language-agnostic |
| Edge memory | Mem0 | Embedded, no cloud, tiny footprint |
| Multi-model routing | LiteLLM | Edge-aware routing based on hardware budget |
