# Contributing to nano-agent

## Most Wanted Contributions

### 1. Hardware Test Reports
Run the benchmark suite on your device and open a PR adding results to the hardware table:

```bash
python scripts/benchmark.py --model ~/models/qwen2.5-1.5b-q4_k_m.gguf --task agent_10step
```

Include: device name, OS, llama.cpp version, tok/s, RAM used, power draw (if measurable).

### 2. Tool Plugins
Build a tool and submit to `tools/community/`. Requirements:
- Reads JSON from stdin, writes JSON to stdout
- Works on Linux ARM64 (Jetson/Pi target)
- Includes a `tool.json` schema descriptor
- Has a test that runs without hardware-specific deps

### 3. Model Compatibility Reports
Tested a new model for tool-calling reliability? Open an issue with:
- Model name + quantization
- Hardware used
- Tool-calling success rate (out of 20 structured calls)
- Sample failures

### 4. Bug Reports
Include: device, OS, model, full command, full output/traceback.

---

## Development Setup

```bash
git clone https://github.com/your-org/nano-agent
cd nano-agent
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Run tests (no hardware required)
pytest tests/unit/

# Run integration tests (requires GGUF model)
NANO_AGENT_TEST_MODEL=~/models/qwen2.5-1.5b-q4_k_m.gguf pytest tests/integration/
```

## Code Style

- `ruff` for linting (`ruff check .`)
- `mypy` for type checking (`mypy nano_agent/`)
- Black for formatting (`black .`)
- No comments explaining WHAT code does — only WHY if non-obvious
- No docstrings longer than one line

## PR Guidelines

- One PR per feature/fix
- All PRs need a test (unit or integration)
- Hardware-specific PRs need benchmark numbers in the PR body
- Breaking changes need a BREAKING CHANGE note in the commit message

## Architecture Principles

- Core agent loop must fit in <1,000 lines of Python
- No new mandatory dependencies without discussion
- Every feature must have an offline fallback
- RAM budget: total process must stay under 500 MB with 1.5B model loaded
