"""Tests for TOML config loading and CLI-override merging."""

import pytest

from nano_agent.config import Config, load_config, merge_overrides

SAMPLE = """
[model]
path = "/models/x.gguf"
n_gpu_layers = 0
temperature = 0.5

[agent]
max_steps = 7

[tools]
allowed = ["bash", "read_file"]

[logging]
file = "/tmp/trace.jsonl"
"""


def test_defaults_when_no_file(tmp_path, monkeypatch):
    # Point default path at a nonexistent file -> built-in defaults.
    monkeypatch.setattr(
        "nano_agent.config.default_config_path", lambda: tmp_path / "missing.toml"
    )
    cfg = load_config(None)
    assert cfg == Config()


def test_load_toml(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(SAMPLE)
    cfg = load_config(p)
    assert cfg.model == "/models/x.gguf"
    assert cfg.n_gpu_layers == 0
    assert cfg.temperature == 0.5
    assert cfg.max_steps == 7
    assert cfg.tools == ["bash", "read_file"]
    assert cfg.log_file == "/tmp/trace.jsonl"
    # unspecified fields keep defaults
    assert cfg.n_ctx == Config.n_ctx
    assert cfg.max_tokens_per_step == Config.max_tokens_per_step


def test_missing_explicit_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml")


def test_merge_overrides_applies_only_non_none():
    cfg = Config(model="/a", max_steps=20)
    merged = merge_overrides(cfg, model="/b", max_steps=None, temperature=0.9)
    assert merged.model == "/b"        # overridden
    assert merged.max_steps == 20      # None override ignored
    assert merged.temperature == 0.9   # overridden
