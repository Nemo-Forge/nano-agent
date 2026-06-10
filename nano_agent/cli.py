"""Command-line interface for nano-agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nano_agent.agent import Agent
from nano_agent.config import Config, load_config, merge_overrides
from nano_agent.plugins import discover_plugins
from nano_agent.tools import ToolRegistry, default_tools
from nano_agent.trace import Tracer
from nano_agent.types import Observation, Step


def _print_step(step: Step) -> None:
    if step.thought:
        print(f"\033[2m  thinking: {step.thought}\033[0m", file=sys.stderr)
    if step.tool:
        print(f"\033[36m  -> {step.tool} {step.args}\033[0m", file=sys.stderr)


def _print_observation(obs: Observation) -> None:
    if obs.ok:
        preview = obs.result if len(obs.result) < 500 else obs.result[:500] + " ..."
        print(f"\033[2m  <- {preview}\033[0m", file=sys.stderr)
    else:
        print(f"\033[31m  <- error: {obs.error}\033[0m", file=sys.stderr)


def _resolve_config(args: argparse.Namespace) -> Config:
    cfg = load_config(getattr(args, "config", None))
    return merge_overrides(
        cfg,
        model=getattr(args, "model", None),
        n_ctx=getattr(args, "n_ctx", None),
        n_gpu_layers=getattr(args, "n_gpu_layers", None),
        temperature=getattr(args, "temperature", None),
        stream=getattr(args, "stream", None),
        max_steps=getattr(args, "max_steps", None),
        max_tokens_per_step=getattr(args, "max_tokens", None),
        context_window=getattr(args, "context_window", None),
        tools=args.tools.split(",") if getattr(args, "tools", None) else None,
        plugin_dir=getattr(args, "plugin_dir", None),
        log_file=getattr(args, "log_file", None),
    )


def _build_registry(cfg: Config) -> tuple[ToolRegistry, list[str]]:
    tools = default_tools(memory_path=cfg.memory_path, offline_only=cfg.offline_only)
    warnings: list[str] = []
    if cfg.plugin_dir:
        plugin_tools, warnings = discover_plugins(Path(cfg.plugin_dir).expanduser())
        tools += plugin_tools
    return ToolRegistry(tools, allowed=cfg.tools), warnings


def _build_agent(cfg: Config, tracer: Tracer) -> Agent:
    if not cfg.model:
        raise SystemExit("error: no model given. Pass --model or set [model].path in config.")

    from nano_agent.backend import LlamaCppBackend

    backend = LlamaCppBackend(
        model_path=cfg.model,
        n_ctx=cfg.n_ctx,
        n_gpu_layers=cfg.n_gpu_layers,
        temperature=cfg.temperature,
        stream=cfg.stream,
    )
    registry, warnings = _build_registry(cfg)
    for w in warnings:
        print(f"warning: plugin {w}", file=sys.stderr)

    def on_step(step: Step) -> None:
        _print_step(step)
        tracer.event("step", action=step.action.value, tool=step.tool, args=step.args)

    def on_observation(obs: Observation) -> None:
        _print_observation(obs)
        tracer.event("observation", ok=obs.ok, error=obs.error, result_len=len(obs.result))

    return Agent(
        backend=backend,
        registry=registry,
        max_steps=cfg.max_steps,
        max_tokens_per_step=cfg.max_tokens_per_step,
        context_window=cfg.context_window,
        max_consecutive_errors=cfg.max_consecutive_errors,
        on_step=on_step,
        on_observation=on_observation,
    )


def _run_once(cfg: Config, task: str) -> int:
    with Tracer(cfg.log_file) as tracer:
        agent = _build_agent(cfg, tracer)
        tracer.event("run_start", task=task, model=cfg.model)
        result = agent.run(task)
        tracer.event(
            "run_end",
            answer=result.answer,
            steps=result.steps_taken,
            reason=result.stopped_reason,
        )
    print("\n" + "=" * 60)
    print(result.answer)
    print("=" * 60)
    print(f"steps: {result.steps_taken} | stopped: {result.stopped_reason}", file=sys.stderr)
    return 0 if result.stopped_reason == "done" else 1


def _cmd_run(args: argparse.Namespace) -> int:
    return _run_once(_resolve_config(args), args.task)


def _cmd_repl(args: argparse.Namespace) -> int:
    cfg = _resolve_config(args)
    print("nano-agent REPL — type a task, Ctrl-D to exit.")
    while True:
        try:
            task = input("\ntask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if task:
            _run_once(cfg, task)


def _cmd_tools(args: argparse.Namespace) -> int:
    """List every available tool (built-in + plugins) and its argument hint."""
    cfg = _resolve_config(args)
    registry, warnings = _build_registry(cfg)
    for w in warnings:
        print(f"warning: plugin {w}", file=sys.stderr)
    for name in registry.names():
        t = registry.get(name)
        print(f"{name}\n  {t.description}\n  args: {t.args_hint}\n")
    return 0


def _add_model_args(p: argparse.ArgumentParser) -> None:
    # Defaults are None so config-file values survive the merge; see config.py.
    p.add_argument("--model", help="path to a GGUF model file")
    p.add_argument("--n-ctx", type=int)
    p.add_argument("--n-gpu-layers", type=int, help="-1 = all on GPU, 0 = CPU only")
    p.add_argument("--temperature", type=float)
    p.add_argument("--stream", action="store_true", default=None, help="stream tokens to stderr")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--max-tokens", type=int, help="max tokens per step")
    p.add_argument("--context-window", type=int, help="keep only the last N steps in the prompt")
    p.add_argument("--log-file", help="write a JSONL trace of the run here")


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="path to a TOML config file")
    p.add_argument("--tools", help="comma-separated tool allowlist")
    p.add_argument("--plugin-dir", help="directory of *.tool.json plugin manifests")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nano-agent",
        description="Offline-first autonomous LLM agent for edge devices.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run")
    _add_common_args(run_p)
    _add_model_args(run_p)
    run_p.add_argument("--task", required=True, help="the task to perform")

    repl_p = sub.add_parser("repl")
    _add_common_args(repl_p)
    _add_model_args(repl_p)

    tools_p = sub.add_parser("tools", help="list available tools and plugins")
    _add_common_args(tools_p)

    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "repl":
        return _cmd_repl(args)
    if args.command == "tools":
        return _cmd_tools(args)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
