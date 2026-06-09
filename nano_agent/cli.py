"""Command-line interface for nano-agent."""

from __future__ import annotations

import argparse
import sys

from nano_agent.agent import Agent
from nano_agent.tools import BUILTIN_TOOLS, ToolRegistry
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


def _build_agent(args: argparse.Namespace) -> Agent:
    from nano_agent.backend import LlamaCppBackend

    backend = LlamaCppBackend(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        temperature=args.temperature,
    )
    registry = ToolRegistry(BUILTIN_TOOLS, allowed=args.tools.split(",") if args.tools else None)
    return Agent(
        backend=backend,
        registry=registry,
        max_steps=args.max_steps,
        max_tokens_per_step=args.max_tokens,
        on_step=None if args.quiet else _print_step,
        on_observation=None if args.quiet else _print_observation,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    agent = _build_agent(args)
    result = agent.run(args.task)
    print("\n" + "=" * 60)
    print(result.answer)
    print("=" * 60)
    print(
        f"steps: {result.steps_taken} | stopped: {result.stopped_reason}",
        file=sys.stderr,
    )
    return 0 if result.stopped_reason == "done" else 1


def _cmd_repl(args: argparse.Namespace) -> int:
    agent = _build_agent(args)
    print("nano-agent REPL — type a task, Ctrl-D to exit.")
    while True:
        try:
            task = input("\ntask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not task:
            continue
        result = agent.run(task)
        print("\n" + result.answer)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nano-agent",
        description="Offline-first autonomous LLM agent for edge devices.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("run", "repl"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, help="path to a GGUF model file")
        p.add_argument("--n-ctx", type=int, default=4096)
        p.add_argument("--n-gpu-layers", type=int, default=-1, help="-1 = all on GPU, 0 = CPU only")
        p.add_argument("--temperature", type=float, default=0.1)
        p.add_argument("--max-steps", type=int, default=20)
        p.add_argument("--max-tokens", type=int, default=512, help="max tokens per step")
        p.add_argument("--tools", default="", help="comma-separated tool allowlist")
        p.add_argument("--quiet", action="store_true", help="suppress step/observation trace")
        if name == "run":
            p.add_argument("--task", required=True, help="the task to perform")

    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "repl":
        return _cmd_repl(args)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
