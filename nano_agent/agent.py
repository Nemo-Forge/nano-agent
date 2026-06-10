"""The agent loop: plan -> tool -> observe, until done or budget exhausted."""

from __future__ import annotations

from collections.abc import Callable

from nano_agent.backend import Backend
from nano_agent.prompt import build_prompt, parse_step
from nano_agent.tools import BUILTIN_TOOLS, ToolError, ToolRegistry
from nano_agent.types import Action, AgentResult, Observation, Step

# Tokens beyond which a malformed step is retried with a correction nudge.
_STOP = ["\nOBSERVATION:", "\nSTEP:"]


class Agent:
    def __init__(
        self,
        backend: Backend,
        registry: ToolRegistry | None = None,
        max_steps: int = 20,
        max_tokens_per_step: int = 512,
        context_window: int = 0,
        max_consecutive_errors: int = 5,
        on_step: Callable[[Step], None] | None = None,
        on_observation: Callable[[Observation], None] | None = None,
    ) -> None:
        self.backend = backend
        self.registry = registry or ToolRegistry(BUILTIN_TOOLS)
        self.max_steps = max_steps
        self.max_tokens_per_step = max_tokens_per_step
        self.context_window = context_window
        self.max_consecutive_errors = max_consecutive_errors
        self.on_step = on_step
        self.on_observation = on_observation

    def run(self, task: str) -> AgentResult:
        history: list[Observation] = []
        tools_desc = self.registry.describe()
        consecutive_errors = 0

        for _ in range(self.max_steps):
            prompt = build_prompt(task, tools_desc, history, self.context_window)
            raw = self.backend.generate(prompt, self.max_tokens_per_step, _STOP)

            try:
                step = parse_step(raw)
            except ValueError as e:
                # Feed the parse failure back as an observation so the model
                # can self-correct rather than ending the run.
                obs = Observation(
                    step=Step(action=Action.TOOL, thought="", tool="(parse)"),
                    ok=False,
                    result="",
                    error=f"could not parse your output as JSON ({e}). "
                    "Respond with exactly one JSON object.",
                )
                history.append(obs)
                if self.on_observation:
                    self.on_observation(obs)
                consecutive_errors += 1
                if consecutive_errors >= self.max_consecutive_errors:
                    return self._give_up(history)
                continue

            if self.on_step:
                self.on_step(step)

            if step.action == Action.DONE:
                return AgentResult(
                    answer=step.answer or "",
                    steps_taken=len(history),
                    history=history,
                    stopped_reason="done",
                )

            obs = self._exec(step)
            history.append(obs)
            if self.on_observation:
                self.on_observation(obs)

            if obs.ok:
                consecutive_errors = 0
            else:
                consecutive_errors += 1
                if consecutive_errors >= self.max_consecutive_errors:
                    return self._give_up(history)

        return AgentResult(
            answer="(stopped: reached max steps without finishing)",
            steps_taken=len(history),
            history=history,
            stopped_reason="max_steps",
        )

    def _give_up(self, history: list[Observation]) -> AgentResult:
        return AgentResult(
            answer=f"(stopped: {self.max_consecutive_errors} consecutive errors without progress)",
            steps_taken=len(history),
            history=history,
            stopped_reason="too_many_errors",
        )

    def _exec(self, step: Step) -> Observation:
        try:
            tool = self.registry.get(step.tool or "")
            result = tool.run(step.args)
            return Observation(step=step, ok=True, result=result)
        except ToolError as e:
            return Observation(step=step, ok=False, result="", error=str(e))
        except Exception as e:  # tool bugs shouldn't kill the run
            return Observation(step=step, ok=False, result="", error=f"tool crashed: {e}")
