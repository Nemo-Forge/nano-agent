"""Prompt construction and planner-output parsing.

We ask the model to emit a single JSON object per step. Small models (1-3B)
are not perfectly reliable at this, so the parser is forgiving: it extracts
the first balanced JSON object from the text rather than requiring the whole
output to be valid JSON.
"""

from __future__ import annotations

import json

from nano_agent.types import Action, Observation, Step

SYSTEM_TEMPLATE = """You are nano-agent, an autonomous assistant running locally on an edge device.

You solve the task by repeating: think, then either call a tool or finish.
Respond with EXACTLY ONE JSON object and nothing else.

To call a tool:
{{"thought": "why", "action": "tool", "tool": "<name>", "args": {{...}}}}

To finish:
{{"thought": "why", "action": "done", "answer": "<final answer>"}}

Available tools:
{tools}

Rules:
- Output only the JSON object, no prose before or after.
- Use one tool per step. Wait for the OBSERVATION before the next step.
- Finish as soon as you can answer the task.
"""


def build_prompt(
    task: str, tools_desc: str, history: list[Observation], window: int = 0
) -> str:
    """Render the prompt. If window > 0, only the last `window` observations are
    included (older ones are summarized as omitted) to bound context growth on
    long runs — important on edge devices with a 4-8K token window."""
    system = SYSTEM_TEMPLATE.format(tools=tools_desc)
    parts = [system, f"\nTASK: {task}\n"]
    shown = history if window <= 0 else history[-window:]
    omitted = len(history) - len(shown)
    if omitted > 0:
        parts.append(f"[{omitted} earlier step(s) omitted to save context]\n")
    for obs in shown:
        step = obs.step
        if step.action == Action.TOOL:
            parts.append(
                f'STEP: {{"action":"tool","tool":"{step.tool}","args":{json.dumps(step.args)}}}'
            )
        body = obs.result if obs.ok else f"ERROR: {obs.error}"
        parts.append(f"OBSERVATION: {body}\n")
    parts.append("STEP:")
    return "\n".join(parts)


def _extract_json(text: str) -> dict:
    """Return the first balanced {...} object found in text."""
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in model output")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON object in model output")


def parse_step(text: str) -> Step:
    """Parse a planner step from raw model output.

    Raises ValueError if no usable JSON object can be recovered.
    """
    obj = _extract_json(text)
    action = obj.get("action")
    thought = obj.get("thought", "")
    if action == "done":
        return Step(action=Action.DONE, thought=thought, answer=obj.get("answer", ""))
    if action == "tool":
        tool = obj.get("tool")
        if not tool:
            raise ValueError("tool action missing 'tool' field")
        return Step(
            action=Action.TOOL,
            thought=thought,
            tool=tool,
            args=obj.get("args", {}),
        )
    raise ValueError(f"unknown action: {action!r}")
