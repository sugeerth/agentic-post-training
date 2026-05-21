"""Dashboard rendering — moved out of `CoordinatorAgent`.

The legacy Coordinator owned a `print_dashboard` method that mixed
presentation (ANSI colors, emoji, layout) with orchestration. This module
isolates the presentation so the Coordinator can stay focused on dispatch.

`render_dashboard` returns a string and `print_dashboard` writes it to a
stream. Tests assert on the string; the agent prints. No coupling on
asyncio or the bus.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, TextIO

from agents.base_agent import BOLD, RESET, AgentStatus

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent
    from agents.planner import PipelineStage


_STATUS_ICON = {
    AgentStatus.IDLE: "⚪",
    AgentStatus.RUNNING: "🔵",
    AgentStatus.WAITING: "🟡",
    AgentStatus.COMPLETED: "🟢",
    AgentStatus.FAILED: "🔴",
}

_STAGE_ICON = {
    "pending": "⬜",
    "running": "🔄",
    "completed": "✅",
    "failed": "❌",
}


def render_dashboard(
    agents: dict[str, BaseAgent],
    stages: list[PipelineStage],
) -> str:
    """Render the dashboard as a single multi-line string.

    Pure function — no I/O. The Coordinator calls `print_dashboard` which is
    a thin `print(render_dashboard(...))` wrapper.
    """
    lines: list[str] = []
    bar = "═" * 70
    lines.append(f"\n{BOLD}{bar}")
    lines.append("  📊 Pipeline Dashboard")
    lines.append(f"{bar}{RESET}")

    lines.append(f"\n  {BOLD}Agents:{RESET}")
    for name, agent in agents.items():
        icon = _STATUS_ICON.get(agent.status, "⚪")
        caps = ", ".join(c.name for c in getattr(agent, "capabilities", [])[:3])
        color = getattr(agent, "_color", "")
        lines.append(f"    {icon} {color}{name}{RESET} [{agent.role}] - {caps}")

    if stages:
        lines.append(f"\n  {BOLD}Pipeline Stages:{RESET}")
        for i, stage in enumerate(stages):
            icon = _STAGE_ICON.get(stage.status, "⬜")
            dur = f" ({stage.duration:.1f}s)" if stage.duration > 0 else ""
            lines.append(f"    {icon} {i+1}. {stage.name}: {stage.description}{dur}")

    lines.append(f"\n{BOLD}{bar}{RESET}\n")
    return "\n".join(lines)


def print_dashboard(
    agents: dict[str, BaseAgent],
    stages: list[PipelineStage],
    stream: TextIO | None = None,
) -> None:
    print(render_dashboard(agents, stages), file=stream or sys.stdout)
