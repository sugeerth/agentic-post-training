"""The RL environment: one Episode per attempt at one task. Multi-turn, text in / text out, stateful.

The agent writes text; the environment answers with a tool result or ends the episode (gym-style step). Everything the
reward needs to know about HOW the agent got its answer (which tools ran, which succeeded) is recorded here by the
environment itself, never parsed back out of the agent's text. That closes a hole the first support reward had: it
granted "looked the order up" credit whenever the text contained "<result>order", and the agent can type that string
itself. (No saved run ever did, but a policy optimiser finds holes like that eventually.)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from selflearn.tools import Tools, last_call


@dataclass(frozen=True)
class ToolCall:
    name: str
    arg: str
    result: str
    ok: bool


@dataclass
class Episode:
    task: object               # selflearn.usecase.Task
    tools: Tools
    log: list[ToolCall] = field(default_factory=list)
    done: bool = False

    def step(self, text: str, ended: bool = False) -> str | None:
        """The agent's latest turn in; the tool result to show it out, or None when the episode is over
        (it answered, it stopped without calling a tool, or generation ended)."""
        call = None if ("</answer>" in text or ended) else last_call(text, self.tools)
        if call is None:
            self.done = True
            return None
        result = self.tools[call[0]](call[1])
        self.log.append(ToolCall(call[0], call[1], result, not result.startswith("error")))
        return result

    def used(self, tool: str) -> bool:
        return any(c.ok and c.name == tool for c in self.log)


@dataclass
class Knobs:
    """The environment settings the Designer agent may turn (shared with the Verifier, which scores with them).
    balance: 0 = tasks come in their natural answer mix; 1 = within each level every answer is equally common.
    partial: multiplier on partial credit (1 = as the use case defines it, 0 = only fully right answers pay)."""
    balance: float = 0.0
    partial: float = 1.0


def score(usecase, text: str, answer, log: list[ToolCall], partial: float = 1.0) -> float:
    """The environment's reward: 0 unless every tool the use case requires ran successfully in this episode
    (evidence, checked from the environment's own log), otherwise the use case's answer reward, with any partial
    credit scaled by `partial`."""
    for need in usecase.requires:
        if not any(c.ok and c.name == need for c in log):
            return 0.0
    r = usecase.reward(text, answer)
    return r if r >= 1.0 else r * partial


def answer_key(answer) -> str:
    """The kind of answer, for spotting collapse: 'refund 35.00' -> 'refund'; a number stays itself."""
    s = str(answer).strip().lower()
    return s.split()[0] if s else ""
