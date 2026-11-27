"""The five agents of the self-learning loop. One job each; dependencies come in through the constructor;
each reports what it did on the bus. None of them knows about the others (the Coordinator wires them).

  Curriculum  proposes what to practise next, from the agent's own success rates
  Actor       attempts tasks with tools (multi-turn), returns trajectories
  Verifier    scores trajectories against the true answers (binary, exact)
  Learner     GRPO update on the scored trajectories
  Assessor    greedy accuracy on the fixed held-out set, per level
"""
from __future__ import annotations

import random
from collections.abc import Callable

from selflearn import tasks as T
from selflearn.bus import Bus
from selflearn.curriculum import Curriculum
from selflearn.grpo import group_advantages
from selflearn.rollout import Trajectory, rollout
from selflearn.verifier import parse_answer


class CurriculumAgent:
    name = "Curriculum"

    def __init__(self, group: int, rng: random.Random, exclude: set[str], bus: Bus):
        self.group, self.rng, self.exclude, self.bus = group, rng, exclude, bus
        self.cur = Curriculum({}, group)

    def calibrate(self, success_by_level: dict[int, float]) -> None:
        self.cur = Curriculum(success_by_level, self.group)
        w = self.cur.weights()
        self.bus.send(self.name, "Coordinator", "practice weights from the baseline: "
                      + " ".join(f"L{k} {v:.0%}" for k, v in w.items()))

    def propose(self, n: int) -> list[T.Task]:
        levels = self.cur.sample(n, self.rng)
        self.bus.send(self.name, "Actor", f"practise levels {sorted(levels)}", levels=levels)
        return [T.make_train_task(lv, self.rng, self.exclude) for lv in levels]

    def observe(self, trajs: list[Trajectory]) -> None:
        for g in range(0, len(trajs), self.group):
            grp = trajs[g:g + self.group]
            self.cur.update(grp[0].task.level, sum(t.reward for t in grp) / len(grp))


class ActorAgent:
    name = "Actor"

    def __init__(self, model, tok, tools: dict[str, Callable[[str], str]], call_tool: Callable, bus: Bus,
                 max_turns: int = 4, max_new: int = 96):
        self.model, self.tok, self.tools, self.call_tool, self.bus = model, tok, tools, call_tool, bus
        self.max_turns, self.max_new = max_turns, max_new

    def act(self, tasks: list[T.Task], group: int, temp: float, report: bool = True) -> list[Trajectory]:
        trajs = rollout(self.model, self.tok, tasks, group=group, temp=temp, tools=self.tools, call_tool=self.call_tool,
                        max_turns=self.max_turns, max_new=self.max_new)
        if report:
            calls = sum(t.calls for t in trajs) / len(trajs)
            self.bus.send(self.name, "Verifier", f"{len(trajs)} attempts, {calls:.1f} tool calls each")
        return trajs


class VerifierAgent:
    name = "Verifier"

    def __init__(self, reward_fn: Callable[[str, int], float], bus: Bus):
        self.reward_fn, self.bus = reward_fn, bus

    def score(self, trajs: list[Trajectory], report: bool = True) -> float:
        for t in trajs:
            t.reward = self.reward_fn(t.text, t.task.answer)
        mean = sum(t.reward for t in trajs) / len(trajs)
        if report:
            self.bus.send(self.name, "Learner", f"mean reward {mean:.2f}", reward=mean)
        return mean


class LearnerAgent:
    name = "Learner"

    def __init__(self, learner, group: int, bus: Bus):
        self.learner, self.group, self.bus = learner, group, bus

    def learn(self, trajs: list[Trajectory]) -> dict:
        advs = group_advantages([t.reward for t in trajs], self.group)
        signal = []
        for t, a in zip(trajs, advs, strict=True):
            if a is not None:
                t.adv = a
                signal.append(t)
        loss, gnorm = self.learner.step(signal)
        informative = len(signal) // self.group
        n_groups = len(trajs) // self.group
        text = (f"updated on {informative}/{n_groups} groups, grad norm {gnorm:.2f}" if signal
                else f"no update: all {n_groups} groups agreed (no signal)")
        self.bus.send(self.name, "Coordinator", text, informative=informative, grad_norm=gnorm)
        return {"informative_groups": informative, "groups": n_groups, "loss": loss, "grad_norm": gnorm}


class AssessorAgent:
    name = "Assessor"

    def __init__(self, actor: ActorAgent, verifier: VerifierAgent, held_out: list[T.Task], bus: Bus):
        self.actor, self.verifier, self.held_out, self.bus = actor, verifier, held_out, bus

    def assess(self, tag: str) -> tuple[dict, list[dict]]:
        trajs = self.actor.act(self.held_out, group=1, temp=0.0, report=False)
        self.verifier.score(trajs, report=False)
        by = {lv: [t.reward for t in trajs if t.task.level == lv] for lv in T.LEVELS}
        summary = {"accuracy": sum(t.reward for t in trajs) / len(trajs),
                   "by_level": {lv: sum(v) / len(v) for lv, v in by.items() if v},
                   "used_tool": sum(t.calls > 0 for t in trajs) / len(trajs),
                   "answered": sum(parse_answer(t.text) is not None for t in trajs) / len(trajs),
                   "mean_calls": sum(t.calls for t in trajs) / len(trajs)}
        self.bus.send(self.name, "Coordinator", f"{tag}: held-out accuracy {summary['accuracy']:.1%} ("
                      + " ".join(f"L{k} {v:.0%}" for k, v in summary["by_level"].items()) + ")", **{"tag": tag})
        records = [{"level": t.task.level, "question": t.task.question, "answer": t.task.answer, "text": t.text,
                    "reward": t.reward, "calls": t.calls} for t in trajs]
        return summary, records
