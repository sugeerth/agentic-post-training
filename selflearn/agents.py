"""The five agents of the self-learning loop. One job each; dependencies come in through the constructor;
each reports what it did on the bus. None of them knows about the others (the Coordinator wires them).

  Curriculum  proposes what to practise next, from the agent's own success rates
  Actor       attempts tasks with tools (multi-turn), returns trajectories
  Verifier    scores trajectories against the true answers (binary, exact)
  Learner     GRPO update on the scored trajectories
  Assessor    greedy accuracy on the fixed held-out set, per level
  Selector    (optional) keeps the best checkpoint by validation accuracy and restores it at the end
"""
from __future__ import annotations

import random
from collections.abc import Callable

from selflearn.bus import Bus
from selflearn.curriculum import Curriculum
from selflearn.grpo import group_advantages
from selflearn.rollout import Trajectory, rollout
from selflearn.usecase import Task, UseCase
from selflearn.verifier import final_answer


class CurriculumAgent:
    name = "Curriculum"

    def __init__(self, usecase: UseCase, group: int, rng: random.Random, exclude: set[str], bus: Bus):
        self.uc, self.group, self.rng, self.exclude, self.bus = usecase, group, rng, exclude, bus
        self.cur = Curriculum({}, group, usecase.levels)

    def calibrate(self, success_by_level: dict[int, float]) -> None:
        self.cur = Curriculum(success_by_level, self.group, self.uc.levels)
        w = self.cur.weights()
        self.bus.send(self.name, "Coordinator", "practice weights from the baseline: "
                      + " ".join(f"L{k} {v:.0%}" for k, v in w.items()))

    def propose(self, n: int) -> list[Task]:
        levels = self.cur.sample(n, self.rng)
        self.bus.send(self.name, "Actor", f"practise levels {sorted(levels)}", levels=levels)
        return [self.uc.train_task(lv, self.rng, self.exclude) for lv in levels]

    def observe(self, trajs: list[Trajectory]) -> None:
        for g in range(0, len(trajs), self.group):
            grp = trajs[g:g + self.group]
            self.cur.update(grp[0].task.level, sum(t.reward for t in grp) / len(grp))


class ActorAgent:
    name = "Actor"

    def __init__(self, model, tok, usecase: UseCase, call_tool: Callable, bus: Bus, max_turns: int = 4):
        self.model, self.tok, self.uc, self.call_tool, self.bus = model, tok, usecase, call_tool, bus
        self.max_turns = max_turns

    def act(self, tasks: list[Task], group: int, temp: float, report: bool = True) -> list[Trajectory]:
        uc = self.uc
        trajs = rollout(self.model, self.tok, tasks, group=group, temp=temp, system_prompt=uc.system_prompt,
                        tools=uc.tools, call_tool=self.call_tool, user_prefix=uc.user_prefix,
                        max_turns=self.max_turns, max_new=uc.max_new)
        if report:
            calls = sum(t.calls for t in trajs) / len(trajs)
            self.bus.send(self.name, "Verifier", f"{len(trajs)} attempts, {calls:.1f} tool calls each")
        return trajs


class VerifierAgent:
    name = "Verifier"

    def __init__(self, reward_fn: Callable[[str, int | float | str], float], bus: Bus):
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

    def __init__(self, actor: ActorAgent, verifier: VerifierAgent, held_out: list[Task], bus: Bus):
        self.actor, self.verifier, self.held_out, self.bus = actor, verifier, held_out, bus

    def assess(self, tag: str) -> tuple[dict, list[dict]]:
        trajs = self.actor.act(self.held_out, group=1, temp=0.0, report=False)
        self.verifier.score(trajs, report=False)
        correct = [t.reward >= 1.0 for t in trajs]   # accuracy counts only fully correct answers, never partial credit
        by = {lv: [c for c, t in zip(correct, trajs, strict=True) if t.task.level == lv]
              for lv in sorted({t.task.level for t in trajs})}
        summary = {"accuracy": sum(correct) / len(trajs),
                   "by_level": {lv: sum(v) / len(v) for lv, v in by.items() if v},
                   "used_tool": sum(t.calls > 0 for t in trajs) / len(trajs),
                   "answered": sum(final_answer(t.text) is not None for t in trajs) / len(trajs),
                   "mean_calls": sum(t.calls for t in trajs) / len(trajs)}
        self.bus.send(self.name, "Coordinator", f"{tag}: held-out accuracy {summary['accuracy']:.1%} ("
                      + " ".join(f"L{k} {v:.0%}" for k, v in summary["by_level"].items()) + ")", **{"tag": tag})
        records = [{"level": t.task.level, "question": t.task.question, "answer": t.task.answer, "text": t.text,
                    "reward": t.reward, "calls": t.calls} for t in trajs]
        return summary, records


class SelectorAgent:
    """Early stopping without touching the test set: scores a separate validation set every `every` steps, keeps the
    best LoRA weights, and restores them at the end. Step 0 is the untouched base model, so a run that only gets worse
    ships the base model instead of a collapsed one."""
    name = "Selector"

    def __init__(self, model, actor: ActorAgent, verifier: VerifierAgent, validation: list[Task], bus: Bus,
                 every: int = 5):
        self.model, self.actor, self.verifier, self.validation, self.bus, self.every = model, actor, verifier, validation, bus, every
        self.best_acc, self.best_step, self.best = -1.0, 0, None

    def _accuracy(self) -> float:
        trajs = self.actor.act(self.validation, group=1, temp=0.0, report=False)
        self.verifier.score(trajs, report=False)
        return sum(t.reward >= 1.0 for t in trajs) / len(trajs)

    def consider(self, step: int) -> None:
        if step % self.every:
            return
        from mlx.utils import tree_flatten
        acc = self._accuracy()
        if acc > self.best_acc:
            self.best_acc, self.best_step = acc, step
            self.best = dict(tree_flatten(self.model.trainable_parameters()))  # MLX arrays are immutable: a snapshot
            note = "new best, kept"
        else:
            note = f"best stays step {self.best_step} ({self.best_acc:.1%})"
        self.bus.send(self.name, "Coordinator", f"step {step}: validation accuracy {acc:.1%} ({note})",
                      step=step, validation=acc)

    def restore_best(self) -> int:
        import mlx.core as mx
        self.model.load_weights(list(self.best.items()), strict=False)
        mx.eval(self.model.parameters())
        self.bus.send(self.name, "Coordinator", f"restored step {self.best_step} (validation {self.best_acc:.1%})")
        return self.best_step
