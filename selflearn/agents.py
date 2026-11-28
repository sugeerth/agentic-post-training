"""The five agents of the self-learning loop. One job each; dependencies come in through the constructor;
each reports what it did on the bus. None of them knows about the others (the Coordinator wires them).

  Curriculum  proposes what to practise next, from the agent's own success rates
  Actor       attempts tasks with tools (multi-turn), returns trajectories
  Verifier    scores trajectories against the true answers (binary, exact)
  Learner     GRPO update on the scored trajectories
  Assessor    greedy accuracy on the fixed held-out set, per level
  Selector    (optional) keeps the best checkpoint by validation accuracy and restores it at the end
  Designer    (optional) owns the environment: audits it for shortcuts before training, switches to the variant
              where doing the task beats every shortcut by the most (or refuses), and watches for reward hacking
"""
from __future__ import annotations

import random
from collections import Counter

from selflearn import audit as env_audit
from selflearn.bus import Bus
from selflearn.curriculum import Curriculum
from selflearn.env import Knobs, answer_key, score
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

    def __init__(self, model, tok, usecase: UseCase, bus: Bus, max_turns: int = 4):
        self.model, self.tok, self.uc, self.bus, self.max_turns = model, tok, usecase, bus, max_turns

    def act(self, tasks: list[Task], group: int, temp: float, report: bool = True) -> list[Trajectory]:
        uc = self.uc
        trajs = rollout(self.model, self.tok, tasks, group=group, temp=temp, system_prompt=uc.system_prompt,
                        tools=uc.tools, user_prefix=uc.user_prefix,
                        max_turns=self.max_turns, max_new=uc.max_new)
        if report:
            calls = sum(t.calls for t in trajs) / len(trajs)
            self.bus.send(self.name, "Verifier", f"{len(trajs)} attempts, {calls:.1f} tool calls each")
        return trajs


class VerifierAgent:
    name = "Verifier"

    def __init__(self, usecase: UseCase, bus: Bus, knobs: Knobs | None = None):
        self.uc, self.bus, self.knobs = usecase, bus, knobs or Knobs()

    def score(self, trajs: list[Trajectory], report: bool = True, train: bool = True) -> float:
        """Training scores use the Designer's knobs; evaluation (train=False) always uses the use case's own reward,
        so before/after numbers stay comparable."""
        partial = self.knobs.partial if train else 1.0
        for t in trajs:
            t.reward = score(self.uc, t.text, t.task.answer, t.ep.log if t.ep else [], partial)
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
        self.verifier.score(trajs, report=False, train=False)
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
        self.verifier.score(trajs, report=False, train=False)
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


class DesignerAgent:
    """Owns the environment, the way an RL environment designer would. It turns two knobs (env.Knobs) and nothing else:
    the answer mix of the tasks, and how much partial credit pays.

    review   before training: audits the environment against the agent's baseline (selflearn/audit.py). If a
             shortcut pays about as much as doing the task, it searches the environment variants, auditing each one,
             and switches to the one where doing the task beats the best shortcut by the most. If no variant makes
             the task pay more than a shortcut, it refuses to train (no GPU time spent) unless forced.
    shape    each step: rebalances the batch the Curriculum proposed to the chosen answer mix.
    observe  each step: watches the attempts for reward hacking (one answer for everything, skipped evidence,
             tool results typed by the agent). Collapse onto one answer switches full answer balance on.
    """
    name = "Designer"

    def __init__(self, usecase: UseCase, held_out: list[Task], knobs: Knobs, rng: random.Random, exclude: set[str],
                 bus: Bus, force: bool = False, collapse: float = 0.8):
        self.uc, self.held_out, self.knobs, self.rng, self.exclude, self.bus = usecase, held_out, knobs, rng, exclude, bus
        self.force, self.collapse = force, collapse
        self.keys = env_audit.level_answers(usecase)
        self.audit: env_audit.EnvAudit | None = None
        self.verdict, self.variants = ("PASS", ""), []

    def review(self, baseline: list[dict]) -> bool:
        policy = env_audit.mean_reward(baseline)
        self.audit = env_audit.audit(self.uc, self.held_out)   # the very tasks the baseline was measured on
        self.verdict = v, why = self.audit.verdict(policy)
        self.bus.send(self.name, "Coordinator", f"env audit {v}: {why}", verdict=v,
                      shortcuts={k: round(p["reward"], 3) for k, p in self.audit.probes.items()})
        if v == "PASS":
            return True
        self.variants = env_audit.search(self.uc, baseline, self.held_out, self.exclude)
        best = env_audit.choose(self.variants)
        self.bus.send(self.name, "Coordinator", "variants (balance, partial credit -> agent vs best shortcut): " + "; ".join(
            f"{r['balance']:.0%}/{r['partial']:.1f} -> {r['agent']:.2f} vs {r['shortcut']:.2f}" for r in self.variants[:4]),
            variants=self.variants)
        if best["margin"] <= 0 and not self.force:
            self.verdict = ("BLOCK", f"{why}; no environment variant makes doing the task pay more than a shortcut "
                                     "for this agent")
            self.bus.send(self.name, "Coordinator", "no variant makes doing the task pay more than a shortcut for this "
                          "agent: refusing to train (pass --force 1 to override)")
            return False
        self.knobs.balance, self.knobs.partial = best["balance"], best["partial"]
        self.bus.send(self.name, "Verifier", f"environment changed: answer balance {best['balance']:.0%}, partial credit "
                      f"x{best['partial']:.1f}; the agent now beats the best shortcut ('{best['by']}') by "
                      f"{best['margin']:+.2f}", **best)
        return True

    def shape(self, tasks: list[Task]) -> list[Task]:
        if not self.knobs.balance:
            return tasks
        return env_audit.balance(self.uc, tasks, self.knobs.balance, self.rng, self.exclude, self.keys)

    def observe(self, trajs: list[Trajectory]) -> None:
        said = Counter(answer_key(a) if (a := final_answer(t.text)) is not None else "(none)" for t in trajs)
        truth = Counter(answer_key(t.task.answer) for t in trajs)
        top, n = said.most_common(1)[0]
        share, true_share = n / len(trajs), truth[top] / len(trajs)
        notes = []
        if top != "(none)" and share >= self.collapse and share - true_share >= 0.25:
            notes.append(f"collapse: '{top}' on {share:.0%} of attempts, right on {true_share:.0%}")
            if self.knobs.balance < 1.0 and self.keys:
                self.knobs.balance = 1.0
                notes.append("answer balance -> 100%")
        for need in self.uc.requires:
            skipped = sum(not t.ep.used(need) for t in trajs if t.ep) / len(trajs)
            if skipped > 0.5:
                notes.append(f"{skipped:.0%} of attempts skipped <{need}> (they score 0)")
        forged = sum(t.text.count("<result>") > t.calls for t in trajs)
        if forged:
            notes.append(f"{forged} attempts typed their own <result> (no credit: evidence comes from the env log)")
        mix = " ".join(f"{k} {v / len(trajs):.0%}" for k, v in said.most_common(3))
        self.bus.send(self.name, "Coordinator", f"answers: {mix}" + ("; " + "; ".join(notes) if notes else ""),
                      answers={k: round(v / len(trajs), 3) for k, v in said.items()}, alarms=notes,
                      balance=self.knobs.balance, partial=self.knobs.partial)
