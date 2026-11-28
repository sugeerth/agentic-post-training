"""Environment audit: before any GPU time, the Designer agent plays scripted policies in the environment and asks
three questions of the ENVIRONMENT (not of the model):

  1. Solvable?  The oracle does the job right; it must get full reward on every task, or the verifier is broken.
  2. Hackable?  Each shortcut probe skips part of the job: never looks the order up, always says "deny", types a
                fake tool result, copies a number out of the question. The best one's mean reward is what an agent
                earns for NOT doing the task.
  3. Does the shortcut beat the agent?  If a shortcut pays as much as the agent already earns (its baseline mean
                reward), group-relative RL drifts to it: reward hacking, predicted before training starts.

  python3 -m selflearn.audit --usecase support                      # the audit
  python3 -m selflearn.audit --usecase support --policy runs/<run>  # plus the verdict against that run's baseline
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from selflearn.env import Episode, answer_key, score
from selflearn.verifier import final_answer

STATIC_LIMIT = 0.25   # with no baseline to compare against, a shortcut paying more than this is worth a warning
MARGIN = 0.10         # an agent less than this far above the best shortcut is at risk of drifting to it


def _oracle(task, ep) -> str:
    final = f"<answer>{task.answer}</answer>"
    ep.step(final)
    return final


def _forged(tools: tuple[str, ...], answer: str):
    """Types the tool's result itself instead of calling the tool, then gives the most common answer."""
    def play(task, ep) -> str:
        final = "".join(f"<result>{t} (typed by the agent, no tool ran)</result>\n" for t in tools) + f"<answer>{answer}</answer>"
        ep.step(final)
        return final
    return play


@dataclass
class EnvAudit:
    usecase: str
    tasks: int
    distinct: float                 # share of distinct questions in the sample
    solved: float                   # share of tasks where the oracle gets full reward
    probes: dict[str, dict]         # shortcut name -> {"reward", "by_level", "answers"}

    @property
    def best_shortcut(self) -> tuple[str, float]:
        name = max(self.probes, key=lambda k: self.probes[k]["reward"])
        return name, self.probes[name]["reward"]

    def constant_answer(self, name: str) -> str | None:
        """The one answer a probe always gives (e.g. 'deny'), or None if it varies."""
        answers = self.probes[name]["answers"]
        return next(iter(answers)) if len(answers) == 1 else None

    def verdict(self, policy: float | None = None) -> tuple[str, str]:
        """PASS / WARN / BLOCK, and why, in one sentence."""
        name, best = self.best_shortcut
        if self.solved < 1.0:
            return "BLOCK", f"the verifier rejects the correct answer on {1 - self.solved:.0%} of tasks"
        if policy is None:
            if best > STATIC_LIMIT:
                return "WARN", f"'{name}' earns {best:.2f} without doing the task"
            return "PASS", f"no shortcut earns more than {best:.2f}"
        if best >= policy:
            return "BLOCK", (f"'{name}' earns {best:.2f}, more than the agent's {policy:.2f}: "
                             "RL would drift to the shortcut")
        if best > policy - MARGIN:
            return "WARN", (f"the agent earns {policy:.2f}, only {policy - best:.2f} above '{name}' ({best:.2f}); "
                            "train with a KL anchor and checkpoint selection")
        return "PASS", f"the agent earns {policy:.2f}, well above the best shortcut ({best:.2f}, '{name}')"


def sample(uc, per_level: int = 50, seed: int = 7) -> list:
    rng = random.Random(seed)
    return [uc.make_task(lv, rng) for lv in uc.levels for _ in range(per_level)]


def level_answers(uc, per_level: int = 60, seed: int = 3) -> dict[int, list[str]]:
    """The kinds of answer each level can have, from a sample. Levels with one kind (or numbers) are left out."""
    rng, out = random.Random(seed), {}
    for lv in uc.levels:
        keys = sorted({answer_key(uc.make_task(lv, rng).answer) for _ in range(per_level)})
        if 1 < len(keys) <= 5:
            out[lv] = keys
    return out


def balance(uc, tasks: list, strength: float, rng: random.Random, exclude=frozenset(), keys=None, tries: int = 40):
    """Edit the task mix: with probability `strength`, swap a task for a same-level task whose kind of answer is drawn
    uniformly from that level's kinds. At strength 1 every answer is equally common within a level: the mix that
    minimises what any one constant answer can earn. (Removing the most common answer instead just hands the
    shortcut to the next one: the first version of this Designer did that and the best shortcut went UP.)"""
    keys = level_answers(uc) if keys is None else keys
    out = []
    for t in tasks:
        if t.level in keys and rng.random() < strength:
            want = rng.choice(keys[t.level])
            for _ in range(tries):
                alt = uc.train_task(t.level, rng, exclude)
                if answer_key(alt.answer) == want:
                    t = alt
                    break
        out.append(t)
    return out


def policy_estimate(records: list[dict], tasks: list, partial: float = 1.0) -> float:
    """What the agent would earn on `tasks`, from its scored baseline attempts: its mean reward for each (level, kind
    of answer), partial credit rescaled, weighted by how common that (level, kind) is in `tasks`. On the baseline's
    own tasks with partial = 1 this is exactly its measured mean reward."""
    cell, level = defaultdict(list), defaultdict(list)
    for r in records:
        x = r["reward"] if r["reward"] >= 1.0 else r["reward"] * partial
        cell[(r["level"], answer_key(r["answer"]))].append(x)
        level[r["level"]].append(x)
    vals = [cell.get((t.level, answer_key(t.answer))) or level.get(t.level) or [0.0] for t in tasks]
    return sum(sum(v) / len(v) for v in vals) / len(vals)


def search(uc, records: list[dict], pool: list, exclude=frozenset()) -> list[dict]:
    """Audit every environment variant (answer balance x partial credit) against the agent's baseline. Returns rows,
    best first: margin = what the agent would earn - what the best shortcut earns, in that variant."""
    keys, rows = level_answers(uc), []
    for b in (0.0, 0.5, 1.0):
        tasks = balance(uc, pool, b, random.Random(0), exclude, keys) if b else pool
        for partial in (1.0, 0.5, 0.0):
            name, short = audit(uc, tasks, partial).best_shortcut
            pol = policy_estimate(records, tasks, partial)
            rows.append({"balance": b, "partial": partial, "agent": round(pol, 3), "shortcut": round(short, 3),
                         "by": name, "margin": round(pol - short, 3)})
    return sorted(rows, key=lambda r: (-r["margin"], r["balance"], -r["partial"]))


def choose(rows: list[dict], tol: float = 0.02) -> dict:
    """The smallest change to the environment whose margin is within `tol` (about the estimates' noise) of the best."""
    return min((r for r in rows if r["margin"] >= rows[0]["margin"] - tol), key=lambda r: (r["balance"], -r["partial"]))


def audit(uc, tasks: list, partial: float = 1.0) -> EnvAudit:
    """Play the oracle and every shortcut probe on `tasks`. To compare with an agent's baseline, pass the same tasks
    the baseline was measured on (scripted probes don't learn, so auditing on held-out tasks leaks nothing)."""
    levels = sorted({t.level for t in tasks})
    common = Counter(str(t.answer) for t in tasks).most_common(1)[0][0]
    probes = {"oracle": _oracle, **uc.probes}      # a use case may bring its own oracle (one that uses the tools)
    if uc.requires:
        probes[f"type a fake tool result, answer '{common}'"] = _forged(uc.requires, common)
    results = {}
    for name, play in probes.items():
        rewards, answers = [], Counter()
        for t in tasks:
            ep = Episode(t, uc.tools)
            text = play(t, ep)
            rewards.append(score(uc, text, t.answer, ep.log, partial))
            got = final_answer(text)
            answers[answer_key(got) if got is not None else "(none)"] += 1
        by_level = {lv: sum(r for r, t in zip(rewards, tasks, strict=True) if t.level == lv)
                    / sum(t.level == lv for t in tasks) for lv in levels}
        results[name] = {"reward": sum(rewards) / len(rewards), "full": sum(r >= 1.0 for r in rewards) / len(rewards),
                         "by_level": by_level, "answers": dict(answers.most_common(3))}
    oracle = results.pop("oracle")
    return EnvAudit(uc.name, len(tasks), len({t.question for t in tasks}) / len(tasks), oracle["full"], results)


def baseline(run: Path) -> list[dict]:
    """The agent's baseline attempts from a run's eval_before.json (scored with the reward that run trained on)."""
    return json.loads((run / "eval_before.json").read_text())["records"]


def mean_reward(records: list[dict]) -> float:
    return sum(r["reward"] for r in records) / len(records)


def render(a: EnvAudit, policy: float | None = None) -> str:
    levels = list(next(iter(a.probes.values()))["by_level"])
    lines = [f"ENV AUDIT  {a.usecase}: {a.tasks} tasks ({a.tasks // len(levels)} per level), {a.distinct:.0%} distinct",
             f"  solvable: the oracle gets full reward on {a.solved:.0%} of tasks", "",
             f"  {'shortcut probe':44s} {'reward':>6s}  " + " ".join(f"{'L' + str(lv):>4s}" for lv in levels)]
    for name, p in sorted(a.probes.items(), key=lambda kv: -kv[1]["reward"]):
        lines.append(f"  {name[:44]:44s} {p['reward']:6.2f}  " + " ".join(f"{p['by_level'][lv]:4.2f}" for lv in levels))
    v, why = a.verdict(policy)
    lines += ["", (f"  agent's baseline mean reward: {policy:.2f}" if policy is not None else "  (no agent baseline given)"),
              f"  VERDICT {v}: {why}"]
    return "\n".join(lines)


def main(argv=None):
    from selflearn.usecases import HISTORY, USE_CASES
    envs = {**USE_CASES, **HISTORY}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usecase", default="support", choices=sorted(envs))
    ap.add_argument("--per-level", type=int, default=50, help="sample size per level when no --policy is given")
    ap.add_argument("--policy", default=None, help="a run folder: compare the shortcuts to its baseline mean reward")
    ap.add_argument("--out", default=None, help="also write the audit as JSON here")
    args = ap.parse_args(argv)
    uc, pol = envs[args.usecase], None
    if args.policy:      # audit on the very tasks the baseline was measured on (the run's held-out set, regenerated)
        recs = baseline(Path(args.policy))
        tasks = uc.eval_set(len(recs) // len(uc.levels))
        assert [t.question for t in tasks] == [r["question"] for r in recs], "held-out set differs from the run's"
        pol = mean_reward(recs)
    else:
        tasks = sample(uc, args.per_level)
    a = audit(uc, tasks)
    print(render(a, pol))
    rows = search(uc, recs, tasks) if pol is not None and a.verdict(pol)[0] != "PASS" else []
    if rows:
        print("\n  environment variants (what the Designer would choose from), best first:")
        print(f"  {'answer balance':>14s} {'partial credit':>14s} {'agent':>6s} {'shortcut':>8s} {'margin':>7s}  best shortcut")
        pick = choose(rows)
        for r in rows:
            print(f"  {r['balance']:14.0%} {r['partial']:14.1f} {r['agent']:6.2f} {r['shortcut']:8.2f} {r['margin']:+7.2f}  "
                  f"{r['by']}" + ("   <- Designer's choice" if r is pick and r["margin"] > 0 else ""))
    if args.out:
        v, why = a.verdict(pol)
        Path(args.out).write_text(json.dumps({**asdict(a), "policy": pol, "verdict": v, "why": why, "variants": rows},
                                             indent=1))


if __name__ == "__main__":
    main()
