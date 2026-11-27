"""Entry point: build the model and the five agents, hand them to the Coordinator, save what happened.

  python3 -m selflearn.train --steps 60      (about 15 minutes for Qwen2.5-0.5B on an M3 Pro)
  python3 -m selflearn.train --smoke         (about 1 minute: checks the plumbing)

Writes to runs/<name>/: messages.jsonl (every agent message), steps.jsonl, eval_before/after.json, examples.json,
results.json, adapter/ (LoRA weights, mlx_lm format). Then: python3 -m selflearn.report runs/<name>
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from selflearn import tasks as T
from selflearn.agents import ActorAgent, AssessorAgent, CurriculumAgent, LearnerAgent, VerifierAgent
from selflearn.bus import Bus
from selflearn.coordinator import Coordinator
from selflearn.grpo import GRPOLearner, attach_lora, save_adapter
from selflearn.tools import TOOLS, run_call
from selflearn.verifier import reward


@dataclass
class Config:
    model: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    steps: int = 60
    tasks: int = 16          # tasks per step: more groups -> more groups with mixed outcomes -> more learning signal
    group: int = 8           # attempts per task (GRPO group)
    lr: float = 2e-5
    rank: int = 16
    scale: float = 2.0       # LoRA alpha/rank; mlx's default 20 broke the policy in one step in our smoke test
    temp: float = 1.0
    max_turns: int = 4
    max_new: int = 96        # new tokens per turn
    eval_per_level: int = 20
    seed: int = 0
    out: str = ""


def run(cfg: Config) -> dict:
    import mlx.core as mx
    from mlx_lm import load
    out = Path(cfg.out or f"runs/{time.strftime('%Y%m%d-%H%M')}-{cfg.model.split('/')[-1]}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    rng = random.Random(cfg.seed)
    mx.random.seed(cfg.seed)

    model, tok = load(cfg.model)
    n_params = attach_lora(model, cfg.rank, cfg.scale)
    bus = Bus(out / "messages.jsonl")
    bus.send("Coordinator", "all", f"model {cfg.model.split('/')[-1]}, LoRA {n_params / 1e6:.1f}M params, {cfg.steps} steps")

    held_out = T.make_eval_set(cfg.eval_per_level)
    actor = ActorAgent(model, tok, TOOLS, run_call, bus, cfg.max_turns, cfg.max_new)
    verifier = VerifierAgent(reward, bus)
    curriculum = CurriculumAgent(cfg.group, rng, {t.question for t in held_out}, bus)
    learner = LearnerAgent(GRPOLearner(model, cfg.lr, pad_id=tok.eos_token_id or 0), cfg.group, bus)
    assessor = AssessorAgent(actor, verifier, held_out, bus)

    log = open(out / "steps.jsonl", "w")

    def on_step(step: int, rec: dict) -> None:
        rec["p"] = {str(k): round(v, 3) for k, v in curriculum.cur.p.items()}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % 10 == 0 or step == cfg.steps:
            save_adapter(model, out / "adapter", cfg.model, cfg.rank, cfg.scale)   # survives a sleeping Mac

    res = Coordinator(assessor, curriculum, actor, verifier, learner).run(cfg.steps, cfg.tasks, cfg.group, cfg.temp, on_step)
    log.close()

    b, a = res["before_records"], res["after_records"]
    fixed = sorted([(x, y) for x, y in zip(b, a, strict=True) if x["reward"] < y["reward"]], key=lambda p: -p[0]["level"])
    broken = [(x, y) for x, y in zip(b, a, strict=True) if x["reward"] > y["reward"]]
    (out / "eval_before.json").write_text(json.dumps({"summary": res["before"], "records": b}, indent=1))
    (out / "eval_after.json").write_text(json.dumps({"summary": res["after"], "records": a}, indent=1))
    (out / "examples.json").write_text(json.dumps({"fixed": fixed[:10], "broken": broken[:5]}, indent=1))
    results = {"model": cfg.model, "steps": cfg.steps, "before": res["before"], "after": res["after"],
               "fixed": len(fixed), "broken": len(broken), "held_out": len(held_out)}
    (out / "results.json").write_text(json.dumps(results, indent=1))
    bus.send("Coordinator", "all", f"done: {res['before']['accuracy']:.1%} -> {res['after']['accuracy']:.1%} held-out, "
             f"fixed {len(fixed)}, broke {len(broken)} | {out}")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in asdict(Config()).items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    ap.add_argument("--smoke", action="store_true")
    a = vars(ap.parse_args(argv))
    smoke = a.pop("smoke")
    cfg = Config(**a)
    if smoke:
        cfg.steps, cfg.tasks, cfg.group, cfg.eval_per_level = 2, 2, 4, 2
    run(cfg)


if __name__ == "__main__":
    main()
