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

from selflearn.agents import (
    ActorAgent,
    AssessorAgent,
    CurriculumAgent,
    DesignerAgent,
    LearnerAgent,
    SelectorAgent,
    VerifierAgent,
)
from selflearn.bus import Bus
from selflearn.coordinator import Coordinator
from selflearn.env import Knobs
from selflearn.grpo import GRPOLearner, attach_lora, save_adapter
from selflearn.usecases import USE_CASES


@dataclass
class Config:
    usecase: str = "math"    # math | support  (see selflearn/usecases/)
    model: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    steps: int = 60
    tasks: int = 16          # tasks per step: more groups -> more groups with mixed outcomes -> more learning signal
    group: int = 8           # attempts per task (GRPO group)
    lr: float = 2e-5
    rank: int = 16
    scale: float = 2.0       # LoRA alpha/rank; mlx's default 20 broke the policy in one step in our smoke test
    temp: float = 1.0
    max_turns: int = 4
    eval_per_level: int = 0  # 0 = the use case's default
    kl: float = 0.0          # KL anchor to the base model (0 = off); stops collapse onto one answer
    select_every: int = 0    # score a validation set every N steps and keep the best checkpoint (0 = off)
    val_per_level: int = 6   # validation set size per level (separate from the held-out test)
    designer: int = 1        # 1 = the Designer audits the environment first and reshapes its tasks while training
    force: int = 0           # 1 = train even if the Designer's audit says BLOCK
    seed: int = 0
    out: str = ""


def run(cfg: Config) -> dict:
    import mlx.core as mx
    from mlx_lm import load
    uc = USE_CASES[cfg.usecase]
    out = Path(cfg.out or f"runs/{time.strftime('%Y%m%d-%H%M')}-{uc.name}-{cfg.model.split('/')[-1]}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    rng = random.Random(cfg.seed)
    mx.random.seed(cfg.seed)

    model, tok = load(cfg.model)
    n_params = attach_lora(model, cfg.rank, cfg.scale)
    bus = Bus(out / "messages.jsonl")
    bus.send("Coordinator", "all", f"use case {uc.name}, model {cfg.model.split('/')[-1]}, "
             f"LoRA {n_params / 1e6:.1f}M params, {cfg.steps} steps")

    held_out = uc.eval_set(cfg.eval_per_level or None)
    exclude = {t.question for t in held_out}
    validation = uc.validation_set(cfg.val_per_level, exclude) if cfg.select_every else []
    exclude |= {t.question for t in validation}           # never train on test or validation questions
    actor = ActorAgent(model, tok, uc, bus, cfg.max_turns)
    knobs = Knobs()                                          # the environment settings only the Designer turns
    verifier = VerifierAgent(uc, bus, knobs)
    curriculum = CurriculumAgent(uc, cfg.group, rng, exclude, bus)
    learner = LearnerAgent(GRPOLearner(model, cfg.lr, pad_id=tok.eos_token_id or 0, kl=cfg.kl), cfg.group, bus)
    assessor = AssessorAgent(actor, verifier, held_out, bus)
    selector = SelectorAgent(model, actor, verifier, validation, bus, cfg.select_every) if cfg.select_every else None
    designer = (DesignerAgent(uc, held_out, knobs, random.Random(cfg.seed + 1), exclude, bus, bool(cfg.force))
                if cfg.designer else None)

    log = open(out / "steps.jsonl", "w")

    def on_step(step: int, rec: dict) -> None:
        rec["p"] = {str(k): round(v, 3) for k, v in curriculum.cur.p.items()}
        if designer:
            rec["knobs"] = asdict(knobs)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % 10 == 0 or step == cfg.steps:
            save_adapter(model, out / "adapter", cfg.model, cfg.rank, cfg.scale, uc.name)   # survives a sleeping Mac

    res = Coordinator(assessor, curriculum, actor, verifier, learner, selector, designer).run(
        cfg.steps, cfg.tasks, cfg.group, cfg.temp, on_step)
    log.close()
    if designer:
        v, why = designer.verdict
        (out / "env_audit.json").write_text(json.dumps({**asdict(designer.audit), "verdict": v, "why": why,
                                                        "variants": designer.variants, "knobs": asdict(knobs)}, indent=1))
    if res["blocked"]:
        results = {"usecase": uc.name, "model": cfg.model, "blocked": True, "why": designer.verdict[1],
                   "before": res["before"]}
        (out / "results.json").write_text(json.dumps(results, indent=1))
        return results
    save_adapter(model, out / "adapter", cfg.model, cfg.rank, cfg.scale, uc.name)   # the weights that were assessed

    b, a = res["before_records"], res["after_records"]
    ok = lambda r: r["reward"] >= 1.0   # noqa: E731  (fully correct; partial credit is a training signal only)
    fixed = sorted([(x, y) for x, y in zip(b, a, strict=True) if not ok(x) and ok(y)], key=lambda p: -p[0]["level"])
    broken = [(x, y) for x, y in zip(b, a, strict=True) if ok(x) and not ok(y)]
    (out / "eval_before.json").write_text(json.dumps({"summary": res["before"], "records": b}, indent=1))
    (out / "eval_after.json").write_text(json.dumps({"summary": res["after"], "records": a}, indent=1))
    (out / "examples.json").write_text(json.dumps({"fixed": fixed[:10], "broken": broken[:5]}, indent=1))
    results = {"usecase": uc.name, "model": cfg.model, "steps": cfg.steps, "selected_step": res["selected_step"],
               "kl": cfg.kl, "before": res["before"], "after": res["after"],
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
