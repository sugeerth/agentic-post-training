"""Show one case handled by the base model and by the trained agent, side by side.

  python3 -m selflearn.demo --adapter runs/<run>/adapter --level 6 --seed 99         # a fresh case from the use case
  python3 -m selflearn.demo --adapter runs/<math run>/adapter --question "A farm has 120 eggs. ..." --answer 340
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from selflearn.rollout import rollout
from selflearn.tools import run_call
from selflearn.usecase import Task
from selflearn.usecases import USE_CASES


def main(argv=None):
    from mlx_lm import load
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--question", default=None, help="your own question (math use case)")
    ap.add_argument("--answer", default=None, help="its true answer, to mark the outputs right or wrong")
    ap.add_argument("--level", type=int, default=None, help="generate a fresh case at this level instead")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args(argv)
    cfg = json.loads((Path(args.adapter) / "adapter_config.json").read_text())
    uc = USE_CASES[cfg.get("usecase", "math")]
    if args.question:
        answer = args.answer if args.answer is None or not args.answer.lstrip("-").isdigit() else int(args.answer)
        task = Task(args.question, answer if answer is not None else "", 0)
    else:
        task = uc.make_task(args.level or uc.levels[-1], random.Random(args.seed))
        answer = task.answer
    print(f"[{uc.name}] {uc.user_prefix}{task.question}" + (f"\n(correct: {answer})" if answer is not None else "") + "\n")
    for label, path in (("BASE MODEL", None), ("AFTER SELF-LEARNING", args.adapter)):
        model, tok = load(cfg["model"], adapter_path=path)
        t = rollout(model, tok, [task], group=1, temp=0.0, system_prompt=uc.system_prompt, tools=uc.tools,
                    call_tool=run_call, user_prefix=uc.user_prefix, max_new=uc.max_new)[0]
        verdict = "" if answer is None else ("  -> correct" if uc.reward(t.text, answer) else "  -> wrong")
        print(f"=== {label}{verdict}\n{t.text.strip()}\n")


if __name__ == "__main__":
    main()
