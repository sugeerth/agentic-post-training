"""Show one problem solved by the base model and by the trained agent, side by side.

  python3 -m selflearn.demo --adapter runs/<run>/adapter --question "A farm has 120 eggs. ..." [--answer 340]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from selflearn.rollout import rollout
from selflearn.tasks import Task
from selflearn.tools import TOOLS, run_call
from selflearn.verifier import reward


def main(argv=None):
    from mlx_lm import load
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--answer", type=int, default=None)
    args = ap.parse_args(argv)
    name = json.loads((Path(args.adapter) / "adapter_config.json").read_text())["model"]
    task = Task(args.question, args.answer if args.answer is not None else -1, 0)
    for label, path in (("BASE MODEL", None), ("AFTER SELF-LEARNING", args.adapter)):
        model, tok = load(name, adapter_path=path)
        t = rollout(model, tok, [task], group=1, temp=0.0, tools=TOOLS, call_tool=run_call)[0]
        verdict = "" if args.answer is None else ("  -> correct" if reward(t.text, args.answer) else "  -> wrong")
        print(f"=== {label}{verdict}\n{t.text.strip()}\n")


if __name__ == "__main__":
    main()
