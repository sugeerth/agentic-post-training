"""CLI.

python -m evals list                       # environments and defects
python -m evals run experiments/*.yaml     # run one or more experiments
"""

from __future__ import annotations

import argparse
import sys

from evals.envs import ENVIRONMENTS
from evals.experiment import ExperimentConfig, run_experiment, write_artifacts
from evals.policies import DEFECTS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="List environments and simulated defects.")
    run = sub.add_parser("run", help="Run experiment config(s).")
    run.add_argument("configs", nargs="+")
    run.add_argument("--out", default=None, help="Override output root (default: config's).")
    args = parser.parse_args(argv)

    if args.cmd == "list":
        for name, env in sorted(ENVIRONMENTS.items()):
            print(f"env     {name:<22} {len(env().tasks()):>3} tasks")
        for name in sorted(DEFECTS):
            print(f"defect  {name}")
        return 0

    worst = 0
    for path in args.configs:
        cfg = ExperimentConfig.from_yaml(path)
        result = run_experiment(cfg)
        out = write_artifacts(result, f"{args.out}/{cfg.name}" if args.out else None)
        print(f"{result.decision.verdict:<5}  {cfg.name:<28} -> {out}/report.md")
        worst = max(worst, ("SHIP", "HOLD", "BLOCK").index(result.decision.verdict))
    return worst  # exit code: 0 SHIP, 1 HOLD, 2 BLOCK — usable as a CI gate


if __name__ == "__main__":
    sys.exit(main())
