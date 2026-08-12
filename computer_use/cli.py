"""`agentic-gui` — run the GUI benchmark and collect training data.

Three subcommands, matching the three things you actually do with a GUI agent:

    agentic-gui tasks                       # what's in the suite
    agentic-gui bench --policy noisy        # how good is an agent
    agentic-gui collect --out data.jsonl    # turn episodes into training data

Argparse and dispatch only. The logic lives in `benchmark`, `tasks`, and
`dataset` — same split as `pipeline/cli.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from computer_use.benchmark import (
    DEFAULT_K,
    format_report,
    gold_factory,
    noisy_factory,
)
from computer_use.dataset import (
    save_jsonl,
    to_preference_pairs,
    to_step_preference_pairs,
    to_training_examples,
)
from computer_use.metrics import BenchmarkReport, TaskResult, summarize, summarize_task
from computer_use.rollout import RolloutConfig, run_group
from computer_use.tasks import DIFFICULTIES, SUITE, GUITask, suite
from computer_use.types import Trajectory


def _select(args: argparse.Namespace) -> tuple[GUITask, ...]:
    try:
        return suite(
            difficulty=args.difficulty,
            tags=args.tags,
            names=args.task or None,
        )
    except (ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


def _policy_factory(args: argparse.Namespace, task: GUITask) -> Any:
    """Build the policy factory for one task.

    `gold` and `noisy` are offline baselines built from the task's own
    reference solution, so they need a per-task closure. `claude` drives the
    real model and needs only the environment geometry.
    """
    if args.policy == "gold":
        return gold_factory(task)
    if args.policy == "noisy":
        return noisy_factory(task, miss_rate=args.miss_rate, seed=args.seed)

    from computer_use.policies import ClaudeComputerUsePolicy, PolicyConfig

    config = PolicyConfig(model=args.model, effort=args.effort)
    return lambda env: ClaudeComputerUsePolicy(env.width, env.height, config)


async def _run(
    args: argparse.Namespace, tasks: tuple[GUITask, ...]
) -> tuple[BenchmarkReport, list[Trajectory]]:
    """Run the selected tasks, one policy factory per task."""
    rollout_config = RolloutConfig(max_steps=args.max_steps)
    results: list[TaskResult] = []
    trajectories: list[Trajectory] = []

    for task in tasks:
        group = await run_group(
            task.instruction,
            task.env_factory,
            _policy_factory(args, task),
            group_size=args.attempts,
            verifier=task.verifier,
            config=rollout_config,
            reward_config=task.reward_config(),
            max_concurrency=args.concurrency,
        )
        trajectories.extend(group)
        results.append(summarize_task(
            task.name, task.difficulty, group, k_values=DEFAULT_K
        ))
        if not args.json:
            last = results[-1]
            print(
                f"  {last.name:<20} {last.successes}/{last.attempts}  "
                f"reward {last.mean_reward:+.3f}",
                file=sys.stderr,
            )

    report = summarize(
        results, attempts_per_task=args.attempts, policy=args.policy,
    )
    return report, trajectories


def _cmd_tasks(args: argparse.Namespace) -> int:
    tasks = _select(args)
    if args.json:
        print(json.dumps([
            {
                "name": t.name,
                "instruction": t.instruction,
                "difficulty": t.difficulty,
                "optimal_steps": t.optimal_steps,
                "tags": list(t.tags),
                "gold_steps": len(t.gold),
            }
            for t in tasks
        ], indent=2))
        return 0

    print(f"{len(tasks)} task(s) of {len(SUITE)} in the suite\n")
    for task in tasks:
        print(f"  {task.name:<20} {task.difficulty:<7} "
              f"{task.optimal_steps:>2} steps  [{', '.join(task.tags)}]")
        print(f"    {task.instruction}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    tasks = _select(args)
    if not tasks:
        print("no tasks matched the filters", file=sys.stderr)
        return 2

    report, _ = asyncio.run(_run(args, tasks))
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report, color=not args.no_color))
    # A benchmark run reports; it does not pass or fail. Non-zero exit is
    # reserved for the run itself going wrong.
    return 0


def _cmd_collect(args: argparse.Namespace) -> int:
    tasks = _select(args)
    if not tasks:
        print("no tasks matched the filters", file=sys.stderr)
        return 2

    report, trajectories = asyncio.run(_run(args, tasks))

    out = Path(args.out)
    written = save_jsonl(trajectories, out, include_frames=args.include_frames)
    pairs = to_preference_pairs(trajectories)
    step_pairs = to_step_preference_pairs(trajectories, include_frames=False)
    sft = to_training_examples(trajectories)

    summary = {
        "trajectories": written,
        "path": str(out),
        "preference_pairs": len(pairs),
        "step_preference_pairs": len(step_pairs),
        "training_examples": len(sft),
        "success_rate": report.success_rate,
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"\nWrote {written} trajectories to {out}")
        print(f"  {len(pairs):>4} trajectory preference pairs   → DPO / ORPO / SimPO")
        print(f"  {len(step_pairs):>4} step preference pairs         → grounding")
        print(f"  {len(sft):>4} SFT examples                   → rejection sampling")
        print(f"  success rate: {report.success_rate:.1%}")
        if not pairs:
            print("\n  No preference pairs: every attempt scored the same. Try "
                  "--policy noisy, or raise --attempts, to get a spread to learn from.")
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--difficulty", choices=list(DIFFICULTIES),
                        help="only tasks at this difficulty")
    parser.add_argument("--tags", nargs="+", metavar="TAG",
                        help="only tasks carrying any of these tags")
    parser.add_argument("--task", nargs="+", metavar="NAME",
                        help="only these tasks, by name")
    parser.add_argument("--json", action="store_true", help="machine-readable output")


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy", choices=["gold", "noisy", "claude"], default="gold",
                        help="gold = reference solution, noisy = gold with grounding "
                             "noise, claude = a real vision model (default: gold)")
    parser.add_argument("--attempts", type=int, default=4,
                        help="episodes per task; pass@k needs k <= attempts (default: 4)")
    parser.add_argument("--model", default="claude-opus-5", help="model id for --policy claude")
    parser.add_argument("--effort", default="high",
                        choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--miss-rate", type=float, default=0.25,
                        help="click miss probability for --policy noisy (default: 0.25)")
    parser.add_argument("--seed", type=int, default=0, help="seed for --policy noisy")
    parser.add_argument("--max-steps", type=int, default=20, help="step budget per episode")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="parallel episodes; keep low for real models")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agentic-gui",
        description="Run the GUI benchmark and turn episodes into training data.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tasks = sub.add_parser("tasks", help="List the benchmark suite")
    _add_common(p_tasks)
    p_tasks.set_defaults(func=_cmd_tasks)

    p_bench = sub.add_parser("bench", help="Run the suite and report pass@k")
    _add_common(p_bench)
    _add_run_options(p_bench)
    p_bench.add_argument("--no-color", action="store_true")
    p_bench.set_defaults(func=_cmd_bench)

    p_collect = sub.add_parser("collect", help="Run the suite and write training data")
    _add_common(p_collect)
    _add_run_options(p_collect)
    p_collect.add_argument("--out", default="outputs/gui_trajectories.jsonl")
    p_collect.add_argument("--include-frames", action="store_true",
                           help="embed screenshots in the JSONL (large)")
    p_collect.set_defaults(func=_cmd_collect)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
