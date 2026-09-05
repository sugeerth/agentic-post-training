"""`agentic-gui` — run the GUI benchmark and collect training data.

Three subcommands, matching the three things you actually do with a GUI agent:

    agentic-gui tasks                       # what's in the suite
    agentic-gui bench --policy noisy        # how good is an agent
    agentic-gui collect --out data.jsonl    # turn episodes into training data

Each accepts `--synthetic` (tasks searched out of the environments) or
`--worlds N` (apps generated first, then searched). With `--worlds N
--held-out` the apps come from a seed range no training run uses, which is the
only configuration where a score is evidence about operating a GUI rather than
about this particular GUI.

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

#: Where `--held-out` starts counting worlds. Any offset larger than the
#: number of worlds anyone trains on would do; this one is round and obvious in
#: a task name, so `world1000_*` reads as "an app the data never saw".
HELD_OUT_OFFSET = 1000


def _filter(args: argparse.Namespace, tasks: tuple[GUITask, ...]) -> tuple[GUITask, ...]:
    """Apply the shared selectors to a generated set.

    `--task` is not applied here: generated names come out of a search, so
    naming one is only meaningful for the curated suite.
    """
    if args.difficulty:
        tasks = tuple(t for t in tasks if t.difficulty == args.difficulty)
    if args.tags:
        wanted = set(args.tags)
        tasks = tuple(t for t in tasks if wanted & set(t.tags))
    return tasks


def _select(args: argparse.Namespace) -> tuple[GUITask, ...]:
    """The tasks to run.

    Three sources, in increasing order of how little of it a human wrote:
    the curated suite, tasks searched out of the hand-written environments
    (`--synthetic`), and tasks searched out of generated apps (`--worlds`).
    """
    if getattr(args, "worlds", 0):
        from computer_use.worlds import curriculum

        start = HELD_OUT_OFFSET if args.held_out else 0
        return _filter(args, tuple(curriculum(
            range(start, start + args.worlds),
            per_world=args.per_environment,
            max_depth=args.max_depth,
            min_depth=args.min_depth,
            sample_seed=args.seed,
            hard=getattr(args, "hard", False),
        )))

    if getattr(args, "synthetic", False):
        from computer_use.synthesis import synthesize_suite

        return _filter(args, tuple(synthesize_suite(
            per_environment=args.per_environment,
            max_depth=args.max_depth,
            seed=args.seed,
        )))

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

    if getattr(args, "worlds", 0):
        seeds = sorted({t.metadata["world_seed"] for t in tasks})
        origin = "held-out" if args.held_out else "training"
        print(f"{len(tasks)} task(s) synthesized in {len(seeds)} generated "
              f"{origin} app(s): seeds {seeds[0]}–{seeds[-1]}\n")
    elif getattr(args, "synthetic", False):
        print(f"{len(tasks)} task(s) synthesized by searching the environments\n")
    else:
        print(f"{len(tasks)} task(s) of {len(SUITE)} in the curated suite\n")
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


def _cmd_learn(args: argparse.Namespace) -> int:
    """Generate apps, search tasks, collect rollouts, train, score on new apps."""
    from computer_use.learn import closed_loop, top_weights

    result = closed_loop(
        train_worlds=range(args.train_worlds),
        test_worlds=range(HELD_OUT_OFFSET, HELD_OUT_OFFSET + args.test_worlds),
        per_world=args.per_environment,
        max_steps=args.max_steps,
        epochs=args.epochs,
        seed=args.seed,
        hard=args.hard,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print(
        f"\n  trained on {result.train_tasks} tasks from {args.train_worlds} generated apps"
        f"\n  {result.decisions} grounding decisions, "
        f"train accuracy {result.train_accuracy:.1%}\n"
    )
    print(f"  {'held out (apps never seen)':<32} {result.test_tasks} tasks")
    print(f"  {'  trained policy':<32} {result.trained}/{result.test_tasks}"
          f"  {result.trained_rate:.0%}")
    controls = ", ".join(str(u) for u in result.untrained)
    print(f"  {'  same features, random weights':<32} "
          f"{controls} of {result.test_tasks}  {result.untrained_rate:.0%}")
    print("\n  what it decided mattered:")
    for name, weight in top_weights(result.model, 8):
        print(f"    {name:<34} {weight:+.2f}")
    return 0


def _cmd_pretrain(args: argparse.Namespace) -> int:
    """Train a transformer on interaction tokens; score it by execution."""
    from computer_use.diagnose import (
        attention_to_target,
        baselines,
        ceiling,
        field_scores,
        format_diagnosis,
        predictions,
    )
    from computer_use.pretrain import (
        DEFAULT_CORPUS,
        build_corpus,
        evaluate,
        format_report,
        train,
    )
    from computer_use.transformer import GPT, ModelConfig, save_model
    from computer_use.worlds import split

    train_tasks, test_tasks = split(
        train=range(args.train_worlds),
        test=range(HELD_OUT_OFFSET, HELD_OUT_OFFSET + args.test_worlds),
        per_world=args.per_environment,
    )
    from dataclasses import replace

    corpus_config = replace(
        DEFAULT_CORPUS, label_first=args.label_first, marks=args.marks,
        vision=args.vision,
    )
    corpus = build_corpus(train_tasks, config=corpus_config)
    held = build_corpus(test_tasks, config=corpus_config)
    longest = max(len(e) for e in (*corpus, *held))
    from computer_use.vision import VISUAL_DIM

    config = ModelConfig(
        visual_dim=VISUAL_DIM if args.vision else 0,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        d_ff=args.d_model * 2,
        max_len=longest + 8,
    )
    print(
        f"  {len(train_tasks)} tasks / {len(corpus)} decisions from "
        f"{args.train_worlds} generated apps",
        flush=True,
    )
    print(f"  longest sequence {longest} tokens, model {config.d_model}d "
          f"x {config.n_layers}L", flush=True)

    def progress(epoch: int, report: object) -> None:
        losses = report.epoch_loss  # type: ignore[attr-defined]
        held_out = report.held_out_loss  # type: ignore[attr-defined]
        tail = f" held-out {held_out[-1]:.3f}" if held_out else ""
        print(f"    epoch {epoch + 1:2d}  train {losses[-1]:.3f}{tail}"
              f"  ({report.seconds / 60:.1f} min)", flush=True)  # type: ignore[attr-defined]
        if args.out:
            save_model(model_ref["model"], args.out)

    model_ref: dict[str, GPT] = {}
    model = GPT(config, seed=args.seed)
    model_ref["model"] = model
    model, report = train(
        corpus,
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        held_out=held,
        on_epoch=progress,
    )
    if args.out:
        save_model(model, args.out)

    results = [evaluate(model, test_tasks, config=corpus_config,
                        label="trained (apps never seen)")]
    if not args.skip_control:
        control = GPT(config, seed=args.seed + 977)
        results.append(evaluate(control, test_tasks, config=corpus_config,
                                label="untrained, same shape"))

    # The score is not interpretable on its own, so it is never printed on its
    # own: the ceiling says how much of it was available, and the floor says
    # what it had to beat to mean anything.
    limit = ceiling(held)
    floor = baselines(held)
    fields = field_scores(predictions(model, held))
    gaze = attention_to_target(model, held)

    if args.json:
        print(json.dumps({
            "train": report.to_dict(),
            "results": [r.to_dict() for r in results],
            "ceiling": {"per_kind": limit.per_kind, "reasons": limit.reasons},
            "baselines": [
                {"name": b.name, "correct": b.correct, "total": b.total}
                for b in floor
            ],
            "attention": {
                "on_target": gaze.on_target, "if_uniform": gaze.if_uniform,
                "on_screen": gaze.on_screen, "ratio": gaze.ratio,
                "examples": gaze.examples,
            },
            "fields": {
                "kind": fields.kind, "column": fields.column, "row": fields.row,
                "point": fields.point, "characters": fields.characters,
                "text_exact": fields.text_exact, "malformed": fields.malformed,
            },
            "config": {"d_model": config.d_model, "n_layers": config.n_layers,
                       "n_heads": config.n_heads, "max_len": config.max_len},
        }, indent=2))
        return 0
    print()
    print(format_report(report, results))
    print(format_diagnosis(limit, floor, fields, gaze))
    return 0


def _cmd_evolve(args: argparse.Namespace) -> int:
    """Practise on undemonstrated apps; keep only what the verifier passes."""
    from computer_use.evolve import evolve, format_report

    seed_end = args.seed_worlds
    practice_end = seed_end + args.practice_worlds
    report = evolve(
        seed_worlds=range(seed_end),
        practice_worlds=range(seed_end, practice_end),
        test_worlds=range(HELD_OUT_OFFSET, HELD_OUT_OFFSET + args.test_worlds),
        rounds=args.rounds,
        group_size=args.group_size,
        temperature=args.temperature,
        per_world=args.per_environment,
        max_steps=args.max_steps,
        use_forks=not args.no_forks,
        seed=args.seed,
        hard=args.hard,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report))
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--difficulty", choices=list(DIFFICULTIES),
                        help="only tasks at this difficulty")
    parser.add_argument("--tags", nargs="+", metavar="TAG",
                        help="only tasks carrying any of these tags")
    parser.add_argument("--task", nargs="+", metavar="NAME",
                        help="only these tasks, by name")
    parser.add_argument("--synthetic", action="store_true",
                        help="synthesize tasks by searching the environments "
                             "instead of using the curated suite")
    parser.add_argument("--worlds", type=int, default=0, metavar="N",
                        help="generate N applications and synthesize tasks in "
                             "them; implies --synthetic over generated GUIs")
    parser.add_argument("--held-out", action="store_true",
                        help="draw --worlds from a disjoint seed range, so the "
                             "apps are ones no training run has seen")
    parser.add_argument("--hard", action="store_true",
                        help="generate apps with near-duplicate captions and a "
                             "decoy primary action, so matching the goal's "
                             "words against the screen is not a whole strategy")
    parser.add_argument("--per-environment", type=int, default=6,
                        help="synthesized tasks per environment (default: 6)")
    parser.add_argument("--max-depth", type=int, default=5,
                        help="search depth for --synthetic / --worlds (default: 5)")
    parser.add_argument("--min-depth", type=int, default=2,
                        help="shortest task to keep. Raising this is the knob "
                             "that actually makes the benchmark harder: success "
                             "falls off with horizon length, not with distractors")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for --synthetic sampling and --policy noisy")
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

    p_learn = sub.add_parser(
        "learn", help="Train a pixel-only policy on generated data and score it "
                      "on generated apps it has never seen",
    )
    p_learn.add_argument("--train-worlds", type=int, default=30)
    p_learn.add_argument("--test-worlds", type=int, default=12)
    p_learn.add_argument("--per-environment", type=int, default=3)
    p_learn.add_argument("--epochs", type=int, default=30)
    p_learn.add_argument("--max-steps", type=int, default=24)
    p_learn.add_argument("--seed", type=int, default=0)
    p_learn.add_argument("--hard", action="store_true",
                         help="generate apps with near-duplicate captions and a "
                              "decoy primary action")
    p_learn.add_argument("--json", action="store_true")
    p_learn.set_defaults(func=_cmd_learn)

    p_pretrain = sub.add_parser(
        "pretrain", help="Train a transformer on tokenized interactions and "
                         "score it by executing what it generates",
    )
    p_pretrain.add_argument("--train-worlds", type=int, default=24)
    p_pretrain.add_argument("--test-worlds", type=int, default=8)
    p_pretrain.add_argument("--per-environment", type=int, default=4)
    p_pretrain.add_argument("--epochs", type=int, default=10)
    p_pretrain.add_argument("--batch-size", type=int, default=8)
    p_pretrain.add_argument("--lr", type=float, default=3e-3)
    p_pretrain.add_argument("--d-model", type=int, default=48)
    p_pretrain.add_argument("--heads", type=int, default=3)
    p_pretrain.add_argument("--layers", type=int, default=2)
    p_pretrain.add_argument("--seed", type=int, default=0)
    p_pretrain.add_argument(
        "--vision", action="store_true",
        help="replace each control's decoded label with the pixels it was "
             "drawn from, so the model reads glyph shapes rather than being "
             "handed the parser's transcription",
    )
    p_pretrain.add_argument(
        "--marks", action="store_true",
        help="emit a click as the index of the control it lands on, rather "
             "than as a coordinate pair the model has to copy",
    )
    p_pretrain.add_argument(
        "--label-first", action="store_true",
        help="encode each control as label-then-coordinate, so copying the "
             "answer runs forwards through the context instead of backwards",
    )
    p_pretrain.add_argument("--out", type=str, default="",
                            help="write the checkpoint here after every epoch")
    p_pretrain.add_argument("--skip-control", action="store_true",
                            help="skip the untrained baseline (it costs a full "
                                 "evaluation pass)")
    p_pretrain.add_argument("--json", action="store_true")
    p_pretrain.set_defaults(func=_cmd_pretrain)

    p_evolve = sub.add_parser(
        "evolve", help="Practise on applications with no demonstrations, keeping "
                       "only the attempts a verifier passes",
    )
    p_evolve.add_argument("--seed-worlds", type=int, default=3,
                          help="apps that get demonstrations (default: 3)")
    p_evolve.add_argument("--practice-worlds", type=int, default=15,
                          help="apps the agent only ever practises on")
    p_evolve.add_argument("--test-worlds", type=int, default=12)
    p_evolve.add_argument("--rounds", type=int, default=3)
    p_evolve.add_argument("--group-size", type=int, default=4,
                          help="attempts per task per round")
    p_evolve.add_argument("--temperature", type=float, default=0.8,
                          help="0 makes every attempt in a group identical")
    p_evolve.add_argument("--no-forks", action="store_true",
                          help="ablation: learn only from successes, discarding "
                               "the corrections that failures carry")
    p_evolve.add_argument("--per-environment", type=int, default=3)
    p_evolve.add_argument("--max-steps", type=int, default=24)
    p_evolve.add_argument("--seed", type=int, default=0)
    p_evolve.add_argument("--hard", action="store_true",
                          help="practise on apps with near-duplicate captions "
                               "and a decoy primary action")
    p_evolve.add_argument("--json", action="store_true")
    p_evolve.set_defaults(func=_cmd_evolve)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
