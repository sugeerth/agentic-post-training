"""Running the GUI suite, and exposing it as a registered evaluator.

`run_benchmark` is the direct entry point. `GUIBenchEvaluator` is the same
thing wearing the `core.Evaluator` protocol, registered as `gui_bench` so the
pipeline and the evaluation agent can reach it by name alongside MMLU and
HumanEval.

It is the framework's first evaluator whose score is a *measurement* rather
than a simulation — the number comes from an agent actually completing tasks,
verified against ground truth. It is also the first to populate
`EvalResult.ci_low` / `ci_high`, which matters here more than usual: a suite
run is a few dozen episodes, so the interval is often more informative than
the point estimate.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import count
from typing import Any, cast

from computer_use.metrics import BenchmarkReport, TaskResult, summarize, summarize_task
from computer_use.policies import NoisyPolicy, ScriptedPolicy, VLMPolicy
from computer_use.rollout import PolicyFactory, RolloutConfig, run_group
from computer_use.tasks import SUITE, GUITask
from computer_use.types import Trajectory
from core.registry import register_evaluator
from core.types import EvalResult

#: k values reported by default. 1 is "does it work", 8 is "does it work if you
#: let it retry", and the distance between them is the interesting part.
DEFAULT_K = (1, 2, 4, 8)


def gold_factory(task: GUITask) -> PolicyFactory:
    """Replay a task's reference solution. Deterministic, always succeeds."""
    return lambda _env: ScriptedPolicy(list(task.gold))


def noisy_factory(
    task: GUITask, *, miss_rate: float = 0.25, seed: int = 0
) -> PolicyFactory:
    """The reference solution with grounding noise, varied per attempt.

    The per-attempt seed is the load-bearing part. A factory that handed every
    attempt the same seed would produce a group of identical trajectories —
    same actions, same reward, zero variance — which makes pass@k meaningless
    and yields no preference pairs at all, since a pair needs a better attempt
    and a worse one. Seeds advance from `seed`, so a run stays reproducible.
    """
    counter = count()

    def _build(_env: Any) -> VLMPolicy:
        return NoisyPolicy(
            list(task.gold), miss_rate=miss_rate, seed=seed + next(counter)
        )

    return _build


async def run_benchmark(
    tasks: Sequence[GUITask] = SUITE,
    policy_factory: PolicyFactory | None = None,
    *,
    attempts: int = 4,
    rollout_config: RolloutConfig | None = None,
    reward_overrides: dict[str, Any] | None = None,
    max_concurrency: int = 4,
    k_values: Sequence[int] = DEFAULT_K,
) -> tuple[BenchmarkReport, list[Trajectory]]:
    """Run each task `attempts` times and summarize.

    Returns the report *and* the raw trajectories, because on this benchmark
    the episodes are worth as much as the score — they are the training data
    (see `computer_use.dataset`). Throwing them away to return a single number
    would be the wasteful choice.

    With no `policy_factory`, each task's own gold solution runs. That makes a
    zero-argument call a self-check of the suite rather than an error.
    """
    if policy_factory is None:
        report, trajectories = await _run_gold(tasks, attempts, rollout_config,
                                               reward_overrides, max_concurrency, k_values)
        return report, trajectories

    results: list[TaskResult] = []
    collected: list[Trajectory] = []

    for task in tasks:
        group = await run_group(
            task.instruction,
            task.env_factory,
            policy_factory,
            group_size=attempts,
            verifier=task.verifier,
            config=rollout_config,
            reward_config=task.reward_config(**(reward_overrides or {})),
            max_concurrency=max_concurrency,
        )
        collected.extend(group)
        results.append(summarize_task(task.name, task.difficulty, group, k_values=k_values))

    return summarize(results, attempts_per_task=attempts), collected


async def _run_gold(
    tasks: Sequence[GUITask],
    attempts: int,
    rollout_config: RolloutConfig | None,
    reward_overrides: dict[str, Any] | None,
    max_concurrency: int,
    k_values: Sequence[int],
) -> tuple[BenchmarkReport, list[Trajectory]]:
    """Every task solved by its own reference script — the suite's self-check."""
    results: list[TaskResult] = []
    collected: list[Trajectory] = []

    for task in tasks:
        group = await run_group(
            task.instruction,
            task.env_factory,
            gold_factory(task),
            group_size=attempts,
            verifier=task.verifier,
            config=rollout_config,
            reward_config=task.reward_config(**(reward_overrides or {})),
            max_concurrency=max_concurrency,
        )
        collected.extend(group)
        results.append(summarize_task(task.name, task.difficulty, group, k_values=k_values))

    return summarize(results, attempts_per_task=attempts, policy="gold"), collected


# --------------------------------------------------------------------------- #
# Registered evaluator
# --------------------------------------------------------------------------- #


@register_evaluator("gui_bench")
@dataclass
class GUIBenchEvaluator:
    """The GUI suite as a `core.Evaluator`.

    The protocol's `evaluate(model, tokenizer)` shape is built for a local
    checkpoint, and a computer-use agent is driven by a policy instead. The
    adaptation is deliberate and narrow: pass a `policy_factory` at
    construction for full control, or let `evaluate` treat a string `model` as
    a model id and build a Claude policy for it. `tokenizer` is unused — a GUI
    agent consumes pixels.
    """

    name: str = "gui_bench"
    tasks: Sequence[GUITask] = SUITE
    attempts: int = 4
    policy_factory: PolicyFactory | None = None
    max_concurrency: int = 4
    rollout_config: RolloutConfig | None = None
    #: Populated by the last `evaluate` call, so a caller that wants the
    #: episodes (to train on) can reach them without re-running the suite.
    last_report: BenchmarkReport | None = field(default=None, init=False)
    last_trajectories: list[Trajectory] = field(default_factory=list, init=False)

    async def evaluate_async(
        self, model: Any = None, tokenizer: Any = None
    ) -> EvalResult:
        """The real implementation. Await this from async code.

        Every agent in this framework is async, so this — not `evaluate` — is
        the method the pipeline and the evaluation agent should call.
        """
        factory = self.policy_factory or _factory_for(model)
        report, trajectories = await run_benchmark(
            self.tasks,
            factory,
            attempts=self.attempts,
            rollout_config=self.rollout_config,
            max_concurrency=self.max_concurrency,
        )
        self.last_report = report
        self.last_trajectories = trajectories

        low, high = report.ci
        return EvalResult(
            metric_name="gui_bench",
            value=round(report.success_rate * 100, 2),
            n=report.episodes,
            ci_low=round(low * 100, 2),
            ci_high=round(high * 100, 2),
            metadata=report.to_dict(),
        )

    def evaluate(self, model: Any = None, tokenizer: Any = None) -> EvalResult:
        """Synchronous entry point, for `core.Evaluator` conformance.

        Refuses to run inside an existing event loop rather than doing
        something worse. `asyncio.run` would raise an opaque RuntimeError, and
        offloading to a worker thread would silently block the caller's loop
        for the length of a benchmark run. Awaiting `evaluate_async` is the
        correct fix and the error says so.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.evaluate_async(model, tokenizer))
        raise RuntimeError(
            "gui_bench.evaluate() was called from inside a running event loop. "
            "Await evaluate_async(...) instead — every agent in this framework "
            "is async, so that is almost certainly the method you want."
        )


def _factory_for(model: Any) -> PolicyFactory | None:
    """Build a policy factory from whatever `evaluate` was handed.

    A string is a model id. `None` falls through to the gold reference, which
    makes `get_evaluator("gui_bench")().evaluate()` a working self-check rather
    than a crash — useful when wiring the pipeline before a model exists.
    """
    if model is None:
        return None
    if isinstance(model, str):
        from computer_use.policies import ClaudeComputerUsePolicy, PolicyConfig

        def _build(env: Any) -> VLMPolicy:
            return ClaudeComputerUsePolicy(
                env.width, env.height, PolicyConfig(model=model)
            )

        return _build
    if callable(model):
        return cast("PolicyFactory", model)
    raise TypeError(
        "gui_bench needs a model id string, a policy factory, or None (gold "
        f"reference); got {type(model).__name__}"
    )


# --------------------------------------------------------------------------- #
# Text rendering
# --------------------------------------------------------------------------- #


def format_report(report: BenchmarkReport, *, color: bool = True) -> str:
    """A terminal table. Pure formatting — no I/O, so it is testable."""
    bold = "\033[1m" if color else ""
    dim = "\033[2m" if color else ""
    reset = "\033[0m" if color else ""

    lines = [
        f"{bold}{'═' * 78}",
        "  🖥  GUI Benchmark",
        f"{'═' * 78}{reset}",
        f"  {'task':<20} {'diff':<7} {'pass':>7} {'reward':>8} {'steps':>6} "
        f"{'ground':>7}  {'pass@k':<22}",
        f"  {dim}{'─' * 74}{reset}",
    ]

    for task in report.tasks:
        bar_len = int(task.success_rate * 10)
        bar = "█" * bar_len + "░" * (10 - bar_len)
        pass_at = " ".join(f"@{k}:{v:.2f}" for k, v in sorted(task.pass_at.items()))
        lines.append(
            f"  {task.name:<20} {task.difficulty:<7} "
            f"{task.successes}/{task.attempts:<5} {task.mean_reward:>8.3f} "
            f"{task.mean_steps:>6.1f} {task.mean_grounding:>7.2f}  {bar} {dim}{pass_at}{reset}"
        )

    low, high = report.ci
    lines += [
        f"  {dim}{'─' * 74}{reset}",
        f"  {bold}Overall{reset} {report.successes}/{report.episodes} "
        f"({report.success_rate:.1%})  "
        f"{dim}95% CI [{low:.1%}, {high:.1%}]{reset}  "
        f"mean reward {report.mean_reward:+.3f}",
    ]
    if report.pass_at:
        lines.append(
            f"  {bold}pass@k{reset}  "
            + "   ".join(f"@{k}: {v:.3f}" for k, v in sorted(report.pass_at.items()))
        )

    tiers = report.by_difficulty()
    if tiers:
        ordered = [t for t in ("easy", "medium", "hard") if t in tiers]
        lines.append(
            f"  {bold}By difficulty{reset}  "
            + "   ".join(f"{t}: {tiers[t]['success_rate']:.0%}" for t in ordered)
        )
    lines.append(f"{bold}{'═' * 78}{reset}")
    return "\n".join(lines)
