"""The computer-use agent on the message bus.

Same contract as the training, optimization, and evaluation agents: it takes a
job, narrates what it is doing to the other agents, and returns a result dict.
What it contributes to the pipeline is *data* — it drives a GUI, verifies the
outcome, and hands back trajectories already converted into the training shapes
the rest of the framework consumes.

    Coordinator ──▶ Operator ──▶ (GUI rollouts) ──▶ preference pairs / rollout batch
                                                          │
                                                          ▼
                                             Trainer (DPO / GRPO / …)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from agents.base_agent import BOLD, RESET, BaseAgent
from computer_use.dataset import (
    to_preference_pairs,
    to_rollout_batch,
    to_step_preference_pairs,
    to_training_examples,
)
from computer_use.environments import ComputerEnvironment, MockComputer
from computer_use.policies import VLMPolicy
from computer_use.rewards import RewardConfig, Verifier
from computer_use.rollout import RolloutConfig, report, run_group
from computer_use.types import Step, Trajectory

#: One entry per task: the instruction, and the check that says it worked.
TaskSpec = tuple[str, Verifier | None]


class ComputerUseAgent(BaseAgent):
    """Collects verified GUI trajectories and converts them to training data.

    Runs each task as a *group* of independent attempts rather than a single
    rollout. Groups are what make the data useful: a group with mixed outcomes
    yields preference pairs, and group-relative advantage is what GRPO needs.
    A single attempt per task gives you a benchmark number and nothing to
    train on.
    """

    def __init__(
        self,
        name: str = "Operator",
        *,
        env_factory: Callable[[], ComputerEnvironment] | None = None,
        policy_factory: Callable[[ComputerEnvironment], VLMPolicy] | None = None,
    ) -> None:
        super().__init__(name, role="operator")
        self._env_factory = env_factory or MockComputer.settings_form
        self._policy_factory = policy_factory
        self.trajectories: list[Trajectory] = []
        self.register_capability("gui_rollout", "Drive a GUI with a vision model")
        self.register_capability("verification", "Check GUI outcomes against ground truth")
        self.register_capability("data_synthesis", "Convert trajectories into training data")

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        tasks: Sequence[TaskSpec] = kwargs.get("tasks") or []
        if not tasks:
            raise ValueError("ComputerUseAgent.run needs `tasks=[(instruction, verifier), ...]`")

        policy_factory = kwargs.get("policy_factory") or self._policy_factory
        if policy_factory is None:
            raise ValueError(
                "ComputerUseAgent needs a `policy_factory`. Pass "
                "`lambda env: ClaudeComputerUsePolicy(env.width, env.height)` for a live "
                "model, or a ScriptedPolicy factory for an offline run."
            )

        env_factory = kwargs.get("env_factory") or self._env_factory
        group_size: int = kwargs.get("group_size", 4)
        rollout_config: RolloutConfig = kwargs.get("rollout_config") or RolloutConfig()
        reward_config: RewardConfig = kwargs.get("reward_config") or RewardConfig()
        max_concurrency: int = kwargs.get("max_concurrency", 4)

        await self.send_message("task_request", {
            "message": (
                f"Starting {len(tasks)} GUI task(s) x {group_size} attempts "
                f"({len(tasks) * group_size} episodes)"
            ),
        }, target="broadcast")

        collected: list[Trajectory] = []
        per_task: dict[str, dict[str, Any]] = {}

        for instruction, verifier in tasks:
            self.log(f"Task: {instruction}")
            group = await run_group(
                instruction,
                env_factory,
                policy_factory,
                group_size=group_size,
                verifier=verifier,
                config=rollout_config,
                reward_config=reward_config,
                max_concurrency=max_concurrency,
            )
            collected.extend(group)
            summary = report(group)
            per_task[instruction] = summary

            await self.send_message("status_update", {
                "message": (
                    f"{instruction[:48]}: {summary['success_rate']:.0%} success "
                    f"over {group_size} attempts, mean reward {summary['mean_reward']:.3f}"
                ),
            }, target="Coordinator")

        self.trajectories = collected
        overall = report(collected)

        # Convert once, here, so the trainer downstream never touches a
        # Trajectory — it receives the same types any other data source
        # produces.
        preference_pairs = to_preference_pairs(collected)
        step_pairs = to_step_preference_pairs(collected)
        sft_examples = to_training_examples(collected)
        rollout_batch = to_rollout_batch(
            {task: [t for t in collected if t.task == task] for task, _ in tasks}
        )

        result = {
            "report": overall,
            "per_task": per_task,
            "trajectories": collected,
            "preference_pairs": preference_pairs,
            "step_preference_pairs": step_pairs,
            "training_examples": sft_examples,
            "rollout_batch": rollout_batch,
        }

        self.memory.set("gui_report", overall)
        self._print_report(overall, per_task, result)

        await self.send_message("data_share", {
            "message": (
                f"GUI data ready: {len(preference_pairs)} trajectory pairs, "
                f"{len(step_pairs)} step pairs, {len(sft_examples)} SFT examples"
            ),
            "counts": {
                "preference_pairs": len(preference_pairs),
                "step_preference_pairs": len(step_pairs),
                "training_examples": len(sft_examples),
                "groups": len(rollout_batch.prompts),
            },
        }, target="Coordinator")

        return result

    async def step(self, **kwargs: Any) -> dict[str, Any]:
        return await self.run(**kwargs)

    async def narrate(self, task: str, step: Step) -> None:
        """A `StepHook` that streams each action onto the bus.

        Pass as `on_step=agent.narrate` to `run_episode` when you want a live
        trace. Off by default — a 20-step episode times a group of 8 is a lot
        of messages, and the bus history is meant to stay readable.
        """
        detail = step.action.describe()
        if step.error:
            detail += f"  ⚠ {step.error}"
        await self.send_message("status_update", {
            "message": f"[{task[:28]}] step {step.index}: {detail}",
        }, target="broadcast")

    def _print_report(
        self, overall: dict[str, Any], per_task: dict[str, dict[str, Any]], result: dict[str, Any]
    ) -> None:
        print(f"\n{BOLD}{'═' * 70}")
        print("  🖥  Computer-Use Rollout Report")
        print(f"{'═' * 70}{RESET}")

        for task, stats in per_task.items():
            rate = stats["success_rate"]
            bar_len = int(rate * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            label = task if len(task) <= 34 else task[:31] + "..."
            print(f"  {label:36s} [{bar}] {rate:5.0%}  r={stats['mean_reward']:+.3f}")

        print(f"\n  {BOLD}Episodes:{RESET} {overall['episodes']}   "
              f"{BOLD}Success:{RESET} {overall['success_rate']:.0%}   "
              f"{BOLD}Mean reward:{RESET} {overall['mean_reward']:+.3f}   "
              f"{BOLD}Mean steps:{RESET} {overall['mean_steps']:.1f}")
        print(f"  {BOLD}Statuses:{RESET} " + ", ".join(
            f"{k}={v}" for k, v in overall.get("statuses", {}).items()
        ))
        print(f"\n  {BOLD}Training data produced:{RESET}")
        print(f"    → {len(result['preference_pairs'])} trajectory preference pairs (DPO/ORPO/SimPO)")
        print(f"    → {len(result['step_preference_pairs'])} step preference pairs (grounding)")
        print(f"    → {len(result['training_examples'])} SFT examples (rejection sampling)")
        print(f"    → {len(result['rollout_batch'].prompts)} rollout groups (GRPO/PPO)")
        print(f"\n{BOLD}{'═' * 70}{RESET}\n")
