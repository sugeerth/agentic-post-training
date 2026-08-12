"""The rollout loop: screenshot → VLM → action → screenshot, until done.

Everything here is deliberately small. The loop is the least interesting part
of a computer-use agent and the easiest place to hide bugs, so it does exactly
four things: step the policy, execute in the environment, record what happened,
and stop for one of a bounded set of reasons.

Three entry points, in increasing width:

    run_episode  — one task, one attempt, one trajectory
    run_group    — one task, N independent attempts (a GRPO group)
    run_suite    — many tasks, bounded concurrency
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from computer_use.actions import ActionError
from computer_use.environments import ComputerEnvironment
from computer_use.policies import RefusalError, VLMPolicy
from computer_use.rewards import RewardConfig, Verdict, Verifier, score_trajectory
from computer_use.types import (
    POINTING_ACTIONS,
    Action,
    Step,
    Trajectory,
    TrajectoryStatus,
)

#: Called after each step with (task, step). Use it to stream progress to a UI
#: or a message bus without threading a logger through the loop.
StepHook = Callable[[str, Step], Any]

#: Builds a policy for one attempt. It receives the environment that attempt
#: will drive, because a policy needs the display geometry to declare the
#: computer tool — and because one policy per attempt is what keeps the
#: attempts in a group independent.
PolicyFactory = Callable[[ComputerEnvironment], VLMPolicy]


@dataclass
class RolloutConfig:
    """How an episode runs.

    `store_frames` is on because the observation is half of a training example
    — an image/action pair is what a VLM policy learns from. Turn it off when
    you only need the reward signal and are collecting thousands of episodes,
    since every frame is a PNG held in memory until the trajectory is written
    out.
    """

    max_steps: int = 20
    store_frames: bool = True
    step_timeout: float = 120.0
    #: Stop early once the verifier is satisfied, instead of waiting for the
    #: model to say it is done. Saves a turn per episode during data
    #: collection; leave off when you want to measure whether the agent knows
    #: it finished.
    stop_on_verified: bool = False


async def run_episode(
    task: str,
    env: ComputerEnvironment,
    policy: VLMPolicy,
    *,
    verifier: Verifier | None = None,
    config: RolloutConfig | None = None,
    reward_config: RewardConfig | None = None,
    on_step: StepHook | None = None,
) -> Trajectory:
    """Run one task to completion and return a scored trajectory.

    The episode ends when the policy stops calling the tool (it believes it is
    done), the step budget runs out, the verifier passes with
    `stop_on_verified`, or something raises. Every one of those is a
    `TrajectoryStatus`, so a caller never has to distinguish "returned normally"
    from "worked".
    """
    cfg = config or RolloutConfig()
    steps: list[Step] = []
    status = TrajectoryStatus.MAX_STEPS
    final_response = ""

    try:
        observation = await env.reset()
        decision = await asyncio.wait_for(
            policy.begin(task, observation), timeout=cfg.step_timeout
        )

        while decision.action is not None:
            if len(steps) >= cfg.max_steps:
                status = TrajectoryStatus.MAX_STEPS
                break

            action = decision.action
            # Hit-test before executing: grounding is a question about the
            # frame the agent was looking at. A click that navigates to another
            # screen would otherwise be scored against the screen it produced,
            # marking every correct navigation as a miss.
            grounding = _grounding_metadata(env, action)

            error: str | None = None
            try:
                result = await env.execute(action)
            except ActionError as exc:
                # A malformed or impossible action is signal, not a crash: it
                # gets recorded, scored, and reported back so the model can
                # correct itself on the next turn.
                error = str(exc)
                result = await env.screenshot()

            step = Step(
                index=len(steps),
                action=action,
                observation=observation if cfg.store_frames else None,
                result=result if cfg.store_frames else None,
                rationale=decision.rationale,
                error=error,
                metadata={
                    "tool_use_id": decision.tool_use_id,
                    "usage": dict(decision.usage),
                    **grounding,
                },
            )
            steps.append(step)
            if on_step is not None:
                await _maybe_await(on_step(task, step))

            observation = result

            if cfg.stop_on_verified and verifier is not None:
                probe = _build(task, steps, TrajectoryStatus.SUCCESS, "")
                if verifier(probe, env.state()).success:
                    status = TrajectoryStatus.SUCCESS
                    final_response = "(stopped early: goal state reached)"
                    break

            decision = await asyncio.wait_for(
                policy.observe(result, error=error), timeout=cfg.step_timeout
            )
        else:
            # Loop exited because the policy stopped calling the tool.
            status = TrajectoryStatus.SUCCESS
            final_response = decision.rationale

    except RefusalError as exc:
        status = TrajectoryStatus.REFUSED
        final_response = str(exc)
    except (TimeoutError, asyncio.TimeoutError):
        status = TrajectoryStatus.ERROR
        final_response = f"policy did not respond within {cfg.step_timeout}s"
    except Exception as exc:  # environment or transport failure
        status = TrajectoryStatus.ERROR
        final_response = f"{type(exc).__name__}: {exc}"

    trajectory = _build(task, steps, status, final_response)

    # The model claiming success is a hypothesis; the verifier decides.
    if verifier is not None:
        verdict = verifier(trajectory, env.state())
    else:
        verdict = Verdict(status is TrajectoryStatus.SUCCESS, "no verifier configured")

    return score_trajectory(trajectory, verdict, reward_config)


async def run_group(
    task: str,
    env_factory: Callable[[], ComputerEnvironment],
    policy_factory: PolicyFactory,
    *,
    group_size: int = 8,
    verifier: Verifier | None = None,
    config: RolloutConfig | None = None,
    reward_config: RewardConfig | None = None,
    max_concurrency: int = 4,
) -> list[Trajectory]:
    """Run the same task `group_size` times independently.

    This is the sampling step for group-relative methods: GRPO's advantage is
    each attempt's reward measured against the group mean, which only means
    anything if the attempts are genuinely independent. Hence the factories —
    every attempt gets a fresh environment and a fresh conversation, so one
    attempt cannot see another's mistakes.

    `max_concurrency` bounds parallel episodes. Raise it for the mock, keep it
    low for real environments and for API rate limits.
    """
    gate = asyncio.Semaphore(max_concurrency)

    async def _one() -> Trajectory:
        async with gate:
            env = env_factory()
            try:
                return await run_episode(
                    task, env, policy_factory(env),
                    verifier=verifier, config=config, reward_config=reward_config,
                )
            finally:
                with contextlib.suppress(Exception):
                    await env.close()

    return list(await asyncio.gather(*(_one() for _ in range(group_size))))


async def run_suite(
    tasks: Sequence[tuple[str, Verifier | None]],
    env_factory: Callable[[], ComputerEnvironment],
    policy_factory: PolicyFactory,
    *,
    config: RolloutConfig | None = None,
    reward_config: RewardConfig | None = None,
    max_concurrency: int = 4,
) -> list[Trajectory]:
    """Run a benchmark suite: `(task, verifier)` pairs, one attempt each."""
    gate = asyncio.Semaphore(max_concurrency)

    async def _one(task: str, verifier: Verifier | None) -> Trajectory:
        async with gate:
            env = env_factory()
            try:
                return await run_episode(
                    task, env, policy_factory(env),
                    verifier=verifier, config=config, reward_config=reward_config,
                )
            finally:
                with contextlib.suppress(Exception):
                    await env.close()

    return list(await asyncio.gather(*(_one(t, v) for t, v in tasks)))


def success_rate(trajectories: Sequence[Trajectory]) -> float:
    if not trajectories:
        return 0.0
    return sum(1 for t in trajectories if t.succeeded) / len(trajectories)


def report(trajectories: Sequence[Trajectory]) -> dict[str, Any]:
    """Aggregate stats for a batch of rollouts."""
    if not trajectories:
        return {"episodes": 0}
    rewards = [t.reward for t in trajectories]
    steps = [t.num_steps for t in trajectories]
    return {
        "episodes": len(trajectories),
        "success_rate": success_rate(trajectories),
        "mean_reward": sum(rewards) / len(rewards),
        "max_reward": max(rewards),
        "min_reward": min(rewards),
        "mean_steps": sum(steps) / len(steps),
        "statuses": {
            status.value: sum(1 for t in trajectories if t.status is status)
            for status in TrajectoryStatus
            if any(t.status is status for t in trajectories)
        },
    }


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _build(
    task: str, steps: Sequence[Step], status: TrajectoryStatus, final_response: str
) -> Trajectory:
    return Trajectory(
        task=task,
        steps=tuple(steps),
        status=status,
        final_response=final_response,
    )


def _grounding_metadata(env: ComputerEnvironment, action: Action) -> dict[str, Any]:
    """Record whether a click landed on something, when the env can tell us.

    Grounding — turning "the save button" into the right pixel — is where GUI
    agents most often fail, and it is invisible in a success/failure signal
    alone. Environments that can answer expose `_hit_test`; the rest opt out.

    Only pointing actions count. A scroll carries a coordinate too, but
    scrolling over empty background is normal and scoring it as a missed click
    would penalize correct behavior.
    """
    hit_test = getattr(env, "_hit_test", None)
    if hit_test is None or action.coordinate is None or action.kind not in POINTING_ACTIONS:
        return {}
    try:
        return {"hit": hit_test(*action.coordinate) is not None}
    except Exception:
        return {}


async def _maybe_await(value: Any) -> Any:
    if isinstance(value, Awaitable):
        return await value
    return value


__all__ = [
    "RolloutConfig",
    "report",
    "run_episode",
    "run_group",
    "run_suite",
    "success_rate",
]
