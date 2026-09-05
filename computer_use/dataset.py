"""Turning GUI trajectories into post-training data.

This is the module that earns computer use a place in a post-training
framework. A rollout is not the product — the product is the preference pair,
the rollout batch, or the SFT example that comes out of it, in exactly the
shapes `core.types` already defines. Once a trajectory is converted here, every
technique in `techniques/` consumes it without knowing a GUI was involved:

    trajectories  ──▶ to_rollout_batch      ──▶ GRPOTechnique.step()   (RL)
                  ──▶ to_preference_pairs   ──▶ DPO / ORPO / SimPO     (preference)
                  ──▶ to_step_preference_pairs ──▶ DPO on single decisions
                  ──▶ to_training_examples  ──▶ SFT / rejection sampling

The conversions are deliberately lossy in one direction only: the text carries
the trajectory, and the frames ride along in `metadata` so a VLM trainer can
recover the images while a text-only trainer can ignore them.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from computer_use.types import (
    Action,
    Screenshot,
    Step,
    Trajectory,
    TrajectoryStatus,
)
from core.types import PreferencePair, RolloutBatch, TrainingExample

# --------------------------------------------------------------------------- #
# Text rendering — the shared serialization every conversion builds on
# --------------------------------------------------------------------------- #


def render_action(action: Action) -> str:
    """One action as a single line of JSON — the format a policy emits."""
    return json.dumps(action.to_tool_input(), sort_keys=True)


def render_transcript(trajectory: Trajectory, *, include_rationale: bool = True) -> str:
    """Render a trajectory as the assistant-side text of a training example.

    Rationale is included by default because the reasoning is part of what you
    want the policy to learn — an action list without the "why" trains a model
    to guess coordinates, not to read a screen.
    """
    lines: list[str] = []
    for step in trajectory.steps:
        if include_rationale and step.rationale:
            lines.append(f"# {step.rationale.strip()}")
        lines.append(render_action(step.action))
        if step.error:
            lines.append(f"# error: {step.error}")
    if trajectory.final_response:
        lines.append(f"# done: {trajectory.final_response.strip()}")
    return "\n".join(lines)


def render_prompt(task: str, *, prefix: Sequence[Step] = ()) -> str:
    """Render the user-side text: the task, plus any actions already taken."""
    lines = [f"Task: {task}"]
    if prefix:
        lines.append("Actions so far:")
        lines.extend(render_action(step.action) for step in prefix)
    lines.append("Next action:")
    return "\n".join(lines)


def _frames(steps: Sequence[Step]) -> list[dict[str, Any]]:
    """Base64 frames for the steps that have one, for VLM-aware trainers."""
    out: list[dict[str, Any]] = []
    for step in steps:
        if step.observation is None:
            continue
        out.append({
            "step": step.index,
            "media_type": step.observation.media_type,
            "width": step.observation.width,
            "height": step.observation.height,
            "data": step.observation.to_base64(),
        })
    return out


# --------------------------------------------------------------------------- #
# Preference data
# --------------------------------------------------------------------------- #


def to_preference_pairs(
    trajectories: Iterable[Trajectory],
    *,
    min_margin: float = 0.05,
    max_pairs_per_task: int = 4,
    include_frames: bool = False,
) -> list[PreferencePair]:
    """Pair better trajectories against worse ones on the same task.

    `min_margin` is the guard that makes this data worth training on. Two
    trajectories that both succeeded and differ by 0.02 of shaped reward are
    noise, and a preference method will happily learn that noise. Requiring a
    real gap keeps only the pairs where one attempt is genuinely better.

    Pairs are formed best-against-worst outward (best vs worst, 2nd vs 2nd
    worst, …) rather than over every combination, so a single lucky rollout
    can't dominate the dataset.
    """
    pairs: list[PreferencePair] = []

    for task, group in _by_task(trajectories).items():
        ranked = sorted(group, key=lambda t: t.reward, reverse=True)
        prompt = render_prompt(task)

        for i in range(min(max_pairs_per_task, len(ranked) // 2)):
            chosen, rejected = ranked[i], ranked[-(i + 1)]
            if chosen.reward - rejected.reward < min_margin:
                break  # ranked order means every later pair is closer still

            metadata: dict[str, Any] = {
                "source": "computer_use",
                "chosen_reward": chosen.reward,
                "rejected_reward": rejected.reward,
                "margin": chosen.reward - rejected.reward,
                "chosen_status": str(chosen.status),
                "rejected_status": str(rejected.status),
                "chosen_steps": chosen.num_steps,
                "rejected_steps": rejected.num_steps,
            }
            if include_frames:
                metadata["chosen_frames"] = _frames(chosen.steps)
                metadata["rejected_frames"] = _frames(rejected.steps)

            pairs.append(PreferencePair(
                prompt=prompt,
                chosen=render_transcript(chosen),
                rejected=render_transcript(rejected),
                metadata=metadata,
            ))

    return pairs


def to_step_preference_pairs(
    trajectories: Iterable[Trajectory],
    *,
    min_margin: float = 0.05,
    include_frames: bool = True,
    require_outcome_difference: bool = True,
) -> list[PreferencePair]:
    """Pair individual *decisions* taken from the same state.

    Sharper signal than trajectory-level pairs: when a successful attempt and a
    failed one share a prefix and then part ways, the difference is
    attributable to that one action rather than smeared across twenty. This is
    where a GUI agent's grounding improves — the pair is "from this screen,
    click here, not there".

    Two guards keep that claim honest, and both matter more than they look:

    `require_outcome_difference` (default on) keeps only pairs where the chosen
    run succeeded and the rejected one did not. Between two *successful* runs
    the reward gap is about efficiency, and attributing it to a single action
    would be wrong — the rejected action there is not a mistake.

    Divergence is judged by **effect, not by text**. Two clicks 20px apart on
    the same button are the same decision, and the first action that merely
    *looks* different is usually not the one that lost the episode. Blaming it
    would emit a pair whose rejected side is a perfectly good action — training
    data that actively degrades grounding rather than improving it. See
    `_blame_step`.

    Frames are included by default: a single-decision example is almost useless
    to a VLM without the screenshot it was conditioned on.
    """
    pairs: list[PreferencePair] = []

    for task, group in _by_task(trajectories).items():
        ranked = sorted(group, key=lambda t: t.reward, reverse=True)
        if len(ranked) < 2:
            continue
        best = ranked[0]

        for other in ranked[1:]:
            if best.reward - other.reward < min_margin:
                continue
            if require_outcome_difference and not (best.succeeded and not other.succeeded):
                continue
            divergence = _blame_step(best.steps, other.steps)
            if divergence is None:
                continue

            prefix = best.steps[:divergence]
            good, bad = best.steps[divergence], other.steps[divergence]
            metadata: dict[str, Any] = {
                "source": "computer_use_step",
                "task": task,
                "step_index": divergence,
                "margin": best.reward - other.reward,
                "chosen_step_reward": good.reward,
                "rejected_step_reward": bad.reward,
            }
            if include_frames and good.observation is not None:
                metadata["observation"] = _frames([good])[0]

            pairs.append(PreferencePair(
                prompt=render_prompt(task, prefix=prefix),
                chosen=render_action(good.action),
                rejected=render_action(bad.action),
                metadata=metadata,
            ))

    return pairs


# --------------------------------------------------------------------------- #
# RL data
# --------------------------------------------------------------------------- #


def to_rollout_batch(
    groups: Mapping[str, Sequence[Trajectory]] | Sequence[Sequence[Trajectory]],
    *,
    normalize_rewards: bool = False,
) -> RolloutBatch:
    """Pack task groups into the `RolloutBatch` GRPO and PPO consume.

    One group per prompt, `group_size` attempts inside it — which is exactly
    the shape group-relative advantage estimation expects, so the output goes
    straight into `GRPOTechnique.step()` with no adapter.

    Leave `normalize_rewards` off unless you have a reason: GRPO already
    centers rewards within the group, and normalizing twice shrinks the signal.
    """
    if isinstance(groups, Mapping):
        items = list(groups.items())
    else:
        items = [(group[0].task if group else "", list(group)) for group in groups]

    prompts: list[str] = []
    responses: list[list[str]] = []
    rewards: list[list[float]] = []
    stats: list[dict[str, Any]] = []

    for task, group in items:
        if not group:
            continue
        prompts.append(render_prompt(task))
        responses.append([render_transcript(t) for t in group])
        group_rewards = [t.reward for t in group]
        if normalize_rewards:
            group_rewards = _standardize(group_rewards)
        rewards.append(group_rewards)
        stats.append({
            "task": task,
            "group_size": len(group),
            "success_rate": sum(1 for t in group if t.succeeded) / len(group),
            "mean_steps": sum(t.num_steps for t in group) / len(group),
        })

    return RolloutBatch(
        prompts=prompts,
        responses=responses,
        rewards=rewards,
        metadata={"source": "computer_use", "groups": stats},
    )


def to_training_examples(
    trajectories: Iterable[Trajectory],
    *,
    min_reward: float = 0.5,
    best_per_task: bool = True,
) -> list[TrainingExample]:
    """Keep the good trajectories as SFT data — rejection sampling, in effect.

    The cheapest way to use a rollout budget: sample N attempts, keep the ones
    that verified, fine-tune on those. It is often most of the gain a
    preference method would give you, for a fraction of the machinery.
    """
    examples: list[TrainingExample] = []
    for task, group in _by_task(trajectories).items():
        eligible = [t for t in group if t.succeeded and t.reward >= min_reward]
        if not eligible:
            continue
        selected = [max(eligible, key=lambda t: t.reward)] if best_per_task else eligible
        for trajectory in selected:
            examples.append(TrainingExample(
                prompt=render_prompt(task),
                response=render_transcript(trajectory),
                metadata={
                    "source": "computer_use",
                    "reward": trajectory.reward,
                    "steps": trajectory.num_steps,
                },
            ))
    return examples


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def to_dict(trajectory: Trajectory, *, include_frames: bool = False) -> dict[str, Any]:
    """JSON-serializable form. Frames are opt-in — they dominate the file size."""
    return {
        "task": trajectory.task,
        "status": str(trajectory.status),
        "reward": trajectory.reward,
        "final_response": trajectory.final_response,
        "metadata": dict(trajectory.metadata),
        "steps": [
            {
                "index": step.index,
                "action": step.action.to_tool_input(),
                "rationale": step.rationale,
                "error": step.error,
                "reward": step.reward,
                "metadata": dict(step.metadata),
                **(
                    {"observation": _frames([step])[0]}
                    if include_frames and step.observation is not None
                    else {}
                ),
            }
            for step in trajectory.steps
        ],
    }


def from_dict(payload: Mapping[str, Any]) -> Trajectory:
    """Inverse of `to_dict`. Frames are restored when present."""
    steps: list[Step] = []
    for raw in payload.get("steps", []):
        frame = raw.get("observation")
        steps.append(Step(
            index=raw["index"],
            action=Action.from_tool_input(raw["action"]),
            observation=(
                Screenshot.from_base64(
                    frame["data"], frame["width"], frame["height"],
                    frame.get("media_type", "image/png"),
                )
                if frame else None
            ),
            rationale=raw.get("rationale", ""),
            error=raw.get("error"),
            reward=raw.get("reward", 0.0),
            metadata=raw.get("metadata", {}),
        ))
    return Trajectory(
        task=payload["task"],
        steps=tuple(steps),
        status=TrajectoryStatus(payload["status"]),
        reward=payload.get("reward", 0.0),
        final_response=payload.get("final_response", ""),
        metadata=payload.get("metadata", {}),
    )


def save_jsonl(
    trajectories: Iterable[Trajectory],
    path: str | Path,
    *,
    include_frames: bool = False,
) -> int:
    """Write trajectories as JSONL. Returns the count written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(json.dumps(to_dict(trajectory, include_frames=include_frames)) + "\n")
            count += 1
    return count


def load_jsonl(path: str | Path) -> Iterator[Trajectory]:
    """Stream trajectories back from JSONL."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield from_dict(json.loads(line))


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _by_task(trajectories: Iterable[Trajectory]) -> dict[str, list[Trajectory]]:
    grouped: dict[str, list[Trajectory]] = defaultdict(list)
    for trajectory in trajectories:
        grouped[trajectory.task].append(trajectory)
    return dict(grouped)


def _same_effect(a: Step, b: Step) -> bool:
    """Whether two differently-written actions do the same thing.

    Uses the widget each click landed on, recorded by the rollout. Without that
    the only available test is textual equality, which calls two clicks on
    opposite corners of the same button "different decisions".

    Falls back to exact equality when no target was recorded (scrolls, typing,
    environments that can't hit-test) — conservative in the right direction:
    an unrecognized equivalence costs a training pair, an unrecognized
    difference costs a wrong one.
    """
    if a.action.kind is not b.action.kind or a.action.text != b.action.text:
        return False
    if a.failed != b.failed:
        return False
    target_a, target_b = a.metadata.get("target"), b.metadata.get("target")
    if target_a is None and target_b is None:
        return a.action.to_tool_input() == b.action.to_tool_input()
    return target_a == target_b


def _blame_step(good: Sequence[Step], bad: Sequence[Step]) -> int | None:
    """The step where two runs actually parted ways.

    Scans forward past divergences that had the same effect — those leave both
    runs in the same state, so the comparison stays valid — and stops at the
    first one that did something different. That step is the last point at
    which both agents faced the same screen, which is what makes the pair a
    statement about one decision.

    Returns `None` when the runs never diverge consequentially, in which case
    the outcome gap belongs to something other than a single choice and no
    honest step pair can be made.
    """
    for i in range(min(len(good), len(bad))):
        if good[i].action.to_tool_input() == bad[i].action.to_tool_input():
            continue
        if _same_effect(good[i], bad[i]):
            continue
        return i
    return None


def _standardize(values: Sequence[float]) -> list[float]:
    n = len(values)
    if n < 2:
        return list(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = var**0.5 + 1e-8
    return [(v - mean) / std for v in values]
