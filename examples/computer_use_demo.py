#!/usr/bin/env python3
"""End-to-end computer-use demo: GUI rollouts → verified rewards → training data.

Runs three agents of deliberately different quality against the same GUI task,
verifies each against ground truth, and shows the training data that falls out
of the spread between them. That spread is the point — a preference pair needs
a better attempt and a worse one, and a group-relative advantage needs a group.

    python3 examples/computer_use_demo.py            # offline, no dependencies
    python3 examples/computer_use_demo.py --live     # drive a real vision model
    python3 examples/computer_use_demo.py --save     # write trajectories + frames

The offline path needs no API key, no GPU, and no browser: `MockComputer`
renders real PNG frames and exposes ground-truth state, so the reward is an
exact check rather than a guess.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BOLD, DIM, RESET
from computer_use import (
    Action,
    ActionKind,
    ComputerUseAgent,
    MockComputer,
    RewardConfig,
    RolloutConfig,
    ScriptedPolicy,
    StateVerifier,
    render_transcript,
    report,
    run_episode,
    save_jsonl,
    to_preference_pairs,
    to_rollout_batch,
    to_step_preference_pairs,
    to_training_examples,
)
from techniques.grpo import GRPOTechnique

TASK = "Set the email to ada@example.com, turn on failure notifications, and save."

#: Ground truth. Exact-match on environment state — no judge, no ambiguity.
VERIFIER = StateVerifier({
    "email": "ada@example.com",
    "notify": True,
    "saved": True,
})

# Screen geometry of `MockComputer.settings_form()`. The SAVE button sits below
# the fold, so reaching it requires a scroll — which is what separates the
# agents below.
EMAIL_FIELD = (290, 218)
NOTIFY_BOX = (76, 382)
SAVE_AFTER_SCROLL = (305, 624)
DEAD_SPACE = (900, 500)


def _click(x: int, y: int) -> Action:
    return Action(ActionKind.LEFT_CLICK, coordinate=(x, y))


def careful_agent() -> ScriptedPolicy:
    """Reads the screen, acts once per intent, scrolls before reaching."""
    return ScriptedPolicy(
        [
            Action(ActionKind.SCREENSHOT),
            _click(*EMAIL_FIELD),
            Action(ActionKind.TYPE, text="ada@example.com"),
            _click(*NOTIFY_BOX),
            Action(ActionKind.SCROLL, coordinate=(640, 400),
                   scroll_direction="down", scroll_amount=3),
            _click(*SAVE_AFTER_SCROLL),
        ],
        rationales=[
            "Looking at the current state of the settings dialog.",
            "The email field is empty; focusing it before typing.",
            "Entering the requested address.",
            "Enabling failure notifications via the checkbox.",
            "SAVE is below the fold — scrolling down to reach it.",
            "Clicking SAVE to commit the changes.",
        ],
        final_response="Email set, notifications enabled, settings saved.",
    )


def sloppy_agent() -> ScriptedPolicy:
    """Gets there, but wastes actions and misses a click on the way.

    Same outcome as the careful agent, worse path — exactly the pair that
    teaches efficiency and grounding rather than just task completion.
    """
    return ScriptedPolicy(
        [
            Action(ActionKind.SCREENSHOT),
            Action(ActionKind.SCREENSHOT),          # redundant: nothing changed
            _click(*DEAD_SPACE),                    # missed the field entirely
            _click(*EMAIL_FIELD),
            Action(ActionKind.TYPE, text="ada@example.com"),
            _click(*NOTIFY_BOX),
            Action(ActionKind.SCROLL, coordinate=(640, 400),
                   scroll_direction="down", scroll_amount=1),
            Action(ActionKind.SCROLL, coordinate=(640, 400),
                   scroll_direction="down", scroll_amount=3),
            _click(*SAVE_AFTER_SCROLL),
        ],
        final_response="I think that's saved now.",
    )


def failing_agent() -> ScriptedPolicy:
    """Types into an unfocused field and never scrolls to SAVE."""
    return ScriptedPolicy(
        [
            Action(ActionKind.SCREENSHOT),
            Action(ActionKind.TYPE, text="ada@example.com"),  # nothing is focused
            _click(*NOTIFY_BOX),
            _click(305, 804),      # SAVE's page coordinate, off-screen → error
        ],
        final_response="Saved the settings.",  # confidently wrong
    )


async def collect_offline() -> list:
    """Run the three agents against fresh environments and score every episode."""
    reward_config = RewardConfig(optimal_steps=6)
    rollout_config = RolloutConfig(max_steps=12, store_frames=True)

    trajectories = []
    for label, factory in (
        ("careful", careful_agent),
        ("sloppy", sloppy_agent),
        ("failing", failing_agent),
    ):
        trajectory = await run_episode(
            TASK,
            MockComputer.settings_form(),
            factory(),
            verifier=VERIFIER,
            config=rollout_config,
            reward_config=reward_config,
        )
        print(f"  {BOLD}{label:<8}{RESET} {trajectory.summary()}")
        breakdown = trajectory.metadata["reward_breakdown"]
        print(f"           {DIM}" + "  ".join(
            f"{k}={v:+.3f}" for k, v in breakdown.items() if v
        ) + RESET)
        if not trajectory.succeeded:
            print(f"           {DIM}why: {trajectory.metadata['verdict']['reason']}{RESET}")
        trajectories.append(trajectory)

    return trajectories


async def collect_live(group_size: int) -> list:
    """Drive an actual vision model through the same environment and task."""
    from computer_use import ClaudeComputerUsePolicy, PolicyConfig

    agent = ComputerUseAgent()
    result = await agent.execute(
        tasks=[(TASK, VERIFIER)],
        env_factory=MockComputer.settings_form,
        policy_factory=lambda env: ClaudeComputerUsePolicy(
            env.width, env.height,
            PolicyConfig(effort="high", keep_frames=3),
        ),
        group_size=group_size,
        rollout_config=RolloutConfig(max_steps=15),
        reward_config=RewardConfig(optimal_steps=6),
        max_concurrency=2,
    )
    return result["trajectories"]


def show_training_data(trajectories: list) -> None:
    """The payoff: the same rollouts, in every shape the framework trains on."""
    pairs = to_preference_pairs(trajectories)
    step_pairs = to_step_preference_pairs(trajectories, include_frames=False)
    sft = to_training_examples(trajectories)
    batch = to_rollout_batch({TASK: trajectories})

    print(f"\n{BOLD}{'─' * 70}{RESET}")
    print(f"{BOLD}Training data extracted from {len(trajectories)} episodes{RESET}")
    print(f"{BOLD}{'─' * 70}{RESET}")
    print(f"  {len(pairs):>3} trajectory preference pairs   → DPO, ORPO, SimPO, KTO")
    print(f"  {len(step_pairs):>3} step preference pairs         → grounding (which pixel, from this screen)")
    print(f"  {len(sft):>3} SFT examples                   → rejection-sampling fine-tune")
    print(f"  {len(batch.prompts):>3} rollout group(s)               → GRPO, PPO, RLHF")

    for pair in pairs:
        print(f"\n{BOLD}Trajectory pair{RESET} {DIM}(margin {pair.metadata['margin']:+.3f}: "
              f"{pair.metadata['chosen_reward']:.3f} over "
              f"{pair.metadata['rejected_reward']:.3f}){RESET}")
        for label, text in (("chosen", pair.chosen), ("rejected", pair.rejected)):
            actions = [ln for ln in text.splitlines() if not ln.startswith("#")]
            print(f"  {DIM}{label:<9}{RESET}" + f"\n  {' ' * 9}".join(actions))

    if step_pairs:
        step_pair = step_pairs[0]
        print(f"\n{BOLD}Sample step pair{RESET} "
              f"{DIM}(diverged at step {step_pair.metadata['step_index']}){RESET}")
        print(f"  {DIM}chosen:{RESET}   {step_pair.chosen}")
        print(f"  {DIM}rejected:{RESET} {step_pair.rejected}")

    # Close the loop: the batch goes straight into a real technique. No torch
    # here, so GRPO takes its simulation path — the point is that the shapes
    # line up with no adapter in between.
    technique = GRPOTechnique()
    technique.prepare(model=None, tokenizer=None, cfg=None)
    metrics = technique.step(batch)
    print(f"\n{BOLD}Handed to GRPOTechnique.step(){RESET} "
          f"{DIM}(simulation path — no GPU in this demo){RESET}")
    print(f"  loss={metrics.loss:.4f}  " + "  ".join(
        f"{k}={v:.3f}" for k, v in metrics.extras.items()
    ))


def save_artifacts(trajectories: list, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "trajectories.jsonl"
    count = save_jsonl(trajectories, path, include_frames=False)

    frames_written = 0
    for trajectory in trajectories[:1]:
        for step in trajectory.steps:
            if step.observation is None:
                continue
            frame_path = outdir / f"frame_{step.index:02d}.png"
            frame_path.write_bytes(step.observation.data)
            frames_written += 1

    transcript = outdir / "transcript.txt"
    transcript.write_text(
        "\n\n".join(f"### {t.summary()}\n{render_transcript(t)}" for t in trajectories),
        encoding="utf-8",
    )
    print(f"\n{BOLD}Wrote{RESET} {count} trajectories, {frames_written} frames, "
          f"and a transcript to {outdir}/")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true",
                        help="drive a real vision model instead of scripted agents")
    parser.add_argument("--group-size", type=int, default=4,
                        help="attempts per task in --live mode (default: 4)")
    parser.add_argument("--save", action="store_true",
                        help="write trajectories, frames, and a transcript to ./outputs")
    args = parser.parse_args()

    print(f"\n{BOLD}{'═' * 70}")
    print("  🖥  Computer Use → Post-Training Data")
    print(f"{'═' * 70}{RESET}")
    print(f"  {DIM}task:{RESET} {TASK}")
    print(f"  {DIM}env:{RESET}  MockComputer.settings_form() — 1280x720, SAVE below the fold\n")

    if args.live:
        print(f"{BOLD}Running live rollouts{RESET} "
              f"{DIM}(group of {args.group_size}){RESET}\n")
        trajectories = await collect_live(args.group_size)
    else:
        print(f"{BOLD}Running three scripted agents of differing quality{RESET}\n")
        trajectories = await collect_offline()
        print(f"\n  {DIM}aggregate:{RESET} {report(trajectories)}")

    show_training_data(trajectories)

    if args.save:
        save_artifacts(trajectories, Path("outputs") / "computer_use")

    print(f"\n{DIM}Next: swap MockComputer for PlaywrightComputer to run this "
          f"against a real browser.{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
