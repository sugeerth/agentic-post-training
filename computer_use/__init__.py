"""Computer use with vision models, wired into post-training.

A GUI agent produces something a text corpus cannot: attempts at a task whose
outcome can be *checked*. That makes computer use a data source, not just a
capability — every episode is a verified success or a verified failure, which
is exactly the supervision preference and RL methods need.

The pipeline this package implements:

    task ──▶ screenshot ──▶ VLM ──▶ action ──▶ environment ──┐
              ▲                                              │
              └──────────────── new screenshot ◀─────────────┘
                                     │
                              verifier (ground truth)
                                     │
                            shaped reward per trajectory
                                     │
        ┌────────────────────────────┼─────────────────────────────┐
        ▼                            ▼                             ▼
   PreferencePair             RolloutBatch                 TrainingExample
   (DPO/ORPO/SimPO)           (GRPO/PPO)                   (SFT / rejection)

Quick start — no API key, no GPU, no browser:

    import asyncio
    from computer_use import MockComputer, ScriptedPolicy, StateVerifier, run_episode

    env = MockComputer.settings_form()
    policy = ScriptedPolicy([...])
    trajectory = asyncio.run(run_episode(
        "Set the email to ada@example.com and save",
        env, policy, verifier=StateVerifier({"email": "ADA@EXAMPLE.COM", "saved": True}),
    ))
    print(trajectory.summary())

Swap `ScriptedPolicy` for `ClaudeComputerUsePolicy` and the same code drives a
real vision model. Swap `MockComputer` for `PlaywrightComputer` and it drives a
real browser.
"""

from computer_use.actions import (
    DEFAULT_TOOL_BETA,
    DEFAULT_TOOL_VERSION,
    RESOLUTIONS,
    ActionError,
    computer_tool,
    system_prompt,
    validate,
)
from computer_use.agent import ComputerUseAgent
from computer_use.benchmark import (
    GUIBenchEvaluator,
    format_report,
    gold_factory,
    noisy_factory,
    run_benchmark,
)
from computer_use.dataset import (
    load_jsonl,
    render_transcript,
    save_jsonl,
    to_preference_pairs,
    to_rollout_batch,
    to_step_preference_pairs,
    to_training_examples,
)
from computer_use.environments import (
    ComputerEnvironment,
    MockComputer,
    PlaywrightComputer,
    Widget,
)
from computer_use.metrics import (
    BenchmarkReport,
    TaskResult,
    pass_at_k,
    summarize,
    summarize_task,
    wilson_interval,
)
from computer_use.policies import (
    ClaudeComputerUsePolicy,
    Decision,
    NoisyPolicy,
    PolicyConfig,
    RefusalError,
    ScriptedPolicy,
    VLMPolicy,
)
from computer_use.rewards import (
    AllOf,
    RewardConfig,
    StateVerifier,
    Verdict,
    Verifier,
    assign_step_credit,
    score_trajectory,
)
from computer_use.rollout import (
    RolloutConfig,
    report,
    run_episode,
    run_group,
    run_suite,
    success_rate,
)
from computer_use.synthesis import (
    AbstractAction,
    Discovery,
    SynthesisConfig,
    explore,
    synthesize,
    synthesize_suite,
)
from computer_use.tasks import SUITE, GUITask, get_task, suite
from computer_use.types import (
    Action,
    ActionKind,
    Screenshot,
    Step,
    Trajectory,
    TrajectoryStatus,
)

__all__ = [
    "DEFAULT_TOOL_BETA",
    "DEFAULT_TOOL_VERSION",
    "RESOLUTIONS",
    "SUITE",
    "AbstractAction",
    "Action",
    "ActionError",
    "ActionKind",
    "AllOf",
    "BenchmarkReport",
    "ClaudeComputerUsePolicy",
    "ComputerEnvironment",
    "ComputerUseAgent",
    "Decision",
    "Discovery",
    "GUIBenchEvaluator",
    "GUITask",
    "MockComputer",
    "NoisyPolicy",
    "PlaywrightComputer",
    "PolicyConfig",
    "RefusalError",
    "RewardConfig",
    "RolloutConfig",
    "Screenshot",
    "ScriptedPolicy",
    "StateVerifier",
    "Step",
    "SynthesisConfig",
    "TaskResult",
    "Trajectory",
    "TrajectoryStatus",
    "VLMPolicy",
    "Verdict",
    "Verifier",
    "Widget",
    "assign_step_credit",
    "computer_tool",
    "explore",
    "format_report",
    "get_task",
    "gold_factory",
    "load_jsonl",
    "noisy_factory",
    "pass_at_k",
    "render_transcript",
    "report",
    "run_benchmark",
    "run_episode",
    "run_group",
    "run_suite",
    "save_jsonl",
    "score_trajectory",
    "success_rate",
    "suite",
    "summarize",
    "summarize_task",
    "synthesize",
    "synthesize_suite",
    "system_prompt",
    "to_preference_pairs",
    "to_rollout_batch",
    "to_step_preference_pairs",
    "to_training_examples",
    "validate",
    "wilson_interval",
]
