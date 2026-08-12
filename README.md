# Agentic Post-Training Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sugeerth/agentic-post-training/blob/main/notebooks/agentic_post_training_colab.ipynb)
[![GitHub Pages](https://img.shields.io/badge/Demo-GitHub%20Pages-blue)](https://sugeerth.github.io/agentic-post-training/)

**Orchestrating LLM alignment through autonomous agent collaboration.**

A modular framework where specialized AI agents coordinate to execute post-training pipelines — from technique selection through training, optimization, and evaluation. Agents communicate via a structured message bus, making the entire process observable and debuggable.

## Architecture

```
                        ┌─────────────────┐
                        │   Coordinator   │
                        │     Agent       │
                        └────────┬────────┘
                                 │ Message Bus
        ┌──────────────┬─────────┴────┬──────────────┐
        │              │              │              │
┌───────▼────────┐ ┌───▼──────────┐ ┌─▼────────────┐ ┌▼───────────────┐
│ Training Agent │ │ Optimization │ │  Evaluation  │ │ Computer-Use   │
│                │ │    Agent     │ │    Agent     │ │    Agent       │
│ PPO,GRPO,DPO,  │ │ Quantization │ │ MMLU,MT-Bench│ │ VLM GUI        │
│ SPO,RLHF,...   │ │ Pruning,     │ │ HumanEval,...│ │ rollouts →     │
│                │ │ Distillation │ │              │ │ training data  │
└────────────────┘ └──────────────┘ └──────────────┘ └────────────────┘
```

## Supported Techniques

| Priority | Technique | Description | Key Advantage |
|----------|-----------|-------------|---------------|
| 1 | **PPO** | Proximal Policy Optimization | Stable RL with clipped objectives |
| 1 | **GRPO** | Group Relative Policy Opt. | No value model needed (DeepSeek-R1) |
| 1 | **DPO** | Direct Preference Opt. | Simple, no reward model needed |
| 1 | **SPO** | Self-Play Optimization | Iterative self-improvement |
| 1 | **RLHF** | RL from Human Feedback | Full proven pipeline (InstructGPT) |
| 2 | **KTO** | Kahneman-Tversky Opt. | Works with binary feedback |
| 2 | **ORPO** | Odds Ratio Preference Opt. | Combined SFT + alignment |
| 2 | **RLAIF** | RL from AI Feedback | No human labelers needed |
| 2 | **SPIN** | Self-Play Fine-Tuning | Only needs SFT data |
| 2 | **SimPO** | Simple Preference Opt. | Reference-free, length-normalized |
| 3 | **IPO** | Identity Preference Opt. | Regularized DPO variant |

## Demos

| Demo | Platform | Link |
|------|----------|------|
| **Interactive Browser Demo** | GitHub Pages | [Launch Demo](https://sugeerth.github.io/agentic-post-training/demo.html) |
| **Full Training on A100** | Google Colab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sugeerth/agentic-post-training/blob/main/notebooks/agentic_post_training_colab.ipynb) |
| **Multi-GPU Training (T4x2)** | Kaggle | [Open in Kaggle](https://www.kaggle.com/kernels/welcome?src=https://github.com/sugeerth/agentic-post-training/blob/main/notebooks/agentic_post_training_kaggle_t4x2.ipynb) |
| **Agent Communication** | Local terminal | `python3 examples/agent_demo.py` |
| **Computer Use → Training Data** | Local terminal | `python3 examples/computer_use_demo.py` |

## Quick Start

```bash
# Install
pip install -e .

# Run the agent demo (no GPU needed)
python3 examples/agent_demo.py

# Run full pipeline
python3 examples/run_pipeline.py --technique grpo --model gpt2 --epochs 3

# Compare techniques
python3 examples/compare_techniques.py --techniques ppo dpo grpo
```

### Python API

```python
import asyncio
from pipeline.config import PipelineConfig
from pipeline.pipeline import AgenticPipeline

config = PipelineConfig(
    technique="grpo",
    model_name="gpt2",
    epochs=3,
    quantization="gptq",
    benchmarks=["mmlu", "mt_bench", "humaneval"],
)

pipeline = AgenticPipeline(config)
results = asyncio.run(pipeline.run())
```

## Pipeline Stages

1. **Data Preparation** — Validate and preprocess training data
2. **Technique Selection** — Choose optimal technique for the task
3. **Training** — Execute post-training with the selected technique
4. **Optimization** — Quantize (GPTQ/AWQ/GGUF), prune, or distill
5. **Evaluation** — Benchmark on MMLU, MT-Bench, HumanEval, etc.

## Computer Use with Vision Models

A GUI agent produces something a text corpus cannot: attempts at a task whose
outcome can be **checked**. Every episode is a verified success or a verified
failure, which is exactly the supervision that preference and RL methods need.
That is why computer use lives in a post-training framework and not beside one.

```
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
PreferencePair              RolloutBatch                TrainingExample
(DPO/ORPO/SimPO)            (GRPO/PPO)                  (SFT / rejection)
```

Run the whole loop with no API key, no GPU, and no browser:

```bash
python3 examples/computer_use_demo.py     # or: make demo-gui
```

The offline demo runs three agents of deliberately different quality against
the same task and shows the training data that falls out of the spread between
them — which is the point, since a preference pair needs a better attempt and a
worse one.

```python
import asyncio
from computer_use import MockComputer, StateVerifier, run_episode
from computer_use import ClaudeComputerUsePolicy

env = MockComputer.settings_form()

trajectory = asyncio.run(run_episode(
    task="Set the email to ada@example.com, enable notifications, and save.",
    env=env,
    policy=ClaudeComputerUsePolicy(env.width, env.height),
    # Ground truth, not a judge: the reward is an exact state check.
    verifier=StateVerifier({"email": "ada@example.com", "notify": True, "saved": True}),
))

print(trajectory.summary())
# [success] 'Set the email to ...' — 6 steps (5 effective), reward 1.000
```

### From rollouts to training data

```python
from computer_use import run_group, to_preference_pairs, to_rollout_batch
from techniques.grpo import GRPOTechnique

# Independent attempts at one task — a GRPO group.
group = asyncio.run(run_group(task, MockComputer.settings_form,
                              lambda env: ClaudeComputerUsePolicy(env.width, env.height),
                              group_size=8, verifier=verifier))

pairs = to_preference_pairs(group)          # → DPO, ORPO, SimPO, KTO
batch = to_rollout_batch({task: group})     # → GRPO, PPO, RLHF

technique = GRPOTechnique()
technique.prepare(model, tokenizer, cfg)
metrics = technique.step(batch)             # no adapter — the shapes line up
```

### The benchmark

Eight verified tasks across three applications, three difficulty tiers, and a
working reference solution for every one of them.

```bash
agentic-gui tasks                          # what's in the suite
agentic-gui bench --policy noisy           # or: make bench-gui
agentic-gui bench --policy claude --attempts 8
agentic-gui collect --out data.jsonl       # episodes → training data
```

```
  task                 diff       pass   reward  steps  ground  pass@k
  ──────────────────────────────────────────────────────────────────────────
  settings.notify      easy    4/8        0.562    3.0    0.75  █████░░░░░ @1:0.50 @8:1.00
  settings.email       medium  6/8        0.781    5.0    0.87  ███████░░░ @1:0.75 @8:1.00
  checkout.express     hard    5/8        0.655    4.0    0.78  ██████░░░░ @1:0.62 @8:1.00
  files.rename         hard    3/8        0.473    5.0    0.78  ███░░░░░░░ @1:0.37 @8:1.00
  ──────────────────────────────────────────────────────────────────────────
  Overall 38/64 (59.4%)  95% CI [47.1%, 70.5%]  mean reward +0.640
  pass@k  @1: 0.594   @2: 0.853   @4: 0.988   @8: 1.000
  By difficulty  easy: 56%   medium: 67%   hard: 54%
```

Three deliberate choices:

- **Every task ships a reference solution**, and a test asserts each one still
  scores a perfect 1.000. A benchmark whose gold solutions have rotted reports
  failures that belong to the harness and blames the agent.
- **pass@k uses the unbiased estimator**, not "did any of my *n* samples pass".
  The distance between pass@1 and pass@8 is the most informative number here: a
  wide gap means the agent knows the task but executes it unreliably — usually
  grounding, not planning.
- **Wilson score intervals**, because a suite run is a few dozen episodes. At
  32/32 the normal approximation reports an interval above 1.0; Wilson reports
  `[89.3%, 100%]`, which is the honest answer.

`gui_bench` is registered in the evaluator registry, so the pipeline reaches it
by name — and it's the framework's first evaluator whose score is a measurement
rather than a simulation, and the first to populate `EvalResult.ci_low/ci_high`.

```python
from core.registry import get_evaluator

result = get_evaluator("gui_bench")(attempts=8).evaluate("claude-opus-5")
print(result.value, result.ci_low, result.ci_high)
```

### What's in the box

| Piece | What it does |
|-------|--------------|
| `MockComputer` | Deterministic widget GUI across three apps — typing, toggles, radio groups, scrolling, and multi-screen navigation. Renders real PNG frames (legible to an actual VLM) and exposes ground-truth state. Zero dependencies. |
| `PlaywrightComputer` | A real browser, same `ComputerEnvironment` protocol. |
| `ClaudeComputerUsePolicy` | Drives Claude through the computer-use tool. Prunes stale frames, caches the system prompt, opts into refusal fallbacks. |
| `ScriptedPolicy` | Offline baseline and gold-trajectory recorder. |
| `NoisyPolicy` | Reference solution plus grounding noise — an imperfect baseline that generates real failure trajectories with zero API calls. |
| `StateVerifier` | Exact-match verification against ground truth — no judge in the reward path. |
| `score_trajectory` | Shaped reward: success, efficiency, grounding, redundancy, invalid actions. |
| `tasks.SUITE` | Eight verified benchmark tasks, each with a reference solution. |
| `metrics.*` | Unbiased pass@k and Wilson score intervals. |
| `dataset.*` | Trajectories → `PreferencePair` / `RolloutBatch` / `TrainingExample`. |
| `ComputerUseAgent` | The whole thing as an agent on the message bus. |
| `agentic-gui` | CLI: `tasks`, `bench`, `collect`. |

### Reward design

The shaped reward keeps two properties that are easy to lose when retuning
weights, and both are enforced by tests:

- **Successes never saturate.** Weights sum to exactly the clip ceiling, so a
  flawless run scores 1.0 and every other success lands strictly below it.
  Overflowing weights would collapse all successes to the same number and
  silently delete the shaping signal.
- **Success dominates.** No amount of efficient, well-aimed failing outranks a
  clumsy success.

A step-level breakdown lands in `trajectory.metadata["reward_breakdown"]`, so a
regression in the aggregate traces to the term that moved.

### Live model

```bash
pip install "agentic-post-training[computer-use]"
export ANTHROPIC_API_KEY=...          # or run `ant auth login`
python3 examples/computer_use_demo.py --live
```

Defaults to `claude-opus-5`; the tool version, beta flag, effort, and frame
budget are all constructor arguments on `PolicyConfig`, so pointing the policy
at a different model is a config change rather than a code change.

## Agent Communication

Agents communicate via a structured message bus. Watch them coordinate in real-time:

```
[14:23:01] 🔗 Coordinator → broadcast: Starting stage: training
[14:23:01] 📊 Trainer → Coordinator: Loading model: gpt2
[14:23:02] 📊 Trainer → broadcast: Beginning GRPO training for 3 epochs
[14:23:02] 📊 Trainer → broadcast: Epoch 1/3 | Loss: 1.8432 | Reward: 0.5700
[14:23:03] 📊 Trainer → broadcast: Epoch 2/3 | Loss: 0.9821 | Reward: 0.7900
[14:23:03] ✅ Trainer → Coordinator: Training complete! Final loss: 0.5513
[14:23:04] ⚡ Optimizer → broadcast: Starting GPTQ quantization (4-bit)
[14:23:04] ✅ Optimizer → Coordinator: Quantization complete: 8.0x compression
[14:23:05] 📈 Evaluator → broadcast: Starting evaluation suite: mmlu, mt_bench
```

## Configuration Presets

```python
from pipeline.config import PipelineConfig

# Quick DPO alignment
config = PipelineConfig.preset("quick_dpo")

# Full RLHF pipeline
config = PipelineConfig.preset("full_rlhf")

# GRPO with quantization
config = PipelineConfig.preset("efficient_grpo")

# Production deployment
config = PipelineConfig.preset("production")
```

## Optimization

| Method | Compression | Quality | Use Case |
|--------|-------------|---------|----------|
| GPTQ | 8x | ~99% | Production inference |
| AWQ | 8x | ~99.5% | Best quality at 4-bit |
| GGUF | 8x | ~98% | llama.cpp deployment |
| NF4 | 8x | ~99% | QLoRA training |
| INT8 | 4x | ~99.9% | Minimal quality loss |

## Tests

```bash
python3 -m pytest tests/ -v
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
