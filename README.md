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
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼───────┐ ┌───▼──────────┐ ┌─▼──────────────┐
     │  Training Agent │ │ Optimization │ │  Evaluation    │
     │                 │ │    Agent     │ │    Agent       │
     │ PPO,GRPO,DPO,  │ │ Quantization │ │ MMLU,MT-Bench  │
     │ SPO,RLHF,...   │ │ Pruning,     │ │ HumanEval,...  │
     │                 │ │ Distillation │ │                │
     └─────────────────┘ └──────────────┘ └────────────────┘
```

## Repository Layout

```
core/           contracts: typed data, Protocols, registry, config base (imports nothing else)
techniques/     post-training math: PPO, GRPO, DPO, ORPO, KTO, … (+ _base/ shared losses)
optimization/   quantization, pruning, distillation
agents/         coordinator + training/optimization/evaluation agents on a message bus
pipeline/       end-to-end pipeline, presets, CLI
backends/       where jobs run (local today)
evals/          behavior evals: envs, graders, metrics, gate, experiments   ← new
experiments/    experiment specs (YAML), one hypothesis per file
examples/       runnable demos      notebooks/  Colab + Kaggle
tests/          pytest suite        docs/       Pages site + design/ (audit, refactor plan)
```

## Behavior Evals & Experiments (`evals/`)

Training is half the loop. The other half is **finding what the model gets wrong,
proving a fix helped, and deciding whether it ships**. `evals/` does that, with no
GPU and no dependencies beyond `pydantic` + `pyyaml`.

```
 hypothesis ─► experiment.yaml ─► run baseline & candidate on every env
                                          │
             ┌────────────────────────────┴───────────────────────────┐
             ▼                                                        ▼
   graders tag each failure                          paired bootstrap Δ per env
   (missing_arg, hallucination, …)                               │
             │                                                   ▼
             ▼                                         gate: SHIP / HOLD / BLOCK
   failures ─► preference pairs ─► DPO / ORPO / KTO in techniques/
```

### Environments — one file per behavior

| Env | Behavior | Grader checks | Failure tags |
|---|---|---|---|
| `function_calling` | tool choice + argument filling | schema, tool, values | `no_call` `hallucinated_tool` `missing_arg` `bad_arg_type` `wrong_tool` `wrong_arg_value` |
| `multistep_tools` | tool use over long horizons (depth 1–8) | final answer within step budget | `answered_early` `exceeded_budget` `wrong_answer` + schema tags |
| `coding` | write a function | hidden unit tests in an isolated subprocess | `syntax_error` `test_failure` `runtime_error` `timeout` |
| `instruction_following` | verifiable constraints (IFEval-style) | pure-function checks | `violated:<constraint>` |
| `factuality` | answer or abstain, with stated confidence | exact match + abstention; ECE / Brier | `hallucination` `over_abstain` `wrong_answer` |

Every environment implements the same three methods (`tasks`, `respond`, `grade`) and runs
through one shared episode loop. Adding a behavior = one new file + one line in
`evals/envs/__init__.py`.

### Run the experiments

```bash
make experiments              # = python -m evals run experiments/*.yaml
python -m evals list          # environments + simulated defects
```

Each run writes `runs/<name>/`: `report.md`, `manifest.json` (config hash, git SHA),
per-arm `*.grades.jsonl` / `*.trajectories.jsonl`, and `preference_pairs.jsonl`.
The exit code is the gate verdict (0 SHIP · 1 HOLD · 2 BLOCK), so it drops straight into CI.

| Experiment | Target | Result | Why |
|---|---|---|---|
| `01_abstention_training` | factuality | **SHIP** | +22pt (CI [+8, +36]); hallucinations 8 → 0; ECE 0.20 → 0.06; nothing else moved |
| `02_tool_schema_grounding` | function_calling, multistep_tools | **BLOCK** | function calling +33pt, but a chatty sign-off cost −19pt on instruction following **and** broke strict-format answers in multistep (−20pt) |
| `03_long_horizon_persistence` | multistep_tools | **HOLD** | +1.9pt, CI touches 0: real but too small to show at this sample size, so add tasks/seeds before deciding |

> **Note:** the shipped configs use `SimulatedPolicy`, a test double that replays reference
> solutions and injects known defects at set rates. Use it to check the harness (do the graders
> catch each defect? does the gate catch regressions?) and to show the method. Its numbers
> **say nothing about any real model.**

### Evaluate a real model

Any `Observation -> Action` function is a policy:

```python
# my_model.py
from evals import Action, Observation, ToolCall

def act(obs: Observation) -> Action:
    reply = call_your_model(obs.prompt, obs.tools, obs.history)   # HF, vLLM, API, harness…
    if reply.tool_call:
        return Action(tool_call=ToolCall(reply.name, reply.args))
    return Action(text=reply.text, confidence=reply.confidence)
```

```yaml
# experiments/my_run.yaml
name: my_run
hypothesis: New checkpoint improves tool use without regressing instruction following.
targets: [function_calling, multistep_tools]
baseline:  {kind: callable, target: "my_model:act_v1"}
candidate: {kind: callable, target: "my_model:act_v2"}
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
