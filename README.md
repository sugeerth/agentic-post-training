# Agentic Post-Training Framework

[![CI](https://github.com/sugeerth/agentic-post-training/actions/workflows/ci.yml/badge.svg)](https://github.com/sugeerth/agentic-post-training/actions/workflows/ci.yml)
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

## Supported Techniques

| Priority | Technique | Description | Key Advantage | Status |
|----------|-----------|-------------|---------------|--------|
| 1 | **PPO** | Proximal Policy Optimization | Stable RL with clipped objectives | ✅ Real loss |
| 1 | **GRPO** | Group Relative Policy Opt. | No value model needed (DeepSeek-R1) | ✅ Real loss |
| 1 | **DPO** | Direct Preference Opt. | Simple, no reward model needed | ✅ Real loss |
| 1 | **SPO** | Self-Play Optimization | Iterative self-improvement | ✅ Real loss |
| 1 | **RLHF** | RL from Human Feedback | Full proven pipeline (InstructGPT) | ✅ Real loss |
| 2 | **KTO** | Kahneman-Tversky Opt. | Works with binary feedback | ✅ Real loss |
| 2 | **ORPO** | Odds Ratio Preference Opt. | Combined SFT + alignment | ✅ Real loss |
| 2 | **SimPO** | Simple Preference Opt. | Reference-free, length-normalized | ✅ Real loss |
| 3 | **IPO** | Identity Preference Opt. | Regularized DPO variant | ✅ Real loss |
| 2 | **RLAIF** | RL from AI Feedback | No human labelers needed | ✅ Real loss + `Judge` protocol |
| 2 | **SPIN** | Self-Play Fine-Tuning | Only needs SFT data | ✅ Real loss + pair builder |

All 11 techniques ship real loss implementations. Every technique also ships a
deterministic **simulation mode** so demos, tests, and notebooks run without a
GPU (or even without torch installed) — the simulation path is taken automatically
when no tensors are supplied.

RLAIF's AI labeler is pluggable: implement the one-method `Judge` protocol
(`judge(prompt, response_a, response_b) -> 0 | 1`) or wrap any scoring function
with `ScoreJudge`, then `label_pairs(judge, prompt, responses)` produces
AI-labeled preference pairs. SPIN's pair construction (`build_spin_pairs`,
chosen = human response, rejected = the model's own generation) and self-play
iteration bookkeeping live in `techniques/spin.py`; the generation/snapshot
cadence belongs to the trainer loop.

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

## CLI

```bash
# Dry-run the ORPO quickstart (prints the resolved plan, spends nothing)
python3 -m pipeline.cli run examples/orpo_quickstart/run.yaml --backend local --dry-run

# List what's registered
python3 -m pipeline.cli list techniques
python3 -m pipeline.cli list backends

# Inspect a technique (JSON: paper, priority, pros/cons, experimental flag)
python3 -m pipeline.cli inspect grpo
```

## Tests & Quality Gates

CI runs the full suite on Python 3.10–3.12, plus `ruff` and `mypy --strict`
(on `core/` and `techniques/_base/`) on every push and pull request.

```bash
make test    # pytest — runs without a GPU or torch
make lint    # ruff check .
make mypy    # strict typing on core + techniques/_base
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
