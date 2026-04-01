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
