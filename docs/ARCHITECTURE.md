# Architecture

Detailed architecture of the Agentic Post-Training Framework.

## System Overview

The framework is organized as a **coordinator-worker multi-agent system** communicating over a structured asynchronous message bus. A single `CoordinatorAgent` plans and sequences a five-stage pipeline; three specialized workers (`TrainingAgent`, `OptimizationAgent`, `EvaluationAgent`) execute stage-scoped tasks and publish status back to the bus.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              User / CLI / Python API                         │
│        examples/run_pipeline.py  ·  AgenticPipeline(config).run()            │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         pipeline.AgenticPipeline                             │
│   • Constructs MessageBus                                                    │
│   • Instantiates agents & wires them to the bus                              │
│   • Builds coordinator_config from PipelineConfig                            │
│   • Delegates execution to Coordinator.execute(config=...)                   │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         agents.CoordinatorAgent                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  Pipeline plan (PipelineStage[])                                       │  │
│  │    1. data_prep           → trainer                                    │  │
│  │    2. technique_selection → coordinator (self-handled)                 │  │
│  │    3. training            → trainer                                    │  │
│  │    4. optimization        → optimizer                                  │  │
│  │    5. evaluation          → evaluator                                  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└───────┬─────────────────────┬──────────────────────┬─────────────────────────┘
        │                     │                      │
   (dispatch)            (dispatch)             (dispatch)
        │                     │                      │
        ▼                     ▼                      ▼
┌───────────────┐     ┌────────────────┐    ┌────────────────┐
│ TrainingAgent │     │ OptimizationA. │    │ EvaluationAgt. │
│               │     │                │    │                │
│ data_prep,    │     │ quantization,  │    │ MMLU, MT-Bench,│
│ training,     │     │ pruning,       │    │ HumanEval,     │
│ checkpointing │     │ distillation   │    │ recommendation │
└───────┬───────┘     └────────┬───────┘    └────────┬───────┘
        │                      │                     │
        ▼                      ▼                     ▼
┌─────────────────┐   ┌──────────────────┐  ┌──────────────────┐
│ techniques/     │   │ optimization/    │  │ benchmarks (stub)│
│  ppo, grpo,     │   │  gptq, awq,      │  │                  │
│  dpo, spo,      │   │  gguf, nf4, int8 │  │                  │
│  rlhf, rlaif,   │   │  pruning,        │  │                  │
│  kto, orpo,     │   │  distillation    │  │                  │
│  spin, simpo,   │   │                  │  │                  │
│  ipo            │   │                  │  │                  │
└─────────────────┘   └──────────────────┘  └──────────────────┘

        ▲                      ▲                     ▲
        │                      │                     │
        └──────────────────────┴─────────────────────┘
                               │
                    (publish/subscribe)
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        agents.communication.MessageBus                       │
│  history: list[Message]   agents: {name → agent}   subscribers: {topic → cb} │
│  publish() → direct delivery to inbox  OR  broadcast to all agents           │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### `pipeline/` — orchestration entry point
| File | Role |
|------|------|
| `config.py` | `PipelineConfig` dataclass + presets (`quick_dpo`, `full_rlhf`, `efficient_grpo`, `production`, …) and validation |
| `pipeline.py` | `AgenticPipeline`: builds bus + agents, translates `PipelineConfig` into per-stage config, calls `coordinator.execute(...)` |

### `agents/` — the multi-agent system
| File | Role |
|------|------|
| `base_agent.py` | `BaseAgent` (ABC) with `AgentStatus`, `AgentCapability`, `AgentMemory`, `inbox: asyncio.Queue`, `execute()` lifecycle |
| `communication.py` | `MessageBus`, `Message`, `MessageType` (8 types); direct + broadcast + topic subscriptions; pretty conversation log |
| `coordinator.py` | `CoordinatorAgent`: worker registry, stage plan, `run_stage()` dispatch, dashboard rendering |
| `training_agent.py` | Loads model, selects a `BaseTechnique`, runs epochs, emits loss/reward status updates |
| `optimization_agent.py` | Applies quantization / pruning / distillation via `optimization/` |
| `evaluation_agent.py` | Runs benchmark suite, computes deltas vs. baseline, prints report + recommendations |

### `techniques/` — post-training algorithms
All implement `BaseTechnique`. Priority 1: **PPO, GRPO, DPO, SPO, RLHF**. Priority 2: **KTO, ORPO, RLAIF, SPIN, SimPO**. Priority 3: **IPO**. The trainer selects one at runtime based on `PipelineConfig.technique`.

### `optimization/` — model compression
`GPTQ · AWQ · GGUF · NF4 · INT8` plus pruning and distillation.

## Message Flow (per pipeline run)

```
Coordinator ── coordination ──▶ broadcast   "Starting stage: <name>"
    │
    ├── dispatch ──▶ Worker.execute(**stage.config)
    │                   │
    │                   ├── status_update ──▶ broadcast   "Loading model: …"
    │                   ├── status_update ──▶ broadcast   "Epoch k/N | loss | reward"
    │                   └── task_result   ──▶ Coordinator "Training complete!"
    │
    └── task_result ──▶ broadcast   "Stage <name> completed in Xs"
```

`MessageType` values: `task_request`, `task_result`, `status_update`, `data_share`, `coordination`, `heartbeat`, `evaluation`, `optimization`. Every published message is appended to `MessageBus.history` and rendered with an ANSI-colored, icon-prefixed line by `Message.pretty_print()`.

## Execution Lifecycle

```
 IDLE ──▶ RUNNING ──┬──▶ COMPLETED
                    └──▶ FAILED
```

`BaseAgent.execute()` wraps `run(**kwargs)` and transitions status, logging entry/exit. `CoordinatorAgent.run_stage()` measures per-stage duration and stores the result dict on the `PipelineStage`. Shared state flows through `agent.memory` (per-agent) and `coordinator.memory.set("stage_<name>", result)` (cross-stage).

## Five-Stage Pipeline

| # | Stage | Owner | Output |
|---|-------|-------|--------|
| 1 | `data_prep` | Trainer | validated dataset handles |
| 2 | `technique_selection` | Coordinator | `selected_technique` in memory |
| 3 | `training` | Trainer | final loss, reward, checkpoint path |
| 4 | `optimization` | Optimizer | compressed model, compression ratio, perplexity delta |
| 5 | `evaluation` | Evaluator | per-benchmark scores + overall improvement % |

## Extension Points

- **Add a technique**: subclass `BaseTechnique` in `techniques/` and register it in `TrainingAgent`'s dispatch table.
- **Add an optimization method**: extend `OptimizationAgent` and the `optimization/` folder; accept a new `quant_type` in `PipelineConfig`.
- **Add a benchmark**: extend `EvaluationAgent._run_benchmark(...)` and list the new name in `PipelineConfig.benchmarks`.
- **Add an agent role**: subclass `BaseAgent`, register it via `coordinator.register_worker(agent)`, and add a `PipelineStage` with the new `agent_role`.
- **Add a message type**: extend `MessageType` in `agents/communication.py` and pick an icon in `MSG_ICONS`.
