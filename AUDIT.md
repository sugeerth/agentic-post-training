# AUDIT — framework_agentic_training

**Date:** 2026-05-11
**Phase:** 0 (read-only)
**Audited by:** 5 parallel sub-agents (agents/, techniques/, optimization+pipeline/, examples+docs/, packaging+tests)
**Tests:** 23/23 passing (`pytest` 9.0.2, Python 3.12)
**Git:** branch `main`, clean tree, 3 commits total

---

## Repo Map

```
framework_agentic_training/
├── __init__.py                          (1 line: __version__ = "0.1.0")
├── LICENSE                              Apache 2.0
├── README.md
├── pyproject.toml                       distribution name: agentic-post-training
├── requirements.txt                     duplicates pyproject.dependencies
├── agents/         7 files,  922 LOC    BaseAgent, MessageBus, Coordinator, 3 worker agents
├── techniques/    13 files, 1064 LOC    11 post-training methods + BaseTechnique + registry
├── optimization/   4 files,  163 LOC    Quantizer, Pruner, Distiller   ← ORPHANED (zero callers)
├── pipeline/       3 files,  233 LOC    AgenticPipeline + PipelineConfig
├── examples/       3 files,  177 LOC    agent_demo, run_pipeline, compare_techniques
├── notebooks/      2 .ipynb             Colab A100, Kaggle T4×2
├── tests/          4 files,  239 LOC    23 unittest-style tests, all passing
└── docs/           2 .html               GitHub Pages landing — no markdown API ref
```

Total Python source: **2,621 LOC across 35 files.**

---

## Stated vs. de facto

| | Stated (README / pyproject) | Reality |
|---|---|---|
| **Distribution name** | `agentic-post-training` | No matching dir; top-level packages are generic names (`agents`, `techniques`…) that will collide with any other installed package |
| **Install** | `pip install -e .` | Build backend is `setuptools.backends._legacy:_Backend` — non-standard; standard is `setuptools.build_meta`. Likely fails on fresh `pip` |
| **Console script** | `agentic-train = pipeline.cli:main` | `pipeline/cli.py` does not exist — entry point is broken |
| **Imports** | implicit | All use rootless absolute (`from agents.x`). Works only when CWD is repo root; tests patch `sys.path`. Won't survive `pip install` |
| **Supported techniques** | 11 (PPO, GRPO, DPO, SPO, RLHF, KTO, ORPO, RLAIF, SPIN, SimPO, IPO) | 7 have real torch loss math (priority 1 + KTO, ORPO); 4 are simulation stubs (RLAIF, SPIN, SimPO, IPO) |
| **Optimization** | 5 quant methods + pruning + distillation | `optimization/` package is dead. `OptimizationAgent` reimplements the same dispatch inline with hard-coded mock results |
| **Evaluation** | MMLU, MT-Bench, HumanEval | Mock numbers generated in `EvaluationAgent`; no real harness wired up |

---

## SOLID / KISS Scorecard (1 = bad, 5 = excellent)

| Module | SRP | OCP | LSP/ISP | DIP | KISS | Testability | Notes |
|---|---|---|---|---|---|---|---|
| `agents/base_agent.py` | 3 | 3 | 3 | 2 | 4 | 4 | Reasonable ABC but `**kwargs` on `run/step` defeats LSP/ISP; `print()` in `log()` violates DIP |
| `agents/communication.py` | 4 | 4 | 4 | 4 | 4 | 5 | Cleanest module. Pub/sub bus with typed `Message`. ANSI color import from base_agent is the only smell |
| `agents/coordinator.py` | 2 | 2 | 3 | 2 | 3 | 3 | 190 LOC mixing planner, executor, dashboard printer. Hardcoded step order |
| `agents/{training,optimization,evaluation}_agent.py` | 2 | 2 | 2 | 2 | 3 | 3 | Each agent re-encodes domain dispatch in private dicts and prints. `**kwargs` everywhere |
| `techniques/base_technique.py` | 4 | 4 | 3 | 4 | 5 | 4 | Clean abstract, but `compute_loss(**kwargs)` punts schema to subclasses |
| `techniques/grpo.py` | 4 | 4 | 4 | 4 | 4 | 4 | **Best exemplar.** Group-advantage extracted, torch/numpy fallback handled |
| `techniques/{dpo,ppo,spo,kto,rlhf}.py` | 3 | 3 | 2 | 3 | 3 | 3 | Real loss math, but copy-pasted `β·(chosen_lp − rejected_lp)` and metrics-dict patterns |
| `techniques/{rlaif,spin,simpo,ipo}.py` | 2 | — | — | — | 3 | 2 | **Stubs.** Compute_loss runs a decay formula; configs declared but unused |
| `optimization/*` | 4 | 4 | 4 | 5 | 5 | 2 | Clean design, but dead code — zero callers, so untested |
| `pipeline/config.py` | 4 | 3 | 4 | 4 | 4 | 4 | Solid dataclass + 5 presets. No YAML loader |
| `pipeline/pipeline.py` | 2 | 2 | 3 | 2 | 3 | 3 | Single blocking `run()`, no resumability, no DAG, no progress file |
| `tests/` | 4 | — | — | — | 4 | — | Cover the public API surface but no integration test with real torch tensors |

**Repo-wide averages:** SRP 3.0 / OCP 3.0 / LSP-ISP 3.1 / DIP 3.1 / KISS 3.7 / Testability 3.3.

The strong point is *bones*: protocol-ish base classes, a registry, a real message bus, presets. The weak point is *contracts*: `**kwargs` and string dispatch everywhere mean nothing can be type-checked or extended without reading source.

---

## Public API surface (cannot break without a deprecation shim)

Aggregated from `examples/*.py`, `notebooks/*.ipynb`, and `*/​__init__.py` re-exports.

```python
# agents
from agents import (
    BaseAgent, AgentStatus,
    MessageBus, Message, MessageType,
    CoordinatorAgent, TrainingAgent, OptimizationAgent, EvaluationAgent,
)
# pipeline
from pipeline import PipelineConfig, AgenticPipeline
PipelineConfig.preset("quick_dpo" | "full_rlhf" | "efficient_grpo" | "research_spo" | "production")
AgenticPipeline(config).run()
AgenticPipeline(config).compare_techniques([...])
# technique strings recognized by string-dispatch
"ppo" | "grpo" | "dpo" | "spo" | "rlhf" | "kto" | "orpo" | "rlaif" | "spin" | "simpo" | "ipo"
# quantization strings
"gptq" | "awq" | "gguf" | "nf4" | "int8"
# benchmarks
"mmlu" | "mt_bench" | "humaneval"
```

Any rename or signature change to the above needs a one-version deprecation alias.

---

## Internal Dependency Graph (cross-package only)

```
pipeline ──► agents ──► techniques        (linear, no cycles)
optimization (orphan: no inbound, no outbound)
```

- `agents/training_agent.py:141` is the only `agents → techniques` edge (lazy import inside method — soft dep).
- `optimization/*` is never imported by anything; `OptimizationAgent` reimplements its dispatch.
- All imports are rootless absolute (`from agents.x`), not `from .x` or `from <dist>.agents.x`. This works in dev but **will break after `pip install`** unless layout is fixed.

---

## Red Flags (highest-impact)

1. **`**kwargs` soup on the hot path.** Every agent's `run/step/execute` and every technique's `compute_loss` takes `**kwargs`. Callers must know magic keys (`technique`, `epochs`, `chosen_log_probs`, `quant_type`). No types, no autocomplete, no validation, no docs.
2. **Optimization package is orphan code.** `Quantizer`, `Pruner`, `Distiller` are never instantiated. `OptimizationAgent` re-declares `QUANTIZATION_METHODS` and runs mock dispatch inline.
3. **Simulation-vs-real bifurcation in every technique.** 12 of 13 techniques have a `HAS_TORCH` branch and a decay-formula fallback. The decay constants are unique per file. This doubles the surface area and makes loss math hard to follow.
4. **4 stub techniques** (RLAIF, SPIN, SimPO, IPO) advertise real algorithms but only compute a decay formula. Their configs (`num_principles`, `tau`) are unused.
5. **Pipeline is not resumable.** `AgenticPipeline.run()` is one blocking async call; if it crashes at epoch 9 of 10, the next run starts from epoch 0. No `progress.json`, no DAG.
6. **Coordinator does too much.** 190 LOC mixing pipeline planning, stage execution, dashboard printing, worker discovery.
7. **Print statements throughout the library.** `BaseAgent.log()` calls `print()`. `coordinator.print_dashboard()`, `evaluation_agent._print_report()` all print. No injectable logger / observer.
8. **Broken packaging.** Distribution name doesn't match any importable package; build backend is non-standard `setuptools.backends._legacy:_Backend`; console script points to a missing module; imports are rootless absolute.
9. **`requirements.txt` duplicates `pyproject.dependencies`** — two sources of truth that will drift.
10. **No quality gates.** Ruff config exists but isn't enforced. No mypy, no pre-commit, no GitHub Actions, no coverage. README claims a CI badge but no `.github/`.

---

## What's Already Good (preserve this)

- **`agents/communication.py`** — clean pub/sub bus with a typed `Message`, broadcast and topic subscriptions, conversation logging. Keep the public shape.
- **`techniques/grpo.py`** — best-organized loss implementation. Use as the migration exemplar in Phase 1.
- **`pipeline/config.py`** — typed dataclass with presets and JSON I/O. Solid foundation for the Pydantic v2 schema.
- **`tests/test_agents.py`** — exercises lifecycle, memory, direct + broadcast routing, coordinator wiring. Good safety net for the refactor.
- **The fact that everything actually runs without a GPU** (simulation mode) is a feature — it just needs to be a clearly-labeled `DryRunBackend` instead of a per-technique if-branch.

---

## Conflicts with the original brief

| Brief assumption | Reality |
|---|---|
| "Existing `agents/` may commit to AutoGen/CrewAI/LangGraph" | No. It uses a homegrown asyncio pub/sub bus. No external agent framework imported. |
| "`techniques/` likely has duplicated training loops" | Less duplication than expected — techniques only define `compute_loss`. The duplication is in *loss math primitives* (β·log-ratio, decay simulator), not in trainer loops. There is no shared trainer at all, which means Phase 2's `techniques/_base/trainer.py` is greenfield. |
| "`optimization/` does real model surgery" | No. It returns mock dicts. Replacing it with `apply_optimizations(model, cfg)` is a write, not a refactor. |
| "`pipeline/` has a YAML run-spec" | No. `PipelineConfig` is a Python dataclass with presets; no YAML loader exists. |
| "`docs/` has Markdown API ref" | No. Only `index.html` and `demo.html` (browser landing). MkDocs is greenfield. |

Net effect on the plan: Phases 1 and 2 are mostly *extraction* (we have the bones), Phases 3, 4, 5 are mostly *greenfield* (CI, backends, real demo).
