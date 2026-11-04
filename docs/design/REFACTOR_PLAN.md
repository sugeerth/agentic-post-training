# REFACTOR_PLAN — framework_agentic_training

**Date:** 2026-05-11
**Status:** awaiting GATE 0 approval before any edits

Issues are sorted by **impact × inverse-effort**. Each item lists its rough effort (S/M/L), the modules it touches, and what success looks like.

---

## Tier A — load-bearing, do first (Phases 1–2)

### A1. Replace `**kwargs` with typed configs on every public method (M, touches all)
**Why:** This single change is the highest-leverage one in the repo. Today `BaseAgent.run(**kwargs)`, `Technique.compute_loss(**kwargs)`, and `Coordinator.execute(**kwargs)` all hide their contract from the type system and from readers. Pydantic v2 + per-technique/per-agent config schemas gives autocomplete, validation, JSON/YAML round-trip, and self-documenting APIs in one move.
**Success:** every public callable has a typed config object; `mypy --strict` passes on `core/`, `agents/`, `techniques/`; no `**kwargs` in any public signature.

### A2. Introduce `core/` package with Protocols + registry (M, new code)
**Why:** Anchors everything else. `Protocol`s for `DatasetLoader`, `Technique`, `Evaluator`, `Backend`, `Agent`; a thin `registry.py` with `register_technique` etc. Use entry-point discovery so third parties can plug in without forking. Deliberately Boring™ — no AbstractFactoryBuilder.
**Success:** `core/protocols.py` < 100 LOC; the registry passes a single roundtrip test (`register("foo", Foo); get("foo") is Foo`); one existing technique (GRPO) imports and conforms.

### A3. Extract `techniques/_base/` shared utilities (S, touches all techniques)
**Why:** Five techniques (DPO, SPO, KTO, GRPO, PPO) duplicate the `β · (chosen_logp − rejected_logp)` ratio loss and the metrics-dict update. Lift these into `_base/ratio_loss.py`, `_base/metrics.py`, `_base/simulator.py` (the latter formalizes the `HAS_TORCH` fallback as one named helper instead of 12 copy-pasted branches).
**Success:** ratio-loss math defined once; per-technique files shrink ~30%; tests still pass.

### A4. Fix packaging so `pip install` actually works (S, touches pyproject + imports)
**Why:** Move sources under `src/agentic_post_training/` (matches distribution name `agentic-post-training`); fix `build-backend` to `setuptools.build_meta`; change every `from agents.x` to `from agentic_post_training.agents.x` (or `from ..x`); delete the broken `pipeline.cli:main` entry point until A8 creates it; remove `requirements.txt` (keep pyproject as the single source of truth) or make it a `-r pyproject.toml` shim.
**Success:** `pip install -e .` succeeds in a fresh venv; `python -c "import agentic_post_training"` works from anywhere; tests pass after the move.

### A5. Kill or quarantine the 4 stub techniques (S, touches techniques/)
**Why:** RLAIF, SPIN, SimPO, IPO advertise published algorithms but only compute a decay formula. They confuse users and inflate the technique count in the README.
**Success:** either (a) delete and remove from README support matrix, or (b) move to `techniques/experimental/` with `NotImplementedError` in `compute_loss` and a clear "not yet implemented" note. Recommend (b) — keeps the names reserved.

### A6. Reabsorb the orphan `optimization/` package (S, touches optimization + agents)
**Why:** `OptimizationAgent` re-implements quant/prune/distill dispatch inline; `optimization/` classes are dead. Make `OptimizationAgent` thin: it instantiates `Quantizer(cfg).quantize(model)` from the package. Define one `apply_optimizations(model, cfg) → model` entry point.
**Success:** `OptimizationAgent` < 80 LOC and only orchestrates; all method dicts live once in `optimization/`; `from optimization.quantization import Quantizer` is the only place that knows about bitsandbytes (lazy-imported).

---

## Tier B — quality of life and growth (Phase 2 finish + Phase 3)

### B1. Split `CoordinatorAgent` into Planner + Executor (M, agents/coordinator.py)
**Why:** 190 LOC mixing pipeline planning, stage execution, worker discovery, dashboard printing. Split into `Planner` (DAG construction from a run-spec) + `Executor` (asyncio dispatch + state tracking) + `Dashboard` (observer that prints).
**Success:** each piece < 80 LOC; planner is pure (no asyncio); executor takes a planner output and runs it.

### B2. Add resumable pipeline state (M, pipeline/)
**Why:** Crashed runs replay from epoch 0. Write `progress.json` after each step; on resume, skip completed steps. Minimal: `{"step_id": "training", "status": "completed", "checkpoint": "..."}`.
**Success:** kill `run_pipeline.py` mid-run, re-launch, observe it picks up at the right step.

### B3. YAML run-spec loader (S, pipeline/)
**Why:** Today the only way to configure a run is Python. `pipeline/config.py` is already a dataclass — add `from_yaml(path) → PipelineConfig` and round-trip with `to_yaml()`. Don't invent a new schema language; YAML in, Pydantic-validated.
**Success:** `agentic-train run configs/demo.yaml --backend local` works.

### B4. Inject a logger/observer instead of `print()` (S, agents/, optimization/)
**Why:** Every agent prints directly. Replace with a `Reporter` protocol — `ConsoleReporter` (current behavior), `JSONLReporter` (machine-readable), `NoopReporter` (tests). Inject through agent constructor.
**Success:** `BaseAgent` has no `print()` calls; tests run silently.

### B5. Quality gates (M, root)
**Why:** Ruff config exists but isn't enforced. Add `mypy --strict` on `core/`/`agents/`/`techniques/`; `pre-commit` with ruff + mypy + end-of-file-fixer; `Makefile` (`install/test/lint/demo/docs`); GH Actions on 3.10/3.11/3.12; `pytest --cov` with thresholds.
**Success:** `make test` is green locally and in CI; coverage ≥ 80% on `core/`.

### B6. MkDocs Material site (S–M, docs/)
**Why:** Today `docs/` is two HTML files. Replace with MkDocs + `mkdocstrings` for auto API ref + 3 hand-written guides (Quickstart, Adding a Technique, Adding a Backend). Replace the broken-looking GitHub Pages badge in README with a real link.
**Success:** `mkdocs serve` renders; `make docs` builds the site.

---

## Tier C — capability expansion (Phases 4–5)

### C1. Three backends behind one protocol (L, new `backends/`)
**Why:** "Runs anywhere" is the value prop. LocalBackend (in-process; default), ColabBackend / KaggleBackend (template-generated notebook), VastAIBackend (cheapest matching instance; tear down on completion). Each has a `dry_run()` that prints the plan + cost without spending.
**Success:** the same `agentic-train run configs/demo.yaml --backend X` works for X ∈ {local, colab, vastai}.

### C2. Promote `data/` and `eval/` to first-class packages (M, new code)
**Why:** Today there's no data layer; techniques accept random kwargs. Today there's no eval layer; `EvaluationAgent` makes up numbers. Create `data/loaders/` (HF Hub, JSONL, S3, Parquet), `data/filters/` (length, dedup MinHash, PII), `data/formatters/` (chat templates), and `eval/` wrapping lm-evaluation-harness + IFEval + MT-Bench.
**Success:** Phase 5 demo uses real datasets and real eval; before/after numbers are reproducible.

### C3. ORPO Qwen2.5-0.5B demo end-to-end (L, examples/, notebooks/)
**Why:** Single proof point that the whole library works. ORPO is single-stage (no ref model, fits in 16GB), Qwen2.5-0.5B fits on free T4 in <90 min.
**Success:** `make demo` runs locally; Colab + Kaggle + Vast.ai variants exist; `RESULTS.md` table commits real numbers (base vs post-trained on IFEval + MT-Bench subset + MMLU regression check).

---

## Top 10 (priority order)

| # | Item | Tier | Effort | Phase |
|---|---|---|---|---|
| 1 | A4 — Fix packaging (`src/`, build backend, imports) | A | S | 1 |
| 2 | A2 — `core/` Protocols + registry | A | M | 1 |
| 3 | A1 — Typed configs replace `**kwargs` | A | M | 1–2 |
| 4 | A3 — `techniques/_base/` shared math | A | S | 2 |
| 5 | A6 — Reabsorb `optimization/` package | A | S | 2 |
| 6 | A5 — Quarantine 4 stub techniques | A | S | 2 |
| 7 | B1 — Split Coordinator | B | M | 2 |
| 8 | B5 — Quality gates (ruff/mypy/CI) | B | M | 3 |
| 9 | C1 — Three backends behind one protocol | C | L | 4 |
| 10 | C3 — ORPO Qwen2.5-0.5B demo | C | L | 5 |

`B2` (resumable pipeline), `B3` (YAML loader), `B4` (logger injection), `B6` (MkDocs), `C2` (data + eval packages) are bundled into Phases 2–3 as supporting work for the items above.

---

## Migration exemplar (preview only — not yet started)

The first concrete migration in Phase 1 will be **GRPO**, because it is the cleanest existing technique (highest scorecard, isolated helper functions, real torch math + graceful numpy fallback). Once GRPO conforms to the new `Technique` protocol with a typed `GRPOConfig`, the same template fans out to DPO → SPO → KTO → PPO → RLHF → ORPO. The 4 stubs do not migrate; they go to `experimental/`.

---

## Risks called out

1. **Test suite is shallow.** 23 tests cover the public *shape* but none drive a real torch forward/backward pass. The refactor is safer if we add 2–3 integration tests on `sshleifer/tiny-gpt2` *before* moving any technique code.
2. **Notebooks pin the public API.** Both `.ipynb` files inline a chunk of the library. If we rename anything they reference, those notebooks silently rot. Plan: regenerate notebooks from a Jinja template (part of `ColabBackend`) so they track the API automatically.
3. **`bitsandbytes` is platform-fragile** (no Apple Silicon support). Phase 4's `LocalBackend` on the audit machine (M-series Mac per global instructions) cannot exercise NF4/INT8 — they'll need to be CUDA-only paths gated by a runtime check.
4. **The ORPO demo budget on free Colab T4 is tight.** Qwen2.5-0.5B + ultrafeedback-binarized + ORPO at bf16 fits, but only with batch size 1, grad accumulation 8, max seq 512. Set those defaults in `examples/orpo_quickstart/run.yaml` explicitly.
