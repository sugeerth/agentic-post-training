"""Single boundary for model optimizations.

Every code path that wants to compress / accelerate / sparsify a model goes
through `apply_optimizations(model, cfg) -> model`. Nothing else should
`import bitsandbytes` (or auto-gptq, autoawq) directly — those imports are
lazy and live in the per-method helpers below.

This is the file that closes the gap the audit flagged: `OptimizationAgent`
used to re-implement the dispatch inline while `optimization/` sat unused.
The agent now calls `apply_optimizations(...)` and is responsible only for
messaging — not for what the optimization actually does.

Phase 2 ships the dispatch and the simulated reports. Phase 4 wires
real bitsandbytes / auto-gptq / autoawq paths once Vast.ai / Colab backends
exist to run them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optimization.distillation import DistillationConfig, Distiller
from optimization.pruning import Pruner, PruningConfig
from optimization.quantization import QuantizationConfig, Quantizer


@dataclass
class OptimizationPlan:
    """Declarative description of which optimizations to apply, in order.

    Empty fields mean 'skip that step'. The plan is intentionally a plain
    dataclass — Pipeline configs (in `pipeline/config.py`) already validate
    upstream, so a second validation layer here is redundant.
    """

    quantization: QuantizationConfig | None = None
    pruning: PruningConfig | None = None
    distillation: DistillationConfig | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Aggregated report from one `apply_optimizations` call.

    Each step's per-method result dict (from `Quantizer.quantize`, etc.)
    is preserved under its own key so callers can extract metrics
    individually.
    """

    quantization: dict[str, Any] | None = None
    pruning: dict[str, Any] | None = None
    distillation: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.quantization is not None:
            out["quantization"] = self.quantization
        if self.pruning is not None:
            out["pruning"] = self.pruning
        if self.distillation is not None:
            out["distillation"] = self.distillation
        return out


def apply_optimizations(
    model: Any,
    plan: OptimizationPlan,
    *,
    model_size_gb: float | None = None,
) -> tuple[Any, OptimizationResult]:
    """Apply quantization → pruning → distillation in that fixed order.

    The order matters: quantizing after pruning would waste pruning effort
    on weights that get rounded away; distilling last lets the student
    benefit from a smaller, faster teacher.

    Today the underlying calls are simulated (they return reports, not
    modified weights) because the dependencies (bitsandbytes, auto-gptq,
    autoawq) aren't installed in the test environment. The agent + pipeline
    care about the report shape, not the weight surgery — which is correct.
    When Phase 4's `LocalBackend` runs on a CUDA box with those packages
    available, only `Quantizer.quantize` / `Pruner.prune` / `Distiller.distill`
    need to flip to real behavior; this dispatcher does not change.
    """
    result = OptimizationResult()

    if plan.quantization is not None:
        q = Quantizer(plan.quantization)
        extras = {"model_size_gb": model_size_gb} if model_size_gb is not None else {}
        result.quantization = q.quantize(model_path=str(model), **extras)

    if plan.pruning is not None:
        p = Pruner(plan.pruning)
        result.pruning = p.prune(model_path=str(model))

    if plan.distillation is not None:
        d = Distiller(plan.distillation)
        result.distillation = d.distill()

    return model, result


# Convenience: build an OptimizationPlan from the legacy string-config shape
# the OptimizationAgent used to dispatch on. This is the one place we map
# the old `method=..., quant_type=..., sparsity=...` kwargs to typed configs.
def plan_from_legacy_kwargs(**kwargs: Any) -> OptimizationPlan:
    """Map the legacy OptimizationAgent.run(**kwargs) shape to a plan.

    Legacy keys: method, quant_type, pruning_type, sparsity,
                 teacher_model, student_model, temperature.
    """
    method = kwargs.get("method", "quantization")
    plan = OptimizationPlan()

    if method == "quantization":
        plan.quantization = QuantizationConfig(method=kwargs.get("quant_type", "gptq"))
    elif method == "pruning":
        plan.pruning = PruningConfig(
            method=kwargs.get("pruning_type", "magnitude"),
            sparsity=kwargs.get("sparsity", 0.5),
        )
    elif method == "distillation":
        plan.distillation = DistillationConfig(
            teacher_model=kwargs.get("teacher_model", "meta-llama/Llama-3.1-70B"),
            student_model=kwargs.get("student_model", "meta-llama/Llama-3.1-8B"),
            temperature=kwargs.get("temperature", 2.0),
        )
    return plan
