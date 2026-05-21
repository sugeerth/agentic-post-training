"""Optimization agent — thin delegate over `optimization.apply_optimizations`.

Phase 2 refactor: this agent used to re-implement quant/prune/distill dispatch
inline with its own copy of `QUANTIZATION_METHODS` etc. That duplicated the
`optimization/` package, which sat unused. Now the agent only does what an
agent should: messaging, status reporting, and turning a request into an
`OptimizationPlan` for the package to execute.

If you want to add a new optimization method, do it in `optimization/`, not
here.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agents.base_agent import BaseAgent
from optimization.apply import apply_optimizations, plan_from_legacy_kwargs


class OptimizationAgent(BaseAgent):
    """Agent responsible for orchestrating post-training optimization.

    Concretely: receives a request, builds an OptimizationPlan, calls
    `apply_optimizations`, and broadcasts the result. The agent does not
    know how quantization works — only that it has a place to send the
    request and a place to send the result.
    """

    def __init__(self, name: str = "Optimizer"):
        super().__init__(name, role="optimizer")
        self.register_capability("quantization", "Quantize models to lower precision")
        self.register_capability("pruning", "Prune model weights for efficiency")
        self.register_capability("distillation", "Distill knowledge to smaller models")

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        plan = plan_from_legacy_kwargs(**kwargs)

        method = kwargs.get("method", "quantization")
        await self.send_message(
            "optimization",
            {"message": f"Starting {method}", "method": method},
            target="broadcast",
        )
        self.log(f"Dispatching {method} via optimization.apply_optimizations(...)")
        # The simulated calls are cheap, but we keep the asyncio.sleep
        # latency so existing tests / demos still look interactive.
        await asyncio.sleep(0.5)

        # The agent doesn't have a real model handle here — pipeline/backend
        # will pass one in Phase 4. For now we hand the package a sentinel
        # so its (simulated) report has all the fields downstream expects.
        model_sentinel = kwargs.get("model", "model")
        _, result = apply_optimizations(model_sentinel, plan)

        # Flatten to the legacy result shape so the Coordinator/Pipeline
        # don't need to be updated in the same commit.
        flat: dict[str, Any] = {"status": "completed", "method": method}
        if result.quantization is not None:
            flat.update(result.quantization)
        if result.pruning is not None:
            flat.update(result.pruning)
        if result.distillation is not None:
            flat.update(result.distillation)

        await self.send_message(
            "task_result",
            {"message": f"{method} complete", "result": flat},
            target="Coordinator",
        )
        return flat

    async def step(self, **kwargs: Any) -> dict[str, Any]:
        return await self.run(**kwargs)
