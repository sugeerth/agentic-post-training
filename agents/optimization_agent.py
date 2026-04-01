"""Optimization agent for quantization, pruning, and distillation."""

from __future__ import annotations

import asyncio
from typing import Any

from agents.base_agent import BaseAgent


class OptimizationAgent(BaseAgent):
    """Agent responsible for post-training optimization.

    Handles:
    - Quantization (GPTQ, AWQ, GGUF, bitsandbytes 4/8-bit)
    - Pruning (magnitude, structured, movement, Wanda)
    - Knowledge distillation
    - Reports compression ratios and quality metrics
    """

    QUANTIZATION_METHODS = {
        "gptq": {"bits": 4, "group_size": 128, "desc": "GPTQ 4-bit quantization"},
        "awq": {"bits": 4, "group_size": 128, "desc": "Activation-aware Weight Quantization"},
        "gguf": {"bits": 4, "desc": "GGUF format for llama.cpp inference"},
        "nf4": {"bits": 4, "desc": "NormalFloat4 via bitsandbytes"},
        "int8": {"bits": 8, "desc": "LLM.int8() via bitsandbytes"},
    }

    PRUNING_METHODS = {
        "magnitude": {"desc": "Remove smallest weights by magnitude"},
        "structured": {"desc": "Remove entire channels/heads"},
        "movement": {"desc": "Learn which weights to prune during training"},
        "wanda": {"desc": "Pruning by Weights and Activations"},
    }

    def __init__(self, name: str = "Optimizer"):
        super().__init__(name, role="optimizer")
        self.register_capability("quantization", "Quantize models to lower precision")
        self.register_capability("pruning", "Prune model weights for efficiency")
        self.register_capability("distillation", "Distill knowledge to smaller models")

    async def run(self, **kwargs) -> dict[str, Any]:
        method = kwargs.get("method", "quantization")
        quant_type = kwargs.get("quant_type", "gptq")
        pruning_type = kwargs.get("pruning_type", "magnitude")
        sparsity = kwargs.get("sparsity", 0.5)

        if method == "quantization":
            return await self._quantize(quant_type)
        elif method == "pruning":
            return await self._prune(pruning_type, sparsity)
        elif method == "distillation":
            return await self._distill(**kwargs)
        else:
            return await self._quantize(quant_type)

    async def _quantize(self, method: str = "gptq") -> dict[str, Any]:
        info = self.QUANTIZATION_METHODS.get(method, self.QUANTIZATION_METHODS["gptq"])

        await self.send_message("optimization", {
            "message": f"Starting {method.upper()} quantization ({info['bits']}-bit)",
            "method": method,
            "bits": info["bits"],
        }, target="broadcast")

        # Simulate calibration
        self.log(f"Running calibration dataset for {method.upper()}...")
        await asyncio.sleep(0.5)

        # Simulate quantization
        self.log(f"Quantizing model weights to {info['bits']}-bit...")
        await asyncio.sleep(0.5)

        compression_ratio = 32 / info["bits"]
        perplexity_increase = {4: 0.15, 8: 0.02}.get(info["bits"], 0.1)

        result = {
            "method": method,
            "bits": info["bits"],
            "compression_ratio": f"{compression_ratio:.1f}x",
            "original_size_gb": 14.0,
            "quantized_size_gb": 14.0 / compression_ratio,
            "perplexity_increase": f"+{perplexity_increase:.2f}",
            "status": "completed",
        }

        await self.send_message("task_result", {
            "message": f"Quantization complete: {compression_ratio:.1f}x compression, +{perplexity_increase:.2f} perplexity",
            "result": result,
        }, target="Coordinator")

        return result

    async def _prune(self, method: str = "magnitude", sparsity: float = 0.5) -> dict[str, Any]:
        info = self.PRUNING_METHODS.get(method, self.PRUNING_METHODS["magnitude"])

        await self.send_message("optimization", {
            "message": f"Starting {method} pruning (target: {sparsity*100:.0f}% sparsity)",
            "method": method,
            "sparsity": sparsity,
        }, target="broadcast")

        self.log(f"Analyzing weight importance with {method} method...")
        await asyncio.sleep(0.4)

        self.log(f"Pruning to {sparsity*100:.0f}% sparsity...")
        await asyncio.sleep(0.4)

        quality_retention = 1.0 - (sparsity * 0.15)

        result = {
            "method": method,
            "sparsity": sparsity,
            "speedup": f"{1 / (1 - sparsity * 0.6):.2f}x",
            "quality_retention": f"{quality_retention:.1%}",
            "params_removed": f"{sparsity:.0%}",
            "status": "completed",
        }

        await self.send_message("task_result", {
            "message": f"Pruning complete: {sparsity:.0%} sparsity, {quality_retention:.1%} quality retained",
            "result": result,
        }, target="Coordinator")

        return result

    async def _distill(self, **kwargs) -> dict[str, Any]:
        teacher = kwargs.get("teacher_model", "llama-70b")
        student = kwargs.get("student_model", "llama-8b")
        temperature = kwargs.get("temperature", 2.0)

        await self.send_message("optimization", {
            "message": f"Starting knowledge distillation: {teacher} → {student}",
            "teacher": teacher,
            "student": student,
        }, target="broadcast")

        self.log(f"Distilling {teacher} → {student} (T={temperature})")
        await asyncio.sleep(0.6)

        result = {
            "teacher": teacher,
            "student": student,
            "temperature": temperature,
            "student_quality": "92% of teacher",
            "size_reduction": "8.75x",
            "status": "completed",
        }

        await self.send_message("task_result", {
            "message": f"Distillation complete: student retains 92% of teacher quality",
            "result": result,
        }, target="Coordinator")

        return result

    async def step(self, **kwargs) -> dict[str, Any]:
        return await self.run(**kwargs)
