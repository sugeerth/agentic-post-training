"""Pipeline configuration with presets and validation.

Phase 3 adds YAML round-trip (`from_yaml` / `to_yaml`) so a run is fully
describable as a single text file. JSON `load` / `save` stay for back-compat.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""

    # Model
    model_name: str = "gpt2"
    model_revision: str = "main"

    # Technique
    technique: str = "grpo"
    technique_config: dict[str, Any] = field(default_factory=dict)

    # Training
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    max_length: int = 512
    gradient_accumulation_steps: int = 4

    # Optimization
    quantization: str | None = None  # gptq, awq, nf4, int8, gguf
    pruning: str | None = None
    pruning_sparsity: float = 0.5

    # Evaluation
    benchmarks: list[str] = field(default_factory=lambda: ["mmlu", "mt_bench", "humaneval"])

    # Infrastructure
    output_dir: str = "./output"
    logging_steps: int = 10
    save_steps: int = 100
    seed: int = 42
    fp16: bool = False
    bf16: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> PipelineConfig:
        data = json.loads(Path(path).read_text())
        return cls(**data)

    # ---- YAML round-trip (Phase 3) -------------------------------------- #

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Build a config from a YAML run-spec.

        Unknown keys raise — typos in a run-spec silently degrade long runs,
        so we fail loud at load time instead.
        """
        data = yaml.safe_load(Path(path).read_text()) or {}
        cls_fields = {f.name for f in cls.__dataclass_fields__.values()}
        extras = set(data) - cls_fields
        if extras:
            raise ValueError(
                f"Unknown keys in run-spec {path}: {sorted(extras)}. "
                f"Allowed: {sorted(cls_fields)}"
            )
        return cls(**data)

    def to_yaml(self, path: str | Path | None = None) -> str:
        text = yaml.safe_dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def preset(cls, name: str) -> PipelineConfig:
        presets = {
            "quick_dpo": cls(technique="dpo", epochs=1, model_name="gpt2"),
            "full_rlhf": cls(technique="rlhf", epochs=3, model_name="meta-llama/Llama-3.1-8B"),
            "efficient_grpo": cls(technique="grpo", epochs=3, quantization="nf4"),
            "research_spo": cls(technique="spo", epochs=5),
            "production": cls(technique="grpo", epochs=3, quantization="gptq",
                            benchmarks=["mmlu", "mt_bench", "humaneval", "gsm8k", "truthfulqa"]),
        }
        if name not in presets:
            raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")
        return presets[name]

    def validate(self) -> list[str]:
        errors = []
        valid_techniques = ["ppo", "grpo", "dpo", "spo", "rlhf", "rlaif", "kto", "orpo", "spin", "simpo", "ipo"]
        if self.technique not in valid_techniques:
            errors.append(f"Unknown technique: {self.technique}")
        if self.epochs < 1:
            errors.append("epochs must be >= 1")
        if self.learning_rate <= 0:
            errors.append("learning_rate must be > 0")
        return errors
