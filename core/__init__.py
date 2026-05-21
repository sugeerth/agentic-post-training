"""Core abstractions for the agentic post-training framework.

This package defines the *contracts* — typed data carriers, Protocols, the
plugin registry, and the config base — that everything else conforms to.
It is intentionally small and dependency-light.

Nothing in `core/` may import from `agents/`, `techniques/`, `optimization/`,
`pipeline/`, `data/`, or `eval/`. The arrow always points inward.
"""

from core.config import BaseConfig
from core.protocols import (
    Agent,
    Backend,
    DatasetLoader,
    Evaluator,
    Reporter,
    Technique,
)
from core.registry import (
    Registry,
    get_backend,
    get_dataset,
    get_evaluator,
    get_technique,
    list_backends,
    list_datasets,
    list_evaluators,
    list_techniques,
    register_backend,
    register_dataset,
    register_evaluator,
    register_technique,
)
from core.types import (
    EvalResult,
    JobHandle,
    JobStatus,
    PreferencePair,
    RolloutBatch,
    StepMetrics,
    TrainingExample,
    TrainingJob,
)

__all__ = [
    # protocols
    "Agent",
    "Backend",
    # config
    "BaseConfig",
    "DatasetLoader",
    "EvalResult",
    "Evaluator",
    "JobHandle",
    "JobStatus",
    "PreferencePair",
    # registry
    "Registry",
    "Reporter",
    "RolloutBatch",
    "StepMetrics",
    "Technique",
    # types
    "TrainingExample",
    "TrainingJob",
    "get_backend",
    "get_dataset",
    "get_evaluator",
    "get_technique",
    "list_backends",
    "list_datasets",
    "list_evaluators",
    "list_techniques",
    "register_backend",
    "register_dataset",
    "register_evaluator",
    "register_technique",
]
