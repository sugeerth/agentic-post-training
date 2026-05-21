"""Pydantic v2 base config for everything user-facing.

Why Pydantic here but frozen dataclasses in `core.types`:

  - `core.types` carries data through hot loops (one StepMetrics per
    training step). Pydantic's validation overhead is dead weight.
  - `core.config` is constructed once per run from YAML/JSON/CLI flags,
    and a wrong config silently degrades a 4-hour training run. The
    validation cost pays for itself in the first typo.

Conventions every subclass should follow:

  - Inherit from `BaseConfig`, not `pydantic.BaseModel` directly. That
    centralizes our model_config (forbid extras, frozen-after-build,
    json-serializable).
  - Use `Field(..., description="…")` on every field. The description
    becomes the CLI `--help` string and the YAML doc comment.
  - Numeric ranges go in `Field(ge=…, le=…)`. Don't invent custom
    validators when a constraint will do.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict

C = TypeVar("C", bound="BaseConfig")


class BaseConfig(BaseModel):
    """Base for every user-facing config in the framework.

    `model_config` enforces:
      - `extra="forbid"`: a typo'd field is an error, not a silent ignore.
      - `frozen=True`: once built, the config is immutable. Mutation is a
        sign you should be using `model_copy(update=...)` to track changes.
      - `validate_assignment`: paranoia, since frozen prevents writes, but
        also documents intent.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        arbitrary_types_allowed=False,
    )

    # ------------------------------------------------------------------ #
    # Serialization helpers — all configs round-trip through YAML / JSON.
    # ------------------------------------------------------------------ #

    @classmethod
    def from_yaml(cls: type[C], path: str | Path) -> C:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    @classmethod
    def from_json(cls: type[C], path: str | Path) -> C:
        with open(path) as f:
            data = json.load(f)
        return cls.model_validate(data)

    @classmethod
    def from_dict(cls: type[C], data: dict[str, Any]) -> C:
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path | None = None) -> str:
        text = yaml.safe_dump(
            self.model_dump(mode="json"),
            default_flow_style=False,
            sort_keys=False,
        )
        if path is not None:
            Path(path).write_text(text)
        return text

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        text = self.model_dump_json(indent=indent)
        if path is not None:
            Path(path).write_text(text)
        return text
