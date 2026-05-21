"""Plugin registry — one source of truth for techniques, datasets, evaluators, backends.

Deliberately boring. A registry is a dict with two safety properties:

  1. Names are unique within a kind. Registering a duplicate is an error
     unless the caller passes `replace=True` (used by tests).
  2. Lookup of an unknown name raises a clear `KeyError` listing the
     registered alternatives, not a silent `None`.

Third-party plug-ins can register via Python entry points
(`agentic_post_training.techniques`). See `discover_entry_points`.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")

_ENTRY_POINT_GROUPS = {
    "technique": "agentic_post_training.techniques",
    "dataset": "agentic_post_training.datasets",
    "evaluator": "agentic_post_training.evaluators",
    "backend": "agentic_post_training.backends",
}


class Registry(Generic[T]):
    """Typed registry of pluggable components of a single kind.

    Generic on `T` so mypy knows `get_technique("grpo")` returns something
    that conforms to `Technique` (when the registry was constructed with
    `Registry[type[Technique]]`).
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}
        self._discovered = False

    def register(
        self,
        name: str,
        item: T,
        *,
        replace: bool = False,
    ) -> T:
        if not replace and name in self._items:
            raise ValueError(
                f"{self.kind} '{name}' already registered. "
                f"Pass replace=True to override, or pick a unique name."
            )
        self._items[name] = item
        return item

    def get(self, name: str) -> T:
        self._ensure_discovered()
        if name not in self._items:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Registered: {available}"
            )
        return self._items[name]

    def list(self) -> list[str]:
        self._ensure_discovered()
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)

    def _ensure_discovered(self) -> None:
        if self._discovered:
            return
        self._discovered = True
        group = _ENTRY_POINT_GROUPS.get(self.kind)
        if group is None:
            return
        try:
            eps = importlib_metadata.entry_points(group=group)
        except TypeError:  # pragma: no cover  (older importlib_metadata)
            eps = importlib_metadata.entry_points().get(group, [])  # type: ignore[attr-defined]
        for ep in eps:
            if ep.name in self._items:
                continue
            try:
                self._items[ep.name] = ep.load()
            except Exception:
                # A broken third-party plug-in must not crash the host.
                # Phase 3 will add a `--warnings strict` mode that re-raises.
                continue


# Singletons. Importing `core` does NOT trigger entry-point discovery —
# discovery is lazy on first `get`/`list` call.
TECHNIQUES: Registry = Registry("technique")
DATASETS: Registry = Registry("dataset")
EVALUATORS: Registry = Registry("evaluator")
BACKENDS: Registry = Registry("backend")


# Convenience module-level functions. These are the public API; the
# singletons above are an implementation detail.
def register_technique(name: str, *, replace: bool = False) -> Callable[[T], T]:
    """Decorator: `@register_technique("grpo")` registers a Technique class."""

    def _decorator(item: T) -> T:
        TECHNIQUES.register(name, item, replace=replace)
        return item

    return _decorator


def register_dataset(name: str, *, replace: bool = False) -> Callable[[T], T]:
    def _decorator(item: T) -> T:
        DATASETS.register(name, item, replace=replace)
        return item

    return _decorator


def register_evaluator(name: str, *, replace: bool = False) -> Callable[[T], T]:
    def _decorator(item: T) -> T:
        EVALUATORS.register(name, item, replace=replace)
        return item

    return _decorator


def register_backend(name: str, *, replace: bool = False) -> Callable[[T], T]:
    def _decorator(item: T) -> T:
        BACKENDS.register(name, item, replace=replace)
        return item

    return _decorator


def get_technique(name: str): return TECHNIQUES.get(name)
def get_dataset(name: str): return DATASETS.get(name)
def get_evaluator(name: str): return EVALUATORS.get(name)
def get_backend(name: str): return BACKENDS.get(name)

def list_techniques() -> list[str]: return TECHNIQUES.list()
def list_datasets() -> list[str]: return DATASETS.list()
def list_evaluators() -> list[str]: return EVALUATORS.list()
def list_backends() -> list[str]: return BACKENDS.list()
