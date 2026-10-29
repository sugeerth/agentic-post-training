"""Auto-generated technique mimics.

Every file here is a stub written by `MimicWriter` from a paper's
abstract. Each mimic follows the same API as `techniques/agentic/`:

    class Mimic:
        name = "..."           # short slug
        paper_reference = "..." # arXiv URL
        def step(iteration: int) -> dict[str, float]: ...

The registry is built by scanning this directory at import time, so
new mimics show up in `MIMIC_REGISTRY` without editing any registry
by hand.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

MIMIC_REGISTRY: dict[str, type] = {}


def _discover() -> None:
    for mod_info in pkgutil.iter_modules(__path__):
        if mod_info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{__name__}.{mod_info.name}")
        for attr_name in dir(mod):
            attr: Any = getattr(mod, attr_name)
            if isinstance(attr, type) and hasattr(attr, "step") and hasattr(attr, "name"):
                name = getattr(attr, "name", None)
                if isinstance(name, str) and name:
                    MIMIC_REGISTRY[name] = attr


_discover()

__all__ = ["MIMIC_REGISTRY"]
