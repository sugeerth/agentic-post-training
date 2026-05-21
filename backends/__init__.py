"""Backend implementations — places to run a `TrainingJob`.

Each backend conforms to `core.Backend` (see core/protocols.py):

  - `dry_run(job)` — print the plan + estimated cost without spending money
  - `launch(job)`  — actually run it, return a `JobHandle`

Phase 4 ships `LocalBackend`. ColabBackend, KaggleBackend, VastAIBackend slot
in next with the same surface.
"""

from backends.local import LocalBackend

__all__ = ["LocalBackend"]
