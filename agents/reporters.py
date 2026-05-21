"""Concrete `core.Reporter` implementations.

Replaces the scattered `print()` calls in the legacy agents. The pattern:

  - In tests, inject `NoopReporter` so the output is silent.
  - In examples and the CLI, inject `ConsoleReporter` for ANSI-colored output
    that matches the legacy look.
  - For machine consumption (CI logs, scrapeable artifacts), inject
    `JSONLReporter` — one JSON object per line, stable schema.

Agents take a `Reporter` in their constructor (Phase 2 wires this in for the
new agent set; the legacy `BaseAgent` keeps `print()` as a fallback until
that migration completes). This file is the single source of truth for what
an event payload looks like.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, TextIO

# ANSI color helpers — same palette the legacy `agents/communication.py` used,
# kept identical so the dashboard look doesn't regress.
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
COLORS = {
    "info": "\033[94m",     # blue
    "success": "\033[92m",  # green
    "warn": "\033[93m",     # yellow
    "error": "\033[91m",    # red
    "dim": DIM,
}


class NoopReporter:
    """Silent reporter — used by tests. Records nothing, prints nothing."""

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        return None


class ConsoleReporter:
    """ANSI-colored single-line events for an interactive terminal.

    Each event is rendered as:
        [HH:MM:SS] <event>  key=value  key=value
    Levels are inferred from the payload's `level` key (default "info").
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdout

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        level = payload.get("level", "info")
        color = COLORS.get(level, "")
        ts = time.strftime("%H:%M:%S")
        kvs = " ".join(
            f"{k}={_brief(v)}"
            for k, v in payload.items()
            if k != "level"
        )
        line = f"{DIM}[{ts}]{RESET} {color}{BOLD}{event}{RESET}  {kvs}"
        print(line, file=self.stream, flush=True)


class JSONLReporter:
    """One JSON object per line. Suitable for log aggregators / CI parsing.

    The schema is intentionally minimal:
        {"ts": <unix>, "event": <str>, "payload": {...}}
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._fh: TextIO | None = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        line = json.dumps(
            {"ts": time.time(), "event": event, "payload": payload},
            default=str,
            sort_keys=False,
        )
        if self._fh is not None:
            self._fh.write(line + "\n")
            self._fh.flush()
        else:
            print(line, flush=True)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _brief(v: Any) -> str:
    """Compact value rendering for console output. Truncates long strings."""
    s = repr(v) if not isinstance(v, str) else v
    return s if len(s) <= 80 else s[:77] + "..."
