"""Message bus: agents report what they did as small messages. Synchronous, dependency-free, logged to JSONL.

(The repo's older agents/communication.py bus is asyncio-based and tied to terminal colours; a training loop does
not need either, so this is the simple version of the same idea.)
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Message:
    sender: str
    receiver: str
    text: str
    data: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class Bus:
    def __init__(self, log_path: Path | None = None, echo: bool = True):
        self.history: list[Message] = []
        self.echo = echo
        self._log = open(log_path, "a") if log_path else None

    def send(self, sender: str, receiver: str, text: str, **data) -> Message:
        m = Message(sender, receiver, text, data)
        self.history.append(m)
        if self.echo:
            print(f"[{time.strftime('%H:%M:%S', time.localtime(m.ts))}] {sender:>10} → {receiver:<11} {text}", flush=True)
        if self._log:
            self._log.write(json.dumps(asdict(m)) + "\n")
            self._log.flush()
        return m
