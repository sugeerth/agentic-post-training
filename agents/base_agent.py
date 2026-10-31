"""Base agent class for all agents in the framework."""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


# ANSI colors for different agents
AGENT_COLORS = {
    "coordinator": "\033[1;35m",   # Bold Magenta
    "trainer": "\033[1;36m",       # Bold Cyan
    "optimizer": "\033[1;33m",     # Bold Yellow
    "evaluator": "\033[1;32m",     # Bold Green
    "operator": "\033[1;34m",      # Bold Blue — computer-use agent
    "default": "\033[1;37m",       # Bold White
}
RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"


@dataclass
class AgentCapability:
    """A capability that an agent can perform."""
    name: str
    description: str
    priority: int = 1  # 1=highest


@dataclass
class AgentMemory:
    """Shared memory store for inter-agent state."""
    data: dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def keys(self) -> list[str]:
        return list(self.data.keys())


class BaseAgent(ABC):
    """Base class for all agents in the agentic post-training framework.

    Each agent has:
    - A unique ID and human-readable name
    - A role describing its function
    - Status tracking (idle -> running -> completed/failed)
    - A message inbox for inter-agent communication
    - A shared memory store
    - Registered capabilities
    """

    def __init__(self, name: str, role: str = "worker"):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.role = role
        self.status = AgentStatus.IDLE
        self.capabilities: list[AgentCapability] = []
        self.memory = AgentMemory()
        self.inbox: asyncio.Queue[Any] = asyncio.Queue()
        self.message_bus: Any = None  # Set by coordinator
        self.logger = logging.getLogger(f"agent.{name}")
        self._color = AGENT_COLORS.get(role, AGENT_COLORS["default"])

    def register_capability(self, name: str, description: str, priority: int = 1) -> None:
        self.capabilities.append(AgentCapability(name, description, priority))

    def has_capability(self, name: str) -> bool:
        return any(c.name == name for c in self.capabilities)

    def log(self, message: str, level: str = "info") -> None:
        prefix = f"{self._color}[{self.name}]{RESET}"
        formatted = f"{prefix} {message}"
        print(formatted)
        getattr(self.logger, level)(f"[{self.name}] {message}")

    async def send_message(self, msg_type: str, payload: dict, target: str | None = None) -> None:
        if self.message_bus:
            from agents.communication import Message, MessageType
            msg = Message(
                sender=self.name,
                receiver=target or "broadcast",
                msg_type=MessageType(msg_type),
                payload=payload,
            )
            await self.message_bus.publish(msg)

    async def receive_message(self, timeout: float = 1.0) -> Any:
        try:
            return await asyncio.wait_for(self.inbox.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    @abstractmethod
    async def run(self, **kwargs) -> dict[str, Any]:
        """Main execution method. Returns result dict."""
        ...

    @abstractmethod
    async def step(self, **kwargs) -> dict[str, Any]:
        """Single step of execution."""
        ...

    async def execute(self, **kwargs) -> dict[str, Any]:
        """Full lifecycle: idle -> running -> completed/failed."""
        self.status = AgentStatus.RUNNING
        self.log(f"Starting execution (status: {self.status.value})")
        try:
            result = await self.run(**kwargs)
            self.status = AgentStatus.COMPLETED
            self.log("Completed successfully ✓")
            return result
        except Exception as e:
            self.status = AgentStatus.FAILED
            self.log(f"Failed: {e}", level="error")
            raise

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} status={self.status.value}>"
