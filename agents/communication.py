"""Inter-agent communication bus with publish/subscribe messaging."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agents.base_agent import AGENT_COLORS, BOLD, DIM, RESET


class MessageType(Enum):
    TASK_REQUEST = "task_request"
    TASK_RESULT = "task_result"
    STATUS_UPDATE = "status_update"
    DATA_SHARE = "data_share"
    COORDINATION = "coordination"
    HEARTBEAT = "heartbeat"
    EVALUATION = "evaluation"
    OPTIMIZATION = "optimization"


MSG_ICONS = {
    MessageType.TASK_REQUEST: "📋",
    MessageType.TASK_RESULT: "✅",
    MessageType.STATUS_UPDATE: "📊",
    MessageType.DATA_SHARE: "📦",
    MessageType.COORDINATION: "🔗",
    MessageType.HEARTBEAT: "💓",
    MessageType.EVALUATION: "📈",
    MessageType.OPTIMIZATION: "⚡",
}


@dataclass
class Message:
    """A message between agents."""
    sender: str
    receiver: str  # agent name or "broadcast"
    msg_type: MessageType
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # Higher = more urgent

    def pretty_print(self) -> str:
        icon = MSG_ICONS.get(self.msg_type, "💬")
        sender_color = _get_color(self.sender)
        receiver_color = _get_color(self.receiver)

        ts = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        header = f"{DIM}[{ts}]{RESET} {icon} {sender_color}{BOLD}{self.sender}{RESET} → {receiver_color}{self.receiver}{RESET}"

        summary = self.payload.get("message", self.payload.get("status", str(self.msg_type.value)))
        return f"{header}: {summary}"


def _get_color(name: str) -> str:
    name_lower = name.lower()
    for key, color in AGENT_COLORS.items():
        if key in name_lower:
            return color
    return AGENT_COLORS["default"]


class MessageBus:
    """Central communication bus for agent-to-agent messaging.

    Supports:
    - Direct messaging (sender -> specific receiver)
    - Broadcast messaging (sender -> all subscribers)
    - Topic-based subscriptions
    - Message history with pretty-print conversation view
    """

    def __init__(self, verbose: bool = True):
        self.agents: dict[str, Any] = {}  # name -> agent
        self.subscribers: dict[str, list[Callable]] = {}  # topic -> callbacks
        self.history: list[Message] = []
        self.verbose = verbose

    def register_agent(self, agent: Any) -> None:
        self.agents[agent.name] = agent
        agent.message_bus = self

    def subscribe(self, topic: str, callback: Callable) -> None:
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)

    async def publish(self, message: Message) -> None:
        self.history.append(message)

        if self.verbose:
            print(message.pretty_print())

        # Direct message
        if message.receiver != "broadcast" and message.receiver in self.agents:
            await self.agents[message.receiver].inbox.put(message)
        # Broadcast
        elif message.receiver == "broadcast":
            for name, agent in self.agents.items():
                if name != message.sender:
                    await agent.inbox.put(message)

        # Topic subscribers
        topic = message.msg_type.value
        for callback in self.subscribers.get(topic, []):
            try:
                await callback(message)
            except Exception as e:
                print(f"{DIM}[MessageBus] Subscriber error: {e}{RESET}")

    def print_conversation(self, last_n: int | None = None) -> str:
        """Print a beautiful conversation view of agent communications."""
        messages = self.history[-last_n:] if last_n else self.history
        lines = []
        lines.append(f"\n{BOLD}{'═' * 70}")
        lines.append(f"  🤖 Agent Communication Log ({len(messages)} messages)")
        lines.append(f"{'═' * 70}{RESET}\n")

        for msg in messages:
            lines.append(msg.pretty_print())

        lines.append(f"\n{BOLD}{'═' * 70}{RESET}")
        output = "\n".join(lines)
        print(output)
        return output

    def get_agent_messages(self, agent_name: str) -> list[Message]:
        return [m for m in self.history if m.sender == agent_name or m.receiver == agent_name]

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for msg in self.history:
            key = msg.msg_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts
