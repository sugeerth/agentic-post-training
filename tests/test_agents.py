"""Tests for the agent framework."""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import AgentStatus, BaseAgent
from agents.communication import Message, MessageBus, MessageType
from agents.coordinator import CoordinatorAgent
from agents.training_agent import TrainingAgent


class ConcreteAgent(BaseAgent):
    async def run(self, **kwargs):
        return {"status": "done"}

    async def step(self, **kwargs):
        return {"step": "done"}


class TestBaseAgent(unittest.TestCase):
    def test_create_agent(self):
        agent = ConcreteAgent("test", "worker")
        self.assertEqual(agent.name, "test")
        self.assertEqual(agent.role, "worker")
        self.assertEqual(agent.status, AgentStatus.IDLE)

    def test_register_capability(self):
        agent = ConcreteAgent("test", "worker")
        agent.register_capability("training", "Can train models")
        self.assertTrue(agent.has_capability("training"))
        self.assertFalse(agent.has_capability("flying"))

    def test_memory(self):
        agent = ConcreteAgent("test", "worker")
        agent.memory.set("key", "value")
        self.assertEqual(agent.memory.get("key"), "value")
        self.assertIsNone(agent.memory.get("missing"))

    def test_execute_lifecycle(self):
        agent = ConcreteAgent("test", "worker")
        result = asyncio.run(agent.execute())
        self.assertEqual(agent.status, AgentStatus.COMPLETED)
        self.assertEqual(result["status"], "done")


class TestMessageBus(unittest.TestCase):
    def test_register_agents(self):
        bus = MessageBus(verbose=False)
        agent1 = ConcreteAgent("a1", "worker")
        agent2 = ConcreteAgent("a2", "worker")
        bus.register_agent(agent1)
        bus.register_agent(agent2)
        self.assertIn("a1", bus.agents)
        self.assertIn("a2", bus.agents)

    def test_direct_message(self):
        async def _test():
            bus = MessageBus(verbose=False)
            sender = ConcreteAgent("sender", "worker")
            receiver = ConcreteAgent("receiver", "worker")
            bus.register_agent(sender)
            bus.register_agent(receiver)

            msg = Message("sender", "receiver", MessageType.TASK_REQUEST, {"data": "test"})
            await bus.publish(msg)

            received = await receiver.inbox.get()
            self.assertEqual(received.payload["data"], "test")

        asyncio.run(_test())

    def test_broadcast(self):
        async def _test():
            bus = MessageBus(verbose=False)
            a1 = ConcreteAgent("a1", "worker")
            a2 = ConcreteAgent("a2", "worker")
            a3 = ConcreteAgent("a3", "worker")
            bus.register_agent(a1)
            bus.register_agent(a2)
            bus.register_agent(a3)

            msg = Message("a1", "broadcast", MessageType.STATUS_UPDATE, {"status": "hello"})
            await bus.publish(msg)

            # a1 is the sender, so it should NOT receive the broadcast
            self.assertTrue(a1.inbox.empty())
            r2 = await a2.inbox.get()
            r3 = await a3.inbox.get()
            self.assertEqual(r2.payload["status"], "hello")
            self.assertEqual(r3.payload["status"], "hello")

        asyncio.run(_test())

    def test_history(self):
        async def _test():
            bus = MessageBus(verbose=False)
            a = ConcreteAgent("a", "worker")
            bus.register_agent(a)
            msg = Message("a", "broadcast", MessageType.HEARTBEAT, {})
            await bus.publish(msg)
            self.assertEqual(len(bus.history), 1)

        asyncio.run(_test())


class TestCoordinator(unittest.TestCase):
    def test_create_pipeline(self):
        coord = CoordinatorAgent()
        stages = coord.create_pipeline()
        self.assertEqual(len(stages), 5)
        self.assertEqual(stages[0].name, "data_prep")

    def test_register_workers(self):
        coord = CoordinatorAgent()
        trainer = TrainingAgent()
        coord.register_worker(trainer)
        self.assertIn("Trainer", coord.workers)


if __name__ == "__main__":
    unittest.main()
