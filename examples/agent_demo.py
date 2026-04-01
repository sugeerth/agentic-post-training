#!/usr/bin/env python3
"""Demo showing agents communicating beautifully in the terminal.

Run: python3 examples/agent_demo.py
"""

import asyncio
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.communication import MessageBus, Message, MessageType
from agents.coordinator import CoordinatorAgent
from agents.training_agent import TrainingAgent
from agents.optimization_agent import OptimizationAgent
from agents.evaluation_agent import EvaluationAgent
from agents.base_agent import BOLD, RESET


async def main():
    print(f"\n{BOLD}{'╔' + '═' * 68 + '╗'}")
    print(f"║{'🤖 AGENTIC POST-TRAINING — AGENT COMMUNICATION DEMO':^68s}║")
    print(f"{'╚' + '═' * 68 + '╝'}{RESET}\n")

    # Create message bus and agents
    bus = MessageBus(verbose=True)

    coordinator = CoordinatorAgent("Coordinator")
    trainer = TrainingAgent("Trainer")
    optimizer = OptimizationAgent("Optimizer")
    evaluator = EvaluationAgent("Evaluator")

    # Register all agents on the bus
    coordinator.setup_bus(bus)
    coordinator.register_worker(trainer)
    coordinator.register_worker(optimizer)
    coordinator.register_worker(evaluator)

    print(f"{BOLD}--- Agent Registration Complete ---{RESET}\n")

    # Run the full pipeline
    config = {
        "technique": "grpo",
        "model": "gpt2",
        "epochs": 3,
        "training": {"technique": "grpo", "model": "gpt2", "epochs": 3},
        "optimization": {"method": "quantization", "quant_type": "gptq"},
        "evaluation": {"benchmarks": ["mmlu", "mt_bench", "humaneval"], "model": "gpt2"},
    }

    results = await coordinator.execute(config=config)

    # Show full conversation log
    bus.print_conversation()

    # Summary
    summary = bus.summary()
    print(f"\n{BOLD}Message Summary:{RESET}")
    for msg_type, count in sorted(summary.items()):
        print(f"  {msg_type}: {count}")
    print(f"  Total: {sum(summary.values())} messages exchanged")
    print()


if __name__ == "__main__":
    asyncio.run(main())
