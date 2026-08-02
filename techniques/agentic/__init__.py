"""Agentic post-training techniques.

Unlike the techniques in `techniques/`, these operate on *trajectories* —
multi-turn tool-use rollouts — rather than single-turn (prompt, response)
pairs. This is the layer where post-training an **agent** differs from
post-training a base LLM.

Techniques modeled after:
  • Kimi K2 — multi-turn tool-use RL
  • DeepSeek-R1 — outcome-reward GRPO + rejection-sampling outer loop
  • Math-Shepherd / PRM800K — process reward models
  • STaR / RFT — rejection-sampling fine-tuning
"""

from techniques.agentic.multi_turn_grpo import MultiTurnGRPO
from techniques.agentic.trajectory_dpo import TrajectoryDPO
from techniques.agentic.rejection_sampling_ft import RejectionSamplingFT
from techniques.agentic.process_reward_model import ProcessRewardModel

AGENTIC_TECHNIQUES: dict[str, type] = {
    "multi_turn_grpo": MultiTurnGRPO,
    "trajectory_dpo": TrajectoryDPO,
    "rejection_sampling_ft": RejectionSamplingFT,
    "process_reward_model": ProcessRewardModel,
}

__all__ = [
    "AGENTIC_TECHNIQUES",
    "MultiTurnGRPO",
    "ProcessRewardModel",
    "RejectionSamplingFT",
    "TrajectoryDPO",
]
