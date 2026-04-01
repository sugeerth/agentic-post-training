"""Post-training techniques for LLM alignment and improvement."""

from techniques.base_technique import BaseTechnique, TechniqueConfig
from techniques.ppo import PPO, PPOConfig
from techniques.grpo import GRPO, GRPOConfig
from techniques.spo import SPO, SPOConfig
from techniques.dpo import DPO, DPOConfig
from techniques.kto import KTO, KTOConfig
from techniques.orpo import ORPO, ORPOConfig
from techniques.rlhf import RLHF, RLHFConfig
from techniques.rlaif import RLAIF, RLAIFConfig
from techniques.spin import SPIN, SPINConfig
from techniques.simpo import SimPO, SimPOConfig
from techniques.ipo import IPO, IPOConfig

TECHNIQUE_REGISTRY: dict[str, type[BaseTechnique]] = {
    "ppo": PPO,
    "grpo": GRPO,
    "spo": SPO,
    "dpo": DPO,
    "kto": KTO,
    "orpo": ORPO,
    "rlhf": RLHF,
    "rlaif": RLAIF,
    "spin": SPIN,
    "simpo": SimPO,
    "ipo": IPO,
}

# Priority groupings for details-on-demand display
PRIORITY_1 = ["ppo", "grpo", "dpo", "spo", "rlhf"]  # Most important, show first
PRIORITY_2 = ["kto", "orpo", "rlaif", "spin", "simpo"]  # Advanced
PRIORITY_3 = ["ipo"]  # Experimental

__all__ = [
    "BaseTechnique", "TechniqueConfig", "TECHNIQUE_REGISTRY",
    "PPO", "GRPO", "SPO", "DPO", "KTO", "ORPO",
    "RLHF", "RLAIF", "SPIN", "SimPO", "IPO",
    "PRIORITY_1", "PRIORITY_2", "PRIORITY_3",
]
