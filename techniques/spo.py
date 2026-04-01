"""Self-Play Optimization (SPO).

Priority: 1 (Core technique)
Paper: "Self-Play Preference Optimization for Language Model Alignment" (Wu et al., 2024)

SPO uses self-play to iteratively improve the model. The model plays both
generator and discriminator roles, creating a competitive dynamic that drives
continuous improvement without external reward models.

When to use:
- You want iterative self-improvement
- Limited preference data available
- Want to bootstrap alignment from the model itself
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from techniques.base_technique import BaseTechnique, TechniqueConfig

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class SPOConfig(TechniqueConfig):
    num_rounds: int = 5
    beta: float = 0.1
    elo_k_factor: float = 32.0
    initial_elo: float = 1000.0
    win_rate_threshold: float = 0.55


class SPO(BaseTechnique):
    """Self-Play Optimization — iterative self-improvement through competition.

    Algorithm:
    1. Generate responses from current policy (generator)
    2. Generate responses from previous policy (opponent)
    3. Use a judge (reward model or the model itself) to determine winners
    4. Update policy to prefer winning responses
    5. Track ELO ratings across iterations
    6. Repeat until convergence (win rate stabilizes)
    """

    name = "spo"
    description = "Self-Play Optimization — iterative self-improvement via competitive self-play"
    paper_reference = "Wu et al., 2024 — Self-Play Preference Optimization"
    priority = 1
    recommended_for = ["iterative improvement", "limited data", "bootstrapping alignment"]
    pros = ["Self-improving", "No external reward model needed", "Iterative refinement"]
    cons = ["Can overfit to self", "Needs careful convergence criteria", "Computationally expensive"]

    def __init__(self, config: SPOConfig | None = None):
        super().__init__(config or SPOConfig())
        self.elo_ratings: list[float] = [self.config.initial_elo]
        self.round_results: list[dict] = []

    def compute_loss(self, **kwargs) -> Any:
        cfg = self.config

        if HAS_TORCH and "winner_log_probs" in kwargs:
            winner_log_probs = kwargs["winner_log_probs"]
            loser_log_probs = kwargs["loser_log_probs"]
            ref_winner = kwargs.get("ref_winner_log_probs", torch.zeros_like(winner_log_probs))
            ref_loser = kwargs.get("ref_loser_log_probs", torch.zeros_like(loser_log_probs))

            winner_ratio = winner_log_probs - ref_winner
            loser_ratio = loser_log_probs - ref_loser
            logits = cfg.beta * (winner_ratio - loser_ratio)
            loss = -F.logsigmoid(logits).mean()

            win_rate = (logits > 0).float().mean().item()
            new_elo = self._update_elo(win_rate)

            self.metrics.update({
                "loss": loss.item(),
                "win_rate": win_rate,
                "elo_rating": new_elo,
            })
            return loss

        # Simulation
        epoch = kwargs.get("epoch", self.step_count)
        loss = 1.2 * math.exp(-0.3 * epoch) + 0.3
        win_rate = min(0.85, 0.5 + 0.08 * epoch)
        new_elo = self._update_elo(win_rate)

        self.metrics.update({
            "loss": loss,
            "win_rate": win_rate,
            "elo_rating": new_elo,
            "reward": win_rate,
            "round": epoch,
        })
        return loss

    def _update_elo(self, win_rate: float) -> float:
        cfg = self.config
        current = self.elo_ratings[-1]
        expected = 1 / (1 + 10 ** ((cfg.initial_elo - current) / 400))
        new_elo = current + cfg.elo_k_factor * (win_rate - expected)
        self.elo_ratings.append(new_elo)
        return new_elo
