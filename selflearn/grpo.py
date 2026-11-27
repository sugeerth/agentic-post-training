"""Learning: GRPO on the agent's own trajectories, with LoRA adapters on a (quantized) MLX model.

- group_advantages: (r - mean) / std inside each task's group; groups where every attempt scored the same are dropped
  (they carry zero signal). Pure Python, unit-tested.
- loss: -sum(A * log pi(token)) over model-generated tokens only, normalized by the total generated tokens.
  One update per fresh batch (on-policy), so the PPO importance ratio is 1 and clipping is not needed.
  Optional KL anchor to the base model (kl > 0): off for the math run, where it was not needed; on for the support
  agent, where two runs without it collapsed onto a single answer.
"""
from __future__ import annotations

import json
from pathlib import Path


def group_advantages(rewards: list[float], group: int, eps: float = 1e-4) -> list[float | None]:
    """Advantage per attempt, or None for attempts in a group with no spread (no learning signal)."""
    out: list[float | None] = []
    for g in range(0, len(rewards), group):
        rs = rewards[g:g + group]
        mean = sum(rs) / len(rs)
        std = (sum((r - mean) ** 2 for r in rs) / len(rs)) ** 0.5
        out += [None] * len(rs) if std < 1e-6 else [(r - mean) / (std + eps) for r in rs]
    return out


LORA_KEYS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def attach_lora(model, rank: int, scale: float) -> int:
    """Freeze the base model and add LoRA to every linear layer. LoRA B starts at zero, so the model is unchanged."""
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.utils import linear_to_lora_layers
    model.freeze()
    linear_to_lora_layers(model, len(model.layers), {"rank": rank, "scale": scale, "dropout": 0.0, "keys": LORA_KEYS})
    return sum(v.size for _, v in tree_flatten(model.trainable_parameters()))


def save_adapter(model, out: Path, model_name: str, rank: int, scale: float, usecase: str = "math") -> None:
    """Saved in mlx_lm's adapter format, so `mlx_lm.load(model, adapter_path=out)` can load it."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    (out / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora", "num_layers": len(model.layers), "model": model_name, "usecase": usecase,
        "lora_parameters": {"rank": rank, "scale": scale, "dropout": 0.0, "keys": LORA_KEYS}}, indent=1))


def token_logp(model, ids):
    """log pi(token t+1 | tokens <= t) for every position."""
    import mlx.core as mx
    logits = model(ids[:, :-1]).astype(mx.float32)
    return mx.take_along_axis(logits, ids[:, 1:, None], axis=-1).squeeze(-1) - mx.logsumexp(logits, axis=-1)


def base_logp(model, ids):
    """The same forward pass with every LoRA scale set to 0, i.e. the untouched base model (no second copy needed)."""
    import mlx.core as mx
    layers = [m for _, m in model.named_modules() if hasattr(m, "lora_a")]
    saved = [m.scale for m in layers]
    for m in layers:
        m.scale = 0.0
    try:
        out = token_logp(model, ids)
        mx.eval(out)
    finally:
        for m, sc in zip(layers, saved, strict=True):
            m.scale = sc
    return mx.stop_gradient(out)


class GRPOLearner:
    """Holds the optimizer; `step(trajs)` does one policy-gradient update on trajectories that carry an advantage.

    kl > 0 adds kl * KL(policy || base model) per generated token (the k3 estimator from the GRPO paper). It anchors the
    policy to the base model so it cannot collapse onto one answer, which two support runs without it did."""

    def __init__(self, model, lr: float, pad_id: int, micro: int = 8, max_len: int = 900, clip: float = 1.0,
                 kl: float = 0.0):
        import mlx.nn as nn
        import mlx.optimizers as optim
        self.model, self.pad, self.micro, self.max_len, self.clip, self.kl = model, pad_id, micro, max_len, clip, kl
        self.opt = optim.Adam(learning_rate=lr)
        self.value_grad = nn.value_and_grad(model, self._loss)

    @staticmethod
    def _loss(model, ids, mask, adv, norm, ref, kl):
        import mlx.core as mx
        logp = token_logp(model, ids)
        m = mask[:, 1:]
        loss = -(logp * m * adv[:, None]).sum()
        if kl > 0:
            d = ref - logp
            loss = loss + kl * ((mx.exp(d) - d - 1) * m).sum()
        return loss / norm

    def step(self, trajs) -> tuple[float, float]:
        import mlx.core as mx
        import mlx.optimizers as optim
        from mlx.utils import tree_map
        if not trajs:
            return 0.0, 0.0
        n = self.max_len
        norm = mx.array(float(max(1, sum(sum(t.mask[:n]) for t in trajs))))
        acc, total = None, 0.0
        for i in range(0, len(trajs), self.micro):
            mb = trajs[i:i + self.micro]
            L = min(n, max(len(t.ids) for t in mb))
            ids = mx.array([t.ids[:L] + [self.pad] * (L - len(t.ids[:L])) for t in mb])
            mask = mx.array([t.mask[:L] + [0] * (L - len(t.mask[:L])) for t in mb], dtype=mx.float32)
            adv = mx.array([t.adv for t in mb], dtype=mx.float32)
            ref = base_logp(self.model, ids) if self.kl > 0 else mx.zeros((1, 1))
            loss, grads = self.value_grad(self.model, ids, mask, adv, norm, ref, self.kl)
            acc = grads if acc is None else tree_map(mx.add, acc, grads)
            mx.eval(acc, loss)
            total += loss.item()
        acc, gnorm = optim.clip_grad_norm(acc, self.clip)
        self.opt.update(self.model, acc)
        mx.eval(self.model.parameters(), self.opt.state)
        return total, gnorm.item()
