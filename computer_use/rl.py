"""Post-training the GUI policy on its own verified rollouts.

This is the claim the whole package was built to support, and until now it
was the one thing never run:

> A GUI agent produces attempts whose outcome can be *checked*. Every episode
> is a verified success or a verified failure — exactly the supervision that
> preference and RL methods need, and exactly what a text corpus cannot give.

Everything upstream exists to serve this. `worlds` generates applications,
`synthesis` searches tasks out of them, `perception` reads the screen back
from pixels, `inference` samples k rollouts and reports the log-probability of
every token it drew. What was missing was a trainer that consumes them, and
the reason it was missing is that `techniques.grpo` needs torch. This package
has its own autograd, so it does not.

**Why RL rather than more supervised fine-tuning.** SFT maximizes the
likelihood of the gold token sequence. It never observes a wrong action, so
it cannot be pushed away from one — the model only ever learns what the right
answer looked like, never what a near-miss costs. The diagnosis says that is
precisely where the policy fails: it emits a plausible action kind and then a
coordinate that is eleven cells from the target, and nothing in the
supervised objective ever charged it for that. GRPO samples several actions
per decision, scores each against the environment, and moves probability from
the ones that missed toward the ones that landed. Wrong answers become
training signal for the first time.

**The estimator, and why the group is the unit.** GRPO normalizes a sample's
reward against the other samples of the same decision, which is what lets it
drop the value network. A group whose members all scored alike therefore
carries no gradient — correctly, since it contains no evidence that any one
of them was better. `degenerate_groups` in the report is how much of a round
was spent on exactly that, and it is the number to watch: a policy that has
collapsed to one action produces nothing but degenerate groups and a training
curve that looks calm.

**One update per batch of samples.** The importance ratio is then exactly 1
at the moment the gradient is taken, so the clipped surrogate reduces to the
plain policy-gradient objective and there is nothing to clip. That is a real
choice rather than a simplification: multiple inner epochs would make the
data off-policy against the very weights being updated, and the correction
for that is the machinery in `inference.version` — worth having, not worth
paying for at this scale.
"""

from __future__ import annotations

import random
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from computer_use import nn
from computer_use.nn import Adam
from computer_use.pretrain import (
    DEFAULT_CORPUS,
    CorpusConfig,
    Example,
    decode_generated,
    same_action,
)
from computer_use.tokens import EOS, VOCAB, Vocabulary
from computer_use.transformer import GPT
from inference.engine import LocalEngine, Request
from inference.runtime import PolicyWeights

__all__ = [
    "DEFAULT_RL",
    "RLConfig",
    "RLReport",
    "Round",
    "format_report",
    "reward_for",
    "train_rl",
]


@dataclass(frozen=True)
class RLConfig:
    """How a round of post-training is run."""

    group_size: int = 6
    rounds: int = 8
    #: Decisions sampled per round. The whole corpus every round would be
    #: honest and slow; a fresh random subset keeps rounds short without
    #: training repeatedly on the same states.
    decisions_per_round: int = 24
    max_new: int = 8
    temperature: float = 1.0
    lr: float = 3e-4
    #: Charged for an action that does not decode at all. Small and negative:
    #: producing nonsense is worse than producing a wrong action, but only
    #: slightly, and a large penalty here teaches the model to stay silent.
    malformed_penalty: float = -0.1
    seed: int = 0
    vocab: Vocabulary = VOCAB
    corpus: CorpusConfig = DEFAULT_CORPUS


#: The default run shape, as a singleton — frozen, so sharing it is safe, and
#: a shared instance keeps `RLConfig()` out of argument defaults where it would
#: be built once at import and read like it was built per call.
DEFAULT_RL = RLConfig()


def reward_for(
    tokens: Sequence[int],
    example: Example,
    *,
    config: RLConfig,
) -> float:
    """What one sampled action earned, judged against ground truth.

    The gold action came out of a verified search, so agreeing with it at grid
    resolution is a ground-truth check rather than a proxy — the same test the
    evaluation uses, applied to a sample the policy chose freely instead of to
    one it was told to reproduce.

    That difference is the point. Under SFT this example contributes only the
    gold tokens; here it contributes the gold tokens *and* whatever the policy
    actually did, with the gap between them priced.
    """
    gold = example.action
    if gold is None:
        return 0.0
    action = decode_generated(tokens, vocab=config.vocab, marks=example.marks)
    if action is None:
        return config.malformed_penalty
    return 1.0 if same_action(action, gold) else 0.0


@dataclass
class Round:
    """What one round of sampling and updating produced."""

    index: int
    reward: float
    solved: int
    samples: int
    degenerate: int
    loss: float
    prefix_saved: int
    seconds: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "round": self.index,
            "reward": round(self.reward, 4),
            "solved": self.solved,
            "samples": self.samples,
            "degenerate_groups": self.degenerate,
            "loss": round(self.loss, 4),
            "prefix_tokens_saved": self.prefix_saved,
            "seconds": round(self.seconds, 1),
        }


@dataclass
class RLReport:
    """The run, in enough detail to plot and to argue with."""

    rounds: list[Round] = field(default_factory=list)
    groups_total: int = 0
    groups_degenerate: int = 0

    @property
    def degenerate_share(self) -> float:
        """Share of groups that carried no gradient at all.

        Rising toward 1 means the policy has collapsed onto one action per
        state: every sample in a group agrees, so every advantage is zero and
        the run is spending its entire budget learning nothing. The loss curve
        stays perfectly calm while this happens.
        """
        if not self.groups_total:
            return 0.0
        return self.groups_degenerate / self.groups_total

    def to_dict(self) -> dict[str, object]:
        return {
            "rounds": [r.to_dict() for r in self.rounds],
            "groups": self.groups_total,
            "degenerate_groups": self.groups_degenerate,
            "degenerate_share": round(self.degenerate_share, 4),
        }


def _advantages(rewards: Sequence[float]) -> list[float]:
    """Reward centred on its group and scaled by the group's spread.

    Zero when the group is unanimous, which is the correct answer and not an
    edge case to paper over: identical rewards are no evidence that any sample
    was better than any other.
    """
    if len(rewards) < 2:
        return [0.0] * len(rewards)
    spread = statistics.pstdev(rewards)
    if spread == 0.0:
        return [0.0] * len(rewards)
    mean = statistics.fmean(rewards)
    return [(r - mean) / spread for r in rewards]


def train_rl(
    model: GPT,
    corpus: Sequence[Example],
    *,
    config: RLConfig = DEFAULT_RL,
    on_round: object = None,
) -> tuple[GPT, RLReport]:
    """Post-train `model` by sampling from it and scoring what it produces.

    The model is both the thing being sampled and the thing being updated, so
    the weights are snapshotted for the sampler each round. That snapshot is a
    copy, which is what makes every sample in a round attributable to one
    policy version — see `inference.version` for why that matters more than it
    looks.
    """
    rng = random.Random(config.seed)
    opt = Adam(model.params(), lr=config.lr)
    stop = (config.vocab.id(EOS),)
    report = RLReport()
    usable = [e for e in corpus if e.action is not None]
    if not usable:
        raise ValueError("no examples with a gold action to score against")

    for index in range(config.rounds):
        started = time.monotonic()
        weights = PolicyWeights.snapshot(model, version=index)
        engine = LocalEngine(weights, seed=config.seed + index)

        batch = rng.sample(usable, min(config.decisions_per_round, len(usable)))
        rows: list[tuple[Example, list[int], float]] = []
        total_reward = 0.0
        solved = degenerate = 0

        for example in batch:
            prompt = list(example.ids[: example.prompt_length])
            if len(prompt) + config.max_new > model.config.max_len:
                continue
            requests = [
                Request(
                    prompt=prompt, max_new=config.max_new, stop=stop,
                    temperature=config.temperature, tag=f"{index}",
                )
                for _ in range(config.group_size)
            ]
            completions = engine.generate(requests)
            rewards = [reward_for(c.tokens, example, config=config) for c in completions]
            total_reward += sum(rewards)
            solved += sum(1 for r in rewards if r >= 1.0)
            report.groups_total += 1
            if len(set(rewards)) <= 1:
                degenerate += 1
                report.groups_degenerate += 1
                continue
            for completion, advantage in zip(
                completions, _advantages(rewards), strict=True
            ):
                if advantage != 0.0 and completion.tokens:
                    rows.append((example, completion.tokens, advantage))

        loss = _update(model, opt, rows, config) if rows else 0.0
        samples = len(batch) * config.group_size
        report.rounds.append(Round(
            index=index + 1,
            reward=total_reward / max(1, samples),
            solved=solved,
            samples=samples,
            degenerate=degenerate,
            loss=loss,
            prefix_saved=engine.cache.stats.tokens_saved,
            seconds=time.monotonic() - started,
        ))
        if callable(on_round):
            on_round(index, report)
    return model, report


def _update(
    model: GPT,
    opt: Adam,
    rows: Sequence[tuple[Example, list[int], float]],
    config: RLConfig,
) -> float:
    """One gradient step over everything sampled this round.

    Each sampled sequence is replayed through the model to get the logits at
    the positions its own tokens were drawn at — the sampler's log-probs are
    what the reward was earned under, and are correct for the ratio, but a
    gradient needs the graph, and the sampler deliberately does not build one.
    """
    model.zero_grad()
    total = 0.0
    charged = 0
    for example, tokens, advantage in rows:
        prompt = list(example.ids[: example.prompt_length])
        ids = [*prompt, *tokens]
        if len(ids) > model.config.max_len:
            continue
        # Position i predicts token i+1, so the states that produced the
        # sampled tokens start at the last prompt position.
        positions = list(range(len(prompt) - 1, len(ids) - 1))
        logits = model.logits(ids, positions)
        loss = nn.policy_gradient(
            logits, tokens, [advantage] * len(tokens)
        )
        loss.backward()
        total += loss.data[0]
        charged += 1
    if charged:
        opt.step()
    return total / charged if charged else 0.0


def format_report(report: RLReport) -> str:
    """The run as a table, with the number that says whether it is learning."""
    lines = ["", "  round   reward   solved   degenerate   loss"]
    for r in report.rounds:
        lines.append(
            f"  {r.index:>5}   {r.reward:6.3f}   {r.solved:>3}/{r.samples:<4}"
            f"   {r.degenerate:>4}        {r.loss:7.3f}"
        )
    lines.append("")
    lines.append(
        f"  {report.degenerate_share * 100:.0f}% of groups were unanimous and "
        "carried no gradient"
    )
    return "\n".join(lines)
