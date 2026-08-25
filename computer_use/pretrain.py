"""Train the transformer on interaction tokens, and score it by execution.

The experiment this module runs, stated so it can be argued with:

> Generated applications produce searched tasks; the searched tasks produce
> verified rollouts; the rollouts tokenize into interaction sequences. If that
> chain carries real signal, a transformer trained on nothing else should be
> able to operate applications it has never seen — and the score should be the
> fraction of tasks whose *verifier* passes, not the fraction of tokens it
> guessed right.

Every example is one decision:

    <bos> instruction <obs> [x y label <sep>] x N <act> action <eos>
    ...........................................└ loss starts here ┘

Loss is charged only after `<act>`. The model is never asked to reproduce the
instruction or the screen — those are context, and spending gradient on them
would train a describer of GUIs rather than an operator of one.

Two properties of this setup are worth being explicit about, because they are
what make the result mean anything.

**The observation is in the context, so the answer is too.** Every candidate
coordinate the model could emit is present in its input, as `<x:..> <y:..>`
next to the label it belongs to. So the task is not to *know* where the button
is, it is to work out which label the instruction is asking for and copy the
coordinate beside it. That is a pointer operation, which is why two layers is
enough and why it can transfer to an application whose layout it has never
seen: nothing about the target's position is memorized.

**Scoring runs the model in the environment.** Held-out accuracy on next-token
prediction would be a much friendlier number and would not mean the same
thing — a model can be right about most tokens of an action and still click
6 pixels outside the control. Here the decoded action is executed, the episode
continues from whatever screen results, and the task's verifier decides.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace

from computer_use.nn import Adam
from computer_use.perception import parse_screen
from computer_use.tasks import GUITask
from computer_use.tokens import (
    ACT,
    BOS,
    EOS,
    OBS,
    VOCAB,
    Vocabulary,
    decode_action,
    encode_action,
    encode_screen,
)
from computer_use.transformer import GPT, ModelConfig, generate
from computer_use.types import Action, Trajectory, TrajectoryStatus

__all__ = [
    "CorpusConfig",
    "EvalResult",
    "Example",
    "TrainReport",
    "build_corpus",
    "evaluate",
    "format_report",
    "train",
]

#: Instruction and label budgets. Both truncate rather than summarize, and both
#: were set from the measured corpus (instructions run to 88 characters, labels
#: to 20) so that the *median* example is short while the longest still fits.
#: Sequence length costs quadratically in attention and linearly everywhere
#: else, and at a second per training step that is the difference between an
#: experiment and an intention.
INSTRUCTION_CHARS = 64
LABEL_CHARS = 12
MAX_ELEMENTS = 10


@dataclass(frozen=True)
class CorpusConfig:
    """How a decision becomes a sequence."""

    instruction_chars: int = INSTRUCTION_CHARS
    label_chars: int = LABEL_CHARS
    max_elements: int = MAX_ELEMENTS
    vocab: Vocabulary = VOCAB


#: The default corpus shape, as a singleton. Frozen, so sharing it is safe —
#: and a shared instance keeps `CorpusConfig()` out of argument defaults, where
#: it would be constructed once at import and read like it was per call.
DEFAULT_CORPUS = CorpusConfig()


@dataclass(frozen=True)
class Example:
    """One supervised decision, ready for the model."""

    ids: list[int]
    supervised: list[int]
    #: Where the action starts — the prefix a generator is given at eval time.
    prompt_length: int
    task: str
    world_seed: int | None = None
    action: Action | None = None

    def __len__(self) -> int:
        return len(self.ids)


def context_tokens(
    instruction: str, screen: object, config: CorpusConfig = DEFAULT_CORPUS
) -> list[str]:
    """Everything the model sees before it has to act."""
    tokens: list[str] = [BOS]
    tokens += [f"<c:{c}>" for c in instruction.upper()[: config.instruction_chars]]
    tokens += encode_screen(
        screen, limit=config.max_elements, label_chars=config.label_chars
    )
    tokens.append(ACT)
    return tokens


def make_example(
    instruction: str,
    screen: object,
    action: Action,
    *,
    config: CorpusConfig = DEFAULT_CORPUS,
    world_seed: int | None = None,
) -> Example:
    """Context plus the action that followed it, with the loss mask."""
    prompt = context_tokens(instruction, screen, config)
    tokens = [*prompt, *encode_action(action), EOS]
    ids = config.vocab.encode(tokens)
    # A position is supervised when the token *after* it is part of the answer.
    # That runs from the `<act>` marker (whose successor is the action kind)
    # through the last action token (whose successor is `<eos>`), so the model
    # learns where an action ends as well as what it contains.
    supervised = list(range(len(prompt) - 1, len(ids) - 1))
    return Example(
        ids=ids,
        supervised=supervised,
        prompt_length=len(prompt),
        task=instruction,
        world_seed=world_seed,
        action=action,
    )


def snap_to_screen(action: Action, screen: object) -> Action:
    """Move a click onto the exact point the parser reports for that control.

    The search that produced the gold path picks a pixel inside a widget; the
    parser picks the centre of the rectangle it recovered from the pixels.
    Those are usually the same 20px cell and occasionally not — measured at 10%
    of clicks on generated apps. When they differ, the target token is one the
    context does not contain, so the only way for the model to be "right" is to
    memorize a coordinate, which is precisely the thing this setup is built to
    avoid teaching.

    Both points are inside the same control, so snapping does not change what
    the action does — and `_frames` does not take that on trust, it executes
    the snapped path and keeps it only if the task's own verifier still passes.
    """
    if action.coordinate is None:
        return action
    px, py = action.coordinate
    for element in getattr(screen, "elements", ()):
        if element.box is not None and element.box.contains(px, py):
            if element.click != action.coordinate:
                return replace(action, coordinate=element.click)
            return action
    return action


async def _walk(
    task: GUITask, config: CorpusConfig, *, snap: bool
) -> tuple[list[Example], bool]:
    """Replay the gold path, recording one example per step.

    Replay rather than reuse of a stored trajectory: a click's pixel depends on
    the scroll offset at the moment it happens, so the screen a step was taken
    against only exists while walking the path.
    """
    env = task.env_factory()
    await env.reset()
    out: list[Example] = []
    for action in task.gold:
        shot = await env.screenshot()
        screen = parse_screen(shot.data)
        chosen = snap_to_screen(action, screen) if snap else action
        out.append(make_example(
            task.instruction, screen, chosen,
            config=config, world_seed=task.metadata.get("world_seed"),
        ))
        await env.execute(chosen)
    episode = Trajectory(
        task=task.instruction, steps=(), status=TrajectoryStatus.MAX_STEPS
    )
    solved = bool(task.verifier(episode, env.state()).success)
    await env.close()
    return out, solved


async def _frames(task: GUITask, config: CorpusConfig) -> list[Example]:
    """This task's steps as examples, with copyable targets where possible."""
    snapped, solved = await _walk(task, config, snap=True)
    if solved:
        return snapped
    # Snapping moved a click off its control. Rather than guess which step, fall
    # back to the search's own coordinates for the whole task: a corpus with a
    # few unpointable targets is better than one with a broken trajectory.
    original, _ = await _walk(task, config, snap=False)
    return original


def build_corpus(
    tasks: Iterable[GUITask], *, config: CorpusConfig = DEFAULT_CORPUS
) -> list[Example]:
    """Every step of every task's gold solution, as training examples."""
    async def run() -> list[Example]:
        out: list[Example] = []
        for task in tasks:
            out.extend(await _frames(task, config))
        return out

    return asyncio.run(run())


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


@dataclass
class TrainReport:
    """What happened during a run, in enough detail to plot."""

    losses: list[float] = field(default_factory=list)
    epoch_loss: list[float] = field(default_factory=list)
    held_out_loss: list[float] = field(default_factory=list)
    seconds: float = 0.0
    examples: int = 0
    parameters: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "epochs": len(self.epoch_loss),
            "epoch_loss": [round(v, 4) for v in self.epoch_loss],
            "held_out_loss": [round(v, 4) for v in self.held_out_loss],
            "seconds": round(self.seconds, 1),
            "examples": self.examples,
            "parameters": self.parameters,
        }


def mean_loss(model: GPT, examples: Sequence[Example]) -> float:
    """Average loss without touching gradients — the held-out curve."""
    if not examples:
        return float("nan")
    total = 0.0
    for example in examples:
        total += model.loss(example.ids, example.supervised).data[0]
    return total / len(examples)


def train(
    examples: Sequence[Example],
    *,
    config: ModelConfig | None = None,
    epochs: int = 6,
    batch_size: int = 8,
    lr: float = 3e-3,
    warmup: int = 30,
    seed: int = 0,
    held_out: Sequence[Example] = (),
    model: GPT | None = None,
    on_epoch: Callable[[int, TrainReport], None] | None = None,
) -> tuple[GPT, TrainReport]:
    """Fit the model, accumulating gradients across a minibatch.

    Accumulation rather than a padded batch tensor: examples here vary from 60
    to 190 tokens, and padding them to a common length would spend most of the
    arithmetic on `<pad>` — attention is quadratic in the padded length, not
    the real one.
    """
    if not examples:
        raise ValueError("nothing to train on")
    model = model or GPT(config or ModelConfig(), seed=seed)
    optimizer = Adam(model.params(), lr=lr)
    report = TrainReport(examples=len(examples), parameters=model.n_params)
    rng = random.Random(seed)
    order = list(range(len(examples)))
    started = time.perf_counter()
    total_steps = max(1, epochs * math.ceil(len(examples) / batch_size))
    step = 0

    for epoch in range(epochs):
        rng.shuffle(order)
        epoch_total = 0.0
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            model.zero_grad()
            batch_loss = 0.0
            for index in chunk:
                example = examples[index]
                loss = model.loss(example.ids, example.supervised)
                batch_loss += loss.data[0]
                loss.backward()
            # Gradients accumulated across the chunk are a *sum*; the step
            # should not get larger just because a chunk was fuller. Dividing
            # here rather than scaling each loss keeps the reported loss in
            # nats-per-token, which is the number worth reading.
            inv = 1.0 / len(chunk)
            for p in model.params():
                if p.grad is not None:
                    p.grad = [g * inv for g in p.grad]
            step += 1
            # Linear warmup then cosine decay: at this scale the first few
            # steps of an un-warmed Adam are large enough to matter for where
            # the run ends up, and cosine costs nothing to add.
            if step <= warmup:
                rate = lr * step / max(1, warmup)
            else:
                progress = (step - warmup) / max(1, total_steps - warmup)
                rate = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))
            optimizer.step(lr=rate)
            report.losses.append(batch_loss / len(chunk))
            epoch_total += batch_loss
        report.epoch_loss.append(epoch_total / len(order))
        if held_out:
            report.held_out_loss.append(mean_loss(model, held_out))
        report.seconds = time.perf_counter() - started
        if on_epoch is not None:
            on_epoch(epoch, report)
    return model, report


# --------------------------------------------------------------------------- #
# Evaluation by execution
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvalResult:
    """How a model did when it was actually put in front of the applications."""

    solved: int
    total: int
    action_matches: int
    action_total: int
    invalid: int
    label: str = ""
    detail: list[tuple[str, bool]] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.solved / self.total if self.total else 0.0

    @property
    def action_accuracy(self) -> float:
        return self.action_matches / self.action_total if self.action_total else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "solved": self.solved,
            "total": self.total,
            "rate": round(self.rate, 4),
            "action_accuracy": round(self.action_accuracy, 4),
            "invalid_actions": self.invalid,
        }


def decode_generated(ids: Sequence[int], *, vocab: Vocabulary = VOCAB) -> Action | None:
    """The action a generation describes, or None if it is not one."""
    tokens = [t for t in vocab.decode(ids) if t not in (EOS, BOS, ACT, OBS)]
    if not tokens:
        return None
    try:
        return decode_action(tokens)
    except ValueError:
        # An action kind the vocabulary knows but the enum does not accept, or
        # a malformed span. Counted as an invalid action rather than raised:
        # producing nonsense is a way for a policy to fail, not for the
        # harness to crash.
        return None


async def _run_task(
    model: GPT,
    task: GUITask,
    *,
    config: CorpusConfig,
    step_budget: int,
) -> tuple[bool, int, int, int]:
    """Drive one task to completion. Returns (solved, matched, steps, invalid)."""
    env = task.env_factory()
    await env.reset()
    gold = list(task.gold)
    matched = invalid = 0
    steps = 0
    stop = VOCAB.id(EOS)

    def solved() -> bool:
        # Verifiers take the episode as well as the state, because some of them
        # judge how a goal was reached. Synthesized tasks all use
        # `StateVerifier`, which reads only the state, but passing a real
        # trajectory keeps this callable with any verifier in the package.
        episode = Trajectory(
            task=task.instruction, steps=(), status=TrajectoryStatus.MAX_STEPS
        )
        return bool(task.verifier(episode, env.state()).success)

    for index in range(step_budget):
        shot = await env.screenshot()
        prompt = config.vocab.encode(
            context_tokens(task.instruction, parse_screen(shot.data), config)
        )
        if len(prompt) >= model.config.max_len:
            break
        produced = generate(model, prompt, max_new=24, stop=(stop,))
        action = decode_generated(produced)
        steps += 1
        if action is None:
            invalid += 1
            break
        if index < len(gold) and action == gold[index]:
            matched += 1
        try:
            await env.execute(action)
        except Exception:
            # The environment refused the action — an out-of-range coordinate,
            # a key it does not know. Same treatment as a malformed decode.
            invalid += 1
            break
        if solved():
            await env.close()
            return True, matched, steps, invalid
    done = solved()
    await env.close()
    return done, matched, steps, invalid


def evaluate(
    model: GPT,
    tasks: Sequence[GUITask],
    *,
    config: CorpusConfig = DEFAULT_CORPUS,
    slack: int = 2,
    label: str = "",
) -> EvalResult:
    """Run the model in each application and let the verifiers decide.

    `slack` is how many steps beyond optimal the model is allowed. Two, not
    zero: a policy that recovers from a wrong click is better than one that
    cannot, and a budget of exactly `optimal_steps` scores them the same.
    """
    async def run() -> EvalResult:
        solved = matched = total_actions = invalid = 0
        detail: list[tuple[str, bool]] = []
        for task in tasks:
            ok, hits, _, bad = await _run_task(
                model, task, config=config,
                step_budget=task.optimal_steps + slack,
            )
            solved += int(ok)
            matched += hits
            invalid += bad
            total_actions += len(task.gold)
            detail.append((task.name, ok))
        return EvalResult(
            solved=solved, total=len(tasks), action_matches=matched,
            action_total=total_actions, invalid=invalid, label=label, detail=detail,
        )

    return asyncio.run(run())


def format_report(
    report: TrainReport, results: Sequence[EvalResult], *, width: int = 30
) -> str:
    """The run as a block of text, with the loss curve drawn in place."""
    lines: list[str] = []
    lines.append(
        f"  {report.parameters:,} parameters, {report.examples} decisions, "
        f"{len(report.epoch_loss)} epochs in {report.seconds / 60:.1f} min"
    )
    if report.epoch_loss:
        lines.append("")
        lines.append("  epoch    train    held-out")
        top = max(report.epoch_loss)
        for index, value in enumerate(report.epoch_loss):
            held = (
                f"{report.held_out_loss[index]:8.3f}"
                if index < len(report.held_out_loss) else "        "
            )
            bar = "#" * max(1, round(width * value / top)) if top else ""
            lines.append(f"  {index + 1:5d} {value:8.3f} {held}  {bar}")
    if results:
        lines.append("")
        lines.append(f"  {'':28s}   solved        actions")
        for result in results:
            lines.append(
                f"  {result.label:28s} {result.solved:3d}/{result.total:<3d} "
                f"{result.rate:5.0%}   {result.action_accuracy:5.0%}"
            )
    return "\n".join(lines)
