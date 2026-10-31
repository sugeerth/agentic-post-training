# Agentic Post-Training Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sugeerth/agentic-post-training/blob/main/notebooks/agentic_post_training_colab.ipynb)
[![GitHub Pages](https://img.shields.io/badge/Demo-GitHub%20Pages-blue)](https://sugeerth.github.io/agentic-post-training/)

**Orchestrating LLM alignment through autonomous agent collaboration.**

A modular framework where specialized AI agents coordinate to execute post-training pipelines — from technique selection through training, optimization, and evaluation. Agents communicate via a structured message bus, making the entire process observable and debuggable.

## Architecture

```
                        ┌─────────────────┐
                        │   Coordinator   │
                        │     Agent       │
                        └────────┬────────┘
                                 │ Message Bus
        ┌──────────────┬─────────┴────┬──────────────┐
        │              │              │              │
┌───────▼────────┐ ┌───▼──────────┐ ┌─▼────────────┐ ┌▼───────────────┐
│ Training Agent │ │ Optimization │ │  Evaluation  │ │ Computer-Use   │
│                │ │    Agent     │ │    Agent     │ │    Agent       │
│ PPO,GRPO,DPO,  │ │ Quantization │ │ MMLU,MT-Bench│ │ VLM GUI        │
│ SPO,RLHF,...   │ │ Pruning,     │ │ HumanEval,...│ │ rollouts →     │
│                │ │ Distillation │ │              │ │ training data  │
└────────────────┘ └──────────────┘ └──────────────┘ └────────────────┘
```

## Supported Techniques

| Priority | Technique | Description | Key Advantage |
|----------|-----------|-------------|---------------|
| 1 | **PPO** | Proximal Policy Optimization | Stable RL with clipped objectives |
| 1 | **GRPO** | Group Relative Policy Opt. | No value model needed (DeepSeek-R1) |
| 1 | **DPO** | Direct Preference Opt. | Simple, no reward model needed |
| 1 | **SPO** | Self-Play Optimization | Iterative self-improvement |
| 1 | **RLHF** | RL from Human Feedback | Full proven pipeline (InstructGPT) |
| 2 | **KTO** | Kahneman-Tversky Opt. | Works with binary feedback |
| 2 | **ORPO** | Odds Ratio Preference Opt. | Combined SFT + alignment |
| 2 | **RLAIF** | RL from AI Feedback | No human labelers needed |
| 2 | **SPIN** | Self-Play Fine-Tuning | Only needs SFT data |
| 2 | **SimPO** | Simple Preference Opt. | Reference-free, length-normalized |
| 3 | **IPO** | Identity Preference Opt. | Regularized DPO variant |

## Demos

| Demo | Platform | Link |
|------|----------|------|
| **Interactive Browser Demo** | GitHub Pages | [Launch Demo](https://sugeerth.github.io/agentic-post-training/demo.html) |
| **Full Training on A100** | Google Colab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sugeerth/agentic-post-training/blob/main/notebooks/agentic_post_training_colab.ipynb) |
| **Multi-GPU Training (T4x2)** | Kaggle | [Open in Kaggle](https://www.kaggle.com/kernels/welcome?src=https://github.com/sugeerth/agentic-post-training/blob/main/notebooks/agentic_post_training_kaggle_t4x2.ipynb) |
| **Agent Communication** | Local terminal | `python3 examples/agent_demo.py` |
| **Computer Use → Training Data** | Local terminal | `python3 examples/computer_use_demo.py` |

## Quick Start

```bash
# Install
pip install -e .

# Run the agent demo (no GPU needed)
python3 examples/agent_demo.py

# Run full pipeline
python3 examples/run_pipeline.py --technique grpo --model gpt2 --epochs 3

# Compare techniques
python3 examples/compare_techniques.py --techniques ppo dpo grpo
```

### Python API

```python
import asyncio
from pipeline.config import PipelineConfig
from pipeline.pipeline import AgenticPipeline

config = PipelineConfig(
    technique="grpo",
    model_name="gpt2",
    epochs=3,
    quantization="gptq",
    benchmarks=["mmlu", "mt_bench", "humaneval"],
)

pipeline = AgenticPipeline(config)
results = asyncio.run(pipeline.run())
```

## Pipeline Stages

1. **Data Preparation** — Validate and preprocess training data
2. **Technique Selection** — Choose optimal technique for the task
3. **Training** — Execute post-training with the selected technique
4. **Optimization** — Quantize (GPTQ/AWQ/GGUF), prune, or distill
5. **Evaluation** — Benchmark on MMLU, MT-Bench, HumanEval, etc.

## Computer Use with Vision Models

A GUI agent produces something a text corpus cannot: attempts at a task whose
outcome can be **checked**. Every episode is a verified success or a verified
failure, which is exactly the supervision that preference and RL methods need.
That is why computer use lives in a post-training framework and not beside one.

```
task ──▶ screenshot ──▶ VLM ──▶ action ──▶ environment ──┐
          ▲                                              │
          └──────────────── new screenshot ◀─────────────┘
                                 │
                          verifier (ground truth)
                                 │
                        shaped reward per trajectory
                                 │
    ┌────────────────────────────┼─────────────────────────────┐
    ▼                            ▼                             ▼
PreferencePair              RolloutBatch                TrainingExample
(DPO/ORPO/SimPO)            (GRPO/PPO)                  (SFT / rejection)
```

Run the whole loop with no API key, no GPU, and no browser:

```bash
python3 examples/computer_use_demo.py     # or: make demo-gui
```

The offline demo runs three agents of deliberately different quality against
the same task and shows the training data that falls out of the spread between
them — which is the point, since a preference pair needs a better attempt and a
worse one.

```python
import asyncio
from computer_use import MockComputer, StateVerifier, run_episode
from computer_use import ClaudeComputerUsePolicy

env = MockComputer.settings_form()

trajectory = asyncio.run(run_episode(
    task="Set the email to ada@example.com, enable notifications, and save.",
    env=env,
    policy=ClaudeComputerUsePolicy(env.width, env.height),
    # Ground truth, not a judge: the reward is an exact state check.
    verifier=StateVerifier({"email": "ada@example.com", "notify": True, "saved": True}),
))

print(trajectory.summary())
# [success] 'Set the email to ...' — 6 steps (5 effective), reward 1.000
```

### From rollouts to training data

```python
from computer_use import run_group, to_preference_pairs, to_rollout_batch
from techniques.grpo import GRPOTechnique

# Independent attempts at one task — a GRPO group.
group = asyncio.run(run_group(task, MockComputer.settings_form,
                              lambda env: ClaudeComputerUsePolicy(env.width, env.height),
                              group_size=8, verifier=verifier))

pairs = to_preference_pairs(group)          # → DPO, ORPO, SimPO, KTO
batch = to_rollout_batch({task: group})     # → GRPO, PPO, RLHF

technique = GRPOTechnique()
technique.prepare(model, tokenizer, cfg)
metrics = technique.step(batch)             # no adapter — the shapes line up
```

### The benchmark

Eight verified tasks across three applications, three difficulty tiers, and a
working reference solution for every one of them.

```bash
agentic-gui tasks                          # what's in the suite
agentic-gui bench --policy noisy           # or: make bench-gui
agentic-gui bench --policy claude --attempts 8
agentic-gui collect --out data.jsonl       # episodes → training data
```

```
  task                 diff       pass   reward  steps  ground  pass@k
  ──────────────────────────────────────────────────────────────────────────
  settings.notify      easy    4/8        0.562    3.0    0.75  █████░░░░░ @1:0.50 @8:1.00
  settings.email       medium  6/8        0.781    5.0    0.87  ███████░░░ @1:0.75 @8:1.00
  settings.retries     medium  6/8        0.785    4.0    0.88  ███████░░░ @1:0.75 @8:1.00
  checkout.express     hard    6/8        0.759    4.0    0.81  ███████░░░ @1:0.75 @8:1.00
  checkout.promo       hard    6/8        0.768    5.0    0.81  ███████░░░ @1:0.75 @8:1.00
  files.select         easy    5/8        0.625    1.0    0.62  ██████░░░░ @1:0.62 @8:1.00
  files.delete         medium  4/8        0.569    3.0    0.79  █████░░░░░ @1:0.50 @8:1.00
  files.rename         hard    4/8        0.576    5.0    0.81  █████░░░░░ @1:0.50 @8:1.00
  ──────────────────────────────────────────────────────────────────────────
  Overall 41/64 (64.1%)  95% CI [51.8%, 74.7%]  mean reward +0.678
  pass@k  @1: 0.641   @2: 0.888   @4: 0.995   @8: 1.000
  By difficulty  easy: 56%   medium: 67%   hard: 67%
```

Three deliberate choices:

- **Every task ships a reference solution**, and a test asserts each one still
  scores a perfect 1.000. A benchmark whose gold solutions have rotted reports
  failures that belong to the harness and blames the agent.
- **pass@k uses the unbiased estimator**, not "did any of my *n* samples pass".
  The distance between pass@1 and pass@8 is the most informative number here: a
  wide gap means the agent knows the task but executes it unreliably — usually
  grounding, not planning.
- **Wilson score intervals**, because a suite run is a few dozen episodes. At
  32/32 the normal approximation reports an interval above 1.0; Wilson reports
  `[89.3%, 100%]`, which is the honest answer.

`gui_bench` is registered in the evaluator registry, so the pipeline reaches it
by name — and it's the framework's first evaluator whose score is a measurement
rather than a simulation, and the first to populate `EvalResult.ci_low/ci_high`.

```python
from core.registry import get_evaluator

result = get_evaluator("gui_bench")(attempts=8).evaluate("claude-opus-5")
print(result.value, result.ci_low, result.ci_high)
```

### What's in the box

| Piece | What it does |
|-------|--------------|
| `MockComputer` | Deterministic widget GUI across three apps — typing, toggles, radio groups, scrolling, and multi-screen navigation. Renders real PNG frames (legible to an actual VLM) and exposes ground-truth state. Zero dependencies. |
| `PlaywrightComputer` | A real browser, same `ComputerEnvironment` protocol. Verified end to end against Chromium — all 16 action kinds, X11→Playwright key chords, and a full scored episode. |
| `ClaudeComputerUsePolicy` | Drives Claude through the computer-use tool. Prunes stale frames, caches the system prompt, opts into refusal fallbacks. |
| `ScriptedPolicy` | Offline baseline and gold-trajectory recorder. |
| `NoisyPolicy` | Reference solution plus grounding noise — an imperfect baseline that generates real failure trajectories with zero API calls. |
| `StateVerifier` | Exact-match verification against ground truth — no judge in the reward path. |
| `score_trajectory` | Shaped reward: success, efficiency, grounding, redundancy, invalid actions. |
| `tasks.SUITE` | Eight verified benchmark tasks, each with a reference solution. |
| `synthesis.*` | Derives tasks by searching the environment — provably-optimal gold, measured difficulty, unlimited supply. |
| `worlds.*` | Generates the *applications* — seeded screens, fields, and layouts — so whole apps can be held out instead of only tasks. |
| `perception.*` | Reads a frame back out of its pixels: PNG decode, font-matched text, control detection. 271/271 controls recovered across 43 apps. |
| `learn.*` | Trains a grounding policy on the pipeline's own rollouts and scores it on generated apps it has never seen. Pixels only, pure Python. |
| `evolve.*` | Self-improvement: practise on undemonstrated apps, keep what the verifier passes, and report what the loop can see beside what is true. |
| `tokens.*` | Interactions as a token stream a transformer can read and write — 3 tokens per click, padded batches, optional torch tensors. |
| `nn.*` | A reverse-mode autograd at matrix granularity — 16 ops, every one finite-difference checked. No numpy, no torch. |
| `transformer.*` | A decoder-only transformer over interaction tokens: ~56k parameters, two layers, tied output. |
| `pretrain.*` | Builds the corpus, trains the model, and scores it by executing what it generates against the task's own verifier. |
| `inference.*` | The sampling half of the loop: KV-cached decoding, radix prefix reuse across a GRPO group, policy-version stamping, and rollouts assembled into `RolloutBatch`. |
| `diagnose.*` | Brackets a score from both sides: whether the answer was in the context at all, what policies that do not learn get on the same decisions, and which field of a prediction was wrong. |
| `metrics.*` | Unbiased pass@k and Wilson score intervals. |
| `dataset.*` | Trajectories → `PreferencePair` / `RolloutBatch` / `TrainingExample`. |
| `ComputerUseAgent` | The whole thing as an agent on the message bus. |
| `agentic-gui` | CLI: `tasks`, `bench`, `collect`, `learn`, `evolve`, `pretrain` — over the curated suite, `--synthetic` tasks, or `--worlds` apps. |

### Inference infrastructure

Post-training an agent spends most of its wall clock **generating rollouts**,
not computing gradients. A GRPO step needs *k* trajectories per prompt, each a
multi-turn episode against an environment. Until now this repo had the
gradient half and not the sampling half: `RolloutBatch` arrived fully formed
and where it came from was out of scope. `inference/` is that other half.

**The sampler must agree with its trainer, exactly.** `transformer.generate`
recomputes the whole sequence for every token it emits, and allocates an
autograd graph on a path that never calls `backward()`. The inference runtime
computes each position's keys and values once and carries no graph. The bar is
not that it is faster — it is that it is faster *and indistinguishable*:

```
  logits, inference path vs GPT.logits    214/214 bit-identical
  max |difference|                        0.000e+00
  greedy decode, 12 real GUI decisions    12/12 identical tokens
  speed, same decisions                   26.7s → 4.4s     6.1x
```

A sampler that merely *rounds differently* from its trainer generates rollouts
for a policy that does not exist, and nothing downstream would fail — the loss
still falls, the reward still rises. So `tests/test_inference.py` asserts `==`,
not `approx`.

**A GRPO group is one prompt sampled *k* times.** That is 87.5% redundant work
at group size 8, and a radix trie over token ids removes it. Measured on the
generated GUI corpus:

```
  GRPO group, 57-token prompt sampled 8x
    prompt tokens requested        456
    actually computed               57
    saved by prefix reuse          399   87.5%

  40 multi-turn decisions from 8 apps
    saved by prefix reuse          992   33.0%
    requests finding a usable prefix     70%
```

Reuse is **block-aligned**, at 16 positions, for the reason a paged engine's
is: storing a snapshot at every depth is quadratic memory to save linear work.
An identical prompt reuses all of it; a divergent branch lands on the deepest
block boundary inside what it shares. Storing only at the terminal node — the
first thing I wrote — reports a 0% hit rate on exactly the multi-turn case
this exists for, and a test now says so.

**On-policy is a property of the sample, not of the wall clock.** GRPO's
advantage is only meaningful relative to the policy that drew the sample. The
moment a trainer steps, every rollout still in flight came from a policy that
no longer exists — and nothing crashes. The loss falls, the curve rises, and
the run optimizes something other than the objective. So every completion is
stamped with the version that produced it, weight snapshots are copies rather
than references, and a batch mixing versions raises `StalenessError` unless
the caller passes `max_staleness` to say the drift is intended.

**Two bugs worth naming, both found by running it.**

*The cache reported hits it did not have.* Covered above — terminal-only
storage, silently zero on divergent branches.

*Ragged log-probs cannot become a tensor.* Samples in a group stop at
different lengths — `[12, 12, 12, 12, 12, 12, 12, 7]` is a normal group — and
GRPO reads `log_probs` as `(B, G, T)`. Unpadded that raises inside the trainer,
on a GPU machine, after the expensive sampling is already paid for. Padding is
`0.0`, which is a *log*-probability of 1.0, so a sum without the mask hands
free probability mass to whichever samples stopped early. `response_mask` and
`response_lengths` ship in the batch metadata for that reason.

**What is verified and what is not.** The runtime, cache, versioning, admission
control and batch assembly all run here, in CI, with no GPU — 31 tests. What is
*not* verified is the far side of the seam: torch is not installed in this
environment, so `GRPOTechnique.step()` cannot take its real path and the batch
is checked for shape rather than consumption. A vLLM or SGLang engine
implements the same `Engine` protocol, and neither has been run.

```python
from inference import LocalEngine, PolicyWeights, GroupSpec, sample_group, to_rollout_batch

engine = LocalEngine(PolicyWeights.snapshot(model, version=trainer.step))
group = sample_group(engine, GroupSpec(prompt=ids, size=8, temperature=0.9), verifier)
batch, report = to_rollout_batch([group])     # -> techniques.grpo, with log-probs
```

### Reward design

The shaped reward keeps two properties that are easy to lose when retuning
weights, and both are enforced by tests:

- **Successes never saturate.** Weights sum to exactly the clip ceiling, so a
  flawless run scores 1.0 and every other success lands strictly below it.
  Overflowing weights would collapse all successes to the same number and
  silently delete the shaping signal.
- **Success dominates.** No amount of efficient, well-aimed failing outranks a
  clumsy success.

A step-level breakdown lands in `trajectory.metadata["reward_breakdown"]`, so a
regression in the aggregate traces to the term that moved.

### Synthesized tasks: stop writing the benchmark, search for it

A hand-written benchmark has three problems no amount of care fixes: the tasks
don't scale, the gold solutions rot, and `optimal_steps` — the number
efficiency is scored against — is a human guess about something the
environment actually knows. All three come from one root: the task is authored
*outside* the environment, so nothing keeps them in agreement.

`computer_use.synthesis` inverts it. It breadth-first searches the
environment's reachable state space and reads tasks back out:

```
initial state ──BFS over abstract actions──▶ every reachable state
                                                    │
                      for each reachable state, the shortest path to it
                                                    │
    ┌───────────────────────────────────────────────┼──────────────────┐
    ▼                        ▼                      ▼                  ▼
instruction            verifier                   gold           optimal_steps
(from the delta)  (the goal state itself)   (the BFS path)    (its length — minimal)
```

```bash
agentic-gui tasks --synthetic          # tasks nobody wrote
agentic-gui bench --synthetic --policy noisy --attempts 8
```

```
  synth.settings.02    medium   4 steps  [click, typing, synthetic]
    Set EMAIL to "ada@example.com" and set MAX RETRIES to "5".
  synth.checkout.02    hard     4 steps  [click, navigate, synthetic]
    Choose EXPRESS - 1 DAY - 15.00 and press PLACE ORDER.
```

What this buys is structural rather than a matter of discipline:

- **Gold can never rot.** It's derived from the environment, not written beside
  it. Change a layout and the next run produces correct solutions — there is no
  second artifact to fall out of sync.
- **`optimal_steps` is provably minimal**, because BFS finds the shortest path.
  A test proves it by re-searching with one step less budget and confirming the
  goal is unreachable. A hand-written benchmark cannot make this claim.
- **Difficulty is measured, not labeled** — search depth is ground truth. It
  shows up in the results: easy 88%, medium 75%, hard 61% for the same agent.
- **Tasks become unlimited and hold-out-able.** A benchmark that is a generator
  over a seed lets you hold out whole *environments* — see the next section,
  which generates the environments too.

The search is given no hints. It discovers on its own that SAVE sits below the
fold and must be scrolled to first — the dependency is found, not encoded.

Synthesis and the curated suite cross-validate each other: a test asserts every
hand-written goal is rediscovered by search within its declared step budget. If
a hand-guessed `optimal_steps` were wrong, that test would say so.

### Generated worlds: hold out the application, not the task

Synthesis removes the human from writing tasks, but it still searches
environments a human wrote — so "held out" could only ever mean a fresh task on
a familiar screen. An agent that has memorized where SAVE lives scores the same
as one that can find it. `computer_use.worlds` closes that: it generates the
application first, and synthesis then searches *that*.

```bash
agentic-gui tasks --worlds 8                     # apps that did not exist a second ago
agentic-gui bench --worlds 8 --held-out          # or: make bench-gui-worlds
```

```
6 task(s) synthesized in 3 generated training app(s): seeds 0–2

  world000.00          medium   2 steps  [click, navigate, synthetic]
    Choose CSV.
  world000.01          hard     3 steps  [click, navigate, synthetic]
    Choose CSV and turn on SHARE USAGE DATA.
  world001.00          easy     2 steps  [click, scroll, synthetic]
    Press APPLY.
  world001.01          medium   3 steps  [click, typing, synthetic]
    Set CITY to "BERLIN" and turn on EMAIL ME ON FAILURE.
  world002.00          easy     2 steps  [click, typing, synthetic]
    Set PHONE to "5550142".
  world002.01          medium   3 steps  [click, typing, synthetic]
    Set PHONE to "5550142" and choose UNLISTED.
```

A seed fixes the whole application: how many screens, what fields with which
value vocabularies, which toggles and radio groups, whether the primary action
sits below the fold, and what everything is called. Layout flows downward from a
cursor, so widgets cannot overlap by construction — hit-testing stays
unambiguous, which is what every derived gold path depends on.

```python
from computer_use.worlds import split

train, test = split(train=range(40), test=range(1000, 1010), per_world=4)
# 152 training tasks, 39 test tasks, no shared application
```

`split` refuses overlapping seed ranges rather than quietly leaking the layout,
the labels, and the button position it exists to hold out. Every task carries
its world's seed in `metadata`, so any result traces back to the app it came
from and a split can be audited after the fact.

The chain has no human link left in it: **the app was generated, the task was
searched out of the app, the solution is the search path, and the check is the
goal state**. A test runs 60 tasks across 30 generated worlds end to end and
asserts every one is solved by its own generated gold at a perfect 1.000 — if
any link were wrong, that is where it shows.

It found a reward bug the hand-written suite could not. Generated wizards put
CONTINUE and the final SUBMIT in the same rectangle on consecutive screens, so
the shortest path presses the same pixel twice — and the redundancy penalty,
which compared coordinates, scored that as the click-the-dead-pixel failure
mode. Correct trajectories were being docked, silently, in any app with a
fixed-position Next button. Redundancy now believes the environment's hit-test
over the coordinates.

### Closing the loop: a policy trained on this data, scored on apps it has never seen

Everything above generates. Nothing consumed any of it, which made the central
claim — verified GUI episodes are useful post-training data — an argument
rather than a result. A bug that quietly destroyed the data's value would have
shown up as nothing at all.

```bash
agentic-gui learn --train-worlds 30 --test-worlds 12    # or: make learn-gui
```

```
  trained on 89 tasks from 30 generated apps
  189 grounding decisions, train accuracy 98.4%

  held out (apps never seen)       36 tasks
    trained policy                 35/36  97%
    same features, random weights  16, 0, 11 of 36  25%
```

The chain has no human in it anywhere: the applications were generated, the
tasks were searched out of them, the training data is rollouts of the searched
solutions, and the score is on applications from a disjoint seed range. A
second held-out set scores 89%, and the result is stable across training seeds.

**The policy reads pixels.** `computer_use.perception` decodes the PNG,
recovers the text by matching the renderer's 3×5 font, and finds controls by
pairing their borders — then hands over labelled boxes with a point to click.
The policy is given a `Screenshot` and nothing else; it never calls
`env.state()`, which is what a benchmark score has to mean. Across 43
applications at two scroll positions, **271 of 271 controls are located,
correctly typed, and correctly labelled**, so the parse is not the bottleneck.

That doubles as a mechanical legibility test. The claim that these frames are
readable by a vision model used to be an assertion; now a label that cannot be
recovered from the pixels fails a test.

**What is learned, and what is not** — because a demo that hardcodes the answer
and calls it learning is worse than no demo:

- *Not learned.* Turning the instruction into an ordered list of goals. That is
  template parsing, and pretending to learn it would prove nothing.
- *Learned.* Which thing on screen a goal refers to. No feature says "click the
  element whose label matches" — token overlap is equally high for the caption
  naming a field and for the field itself, so the weights have to find the rule
  in the data.

They do, and the weights are legible:

| weight | learned |
|---|---|
| `kind:text` −2.6 | the words above an input are not the input |
| `goal:set\|above` +3.9 | a form names its field with the caption above it |
| `goal:navigate\|x` +1.3 | to leave a screen, take the button on the right |
| `goal:navigate\|filled` −1.2 | avoid the primary action — that's the commit, not the exit |
| `goal:navigate\|exact` −3.1 | never press the button the task *names* while still looking for it |

The controls matter more than the score. The same feature set with random
weights scores 25%, and it swings from 0% to 44% between seeds — so "the policy
solves held-out apps" is a statement about the data, not about the features.
Deleting the learned vocabulary (`word:*`) changes nothing at all: 97% either
way, so it is not recognizing labels it memorized, it is reading the screen.

The learner is a softmax over candidates trained by SGD in pure Python — no
numpy, no torch. That is the point rather than a limitation: a model small
enough to read end to end makes it obvious the signal is coming from the data.

### Practice instead of demonstrations

Demonstrations are the expensive part of post-training. They are free here
because search produces them, but the question that matters elsewhere is
whether an agent can improve by *practising* — attempting tasks nobody
demonstrated and keeping only what a verifier passes.

```bash
agentic-gui evolve --seed-worlds 3 --practice-worlds 12    # or: make evolve-gui
```

Two stages, following the shape current GUI-agent work has converged on
([UI-Voyager](https://arxiv.org/abs/2603.24533)):

- **Rejection fine-tuning.** Roll the current policy out several times per
  task, keep the attempts the verifier passes, retrain. No labels, no human.
- **Fork-point supervision.** Rejection sampling throws away every failure,
  which early on is most of the data. When one group holds both a success and
  a failure, the two runs agree up to some step and then diverge — and there
  the successful run is a *correction* for the failed one. They were looking at
  the same screen until that point, so the failed run's frame paired with the
  successful run's choice is a labelled decision at the exact moment the
  failure was decided. Divergence is judged by effect, not coordinates.

Starting from **one demonstrated task**:

```
  round    practice    kept  forks   pool   held out
  ──────────────────────────────────────────────────
  0               —       2      0      2     6/12    50%
  1             58%      36      6     44   12/12   100%
  2            100%      56      0    100   12/12   100%
  ──────────────────────────────────────────────────
  held out: 50% → 100% (+50%) with no new demonstrations
```

One demonstration plus practice reaches what thirty demonstrated applications
gave. From one application's worth of gold (three tasks), 56% → 97%. Across
three training seeds the one-demonstration result is 50% → 100% every time.

**Fork-point supervision earns nothing here, and the ablation says so.**
Turning it off (`--no-forks`) changes the outcome at no seed: it contributed
six labelled decisions in one run and zero in the others. The reason is
structural rather than a bug — a fork needs one group to contain both a success
and a failure, and practice pass rates here reach 100% within a round, so
almost no mixed groups exist to mine. It is implemented, tested, and reported
as inert in this setting; it is aimed at the long-horizon case where pass rates
stay low and most of the data is failures.

**The guard is the point, though.** Recent work finds that self-improvement
loops built on self-authored verification degrade quietly, and that
verifier-in-the-loop training stalls with the visible score climbing while
accuracy does not move. Two things here are aimed squarely at that:

The verifier is an exact state check the policy cannot see, so nothing the
agent does can move the bar it is judged against.

Every round reports the practice pass rate — what the loop can see — beside
held-out accuracy, which is the truth. The first full run of this module is
what that guard is for:

```
  round    practice    kept  forks   pool   held out
  0               —      19      0     19    35/36    97%
  1             89%     253      6    278    35/36    97%
  3             94%     277      1    836    35/36    97%
  ⚠ practice is getting easier while held-out accuracy is not moving —
    the loop is feeding on what it already solves
```

Practice climbed, the pool grew from 19 decisions to 836, and held-out accuracy
did not move at all. That case is a ceiling rather than a stall — three
demonstrated applications already saturate this task distribution — but a loop
reporting only what it can see would have shown a rising curve and called it
progress.

### What actually makes this benchmark hard

Three demonstrated applications reach 97% on held-out apps, which means the
easy distribution is saturated and no method can be told from another on it.
Two candidate difficulty knobs, measured rather than assumed:

**Distractors do not work.** `--hard` generates near-duplicate captions
("EMAIL" beside "EMAIL BACKUP") and a decoy primary action ("SAVE" beside
"SAVE DRAFT"), on the theory that matching the goal's words against the screen
would stop being a whole strategy. It barely moves: **97% → 94%**, with an
untrained control at 24% and 29%. Blinding the model to exact string equality
does not change either number, so exact-match is not what saves it either.

The reason is arithmetic. Token overlap here is Jaccard, so a caption that is a
strict superset of the goal is *already* penalised — goal `EMAIL` scores 1.0
against `EMAIL` and 0.5 against `EMAIL BACKUP`. The distractor was designed
against a weakness the metric does not have.

**Horizon length works.** Holding everything else fixed and generating deeper
tasks:

| optimal steps | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| solved | 100% | 88% | 94% | 81% | 75% | 69% | **63%** |

95% over 2–4 steps, 72% over 5–8, declining monotonically after depth 4. Errors
compound: one wrong control early leaves the agent on the wrong screen with no
way to notice. So `--min-depth` is the knob worth turning, and difficulty in
this environment is about how long the chain is rather than how confusable the
labels are.

```bash
agentic-gui tasks --worlds 8 --min-depth 5 --max-depth 8
```

`--hard` stays, because it is honest about what it is and it paid for itself
twice: it exposed two perception bugs that the easy worlds were hiding — a
button clipping its own caption at the edge, and the rows of a filled button's
white text pairing into phantom controls *inside* the button, each wide enough
to be read as a field.

One methodological note, since it nearly fooled me. A single untrained control
run is worthless here: five draws of random weights on the same tasks scored
`[34, 3, 11, 0, 5]` out of 36. The first of those, taken alone, says training
contributes nothing.

### Interactions as tokens

A trajectory reaches a trainer as JSON text:

```json
{"action": "left_click", "coordinate": [290, 218]}
```

A subword tokenizer spends about twenty tokens on that, most of them
punctuation, and spells the coordinate as digits — so the model has to learn
that "2", "9", "0" composes into a horizontal position, and that the same pixel
written `291` means nearly the same thing. Neither is a fact about operating a
GUI. It is an encoding tax paid on every step of every episode.

`computer_use.tokens` gives interactions their own vocabulary: one symbol per
action kind, one per quantized coordinate, one per character. **A click is 3
tokens instead of ~20**, and "nearby pixels are nearby" is built into the
representation instead of inferred from digit strings.

```python
from computer_use import encode_trajectory, to_token_batch

ids = encode_trajectory(trajectory)              # [<bos>, task…, <act>, <k:left_click>, <x:14>, <y:10>, …]
batch = to_token_batch(group, include_screens=True)
tensors = batch.to_torch()                       # input_ids, attention_mask, rewards
```

214 tokens in the vocabulary; 4.1–4.7 tokens per step against a ~12.5 estimate
for the JSON rendering.

**The observation has to carry state, not just layout.** Three of those tokens
— `<s:on>`, `<s:off>`, `<s:focus>` — exist because of a defect this encoding
had until a transformer was trained on it and refused to improve. A screen
encoded as position-and-label alone is *identical before and after a checkbox
is flipped*, and identical before and after a field takes focus. So the same
context carried two different correct actions, and no amount of training could
separate them:

| observation carries | training examples on an ambiguous context |
|---|---|
| position + label | 58.7% |
| ...+ toggle state | 40.0% |
| ...+ focus | **5.3%** |

Of the 5.3% that remain, three quarters are instructions of the form *"do X and
Y"*, where either subgoal legitimately goes first — both actions are correct,
and the verifier accepts either. The irreducible contradiction is ~1.3%.

Both signals were already in the pixels: a checked box is drawn with a filled
middle, a focused control with a differently-coloured outline. `perception` was
throwing them away — `Element.filled` samples 3px inside the top-left corner,
which sits between the border and an inset mark, so it reported "off" for every
checked box on screen. `Element.checked` reads the centre instead, and
`Element.focused` calls a control focused when its outline is the minority
colour among the controls on that screen, so neither needs to be told the
renderer's palette.

**The grid is measured, not chosen.** A lossy encoding of actions is only safe
if the actions still work afterwards, so the resolution is the coarsest one on
which every control in every generated world still hit-tests to itself:

| grid | controls surviving | max shift | vocabulary |
|---|---|---|---|
| 16×9 | 41% | 40px | 110 |
| 32×18 | 92% | 20px | 135 |
| **64×36** | **100%** | 10px | 185 |
| 128×72 | 100% | 5px | 285 |

32×18 is a smaller vocabulary and a corrupted dataset — 8% of controls decode
to a click on a *different* control, and nothing downstream would report it.

The real check is end to end: decode an episode from its ids alone, replay it,
and ask the verifier. **128 of 128 trajectories still pass** across the curated
suite and generated worlds, easy and hard.

That test earned its place immediately. The first version folded text to
uppercase, reasoning that the renderer's font defines the alphabet — it defines
what the environment can *draw*, but a field stores what it was *typed*, so
`ada@example.com` came back as `ADA@EXAMPLE.COM` and failed verification. Every
coordinate in those episodes was still perfect, which is exactly why nothing
else in the suite noticed.

### A transformer that reads those tokens

The section above ends by handing a `TokenBatch` to "a transformer". There
wasn't one — the data was in the right shape for a model nobody had run, which
is the weakest kind of claim a pipeline can make.

`computer_use.nn` and `computer_use.transformer` remove the hand-wave: a
reverse-mode autograd and a decoder-only transformer, in pure Python, with no
numpy and no torch. **56,352 parameters, two layers, no pretraining, no outside
corpus.** Whatever it learns, it learned from this pipeline's own output.

```bash
agentic-gui pretrain --train-worlds 24 --test-worlds 8   # or: make pretrain-gui
```

Every training example is one decision — the instruction, the screen as the
parser recovered it, and the action that followed:

```
<bos> C H O O S E _ C S V _ A N D _ T U R N _ O N _ S H A R E _ U S A G E _ D A T A .
<obs> <x:3> <y:8>   R E Q U I R E _ A P P R <sep>
      <x:3> <y:10>  S H A R E _ U S A G E _ <sep>
      <x:8> <y:13>  C O N T I N U E         <sep>
<act> <k:left_click> <x:3> <y:10> <eos>
      └──────────── loss is charged only here ────────────┘
```

**The answer is in the context, so this is a pointer problem, not a memory
one.** Every coordinate the model could emit is already in its input, sitting
beside the label it belongs to. The work is deciding *which* label the
instruction is asking for and copying the two tokens next to it — which is why
two layers is enough, and why it can transfer to an application whose layout it
has never seen. Nothing about the target's position is memorized.

The same mechanism, isolated, is a test: `test_learns_to_copy_from_context`
trains this architecture on `k a k ?` sequences where the answer differs every
time, so only attending back to the earlier match can score.

**Scoring runs the model in the environment.** Held-out next-token accuracy
would be a friendlier number and would mean less — a model can be right about
most tokens of an action and still click six pixels outside the control. Here
the generated action is decoded, executed, the episode continues from whatever
screen results, and the task's verifier decides.

#### What it scored

Two measures, reported together because they answer different questions.
**Step accuracy** is teacher-forced: given the screen the gold path actually
reached, is the next action right? **Solved** is closed-loop: the model acts on
the screen its own previous action produced, and the task's verifier decides.

```
  held out: 112 decisions across 32 tasks, on 8 generated apps never seen

  training data      params    step accuracy      solved
  ---------------------------------------------------------
  24 apps            56,352    20/112   17.9%     0/32    0%
  24 apps, wider     91,520    24/112   21.4%     0/32    0%
  48 apps            56,352    42/112   37.5%     3/32    9%
  untrained          56,352     0/112    0.0%     0/32    0%
```

**That 0% is the wrong null, and the comparison against it was not evidence.**
A randomly initialized model emits token soup, so it scores zero for a reason
that has nothing to do with grounding — beating it proves only that training
happened. `computer_use/diagnose.py` replaces it with two readings that bracket
the score from both sides, on exactly the same held-out decisions:

```
  ceiling — answers present in their own context
    all                      112/112  100.0%
    left_click                93/93   100.0%
    type                      16/16   100.0%
    scroll                     3/3    100.0%

  floor — policies that do not learn (93 held-out clicks)
    uniform over controls     29/93    31.2%
    always the first control  18/93    19.4%
    label overlapping the instruction
                              44/93    47.3%
```

**The ceiling is 100%, so every point dropped is a learning failure.** Nothing
held out asks the model for something its context does not contain — which is
what `snap_to_screen` was for, and this is the check that says it worked.

**The floor is where the result changes.** The 24-app model gets 19 of those
93 clicks — 20.4%, Wilson 95% CI **[13.5%, 29.7%]**. The uniform policy's
31.2% is not a sample and needs no interval: it is the exact expected value of
picking at random among the controls actually on each screen. It sits *above*
the model's upper bound. **On its own training scale the transformer is
significantly worse than guessing**, and the ten-line label-overlap heuristic —
47.3% — is better than both.

Two things made the old framing easy to believe. The prose said a click has to
name "one of a dozen cells"; the median generated screen offers **three**
controls, so chance is high, not negligible. And the 0% control could not
possibly score otherwise, which made any positive number look like progress.

**The 48-app run is the one still standing, and it is not yet cleared.** Its
42/112 is over every kind, while the floor above is over clicks only, so the
two are not the same denominator and cannot be compared directly. Its click
breakdown was not recorded before `diagnose` existed. `agentic-gui pretrain`
now prints the ceiling and the floor beside every score it reports, so no
future run can be quoted without them.

**Data diversity is the binding constraint, not capacity.** Doubling the
applications at an identical parameter count nearly doubled step accuracy and
took closed-loop from zero to three. Adding 62% more parameters against the
same 24 apps bought 3.5 points and no solved task. The wider model also overfit
harder — held-out loss bottomed at 1.18 and rose, while the 48-app run reached
0.64 and stayed there.

**Closed-loop is much harder than per-step, and that gap is the honest part.**
At 37.5% per decision, a three-step task is unlikely to survive: one wrong
click and every later step is taken from a screen the gold path never visited.

#### What it actually learned, which is not what the encoding assumed

Splitting a prediction into the parts that can fail on their own says what the
score cannot. On the 24-app model:

```
  the model, by field
    action kind               94/112   83.9%
    column | kind             37/89    41.6%
    row | kind                31/89    34.8%
    both | kind               20/89    22.5%
    missed clicks: median 11 cells away, 5 within two

  where it looked when emitting a coordinate
    attention on the screen  0.483
    of that, on the target   0.298
    if spread evenly         0.295
    ratio                     1.01
```

**It learned the action grammar and not the grounding.** The verb is right
five times in six. The coordinate is then close to a guess, and not a near
one — the median missed click is *eleven cells* from the target, with only
five of the misses within two. Column and row are each individually above
chance while the pair is not, which is what fitting the marginal distribution
of where controls sit looks like, as opposed to conditioning on the
instruction.

**The attention says the same thing directly, and it is the sharper result.**
At the moment it emits a coordinate, 0.298 of the model's screen-directed
attention lands on the control the instruction names — against 0.295 for a
gaze spread evenly across the observation. A ratio of **1.01**. The premise
this whole encoding rests on is that emitting a click is a *copy*: find the
label the instruction names, take the cell beside it. The model never learned
to point. That is a mechanism, not an inference from a score, and it explains
the score.

**`type` is not exact-match strictness hiding a near miss.** The obvious
defence of 0/16 is that fifteen character tokens have to be right at once, so
a model that got fourteen would read as zero. Measured, character-level
recall is 13/48 — **27.1%**. It is not close.

#### One experiment on the encoding, which failed as predicted and succeeded elsewhere

If emitting a coordinate is a copy, field order should matter. An attention
head matches a pattern and copies what *follows* the match, and this encoding
puts each control's coordinate *before* the label that identifies it — so the
copy has to run backwards. `--label-first` flips it. Same seed, same corpus,
same 313 decisions, one difference:

```
                          coordinate-first        label-first
  held-out clicks         19/93  20.4%            16/93  17.2%
    95% CI                [13.5%, 29.7%]          [10.9%, 26.1%]
  attention ratio         1.01                    1.10
  action kind             94/112 83.9%            94/112 83.9%
  typed characters        13/48  27.1%            50/66  75.8%
    95% CI                [16.6%, 41.0%]          [64.2%, 84.5%]
```

**The prediction was wrong.** Clicks did not improve — 17.2% against 20.4%,
intervals overlapping, both far below the 31.2% floor. Attention moved from
1.01 to 1.10, which is a nudge and not a mechanism appearing. Field order does
not buy grounding, so backwards-copying was not what was stopping it.

**Something unpredicted did move, and a lot.** Character-level recall on
`type` went from 27.1% to 75.8%, with non-overlapping intervals, and whole
strings from 0 to 2. Changing the *screen* encoding substantially improved
copying into the *action* — and the mechanism for that is not established
here. Characters within a string are not independent draws, so the interval
above is optimistic; the effect is much larger than the caveat.

**What both runs agree on is the finding.** Action-kind accuracy is 94/112 in
both, to the decision. The grammar is learned, robustly and identically; the
grounding is not learned in either. Doubling the applications moved the score
before, and the reading here says why that is the lever: nothing about the
model's attention suggests capacity or field order is what stands between it
and pointing.

**A zero is only evidence if a perfect policy would not score zero.** Replaying
each task's own gold actions through the same verifier the evaluation uses
scores 32/32, so the closed-loop numbers are about the model. There is a test
that asserts it.

#### Two things this needed that weren't obvious

**Gradients are checked, not trusted.** Hand-written backwards are the classic
bug that trains anyway: a transposed index still points roughly downhill, so
the loss falls and nothing looks wrong. `tests/test_nn.py` finite-difference
checks all sixteen ops against their analytic gradients; they agree to ~1e-10.

**One click in ten had an uncopyable target.** The search that produces a gold
path picks some pixel inside a widget; the parser picks the centre of the
rectangle it recovered from the pixels. Usually the same 20px cell — measured
at 10% of clicks, not. On those steps the target token is one the context does
not contain, so the only way to be "right" is to memorize a coordinate, which
is precisely what this setup exists to avoid teaching. `snap_to_screen` moves
the click onto the parser's point, and `_frames` does not take on trust that
this is harmless: it executes the snapped path and keeps it only if the task's
own verifier still passes. Uncopyable click targets: **10% → 3.8%**.

#### Two decisions that made it finish

Pure-Python arithmetic runs at ~20M multiply-accumulates per second, so the
shape of the compute is the difference between an experiment and an intention.

- **Project only where the loss is charged.** Logits at all `T` positions cost
  `T x d x vocab` — the largest matmul in the model. The loss is charged on the
  ~4 tokens of the action, so the hidden states are sliced to the supervised
  positions *before* the projection. Same gradients; a tenth of the time. A
  test asserts the restricted rows equal the full projection's.
- **The inner loop was picked by measuring, not reasoning.** Three candidate
  matmul loop orders, benchmarked on the five shapes the model actually uses.
  The one that looks fastest per element — a list comprehension accumulating
  into an output row — loses by 5-25%, because it re-slices and reallocates on
  every one of its `m*k` iterations. The reasoning had it backwards.

### Step pairs: attribution by effect, not by text

Step-level pairs claim something strong — "from this screen, click *here*, not
*there*" — so two guards keep the claim honest:

- **Divergence is judged by which widget a click landed on**, not by whether the
  coordinates differ. Two clicks 20px apart on the same button are the same
  decision; blaming an outcome on the difference between them emits a pair whose
  rejected side is a perfectly good action. That is training data that degrades
  grounding rather than improving it, and it fails silently.
- **Only pairs where the outcomes differ** are kept by default. Between two
  successful runs the reward gap is efficiency, and the "rejected" action is not
  a mistake. Opt out with `require_outcome_difference=False`.

### Real browser

The same `run_episode` and `StateVerifier` drive Chromium — only the
environment changes. A browser has no ground truth of its own, so
`state_script` supplies it; without something checkable, an episode cannot be
verified and its trajectories are not usable as training data.

```python
env = PlaywrightComputer(
    start_url="https://example.com/settings",
    state_script="""() => ({
        email: document.getElementById('email').value,
        saved: window.__saved === true,
    })""",
    # Point at an existing Chromium instead of downloading one (CI images,
    # devcontainers). Also reads PLAYWRIGHT_CHROMIUM_EXECUTABLE.
    executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
)

trajectory = asyncio.run(run_episode(
    task, env, policy,
    verifier=StateVerifier({"email": "ada@example.com", "saved": True}),
))
```

The browser tests skip themselves when Playwright or a Chromium build is
missing, so CI stays dependency-light while anyone with a browser gets the
real coverage.

### Live model

```bash
pip install "agentic-post-training[computer-use]"
export ANTHROPIC_API_KEY=...          # or run `ant auth login`
python3 examples/computer_use_demo.py --live
```

Defaults to `claude-opus-5`; the tool version, beta flag, effort, and frame
budget are all constructor arguments on `PolicyConfig`, so pointing the policy
at a different model is a config change rather than a code change.

## Agent Communication

Agents communicate via a structured message bus. Watch them coordinate in real-time:

```
[14:23:01] 🔗 Coordinator → broadcast: Starting stage: training
[14:23:01] 📊 Trainer → Coordinator: Loading model: gpt2
[14:23:02] 📊 Trainer → broadcast: Beginning GRPO training for 3 epochs
[14:23:02] 📊 Trainer → broadcast: Epoch 1/3 | Loss: 1.8432 | Reward: 0.5700
[14:23:03] 📊 Trainer → broadcast: Epoch 2/3 | Loss: 0.9821 | Reward: 0.7900
[14:23:03] ✅ Trainer → Coordinator: Training complete! Final loss: 0.5513
[14:23:04] ⚡ Optimizer → broadcast: Starting GPTQ quantization (4-bit)
[14:23:04] ✅ Optimizer → Coordinator: Quantization complete: 8.0x compression
[14:23:05] 📈 Evaluator → broadcast: Starting evaluation suite: mmlu, mt_bench
```

## Configuration Presets

```python
from pipeline.config import PipelineConfig

# Quick DPO alignment
config = PipelineConfig.preset("quick_dpo")

# Full RLHF pipeline
config = PipelineConfig.preset("full_rlhf")

# GRPO with quantization
config = PipelineConfig.preset("efficient_grpo")

# Production deployment
config = PipelineConfig.preset("production")
```

## Optimization

| Method | Compression | Quality | Use Case |
|--------|-------------|---------|----------|
| GPTQ | 8x | ~99% | Production inference |
| AWQ | 8x | ~99.5% | Best quality at 4-bit |
| GGUF | 8x | ~98% | llama.cpp deployment |
| NF4 | 8x | ~99% | QLoRA training |
| INT8 | 4x | ~99.9% | Minimal quality loss |

## Tests

```bash
python3 -m pytest tests/ -v
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
