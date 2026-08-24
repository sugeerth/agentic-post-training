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
| `metrics.*` | Unbiased pass@k and Wilson score intervals. |
| `dataset.*` | Trajectories → `PreferencePair` / `RolloutBatch` / `TrainingExample`. |
| `ComputerUseAgent` | The whole thing as an agent on the message bus. |
| `agentic-gui` | CLI: `tasks`, `bench`, `collect`, `learn`, `evolve` — over the curated suite, `--synthetic` tasks, or `--worlds` apps. |

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
