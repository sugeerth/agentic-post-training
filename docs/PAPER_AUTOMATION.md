# Daily paper-scan automation

A cron job that tracks agent post-training research and drops runnable
mimic stubs into the codebase every day.

## The chain

```
                        ┌──────────────────┐
   cron @ 09:00 UTC ──▶ │   PaperScanner   │  arXiv Atom API (canned fallback)
                        └────────┬─────────┘
                                 │  ~12 recent abstracts
                                 ▼
                        ┌──────────────────┐
                        │  TechniqueScout  │  regex → (name, novelty, claims)
                        └────────┬─────────┘
                                 │  one candidate per paper
                                 ▼
                        ┌──────────────────┐
                        │   MimicWriter    │  emits techniques/mimics/{slug}.py
                        └────────┬─────────┘
                                 │  gain + cap seeded from claims
                                 ▼
                        ┌──────────────────┐
                        │      Digest      │  Markdown → workflow summary + draft PR
                        └──────────────────┘
```

## What each agent owns

| Agent | File | Owns |
|-------|------|------|
| **PaperScanner** | `agents/paper_scanner.py` | arXiv fetch (Atom API), canned offline fallback |
| **TechniqueScout** | `agents/technique_scout.py` | Regex extraction of algorithm names + claims |
| **MimicWriter** | `agents/mimic_writer.py` | Codegen of the technique stub file |
| **PaperScanPipeline** | `pipeline/paper_scan_pipeline.py` | Chain + digest rendering |

## Extraction rules — deliberately auditable

Every extraction rule is one line of code, not a learned model:

- **Method name** — acronym near "we propose / introduce / present" (only the abstract; title used as a fallback with a tighter filter that rejects descriptive-adjective compounds like *Multi-Turn* or *Tool-Using*).
- **"outperforms X"** — captures target + optional percentage.
- **"N% higher on TASK"** — captures gain + task.
- **Flags** — mentions of KL, process reward, reward hacking.

An LLM extractor at 3 AM every day with no eval is a bug factory — the
regex path is boring in exactly the way this problem needs.

## Anatomy of a generated mimic

`techniques/mimics/mt_agrpo.py` (example, auto-generated):

```python
class MtAgrpo:
    name = "mt_agrpo"
    paper_reference = "https://arxiv.org/abs/2501.01234"
    is_mimic = True

    def __init__(self) -> None:
        self._it = 0
        self.gain = 0.127  # scaled from "outperforms GRPO by 8%"
        self.cap = 0.80

    def step(self, iteration=None) -> dict[str, float]:
        self._it = iteration if iteration is not None else self._it + 1
        loss = (1.7 - self.gain) * math.exp(-0.3 * self._it) + 0.28
        reward = min(self.cap + 0.05, 0.30 + self.gain * 2 * self._it)
        kl = max(0.0, (0.20 - self.gain) - 0.02 * self._it)
        return {"loss": ..., "reward": ..., "kl_divergence": ...}
```

The mimic is **not** a working implementation of the paper. It's a fair
stand-in for bake-off comparisons and API testing until someone reads
the paper and writes a real one. The `is_mimic = True` class attribute
lets the bake-off tag it as untrusted.

## Running it

```bash
# Manual — tries arXiv, falls back to canned
python3 examples/run_paper_scan.py

# Offline / deterministic
python3 examples/run_paper_scan.py --no-network

# Regenerate stubs even when the file already exists
python3 examples/run_paper_scan.py --overwrite
```

## CI / cron

`.github/workflows/daily-paper-scan.yml` runs the pipeline daily at 09:00 UTC:

1. Runs the scan.
2. Uploads `output/paper_scan/digest.{md,json}` + `scan.log` as an artifact.
3. Appends the digest to the workflow summary (visible on the Actions run page).
4. If new mimics were written, opens a **draft PR** (`bot/daily-paper-scan`)
   so a human reviews the stubs before they land on `main`.

The workflow also has `workflow_dispatch` — you can trigger it by hand
from the Actions tab with two toggles: `overwrite` and `allow_network`.

## Example digest (canned mode)

```
# 📚 Paper scan · 5 papers · 5 candidate techniques · 5 new mimics

**Source:** canned · **Written:** `mt_agrpo`, `trajdpo`, `vg_rft`, `mc_prm`, `spin_2`

## 🆕 New mimics generated
- **`mt_agrpo`** — Multi-Turn GRPO with Adaptive KL Control for Tool-Using Agents
    - Paper: [2501.01234v1](https://arxiv.org/abs/2501.01234)
    - Novelty: We propose MT-AGRPO, a multi-turn variant of GRPO with an adaptive KL coefficient
    - Claims: beats GRPO by 8% · [KL]
- **`trajdpo`** — TrajDPO: Trajectory-Aware Preference Optimization
    - Paper: [2501.02345v1](https://arxiv.org/abs/2501.02345)
    - Novelty: We introduce TrajDPO, which extends DPO to compare full multi-turn trajectories
- ...
```

Attention budget: 3 lines above the fold, per-paper detail below, all
raw papers collapsed inside `<details>`.
