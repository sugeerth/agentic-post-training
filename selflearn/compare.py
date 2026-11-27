"""Compare runs side by side: learning signal, training reward, and what the agent actually answers.

  python3 -m selflearn.compare runs/a runs/b ... --out docs/compare.png [--labels "v1" "v2" ...]

Collapse shows up as informative groups falling to ~0 (every attempt agrees, so GRPO learns nothing) and as a
decision mix that turns into one bar.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def _steps(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines() if line.strip()]


def _decision(text: str) -> str:
    m = re.findall(r"<answer>(.*?)</answer>", text, re.S)
    words = m[-1].split() if m else []
    return words[0].lower().strip(".,") if words else "no answer"


def _mix(run: Path, which: str) -> Counter | None:
    f = run / f"eval_{which}.json"
    if not f.exists():
        return None
    return Counter(_decision(r["text"]) for r in json.loads(f.read_text())["records"])


def plot(runs: list[Path], labels: list[str], out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for run, label in zip(runs, labels, strict=True):
        s = _steps(run)
        x = [r["step"] for r in s]
        ax[0].plot(x, [r["informative_groups"] / max(1, r.get("groups", 16)) for r in s], lw=1.8, label=label)
        k = 5
        rw = [r["reward"] for r in s]
        ax[1].plot(x, [sum(rw[max(0, i - k + 1):i + 1]) / len(rw[max(0, i - k + 1):i + 1]) for i in range(len(rw))], lw=1.8, label=label)
    ax[0].set_title("Learning signal: share of groups whose attempts disagree")
    ax[0].set_ylim(0, 1)
    ax[1].set_title("Training reward (5-step mean)")
    for a in ax[:2]:
        a.set_xlabel("step")
        a.legend(frameon=False, fontsize=8)
    # decision mix on the held-out set: truth, base, then each run after training
    first = json.loads((runs[0] / "eval_before.json").read_text())["records"]
    truth = Counter(str(r["answer"]).split()[0] for r in first)
    cols = [("truth", truth), ("base", _mix(runs[0], "before"))]
    cols += [(label, _mix(run, "after")) for run, label in zip(runs, labels, strict=True) if _mix(run, "after")]
    kinds = ["refund", "replace", "deny", "no answer"]
    colors = {"refund": "#2563eb", "replace": "#16a34a", "deny": "#dc2626", "no answer": "#9ca3af"}
    bottom = [0.0] * len(cols)
    for kind in kinds:
        vals = [c.get(kind, 0) / max(1, sum(c.values())) for _, c in cols]
        ax[2].bar(range(len(cols)), vals, bottom=bottom, color=colors[kind], label=kind)
        bottom = [b + v for b, v in zip(bottom, vals, strict=True)]
    ax[2].set_xticks(range(len(cols)), [n for n, _ in cols], rotation=20, ha="right", fontsize=8)
    ax[2].set_title("What the agent answers on the held-out set")
    ax[2].legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--out", default="compare.png")
    a = ap.parse_args(argv)
    runs = [Path(r) for r in a.runs]
    print(plot(runs, a.labels or [r.name for r in runs], Path(a.out)))


if __name__ == "__main__":
    main()
