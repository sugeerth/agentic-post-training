"""Turn a run folder into REPORT.md + report.png (before/after by level, training reward, curriculum).

  python3 -m selflearn.report runs/<run>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _chart(run: Path, res: dict, steps: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    lv = sorted(res["before"]["by_level"], key=int)
    x = range(len(lv))
    b = [res["before"]["by_level"][k] for k in lv]
    a = [res["after"]["by_level"][k] for k in lv]
    ax[0].bar([i - 0.2 for i in x], b, 0.4, label=f"base {res['before']['accuracy']:.1%}", color="#9ca3af")
    ax[0].bar([i + 0.2 for i in x], a, 0.4, label=f"trained {res['after']['accuracy']:.1%}", color="#2563eb")
    ax[0].set_xticks(list(x), [f"L{k}" for k in lv])
    ax[0].set_ylim(0, 1)
    ax[0].set_title("Held-out accuracy by difficulty (greedy)")
    ax[0].legend(frameon=False)
    s = [r["step"] for r in steps]
    r = [r["reward"] for r in steps]
    k = 5
    smooth = [sum(r[max(0, i - k + 1):i + 1]) / len(r[max(0, i - k + 1):i + 1]) for i in range(len(r))]
    ax[1].plot(s, r, color="#cbd5e1", lw=1, label="per step")
    ax[1].plot(s, smooth, color="#2563eb", lw=2, label=f"{k}-step mean")
    ax[1].set_title("Training reward (mixed levels, sampled)")
    ax[1].set_xlabel("step")
    ax[1].legend(frameon=False)
    for level in lv:
        ax[2].plot(s, [r["p"][str(level)] for r in steps], lw=1.6, label=f"L{level}")
    ax[2].set_ylim(0, 1)
    ax[2].set_title("Agent's own success estimate per level")
    ax[2].set_xlabel("step")
    ax[2].legend(frameon=False, ncol=3, fontsize=8)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(run / "report.png", dpi=130)


def build(run: Path) -> Path:
    res = json.loads((run / "results.json").read_text())
    steps = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines() if line.strip()]
    ex = json.loads((run / "examples.json").read_text())
    _chart(run, res, steps)
    b, a = res["before"], res["after"]
    lv = sorted(b["by_level"], key=int)
    rows = "\n".join(f"| L{k} | {b['by_level'][k]:.0%} | {a['by_level'][k]:.0%} |" for k in lv)
    md = [f"# Self-learning agentic RL ({res.get('usecase', 'math')}): {res['model']}, {res['steps']} steps", "",
          f"Held-out set: {res['held_out']} problems (fixed seed), greedy decoding, same questions before and after.", "",
          "| | base | trained |", "|---|---|---|",
          f"| **accuracy** | **{b['accuracy']:.1%}** | **{a['accuracy']:.1%}** |", rows,
          f"| used the calculator | {b['used_tool']:.0%} | {a['used_tool']:.0%} |",
          f"| calls per problem | {b['mean_calls']:.2f} | {a['mean_calls']:.2f} |", "",
          f"Problems fixed by training: **{res['fixed']}**; problems broken: **{res['broken']}**.", "",
          "![report](report.png)", ""]
    for i, (before, after) in enumerate(ex["fixed"][:2], 1):
        md += [f"## Example {i} (level {before['level']}, answer {before['answer']})", "", f"> {before['question']}", "",
               "**Base model** (reward 0):", "", "```", before["text"].strip(), "```", "",
               "**After self-learning** (reward 1):", "", "```", after["text"].strip(), "```", ""]
    if ex["broken"]:
        before, after = ex["broken"][0]
        md += [f"## One it got worse on (level {before['level']}, answer {before['answer']})", "", f"> {before['question']}", "",
               "```", after["text"].strip(), "```", ""]
    path = run / "REPORT.md"
    path.write_text("\n".join(md))
    return path


if __name__ == "__main__":
    print(build(Path(sys.argv[1])))
