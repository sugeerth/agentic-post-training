# Sample: ReporterAgent

**Owns:** attention budget. Turns the pipeline's raw result dict into a
TL;DR that fits above the fold.

```markdown
# TL;DR — agentic-tool-use

**Task success 76% (Δ +38%) via multi_turn_grpo.**

curated `32/64` · rm-acc `0.82` · reward `0.71` · success `76%`

## Needs attention
- **multi_turn_grpo**: KL=0.61 exceeds 0.5 — tighten kl_coef → `adjust`

## Next action
- Address `multi_turn_grpo` — KL=0.61 exceeds 0.5 — tighten kl_coef.

<details><summary>Full stage metrics</summary>
...
</details>
```

**What to notice:**
- Headline leads with the **delta** (`Δ +38%`), not the raw value.
- Key metrics fit on **one line**.
- The `<details>` collapse hides everything only CI or a debugging
  human should see.
- If nothing needed attention, the "Needs attention" section becomes
  one sentence: *"All stages within expected envelope."*
