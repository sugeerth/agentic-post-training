# Sample: EvaluationAgent

**Owns:** distribution shift & regression detection. Runs a benchmark
suite in parallel and compares to a baseline.

```
📈 Evaluation Report: agent-checkpoint

  MMLU            [█████████████████████████░░░░░]   82.1/100 (+18.3%)
  GSM8K           [██████████████████████████░░░░]   85.4/100 (+22.7%)
  HumanEval       [████████████████░░░░░░░░░░░░░░]   54.8/100 (+31.2%)

  Overall: +24.1%

  Recommendations:
    → HumanEval shows room for improvement — try technique-specific tuning
```

**What to notice:** the evaluator is deliberately terse — a bar per
benchmark, an overall delta, one recommendation. The full JSON goes
into `output/report.json` for downstream tooling; humans read three
lines.
