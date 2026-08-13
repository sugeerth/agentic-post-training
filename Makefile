.PHONY: install test test-fast lint mypy demo demo-dry demo-gui demo-gui-live bench-gui bench-gui-synth bench-gui-worlds learn-gui evolve-gui clean

PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTEST) --tb=short

test-fast:
	$(PYTEST) --tb=line -q

lint:
	ruff check .

# Targets come from `[tool.mypy] files` so the command and the config cannot
# drift. Runs without torch/pydantic installed; see the overrides there.
mypy:
	mypy

demo-dry:
	$(PYTHON) -m pipeline.cli run examples/orpo_quickstart/run.yaml --backend local --dry-run

demo:
	$(PYTHON) -m pipeline.cli run examples/orpo_quickstart/run.yaml --backend local

# Computer use. `demo-gui` is fully offline — no API key, no browser, no GPU.
demo-gui:
	$(PYTHON) examples/computer_use_demo.py

demo-gui-live:
	$(PYTHON) examples/computer_use_demo.py --live

# The GUI benchmark. `noisy` is the reference solution with grounding noise —
# an imperfect baseline, so pass@k and the confidence interval show real spread.
bench-gui:
	$(PYTHON) -m computer_use.cli bench --policy noisy --attempts 8

# Benchmark on tasks derived by searching the environments — nothing hand-written.
bench-gui-synth:
	$(PYTHON) -m computer_use.cli bench --synthetic --policy noisy --attempts 8

# Benchmark on applications that did not exist until this command ran, from a
# seed range no training run uses. Nobody wrote the app, the task, or the
# solution, and nobody has seen the screen before.
bench-gui-worlds:
	$(PYTHON) -m computer_use.cli bench --worlds 8 --held-out --policy noisy --attempts 8

# The loop closed: generate applications, search tasks out of them, roll out the
# solutions, train a policy on the result, and score it on applications from a
# disjoint seed range. The policy reads pixels — no environment state anywhere.
learn-gui:
	$(PYTHON) -m computer_use.cli learn --train-worlds 30 --test-worlds 12

# Practice instead of demonstrations: a handful of demonstrated apps, then
# rounds of attempts on apps that have none, keeping only what a ground-truth
# verifier passes. Reports what the loop can see next to what is actually true.
evolve-gui:
	$(PYTHON) -m computer_use.cli evolve --seed-worlds 3 --practice-worlds 12 --rounds 3

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
