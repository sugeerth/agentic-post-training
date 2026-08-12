.PHONY: install test test-fast lint mypy demo demo-dry demo-gui demo-gui-live bench-gui clean

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

mypy:
	mypy core techniques/_base

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

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
