.PHONY: install test test-fast lint mypy demo demo-dry clean

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

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
