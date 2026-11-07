PYTHON ?= python

.PHONY: install install-dev test lint folds sample prepare verify pipeline clean

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests scripts

folds:
	cadica-build-folds

sample:
	cadica-sample-frames

prepare:
	cadica-prepare-folds

verify:
	cadica-verify-folds

pipeline: folds sample prepare verify

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
