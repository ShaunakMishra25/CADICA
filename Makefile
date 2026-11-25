PYTHON ?= python
FOLD ?= 1
EPOCHS ?= 60
BATCH_SIZE ?= 16
RUN_TAG ?= baseline
VARIANT ?= last
FOLDS ?= 1 2 3 4 5
EXTRA ?= --no-early-stopping

.PHONY: install install-dev test lint folds sample prepare verify pipeline train train-all train-fixed aggregate clean

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

train:
	cadica-train --fold $(FOLD) --epochs $(EPOCHS) --batch-size $(BATCH_SIZE) \
		--run-tag $(RUN_TAG)

# Fixed-budget protocol: early stopping off, last epoch is the headline result.
train-fixed:
	cadica-train --fold $(FOLD) --epochs $(EPOCHS) --batch-size $(BATCH_SIZE) \
		--run-tag $(RUN_TAG) --no-early-stopping

# Train every fold for one run tag. EXTRA carries protocol flags.
train-all:
	@for f in $(FOLDS); do \
		echo "fold $$f"; \
		cadica-train --fold $$f --epochs $(EPOCHS) --batch-size $(BATCH_SIZE) \
			--run-tag $(RUN_TAG) $(EXTRA) || exit 1; \
	done

aggregate:
	cadica-aggregate --run-tag $(RUN_TAG) --variant $(VARIANT)

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
