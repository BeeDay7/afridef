# AfriDef Makefile -- reproducible entry points

.PHONY: data baseline full all clean lint test

CONFIG ?= configs/default.yaml
SEEDS  ?= 0 1 2 3 4

data:
	python scripts/prepare_data.py --config $(CONFIG)

baseline:
	python scripts/run_baselines.py --config $(CONFIG)

full:
	python scripts/run_full.py --config $(CONFIG) --seed 0

all:
	@for s in $(SEEDS); do \
	  echo "=== seed $$s ==="; \
	  python scripts/run_full.py --config $(CONFIG) --seed $$s; \
	done

lint:
	ruff check src/

test:
	pytest -q tests/

clean:
	rm -rf results/* __pycache__/ .pytest_cache/
