PYTHON ?= python
CONFIG ?= configs/experiment.yaml

.PHONY: setup test ci download smoke experiment ablation stats tables figures paper all clean

setup:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

ci:
	$(PYTHON) -m pip check
	$(PYTHON) -m pytest --cov=src/aldwknn --cov-report=term-missing -q
	$(PYTHON) scripts/run_experiments.py --config configs/smoke.yaml

download:
	$(PYTHON) scripts/download_data.py --config $(CONFIG)

smoke:
	$(PYTHON) scripts/run_experiments.py \
		--config configs/smoke.yaml \
		--run-name smoke

experiment:
	$(PYTHON) scripts/run_experiments.py \
		--config $(CONFIG) \
		--run-name main

ablation:
	$(PYTHON) scripts/run_ablation.py \
		--config $(CONFIG) \
		--run-name ablation

stats:
	$(PYTHON) scripts/analyze_statistics.py \
		--config $(CONFIG)

tables:
	$(PYTHON) scripts/make_tables.py \
		--config $(CONFIG)

figures:
	$(PYTHON) scripts/make_figures.py \
		--config $(CONFIG)

paper:
	$(PYTHON) scripts/build_paper.py \
		--config $(CONFIG) \
		--format markdown

all: test download smoke experiment ablation stats tables figures paper

clean:
	rm -rf artifacts/runs/*
	rm -rf artifacts/tables/*
	rm -rf artifacts/figures/*
	rm -rf artifacts/statistics/*
