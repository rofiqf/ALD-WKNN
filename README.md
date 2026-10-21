# ALD-WKNN Coursework Experiment

Reproducible experimental project for evaluating Adaptive
Local-Density Distance-Weighted KNN.

## Reproducibility seed

Global seed: `20261021`

## Setup with Conda

```bash
conda env create -f environment.yml
conda activate ald-wknn
python -m pip check
pytest -q
```

## Alternative setup with venv

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest -q
```

## Download datasets

```bash
make download
```

## Run smoke test

```bash
make smoke
```

Inspect the smoke artifacts before starting the complete benchmark.

## Run main benchmark

```bash
make experiment
```

## Run ablation

```bash
make ablation
```

## Statistical analysis

```bash
make stats
```

## Generate tables

```bash
make tables
```

## Generate figures

```bash
make figures
```

## Generate Markdown paper

```bash
make paper
```

## Full pipeline

```bash
./scripts/run_all.sh
```

or:

```bash
make all
```

## Run tests and CI checks

```bash
make ci
```

## Generate LaTeX/PDF

If Pandoc and LaTeX are installed:

```bash
pandoc \
  paper/paper.md \
  --citeproc \
  --bibliography paper/references.bib \
  -o paper/paper.tex
```

Then:

```bash
latexmk \
  -pdf \
  -interaction=nonstopmode \
  -halt-on-error \
  -output-directory=paper \
  paper/paper.tex
```

The PDF must never be generated from invented results.
All empirical values must come from files under `artifacts/`.

## Main outputs

- `artifacts/runs/`
- `artifacts/statistics/`
- `artifacts/tables/`
- `artifacts/figures/`
- `paper/paper.md`
- `paper/paper.pdf`

## Novelty warning

ALD-WKNN is an experimental candidate formulation.
Density-aware and locally informed KNN methods already exist.
Do not claim novelty unless formula-level literature verification
has been completed.
