#!/usr/bin/env bash
set -euo pipefail

export PYTHONHASHSEED=20261021
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p artifacts/logs

RUN_LOG="artifacts/logs/run_all_$(date -u +%Y%m%dT%H%M%SZ).log"

{
    echo "Starting ALD-WKNN pipeline"
    date -u

    make ci
    make download
    make smoke
    make experiment
    make ablation
    make stats
    make tables
    make figures
    make paper

    echo "Pipeline completed"
    date -u
} 2>&1 | tee "${RUN_LOG}"
