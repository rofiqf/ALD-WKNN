"""Ablation study: A0 distance-only, A1 density-only, A2 ALD unclipped, A3 ALD full.

Uses outer splits identical to the main run (same config seed). Hyperparameters
are taken from the modal best params of the latest main run when available,
with deterministic parametric defaults as fallback. Output schema matches the
main run artifacts exactly.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import _helpers as helpers

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aldwknn.cv import make_outer_cv
from aldwknn.data import load_dataset
from aldwknn.metrics import compute_metrics
from aldwknn.model import ALDWKNNClassifier, PowerWeightedKNNClassifier
from aldwknn.reporting import ensure_dir, write_csv, write_manifest
from aldwknn.utils import derive_seed, dump_config, load_config, make_run_id, params_to_jsonable

import run_experiments as rexp

VARIANTS = ("A0_distance_only", "A1_density_only", "A2_ald_unclipped", "A3_ald_full")

VARIANT_FLAGS = {
    "A0_distance_only": ("✓", "✗", "N/A"),
    "A1_density_only": ("✗", "✓", "✓"),
    "A2_ald_unclipped": ("✓", "✓", "✗"),
    "A3_ald_full": ("✓", "✓", "✓"),
}

DEFAULT_ALD_PARAMS = {
    "n_neighbors": 5,
    "density_neighbors": 5,
    "distance_power": 1.0,
    "density_alpha": 0.5,
    "density_clip": 3.0,
}
DEFAULT_POWER_PARAMS = {"n_neighbors": 5, "distance_power": 1.0}


def _modal_best_params(run_dir: Path | None, dataset: str, method: str) -> dict | None:
    if run_dir is None:
        return None
    path = run_dir / "best_params.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    subset = df[(df["dataset"] == dataset) & (df["method"] == method)]
    if subset.empty:
        return None
    counts = Counter(subset["best_params"].tolist())
    return json.loads(counts.most_common(1)[0][0])


def _clean_params(params: dict, estimator) -> dict:
    valid = set(estimator.get_params())
    return {k: v for k, v in params.items() if k in valid}


def _build_variant(variant: str, power_params: dict, ald_params: dict):
    if variant == "A0_distance_only":
        return PowerWeightedKNNClassifier(**power_params)
    params = dict(ald_params)
    if variant == "A1_density_only":
        params["distance_power"] = 0.0
    elif variant == "A2_ald_unclipped":
        params["density_clip"] = 1e12
    return ALDWKNNClassifier(**params)


def run_ablation(
    cfg: dict,
    run_name: str,
    dataset_names: list[str] | None = None,
    data_root: str | Path = "data",
) -> Path:
    seed = int(cfg["project"]["seed"])
    artifacts_root = Path(cfg.get("project", {}).get("output_root", "artifacts"))
    run_id = make_run_id(seed, run_name)
    run_dir = ensure_dir(artifacts_root / "runs" / run_id)

    imputer_strategy = cfg.get("preprocessing", {}).get("imputer", {}).get("strategy", "median")
    save_predictions = bool(cfg.get("artifacts", {}).get("save_predictions", True))
    datasets = dataset_names if dataset_names is not None else list(cfg["datasets"])
    outer_cfg = cfg["cv"]["outer"]
    n_outer_splits = int(outer_cfg["splits"])

    main_run = helpers.resolve_main_run(cfg)
    if main_run is None:
        print("[ablation] no main run found; using parametric defaults", flush=True)
    else:
        print(f"[ablation] reusing best params from {main_run.name}", flush=True)

    metrics_rows, pred_rows, params_rows, timing_rows, fold_rows = [], [], [], [], []
    t_start = time.perf_counter()

    for dataset_index, name in enumerate(datasets):
        frame = load_dataset(name, data_root)
        rexp._validate_dimensions(name, frame)
        X = frame.drop(columns=["target"]).to_numpy(dtype=np.float64)
        y = frame["target"].to_numpy()

        power_defaults = _modal_best_params(main_run, name, "power_wknn") or DEFAULT_POWER_PARAMS
        ald_defaults = _modal_best_params(main_run, name, "ald_wknn") or DEFAULT_ALD_PARAMS
        power_params = _clean_params(power_defaults, PowerWeightedKNNClassifier())
        ald_params = _clean_params(ald_defaults, ALDWKNNClassifier())

        outer = make_outer_cv(cfg)
        for variant in VARIANTS:
            estimator = _build_variant(variant, power_params, ald_params)
            fixed_params = params_to_jsonable(estimator.get_params())
            for fold_counter, (train_idx, test_idx) in enumerate(outer.split(X, y)):
                repeat_index = fold_counter // n_outer_splits
                fold_index = fold_counter % n_outer_splits
                fold_seed = derive_seed(seed, dataset_index, repeat_index, fold_index)

                pipe = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy=imputer_strategy)),
                        ("scaler", StandardScaler()),
                        ("classifier", estimator),
                    ]
                )
                key = f"{name}/{variant}/repeat{repeat_index}/fold{fold_index}"
                try:
                    t0 = time.perf_counter()
                    pipe.fit(X[train_idx], y[train_idx])
                    fit_seconds = time.perf_counter() - t0
                    t1 = time.perf_counter()
                    y_pred = pipe.predict(X[test_idx])
                    predict_seconds = time.perf_counter() - t1
                except Exception as exc:
                    raise RuntimeError(f"[hard stop] variant {variant} failed on {key}") from exc

                proba = pipe.predict_proba(X[test_idx])
                metrics = compute_metrics(y[test_idx], y_pred)
                if not np.all(np.isfinite(list(metrics.values()))) or not np.isfinite(proba).all():
                    raise FloatingPointError(f"[hard stop] non-finite output on {key}")

                base = {
                    "run_id": run_id,
                    "dataset": name,
                    "method": variant,
                    "repeat": repeat_index,
                    "outer_fold": fold_index,
                    "seed": fold_seed,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                }
                metrics_rows.append(
                    {**base, "best_params": json.dumps(fixed_params), **metrics,
                     "fit_seconds": fit_seconds, "predict_seconds": predict_seconds}
                )
                params_rows.append({**base, "best_score": np.nan, "best_params": json.dumps(fixed_params)})
                timing_rows.append({**base, "fit_seconds": fit_seconds,
                                    "predict_seconds": predict_seconds})
                fold_rows.append(
                    {**base, "train_idx_sha256": rexp._idx_hash(train_idx),
                     "test_idx_sha256": rexp._idx_hash(test_idx)}
                )
                if save_predictions:
                    classes = pipe.named_steps["classifier"].classes_
                    for row_pos, test_row in enumerate(test_idx):
                        pred_rows.append(
                            {
                                **base,
                                "row_id": int(test_row),
                                "y_true": y[test_row],
                                "y_pred": y_pred[row_pos],
                                **{f"score_class_{j}": float(proba[row_pos, j])
                                   for j in range(len(classes))},
                            }
                        )
                print(f"[{run_id}] {key} f1_macro={metrics['f1_macro']:.4f}", flush=True)

    t_run = time.perf_counter() - t_start
    write_manifest(
        run_dir,
        {
            "run_id": run_id,
            "run_name": run_name,
            "kind": "ablation",
            "seed": seed,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "environment": rexp._environment_info(),
            "datasets": list(datasets),
            "methods": list(VARIANTS),
            "main_run": str(main_run) if main_run else None,
            "wall_seconds": t_run,
        },
    )
    dump_config(cfg, run_dir / "config.snapshot.yaml")
    write_csv(pd.DataFrame(metrics_rows), run_dir / "metrics_outer.csv")
    write_csv(pd.DataFrame(params_rows), run_dir / "best_params.csv")
    write_csv(pd.DataFrame(timing_rows), run_dir / "timing.csv")
    write_csv(pd.DataFrame(fold_rows), run_dir / "fold_metadata.csv")
    if save_predictions and pred_rows:
        ensure_dir(run_dir)
        pd.DataFrame(pred_rows).to_csv(
            run_dir / "predictions_outer.csv.gz", index=False, compression="gzip"
        )
    helpers.update_latest_link(cfg, run_dir)
    print(f"[{run_id}] completed in {t_run:.1f}s -> {run_dir}", flush=True)
    return run_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the ALD-WKNN ablation study.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--run-name", default="ablation")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    run_ablation(cfg, args.run_name, data_root=args.data_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
