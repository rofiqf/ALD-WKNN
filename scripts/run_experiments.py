"""Run the nested cross-validation benchmark for all configured datasets and methods.

Produces one immutable run directory under ``artifacts/runs/<run_id>/`` with
manifest, config snapshot, per-fold metrics, predictions, selected parameters,
timing, and fold metadata, exactly as specified in PROJECT_SPEC.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import _helpers as helpers  # noqa: F401  (sys.path bootstrap)

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aldwknn import __version__ as pkg_version
from aldwknn.cv import make_inner_cv, make_outer_cv
from aldwknn.data import EXPECTED_SCHEMA, DatasetValidationError, load_dataset
from aldwknn.metrics import compute_metrics
from aldwknn.model import ALDWKNNClassifier, PowerWeightedKNNClassifier
from aldwknn.reporting import ensure_dir, write_csv, write_manifest
from aldwknn.utils import derive_seed, dump_config, load_config, make_run_id, params_to_jsonable

METRIC_COLUMNS = (
    "f1_macro",
    "balanced_accuracy",
    "accuracy",
    "precision_macro",
    "recall_macro",
)

PARAM_MAP = {
    "knn_uniform": {"k": "n_neighbors"},
    "knn_distance": {"k": "n_neighbors"},
    "power_wknn": {"k": "n_neighbors", "p": "distance_power"},
    "ald_wknn": {
        "k": "n_neighbors",
        "q": "density_neighbors",
        "p": "distance_power",
        "alpha": "density_alpha",
        "clip_c": "density_clip",
    },
}


def build_estimator(method: str):
    if method == "knn_uniform":
        return KNeighborsClassifier(weights="uniform", metric="euclidean")
    if method == "knn_distance":
        return KNeighborsClassifier(weights="distance", metric="euclidean")
    if method == "power_wknn":
        return PowerWeightedKNNClassifier()
    if method == "ald_wknn":
        return ALDWKNNClassifier()
    raise ValueError(f"Unknown method: {method}")


def build_grid(method: str, spec: dict) -> dict:
    """Map config grid keys to estimator parameters, prefixed for the pipeline."""
    estimator = build_estimator(method)
    valid = set(estimator.get_params())
    grid = {}
    for key, value in spec.items():
        mapped = PARAM_MAP.get(method, {}).get(key, key)
        if mapped not in valid:
            raise ValueError(f"Grid key {key!r} is not a parameter of {method}")
        values = value if isinstance(value, list) else [value]
        grid[f"classifier__{mapped}"] = values
    return grid


def grid_size(grid: dict) -> int:
    size = 1
    for values in grid.values():
        size *= len(values)
    return size


def _idx_hash(idx: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(idx).tobytes()).hexdigest()


def _environment_info() -> dict:
    import sklearn

    from aldwknn import statistics as stats_mod  # noqa: F401

    import scipy

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "aldwknn_version": pkg_version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _validate_dimensions(name: str, frame: pd.DataFrame) -> None:
    expected = EXPECTED_SCHEMA[name]
    n_samples = len(frame)
    n_features = frame.shape[1] - 1
    n_classes = int(frame["target"].nunique())
    if (
        n_samples != expected["n_samples"]
        or n_features != expected["n_features"]
        or n_classes != expected["n_classes"]
    ):
        raise DatasetValidationError(
            f"{name}: dimensions {n_samples}x{n_features}, {n_classes} classes differ "
            f"from expected {expected}"
        )


def run_pipeline(
    cfg: dict,
    run_name: str,
    dataset_names: list[str] | None = None,
    data_root: str | Path = "data",
    kind: str = "main",
) -> Path:
    seed = int(cfg["project"]["seed"])
    artifacts_root = Path(cfg.get("project", {}).get("output_root", "artifacts"))
    run_id = make_run_id(seed, run_name)
    run_dir = ensure_dir(artifacts_root / "runs" / run_id)

    imputer_strategy = cfg.get("preprocessing", {}).get("imputer", {}).get("strategy", "median")
    inner_jobs = int(cfg.get("parallel", {}).get("inner_jobs", 1))
    save_models = bool(cfg.get("artifacts", {}).get("save_models", False))
    save_predictions = bool(cfg.get("artifacts", {}).get("save_predictions", True))
    primary = cfg.get("evaluation", {}).get("primary_metric", "f1_macro")

    datasets = dataset_names if dataset_names is not None else list(cfg["datasets"])
    methods = cfg["models"]
    outer_cfg = cfg["cv"]["outer"]
    n_outer_splits = int(outer_cfg["splits"])

    metrics_rows, pred_rows, params_rows, timing_rows, fold_rows, dataset_rows = (
        [], [], [], [], [], [],
    )
    fits_counter = 0
    t_start = time.perf_counter()

    for dataset_index, name in enumerate(datasets):
        frame = load_dataset(name, data_root)
        _validate_dimensions(name, frame)
        X = frame.drop(columns=["target"]).to_numpy(dtype=np.float64)
        y = frame["target"].to_numpy()

        meta_path = Path(data_root) / "metadata" / f"{name}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        dataset_rows.append(
            {
                "dataset": name,
                "n_samples": int(len(frame)),
                "n_features": int(X.shape[1]),
                "n_classes": int(frame["target"].nunique()),
                "sha256": meta.get("sha256", ""),
                "source_url": meta.get("source_url", ""),
                "doi": meta.get("doi", ""),
            }
        )

        outer = make_outer_cv(cfg)
        n_outer_folds = n_outer_splits * int(outer_cfg.get("repeats", 1))
        for method, spec in methods.items():
            estimator = build_estimator(method)
            grid = build_grid(method, spec)
            n_inner_splits = int(cfg["cv"]["inner"]["splits"])
            fits_counter += (grid_size(grid) * n_inner_splits + 1) * n_outer_folds

            for fold_counter, (train_idx, test_idx) in enumerate(outer.split(X, y)):
                repeat_index = fold_counter // n_outer_splits
                fold_index = fold_counter % n_outer_splits
                inner_seed = derive_seed(seed, dataset_index, repeat_index, fold_index)
                inner = make_inner_cv(cfg, inner_seed)

                pipe = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy=imputer_strategy)),
                        ("scaler", StandardScaler()),
                        ("classifier", estimator),
                    ]
                )
                search = GridSearchCV(
                    pipe,
                    grid,
                    scoring=primary,
                    cv=inner,
                    n_jobs=inner_jobs,
                    refit=True,
                    error_score="raise",
                )
                key = f"{name}/{method}/repeat{repeat_index}/fold{fold_index}"
                try:
                    t0 = time.perf_counter()
                    search.fit(X[train_idx], y[train_idx])
                    fit_seconds = time.perf_counter() - t0
                    best = search.best_estimator_
                    t1 = time.perf_counter()
                    y_pred = best.predict(X[test_idx])
                    predict_seconds = time.perf_counter() - t1
                except Exception as exc:  # hard stop: one method failing on one fold
                    raise RuntimeError(
                        f"[hard stop] method {method} failed on fold {key}"
                    ) from exc

                proba = best.predict_proba(X[test_idx]) if hasattr(best, "predict_proba") else None
                metrics = compute_metrics(y[test_idx], y_pred)
                if not np.all(np.isfinite(list(metrics.values()))):
                    raise FloatingPointError(f"[hard stop] non-finite metrics on {key}")
                if proba is not None and not np.isfinite(proba).all():
                    raise FloatingPointError(f"[hard stop] non-finite scores on {key}")

                best_params = params_to_jsonable(
                    {k.split("__", 1)[-1]: v for k, v in search.best_params_.items()}
                )
                base = {
                    "run_id": run_id,
                    "dataset": name,
                    "method": method,
                    "repeat": repeat_index,
                    "outer_fold": fold_index,
                    "seed": inner_seed,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                }
                metrics_rows.append(
                    {**base, "best_params": json.dumps(best_params), **metrics,
                     "fit_seconds": fit_seconds, "predict_seconds": predict_seconds}
                )
                params_rows.append(
                    {**base, "best_score": float(search.best_score_),
                     "best_params": json.dumps(best_params)}
                )
                timing_rows.append(
                    {**base, "fit_seconds": fit_seconds, "predict_seconds": predict_seconds}
                )
                fold_rows.append(
                    {**base, "train_idx_sha256": _idx_hash(train_idx),
                     "test_idx_sha256": _idx_hash(test_idx)}
                )
                if save_predictions and proba is not None:
                    classes = best.named_steps["classifier"].classes_
                    for row_pos, test_row in enumerate(test_idx):
                        pred_rows.append(
                            {
                                "run_id": run_id,
                                "dataset": name,
                                "method": method,
                                "repeat": repeat_index,
                                "outer_fold": fold_index,
                                "row_id": int(test_row),
                                "y_true": y[test_row],
                                "y_pred": y_pred[row_pos],
                                **{f"score_class_{j}": float(proba[row_pos, j])
                                   for j in range(len(classes))},
                            }
                        )
                if save_models:
                    model_path = (
                        artifacts_root / "models" / run_id / name / method
                        / f"repeat_{repeat_index:02d}_fold_{fold_index:02d}.joblib"
                    )
                    ensure_dir(model_path.parent)
                    joblib.dump(best, model_path)
                print(
                    f"[{run_id}] {key} best={best_params} "
                    f"f1_macro={metrics['f1_macro']:.4f}",
                    flush=True,
                )

    t_run = time.perf_counter() - t_start

    write_manifest(
        run_dir,
        {
            "run_id": run_id,
            "run_name": run_name,
            "kind": kind,
            "seed": seed,
            "config_path": None,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "environment": _environment_info(),
            "datasets": list(datasets),
            "methods": list(methods),
            "primary_metric": primary,
            "n_fits": int(fits_counter),
            "wall_seconds": t_run,
        },
    )
    dump_config(cfg, run_dir / "config.snapshot.yaml")
    write_csv(pd.DataFrame(dataset_rows), run_dir / "dataset_manifest.csv")
    write_csv(pd.DataFrame(metrics_rows), run_dir / "metrics_outer.csv")
    write_csv(pd.DataFrame(params_rows), run_dir / "best_params.csv")
    write_csv(pd.DataFrame(timing_rows), run_dir / "timing.csv")
    write_csv(pd.DataFrame(fold_rows), run_dir / "fold_metadata.csv")
    if save_predictions and pred_rows:
        pred_df = pd.DataFrame(pred_rows)
        ensure_dir(run_dir)
        pred_df.to_csv(run_dir / "predictions_outer.csv.gz", index=False, compression="gzip")

    if kind == "main":
        write_runtime_projection(cfg, t_run, fits_counter)
        helpers.update_latest_link(cfg, run_dir)
    print(f"[{run_id}] completed in {t_run:.1f}s -> {run_dir}", flush=True)
    return run_dir


def _fits_for_config(path: Path) -> int | None:
    if not path.exists():
        return None
    cfg = load_config(path)
    datasets = len(cfg.get("datasets", {}))
    methods = cfg.get("models", {})
    if not datasets or not methods:
        return None
    outer = cfg["cv"]["outer"]
    inner_splits = int(cfg["cv"]["inner"]["splits"])
    total = 0
    for spec in methods.values():
        grid = 1
        for value in spec.values():
            grid *= len(value) if isinstance(value, list) else 1
        total += grid * inner_splits + 1
    return datasets * int(outer["splits"]) * int(outer.get("repeats", 1)) * total


def write_runtime_projection(cfg: dict, t_run: float, fits_this_run: int) -> None:
    """T_projected = T_run x (expected full fits / smoke fits)."""
    logs_dir = ensure_dir(Path(cfg.get("project", {}).get("output_root", "artifacts")) / "logs")
    smoke_fits = _fits_for_config(helpers.REPO_ROOT / "configs" / "smoke.yaml")
    full_fits = _fits_for_config(helpers.REPO_ROOT / "configs" / "experiment.yaml")

    lines = [
        "Runtime projection (see PROJECT_SPEC.md, 'Resource requirements')",
        f"Formula: T_projected = T_run * (F_full / F_smoke)",
        f"F = n_datasets * outer_splits * outer_repeats * sum_methods(grid_size_m * inner_splits + 1)",
        f"T_run (this run, wall seconds): {t_run:.3f}",
        f"F_this_run (counted): {fits_this_run}",
        f"F_smoke (configs/smoke.yaml): {smoke_fits}",
        f"F_full (configs/experiment.yaml): {full_fits}",
    ]
    if smoke_fits and full_fits:
        ratio = full_fits / smoke_fits
        lines.append(f"F_full / F_smoke = {ratio:.1f}")
        lines.append(f"T_projected = {t_run:.3f} * {ratio:.1f} = {t_run * ratio:.1f} seconds")
    else:
        lines.append("T_projected: unavailable (smoke/experiment config missing)")
    (logs_dir / "runtime_projection.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the ALD-WKNN nested-CV benchmark.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--run-name", default="main")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    run_dir = run_pipeline(cfg, args.run_name, data_root=args.data_root)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({"run_id": manifest["run_id"], "run_dir": str(run_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
