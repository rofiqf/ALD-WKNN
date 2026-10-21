"""Dataset acquisition and integrity validation (UCI-first, OpenML fallback).

Raw files land in data/raw/<dataset>/, metadata in data/metadata/, and
SHA256 checksums in data/hashes/. Any schema/shape mismatch hard-stops
the pipeline instead of silently continuing.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from aldwknn.utils import sha256_of_file

# Expected characteristics from PROJECT_SPEC.md (verified against UCI records).
EXPECTED_SCHEMA = {
    "iris": {"n_samples": 150, "n_features": 4, "n_classes": 3},
    "wine": {"n_samples": 178, "n_features": 13, "n_classes": 3},
    "seeds": {"n_samples": 210, "n_features": 7, "n_classes": 3},
    "glass": {"n_samples": 214, "n_features": 9, "n_classes": 6},
    "sonar": {"n_samples": 208, "n_features": 60, "n_classes": 2},
    "ionosphere": {"n_samples": 351, "n_features": 34, "n_classes": 2},
    "breast_cancer_wisconsin_diagnostic": {"n_samples": 569, "n_features": 30, "n_classes": 2},
    "banknote": {"n_samples": 1372, "n_features": 4, "n_classes": 2},
}

DATASET_SOURCES = {
    "iris": {"uci_id": 53, "doi": "10.24432/C56C76", "url": "https://archive.ics.uci.edu/dataset/53/iris"},
    "wine": {"uci_id": 109, "doi": "10.24432/C5PC7J", "url": "https://archive.ics.uci.edu/dataset/109/wine"},
    "seeds": {"uci_id": 236, "doi": "10.24432/C5H30K", "url": "https://archive.ics.uci.edu/dataset/236/seeds"},
    "glass": {"uci_id": 42, "doi": None, "url": "https://archive.ics.uci.edu/dataset/42/glass+identification"},
    "sonar": {"uci_id": 151, "doi": "10.24432/C5T01Q", "url": "https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks"},
    "ionosphere": {"uci_id": 52, "doi": "10.24432/C5W01B", "url": "https://archive.ics.uci.edu/dataset/52/ionosphere"},
    "breast_cancer_wisconsin_diagnostic": {"uci_id": 17, "doi": "10.24432/C5DW2B", "url": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic"},
    "banknote": {"uci_id": 267, "doi": "10.24432/C55P57", "url": "https://archive.ics.uci.edu/dataset/267/banknote+authentication"},
}


class DatasetValidationError(RuntimeError):
    """Raised when downloaded data does not match the expected schema."""


# OpenML fallback ids, used only when the UCI import API does not serve the
# dataset (e.g. Seeds id=236: "exists in the repository, but is not available
# for import"). Identity is enforced by the same schema validation as UCI.
OPENML_FALLBACK_IDS = {
    "seeds": 1499,
    "glass": 41,
    "sonar": 40,
    "ionosphere": 59,
    "breast_cancer_wisconsin_diagnostic": 1510,
    "banknote": 1462,
}


def _fetch_from_openml(name: str, ds_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Fetch features/targets from OpenML as a fallback source."""
    import openml

    openml_id = OPENML_FALLBACK_IDS.get(name)
    if openml_id is None:
        raise DatasetValidationError(f"{name}: no OpenML fallback id configured")
    dataset = openml.datasets.get_dataset(openml_id)
    X, y, _, _ = dataset.get_data(target_name=dataset.default_target_attribute)
    features = pd.DataFrame(X).reset_index(drop=True)
    targets = pd.DataFrame({dataset.default_target_attribute: np.asarray(y)}).reset_index(drop=True)
    return features, targets, dataset.default_target_attribute


def fetch_uci_dataset(name: str, cfg: dict, data_root: str | Path = "data") -> pd.DataFrame:
    """Download one dataset via ucimlrepo (OpenML fallback), validate, persist raw + metadata + hash."""
    data_root = Path(data_root)
    ds_cfg = cfg["datasets"][name]
    expected = EXPECTED_SCHEMA[name]

    source = "UCI"
    try:
        from ucimlrepo import fetch_uci_dataset as _fetch

        ds = _fetch(id=int(ds_cfg["id"]))
        features: pd.DataFrame = ds.data.features.copy()
        targets: pd.DataFrame = ds.data.targets.copy()
        target_col = targets.columns[0]
    except Exception as uci_exc:
        # Spec allows OpenML strictly as fallback; identity is validated below.
        source = "OpenML-fallback"
        try:
            features, targets, target_col = _fetch_from_openml(name, ds_cfg)
        except Exception as openml_exc:
            raise DatasetValidationError(
                f"{name}: UCI import failed ({uci_exc}); "
                f"OpenML fallback failed ({openml_exc})"
            ) from openml_exc

    for col in ds_cfg.get("drop_columns", []):
        if col not in features.columns:
            raise DatasetValidationError(
                f"{name}: configured drop column {col!r} not present in features"
            )
        features = features.drop(columns=[col])

    if features.shape[1] != expected["n_features"]:
        raise DatasetValidationError(
            f"{name}: expected {expected['n_features']} features, got {features.shape[1]}"
        )
    if len(features) != expected["n_samples"]:
        raise DatasetValidationError(
            f"{name}: expected {expected['n_samples']} samples, got {len(features)}"
        )

    target_col = targets.columns[0]
    y = targets[target_col]
    if y.isna().any():
        raise DatasetValidationError(f"{name}: target contains missing values")
    n_classes = int(y.nunique())
    if n_classes != expected["n_classes"]:
        raise DatasetValidationError(
            f"{name}: expected {expected['n_classes']} classes, got {n_classes}"
        )

    frame = features.copy()
    frame["target"] = y.to_numpy()

    raw_dir = data_root / "raw" / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "data.csv"
    frame.to_csv(raw_path, index=False)

    digest = sha256_of_file(raw_path)
    hash_path = data_root / "hashes" / f"{name}.sha256"
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(f"{digest}  {raw_path.name}\n", encoding="utf-8")

    meta = {
        "dataset": name,
        "source": source,
        "uci_id": int(ds_cfg["id"]),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(frame)),
        "n_features": int(features.shape[1]),
        "n_classes": n_classes,
        "target_name": str(target_col),
        "source_url": DATASET_SOURCES[name]["url"],
        "doi": DATASET_SOURCES[name]["doi"],
        "sha256": digest,
    }
    meta_dir = data_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / f"{name}.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return frame


def load_dataset(name: str, data_root: str | Path = "data") -> pd.DataFrame:
    """Load a previously downloaded raw dataset, verifying its checksum first."""
    data_root = Path(data_root)
    raw_path = data_root / "raw" / name / "data.csv"
    hash_path = data_root / "hashes" / f"{name}.sha256"
    if not raw_path.exists():
        raise FileNotFoundError(f"{name}: raw data not found at {raw_path}")
    recorded = hash_path.read_text(encoding="utf-8").split()[0]
    actual = sha256_of_file(raw_path)
    if recorded != actual:
        raise DatasetValidationError(
            f"{name}: checksum mismatch (recorded {recorded}, actual {actual}). "
            "Pipeline stopped; manual validation required."
        )
    return pd.read_csv(raw_path)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Download and validate UCI datasets.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args(argv)

    from aldwknn.utils import load_config

    cfg = load_config(args.config)
    for name in cfg["datasets"]:
        print(f"[download] {name} ...", flush=True)
        try:
            frame = fetch_uci_dataset(name, cfg, args.data_root)
        except DatasetValidationError as exc:
            print(f"[download] HARD STOP: {exc}", file=sys.stderr)
            return 2
        print(f"[download] {name}: OK ({frame.shape[0]} x {frame.shape[1]-1})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
