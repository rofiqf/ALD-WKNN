"""Cross-validation protocol builders (see PROJECT_SPEC.md, 'Cross-validation protocol')."""

from __future__ import annotations

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


def make_outer_cv(cfg: dict) -> RepeatedStratifiedKFold:
    outer = cfg["cv"]["outer"]
    return RepeatedStratifiedKFold(
        n_splits=int(outer["splits"]),
        n_repeats=int(outer.get("repeats", 1)),
        random_state=int(outer.get("random_state", cfg["project"]["seed"])),
    )


def make_inner_cv(cfg: dict, seed: int) -> StratifiedKFold:
    inner = cfg["cv"]["inner"]
    return StratifiedKFold(
        n_splits=int(inner["splits"]),
        shuffle=bool(inner.get("shuffle", True)),
        random_state=seed if inner.get("shuffle", True) else None,
    )
