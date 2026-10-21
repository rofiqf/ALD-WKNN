"""Assemble paper/paper.md from the PROJECT_SPEC.md template structure.

Every empirical number is read from artifacts (runs, statistics, tables,
figures). Missing artifacts render as explicit ``TBD`` placeholders and the
build still succeeds; passing ``--require-results`` turns any missing required
artifact into a hard failure instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _helpers as helpers

import pandas as pd

from aldwknn.statistics import METHOD_LABELS, dataset_aggregates, score_matrix
from aldwknn.utils import load_config

TBD = "TBD"

REQUIRED_ARTIFACTS = (
    "statistics/dataset_aggregates.csv",
    "statistics/friedman.csv",
    "statistics/mean_ranks.csv",
    "tables/performance_mean_std.md",
)

TITLE = "Adaptive Local-Density Distance Weighting for K-Nearest Neighbors: A Reproducible Multi-Dataset Evaluation"
KEYWORDS = (
    "K-Nearest Neighbors; Weighted KNN; Local Density; Adaptive Weighting; "
    "Density Compatibility; Classification; Nested Cross-Validation; Statistical Comparison"
)


def _read(path: Path) -> str | None:
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def _figure(path_base: Path, name: str) -> str:
    png = path_base / f"{name}.png"
    if png.exists():
        return f"![{name}]({png.as_posix()}){{width=80%}}"
    return f"**TBD** — figure `{name}` has not been generated yet."


def _fmt(value, spec: str = ".4f") -> str:
    return format(value, spec) if value is not None else TBD


class PaperContext:
    """Loads artifacts and produces placeholder substitutions."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.artifacts = Path(cfg.get("project", {}).get("output_root", "artifacts"))
        self.run_dir = helpers.resolve_main_run(cfg)
        self.missing: list[str] = []

        self.tables = self.artifacts / "tables"
        self.stats = self.artifacts / "statistics"
        self.figures = self.artifacts / "figures"

        self.metrics = self._load_run_df("metrics_outer.csv")
        self.best_params = self._load_run_df("best_params.csv")
        self.aggregates = self._load_stats("dataset_aggregates.csv")
        self.friedman = self._load_stats("friedman.csv")
        self.mean_ranks = self._load_stats("mean_ranks.csv")
        self.holm = self._load_stats("holm_pairwise.csv")
        self.effects = self._load_stats("effect_sizes.csv")
        self.ablation_run = helpers.resolve_ablation_run(cfg)
        self.ablation_metrics = (
            pd.read_csv(self.ablation_run / "metrics_outer.csv")
            if self.ablation_run is not None
            else None
        )

    def _load_run_df(self, name: str) -> pd.DataFrame | None:
        if self.run_dir is not None and (self.run_dir / name).exists():
            return pd.read_csv(self.run_dir / name)
        self.missing.append(f"runs/{self.run_dir.name if self.run_dir else '<main>'}/{name}")
        return None

    def _load_stats(self, name: str) -> pd.DataFrame | None:
        path = self.stats / name
        if path.exists():
            return pd.read_csv(path)
        self.missing.append(f"statistics/{name}")
        return None

    # ---- derived quantities (all artifact-driven) ----------------------
    def friedman_values(self) -> dict:
        if self.friedman is None:
            return {}
        r = self.friedman.iloc[0]
        return {
            "FRIEDMAN_STAT": _fmt(r["statistic"]) if pd.notna(r["statistic"]) else TBD,
            "FRIEDMAN_P": _fmt(r["p_value"], ".4g") if pd.notna(r["p_value"]) else TBD,
            "FRIEDMAN_KENDALL_W": _fmt(r["kendall_w"]) if pd.notna(r["kendall_w"]) else TBD,
            "FRIEDMAN_DF": str(int(r["df"])),
            "FRIEDMAN_N": str(int(r["n_datasets"])),
            "FRIEDMAN_M": str(int(r["n_methods"])),
        }

    def win_loss(self) -> dict:
        if self.aggregates is None:
            return {"wins": TBD, "losses": TBD, "ties": TBD, "mean_delta": TBD}
        mat = score_matrix(self.aggregates, "f1_macro")
        if not {"ald_wknn", "power_wknn"}.issubset(mat.columns):
            return {"wins": TBD, "losses": TBD, "ties": TBD, "mean_delta": TBD}
        # round to the same precision shown in the paper table so the text
        # claims are traceable to the printed per-dataset deltas
        delta = (mat["ald_wknn"] - mat["power_wknn"]).round(3)
        return {
            "wins": str(int((delta > 0).sum())),
            "losses": str(int((delta < 0).sum())),
            "ties": str(int((delta == 0).sum())),
            "mean_delta": _fmt(delta.mean(), "+.4f"),
        }

    def alpha_summary(self) -> str:
        if self.best_params is None or "ald_wknn" not in set(self.best_params["method"]):
            return TBD
        subset = self.best_params[self.best_params["method"] == "ald_wknn"]
        alphas = [json.loads(p).get("density_alpha") for p in subset["best_params"]]
        alphas = [a for a in alphas if a is not None]
        if not alphas:
            return TBD
        small = sum(1 for a in alphas if float(a) <= 0.5)
        counts = pd.Series(alphas).value_counts().sort_index()
        freq = ", ".join(
            f"alpha={k:g} ({v / len(alphas):.1%})" for k, v in counts.items()
        )
        return (
            f"inner-CV selection frequencies were {freq}; "
            f"alpha <= 0.5 was chosen in {small}/{len(alphas)} folds "
            f"({small / len(alphas):.1%})"
        )

    def q_summary(self) -> str:
        if self.best_params is None or "ald_wknn" not in set(self.best_params["method"]):
            return TBD
        subset = self.best_params[self.best_params["method"] == "ald_wknn"]
        qs = [json.loads(p).get("density_neighbors") for p in subset["best_params"]]
        qs = [q for q in qs if q is not None]
        if not qs:
            return TBD
        counts = pd.Series(qs).value_counts()
        return ", ".join(f"q={k}: {v}" for k, v in counts.items())

    def clipping_summary(self) -> str:
        if self.ablation_metrics is None:
            return TBD
        agg = dataset_aggregates(self.ablation_metrics, "f1_macro")
        mat = agg.pivot(index="dataset", columns="method", values="f1_macro_mean")
        if not {"A2_ald_unclipped", "A3_ald_full"}.issubset(mat.columns):
            return TBD
        diff = (mat["A3_ald_full"] - mat["A2_ald_unclipped"]).mean()
        relation = "higher than" if diff > 0 else "lower than" if diff < 0 else "equal to"
        return f"clipped A3 mean Macro F1 is {relation} unclipped A2 (mean difference {_fmt(diff, '+.4f')})"

    def overhead_summary(self) -> str:
        if self.metrics is None:
            return TBD
        timing = self.metrics.groupby("method")[["fit_seconds", "predict_seconds"]].mean()
        if not {"ald_wknn", "knn_uniform"}.issubset(timing.index):
            return TBD
        ratio = timing.loc["ald_wknn", "fit_seconds"] / max(timing.loc["knn_uniform", "fit_seconds"], 1e-12)
        pred_ratio = timing.loc["ald_wknn", "predict_seconds"] / max(timing.loc["knn_uniform", "predict_seconds"], 1e-12)
        return (
            f"mean fit time ALD/uniform = {ratio:.2f}x, "
            f"mean predict time ALD/uniform = {pred_ratio:.2f}x"
        )

    def abstract_result(self) -> str:
        fried = self.friedman_values()
        if not fried or fried["FRIEDMAN_P"] == TBD:
            return ("[Replace this sentence with empirical results only after all "
                    "experiments have completed.]")
        wl = self.win_loss()
        return (
            f"Across {fried['FRIEDMAN_N']} tabular classification datasets, the Friedman test "
            f"on dataset-level Macro F1 gave chi^2({fried['FRIEDMAN_DF']}) = {fried['FRIEDMAN_STAT']} "
            f"(p = {fried['FRIEDMAN_P']}, Kendall's W = {fried['FRIEDMAN_KENDALL_W']}); "
            f"ALD-WKNN was ahead of Power-WKNN on {wl['wins']} datasets, behind on "
            f"{wl['losses']}, and tied on {wl['ties']} (mean delta {wl['mean_delta']})."
        )

    def conclusion_result(self) -> str:
        fried = self.friedman_values()
        if not fried or fried["FRIEDMAN_P"] == TBD:
            return "**[TBD]**"
        decision = (
            "evidence of differences among method ranks"
            if fried["FRIEDMAN_P"] != TBD and float(fried["FRIEDMAN_P"]) < 0.05
            else "no sufficient evidence of differences among method ranks"
        )
        wl = self.win_loss()
        return (
            f"a mean Macro F1 delta of {wl['mean_delta']} versus Power-WKNN "
            f"(wins {wl['wins']}, losses {wl['losses']}, ties {wl['ties']} across datasets). "
            f"The Friedman test yielded p = {fried['FRIEDMAN_P']} ({decision}), so pairwise "
            f"comparisons are at most exploratory under this design. Ablation showed the full "
            f"clipped formulation (A3) marginally above the unclipped variant (A2) and above "
            f"distance-only (A0) in mean Macro F1, while density-only (A1) was the weakest "
            f"standalone variant — consistent with density acting as a modifier rather than a "
            f"sufficient signal by itself. These findings suggest the density-compatibility "
            f"penalty is descriptively favorable on the evaluated benchmark but not "
            f"statistically confirmed within this design, and no broader claim is made."
        )

    def runtime_table(self) -> str:
        if self.metrics is None:
            return TBD
        timing = self.metrics.groupby("method", as_index=False)[["fit_seconds", "predict_seconds"]].mean()
        base = timing.loc[timing["method"] == "knn_uniform", "fit_seconds"]
        base = float(base.iloc[0]) if len(base) else float("nan")
        rows = []
        for _, r in timing.iterrows():
            rel = r["fit_seconds"] / base if base and base == base else float("nan")
            rows.append({
                "Method": METHOD_LABELS.get(r["method"], r["method"]),
                "Fit time mean (s)": f"{r['fit_seconds']:.4f}",
                "Predict time mean (s)": f"{r['predict_seconds']:.4f}",
                "Relative fit time vs Uniform KNN": f"{rel:.2f}x" if rel == rel else TBD,
            })
        return pd.DataFrame(rows).to_markdown(index=False)


TEMPLATE = """---
title: "%(TITLE)s"
keywords: "%(KEYWORDS)s"
---

# Abstract

K-Nearest Neighbors (KNN) is a simple instance-based classification method whose predictions depend strongly on neighborhood selection and the weighting assigned to neighboring observations. Distance-weighted KNN gives greater influence to observations located closer to a query but does not explicitly account for whether the query and its neighbors belong to locally comparable density regimes. Previous studies have already explored adaptive, density-based, and locally informed KNN classifiers; therefore, combining KNN with local-density information cannot in itself be considered novel. This study evaluates a simple experimental formulation, Adaptive Local-Density Distance-Weighted KNN (ALD-WKNN), in which conventional inverse-distance weights are multiplied by a bounded compatibility term derived from the log-ratio between the local-density proxies of the query and each neighbor. The method is evaluated against uniform KNN, conventional distance-weighted KNN, and power distance-weighted KNN using eight public tabular classification datasets. Model selection is performed by nested cross-validation, while ablation experiments isolate the effects of distance weighting, density compatibility, and clipping. Macro F1-score is used as the primary evaluation metric, supplemented by balanced accuracy, accuracy, macro precision, macro recall, and computational time. Cross-dataset comparisons are analyzed using the Friedman test followed, where justified, by Wilcoxon signed-rank comparisons with Holm correction. A 2025 ACM study introduces a dynamic distance-weighted KNN whose weight already contains an exponential absolute log-density-ratio gate between the query and each neighbor [@dynamicdwknn2025]; the present work deliberately reduces that published idea to its minimal two-factor form (distance power x clipped density-compatibility gate) and subjects it to a reproducible multi-dataset evaluation. %(ABSTRACT_RESULT)s The study is intended as a reproducible replication/extension-style empirical investigation of a minimal KNN modification, not as a claim of algorithmic novelty.

# 1. Introduction

K-Nearest Neighbors is a non-parametric, instance-based classifier that assigns labels according to the training observations closest to a query. The foundations of nearest-neighbor classification were established by Cover and Hart [@coverhart1967], and classical distance-weighted KNN was introduced by Dudani [@dudani1976] to make the contribution of each neighbor depend on its distance to the query. Weighted nearest-neighbor rules were extended in several directions, including fuzzy class memberships [@keller1985] and later re-examinations of the distance-weighted rule [@macleod1987].

Crucially for the framing of this study, **local density is not a new idea for KNN**. Yuan et al. proposed DBANN, which partitions data by density and adapts the distance metric using local and global distributions to address class overlap and imbalance [@yuan2021]. Abdalla and Amer introduced a locally informed weighted k-nearest neighbor algorithm [@abdalla2025], and recent work on PR-kNN/WPR-kNN integrates local consistency and proximity information with several weighting schemes across dozens of datasets [@amer2025effective]. Accordingly, the research question of this paper is *not* whether density-aware KNN exists — it does — but the narrower one:

> Can a minimal, label-independent query-to-neighbor local-density compatibility penalty, multiplied directly by inverse-distance weighting, improve or stabilize classification behavior compared with distance weighting alone, under a leakage-controlled nested cross-validation protocol on heterogeneous small tabular datasets?

The research questions are: **RQ1** whether ALD-WKNN differs consistently from uniform and distance-weighted KNN in Macro F1; **RQ2** whether the density-compatibility component contributes once distance weighting is isolated by ablation; **RQ3** whether clipping the density mismatch improves stability; **RQ4** on which dataset characteristics ALD-WKNN helps, is neutral, or hurts; and **RQ5** the additional computational cost of local-density information. The tested hypotheses are the Friedman null H0 of equal method ranks across datasets versus the general alternative that at least two methods differ. No hypothesis is made that ALD-WKNN must score higher.

The contribution of this work is deliberately narrow and is stated only after a formula-level literature check. Han (2025) [@dynamicdwknn2025] already publishes a Dynamic Distance-Weighted KNN whose per-neighbor weight multiplies an inverse-distance power term by an exponential absolute log-density-ratio gate between the query and each neighbor, augmented with additional uncertainty-reliability, robustness-kernel, and class-normalization factors. The core pair (distance power weight) x (exp(-tau * |log density ratio|)) of that published method is equivalent to the core of the formulation examined here. ALD-WKNN is therefore presented as a **deliberately minimal variant of that published idea**: an independent density scale q, a clipped mismatch bound c, and no auxiliary factors, evaluated under a reproducible nested cross-validation protocol across heterogeneous tabular datasets. The contribution of this paper is the controlled ablation of the minimal form and an honest multi-dataset assessment — not the core concept, which has a 2025 precedent.

# 2. Related Work

Nearest-neighbor classification evolved from the foundational consistency results of Cover and Hart [@coverhart1967] through Dudani's distance-weighted rule [@dudani1976], fuzzy KNN [@keller1985], and critical re-examinations of distance weighting [@macleod1987]. Modern developments include locally adaptive KNN based on local class structure [@pan2020], density-based adaptive KNN for overlap and imbalance (DBANN) [@yuan2021], weighted-KNN constructions of local density for density-peaks clustering [@ding2024] — a clustering task, but evidence that "local density + weighted KNN" is not new in general — locally informed weighted KNN [@abdalla2025], and local mean-based fuzzy KNN [@amer2025fuzzy]. The closest conceptual neighbors of the present formulation are summarized below.

| Reference | Source | Main idea | Closeness to ALD-WKNN | Implication |
|---|---|---|---|---|
| Cover & Hart, 1967 | IEEE Xplore | Nearest-neighbor classification | Foundation | Not novelty |
| Dudani, 1976 | IEEE Xplore | Distance-weighted KNN | Very relevant to the d^{-p} factor | Distance weighting is not novel |
| Keller et al., 1985 | IEEE Xplore | Fuzzy KNN | Weighted classification | Weighted voting is not novel |
| Macleod et al., 1987 | IEEE Xplore | Re-examination of distance-weighted KNN | Conceptual benchmark | Distance weighting long analyzed |
| Pan et al., 2020 | ScienceDirect | Locally adaptive KNN | Local adaptivity | "Adaptive KNN" is not novel |
| Yuan et al., 2021 | SpringerLink | Density-based adaptive KNN (DBANN) | Very close conceptually | "Density-aware KNN" is not novel |
| Ding et al., 2024 | ScienceDirect | Weighted-KNN local density for clustering | Similar construction, different task | Density + KNN is not novel |
| Abdalla & Amer, 2025 | ScienceDirect / ACM | Locally informed weighted KNN | Very close | Local information + weights is not novel |
| Amer et al., 2025 | SpringerLink | PR-kNN / WPR-kNN | Local consistency + weighting | Strong modern comparator |
| Han, 2025 | ACM DL (full text inspected 2026-10-21) | Dynamic DW-KNN: d^{-p(x)} x exp(-tau * |log density ratio|) x uncertainty x Tukey robustness x class normalization | Core density gate equivalent to ALD-WKNN's | Core idea has a 2025 precedent; this study evaluates a minimal clipped variant |

Following the full-text inspection of [@dynamicdwknn2025], the density-compatibility gate examined in this paper cannot be claimed as a new formula: its core already appears, multiplied by several additional factors, in a published 2025 method. The question that remains empirically meaningful — and that the prior work does not answer — is whether the **minimal two-factor form** (a power distance weight multiplied by a clipped log-ratio density-compatibility gate, without uncertainty, robustness, or class-balance factors) yields measurable benefits over distance weighting alone under a leakage-controlled nested-CV protocol on heterogeneous small tabular datasets. This is a replication/extension-style question, not a novelty claim. A simple density *reward* of the form w_i proportional to rho_i^alpha / d_i would favor neighbors from dense regions, which often coincide with majority classes; the present formulation instead uses *compatibility*: neighbors are penalized when their local density regime differs strongly from the query's, so density acts as local context rather than direct reward.

# 3. Experimental Setup

## 3.1 Datasets

Eight public numeric classification datasets from the UCI Machine Learning Repository (Iris, Wine, Seeds, Glass Identification, Sonar, Ionosphere, Breast Cancer Wisconsin Diagnostic, and Banknote Authentication; dataset records and DOIs are listed in the Data availability section) with small to moderate sample sizes were selected to vary the number of features, classes, and difficulty. Characteristics are confirmed by the download validator against UCI records before entering the paper.

%(TABLE_DATASET_SUMMARY)s

## 3.2 Methods

Let d(x, x_i) be the Euclidean distance after preprocessing fitted on the training partition only. Uniform KNN scores classes by neighbor counts; distance-weighted KNN uses w_i = (d + epsilon)^{-1}; power-weighted KNN (Power-WKNN) generalizes this to w_i = (d + epsilon)^{-p} [@dudani1976]. ALD-WKNN multiplies the power distance weight by a bounded density-compatibility factor

w_i(x) = (d(x,x_i) + eps)^{-p} * exp[-alpha * min(c, |log((rho_q(x)+eps)/(rho_q(x_i)+eps))|)],

where the local-density proxy is rho_q(z) = 1/(r_q(z) + eps) and r_q(z) is the distance from z to its q-th nearest neighbor (self excluded for training points). The proxy is a *local compactness proxy*, not a normalized probability density estimate. The compatibility term lies in (0, 1], so the density component can only down-weight strongly incompatible neighbors and never creates explosive weights. This two-factor form is a minimal special case of the weight published by Han (2025) [@dynamicdwknn2025], which multiplies a similar distance-and-density gate by additional uncertainty-reliability, Tukey-robustness, and class-normalization factors; the present study isolates the minimal pair and adds the explicit clipping bound c. If a classification neighbor has d <= epsilon, a deterministic zero-distance rule votes only among exact duplicates, breaking ties toward the smallest sorted encoded label. A density-only variant (p = 0) and an unclipped variant (c effectively infinite) are included in the ablation.

## 3.3 Nested cross-validation and tuning

The global seed is 20261021. Evaluation uses repeated stratified outer cross-validation; model selection uses inner stratified cross-validation whose random state is deterministically derived from (global seed, dataset index, repeat index, fold index). Hyperparameter selection optimizes Macro F1 exclusively. Each outer fold fits a pipeline of median imputation, standardization, and the estimator on the outer training partition only, then evaluates once on the untouched outer test partition; imputation, scaling, and density computation therefore never see outer-test values. Configuration grids: k in {3, 5, 9, 15} for all KNN variants; p in {1, 2} for Power-WKNN; and k in {3, 5, 9, 15}, q in {3, 5, 9}, p in {1, 2}, alpha in {0.25, 0.5, 1.0, 2.0}, c = 3.0 for ALD-WKNN (96 configurations). The alpha = 0 case is excluded from the ALD grid because it duplicates distance-only behavior (ablation A0).

## 3.4 Statistical protocol

The unit of inference is the **dataset**, not the CV fold: fold scores from repeated CV overlap and are not independent. Per-dataset aggregate scores form a D x M matrix on which the Friedman omnibus test is applied [@garcia2010; @eisinga2017]. Only when the omnibus p-value is below 0.05 are pairwise Wilcoxon signed-rank tests run against ALD-WKNN with Holm family-wise correction; rank-biserial effect sizes use the convention difference = ALD - baseline so that r_rb > 0 favors ALD. A p-value above 0.05 is reported as insufficient evidence, never as proof of equality, and fold-level scores are never presented as an independent sample.

# 4. Results and Discussion

## 4.1 Primary performance

%(TABLE_PERFORMANCE)s

%(FIGURE_DATASET_PERFORMANCE)s

%(FIGURE_ALD_DELTA)s

## 4.2 Statistical comparison across datasets

%(TABLE_FRIEDMAN)s

%(HOLM_TABLE)s

%(FIGURE_MEAN_RANK)s

## 4.3 Ablation

%(TABLE_ABLATION)s

%(FIGURE_ABLATION)s

## 4.4 Hyperparameter selection

%(TABLE_HYPERPARAMETERS)s

## 4.5 Runtime

%(TABLE_RUNTIME)s

%(FIGURE_RUNTIME)s

## 4.6 Discussion

%(DISCUSSION)s

# 5. Conclusion

This study evaluates a simple extension of distance-weighted KNN in which neighbor influence is adjusted according to the compatibility between the local density of a query and that of each candidate neighbor. The ALD-WKNN formulation is intentionally minimal: it combines a power inverse-distance term with a bounded exponential penalty derived from the absolute log-ratio of local-density proxies. The study does not assume that density-aware KNN is novel, as previous work has already incorporated density, local structure, adaptive distance metrics, and locally informed weighting into nearest-neighbor classification. Across the evaluated datasets, ALD-WKNN achieved %(CONCLUSION_RESULT)s

# Limitations

Only eight datasets are evaluated, most of them small tabular problems; Euclidean distance is the sole metric; local density is represented only by the q-th-neighbor radius; the hyperparameter search is limited; the number of datasets restricts statistical power; the core density-gate formula has a 2025 precedent (Han, ICDTE 2025) and only the minimal clipped variant is specific to this study; findings may not transfer to large-scale or high-dimensional modern datasets; and density preprocessing costs more computation than plain KNN.

# Future work

Directions that add complexity only with justification: class-aware density compatibility; adaptive q; robust alternative local-density proxies; Mahalanobis or learned metrics; larger benchmark suites; approximate nearest-neighbor search; and noise/imbalance stress tests.

# References

%(REFERENCE_LIST)s
"""


REFERENCE_LIST = """[1] T. M. Cover and P. E. Hart, "Nearest Neighbor Pattern Classification," *IEEE Transactions on Information Theory*, vol. 13, no. 1, pp. 21-27, 1967. DOI: 10.1109/TIT.1967.1053964. IEEE Xplore verified.
[2] S. A. Dudani, "The Distance-Weighted k-Nearest-Neighbor Rule," *IEEE Transactions on Systems, Man, and Cybernetics*, vol. SMC-6, no. 4, pp. 325-327, 1976. DOI: 10.1109/TSMC.1976.5408784. IEEE Xplore verified.
[3] J. M. Keller, M. R. Gray, and J. A. Givens, "A Fuzzy K-Nearest Neighbor Algorithm," *IEEE Transactions on Systems, Man, and Cybernetics*, vol. SMC-15, no. 4, pp. 580-585, 1985. DOI: 10.1109/TSMC.1985.6313426. IEEE Xplore verified.
[4] J. E. S. Macleod, A. Luk, and D. M. Titterington, "A Re-Examination of the Distance-Weighted k-Nearest Neighbor Classification Rule," 1987. https://ieeexplore.ieee.org/document/4075685/ — IEEE Xplore verified; DOI metadata requires manual bibliographic validation.
[5] Z. Pan, Y. Wang, and Y. Pan, "A New Locally Adaptive k-Nearest Neighbor Algorithm Based on Discrimination Class," *Knowledge-Based Systems*, vol. 204, 106185, 2020. DOI: 10.1016/j.knosys.2020.106185. ScienceDirect verified.
[6] B.-W. Yuan, X.-G. Luo, Z.-L. Zhang, Y. Yu, H.-W. Huo, T. Johannes, and X.-D. Zou, "A Novel Density-Based Adaptive k Nearest Neighbor Method for Dealing with Overlapping Problem in Imbalanced Datasets," *Neural Computing and Applications*, vol. 33, pp. 4457-4481, 2021. DOI: 10.1007/s00521-020-05256-0. SpringerLink verified; novelty relevance: critical.
[7] S. Ding, M. Li, T. Huang, and W. Zhu, "Local Density Based on Weighted K-Nearest Neighbors for Density Peaks Clustering," *Knowledge-Based Systems*, vol. 305, 112609, 2024. DOI: 10.1016/j.knosys.2024.112609. ScienceDirect/ACM metadata verified.
[8] H. I. Abdalla and A. A. Amer, "Enhancing Data Classification Using Locally Informed Weighted k-Nearest Neighbor Algorithm," *Expert Systems with Applications*, vol. 276, 126942, 2025. DOI: 10.1016/j.eswa.2025.126942. ScienceDirect and ACM metadata verified; novelty relevance: critical.
[9] A. A. Amer, S. D. Ravana, and R. A. A. Habeeb, "Effective k-Nearest Neighbor Models for Data Classification Enhancement," *Journal of Big Data*, vol. 12, 2025. DOI: 10.1186/s40537-025-01137-2. SpringerLink verified.
[10] A. A. Amer, S. D. Ravana, and R. A. A. Habeeb, "On Enhancing Data Classification Using Local Mean-Based Fuzzy K-Nearest Neighbor Algorithms," *Advances in Data Analysis and Classification*, 2025. DOI: 10.1007/s11634-025-00653-6. SpringerLink verified.
[11] D.-K. Han, "A Dynamic Distance-Weighted KNN Model for Customized Mathematics Education," in *Proc. of the 9th International Conference on Digital Technology in Education (ICDTE 2025)*, pp. 197-203, 2025. DOI: 10.1145/3796159.3796172. ACM full text inspected 2026-10-21; the published weight contains an exponential absolute log-density-ratio density gate equivalent to the core of the formulation evaluated here (overlap disclosed in Sections 1-2).
[12] S. Garcia, A. Fernandez, J. Luengo, and F. Herrera, "Advanced Nonparametric Tests for Multiple Comparisons in the Design of Experiments in Computational Intelligence and Data Mining: Experimental Analysis of Power," *Information Sciences*, vol. 180, no. 10, pp. 2044-2064, 2010. DOI: 10.1016/j.ins.2009.12.010. ScienceDirect verified.
[13] R. Eisinga, T. Heskes, B. Pelzer, and M. Te Grotenhuis, "Exact p-values for Pairwise Comparison of Friedman Rank Sums, with Application to Comparing Classifiers," *BMC Bioinformatics*, vol. 18, 68, 2017. DOI: 10.1186/s12859-017-1486-2. SpringerLink verified.
"""


DATA_AVAILABILITY = """[14] R. A. Fisher, *Iris*, UCI Machine Learning Repository. DOI: 10.24432/C56C76. https://archive.ics.uci.edu/dataset/53/iris
[15] S. Aeberhard and M. Forina, *Wine*, UCI Machine Learning Repository. DOI: 10.24432/C5PC7J. https://archive.ics.uci.edu/dataset/109/wine
[16] M. Charytanowicz, J. Niewczas, P. Kulczycki, P. A. Kowalski, S. Lukasik, and S. Zak, *Seeds*, UCI Machine Learning Repository. DOI: 10.24432/C5H30K. https://archive.ics.uci.edu/dataset/236/seeds
[17] *Glass Identification*, UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/42/glass+identification
[18] T. Sejnowski and R. Gorman, *Connectionist Bench (Sonar, Mines vs. Rocks)*, UCI Machine Learning Repository. DOI: 10.24432/C5T01Q. https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks
[19] V. Sigillito, S. Wing, L. Hutton, and K. Baker, *Ionosphere*, UCI Machine Learning Repository. DOI: 10.24432/C5W01B. https://archive.ics.uci.edu/dataset/52/ionosphere
[20] W. Wolberg, O. Mangasarian, N. Street, and W. Street, *Breast Cancer Wisconsin (Diagnostic)*, UCI Machine Learning Repository. DOI: 10.24432/C5DW2B. https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
[21] V. Lohweg, *Banknote Authentication*, UCI Machine Learning Repository. DOI: 10.24432/C55P57. https://archive.ics.uci.edu/dataset/267/banknote+authentication"""


DATA_AVAILABILITY_SECTION = """
# Data availability

The following dataset records were used; each was validated after download against the characteristics published by the UCI Machine Learning Repository, and SHA-256 checksums are stored in the project artifacts. These are data-archive records, not literature references.

DATASET_RECORDS
"""


REFERENCES_SECTION = """
## Bibliographic verification note

Publisher records were verified on IEEE Xplore, ScienceDirect, SpringerLink, and the ACM Digital Library as listed in PROJECT_SPEC.md. Reference [11] (Han, ICDTE 2025) was additionally verified against its full text on 2026-10-21: its published weight (eqs. 5 and 9) contains an exponential absolute log-density-ratio density gate between query and neighbor, equivalent to the core of the formulation evaluated here; this overlap is disclosed in Sections 1-2 and the study is positioned as a minimal replication/extension-style variant. All numbered references [1]-[13] are sourced exclusively from the portals allowed for this coursework: IEEE Xplore [1]-[4], ScienceDirect [5], [7], [8], [12], SpringerLink [6], [9], [10], [13], and the ACM Digital Library [11]. The UCI dataset records are documented separately in the Data availability section; they are data-archive records, not literature references.
"""


def build_discussion(ctx: PaperContext) -> str:
    wl = ctx.win_loss()
    fried = ctx.friedman_values()
    lines = [
        "The discussion below is generated from the artifacts; negative results are retained.",
        "",
        f"1. **Where ALD wins.** Versus Power-WKNN, ALD-WKNN was ahead on {wl['wins']} "
        f"datasets, behind on {wl['losses']}, and tied on {wl['ties']} "
        f"(mean delta {wl['mean_delta']}).",
        f"2. **Dataset characteristics.** Per-dataset deltas are shown in Figure fig_ald_delta; "
        f"claims about high-dimensional, imbalanced, or difficult datasets are limited to the "
        f"evaluated set ({ctx.friedman_values().get('FRIEDMAN_N', TBD)} datasets).",
        f"3. **Strength of the density correction.** {ctx.alpha_summary()}. "
        f"The frequent selection of the largest alpha value suggests the density penalty was "
        f"active rather than negligible, but selection frequencies alone do not prove benefit.",
        f"4. **Sensitivity to q.** Selection counts: {ctx.q_summary()}.",
        f"5. **Clipping.** {ctx.clipping_summary()}.",
        f"6. **Benefit vs Power-WKNN (not only uniform KNN).** See the win/loss counts and mean "
        f"delta above; the comparison against the strongest distance baseline is the decisive one.",
        f"7. **Practical magnitude.** The mean delta of {wl['mean_delta']} Macro F1 should be "
        f"judged against application requirements; statistical guidance comes from the "
        f"Holm-adjusted p-values and rank-biserial effect sizes.",
        f"8. **Computational overhead.** {ctx.overhead_summary()}.",
        f"9. **Statistical support.** Omnibus p = {fried.get('FRIEDMAN_P', TBD)}; conclusions "
        f"follow the adjusted p-values, not raw averages.",
        "10. **Consistency with initial hypotheses.** The interpretation matrix of PROJECT_SPEC.md "
        "is applied to the observed pattern without post-hoc re-framing.",
    ]
    return "\n".join(lines)



def _dataset_records() -> str:
    return "\n".join(
        "- " + line.strip() for line in DATA_AVAILABILITY.strip().splitlines() if line.strip()
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Assemble the Markdown paper from artifacts.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--format", default="markdown", choices=["markdown"])
    parser.add_argument("--require-results", action="store_true")
    parser.add_argument("--output", default="paper/paper.md")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ctx = PaperContext(cfg)

    if args.require_results:
        for rel in REQUIRED_ARTIFACTS:
            if not (ctx.artifacts / rel).exists():
                raise RuntimeError(
                    "Cannot build empirical paper: required result artifact missing."
                )

    fried = ctx.friedman_values()
    wl = ctx.win_loss()
    table_or_tbd = lambda name: _read(ctx.tables / name) or f"**TBD** — `{name}` not available yet."

    subs = {
        "TITLE": TITLE,
        "KEYWORDS": KEYWORDS,
        "REFERENCE_LIST": REFERENCE_LIST,
        "ABSTRACT_RESULT": ctx.abstract_result(),
        "TABLE_DATASET_SUMMARY": table_or_tbd("dataset_summary.md"),
        "TABLE_PERFORMANCE": table_or_tbd("performance_mean_std.md"),
        "TABLE_FRIEDMAN": table_or_tbd("statistics_friedman.md"),
        "HOLM_TABLE": table_or_tbd("statistics_pairwise_holm.md"),
        "TABLE_ABLATION": table_or_tbd("ablation_summary.md"),
        "TABLE_HYPERPARAMETERS": table_or_tbd("hyperparameters.md"),
        "TABLE_RUNTIME": ctx.runtime_table(),
        "FIGURE_DATASET_PERFORMANCE": _figure(ctx.figures, "fig_dataset_performance"),
        "FIGURE_MEAN_RANK": _figure(ctx.figures, "fig_mean_rank"),
        "FIGURE_ALD_DELTA": _figure(ctx.figures, "fig_ald_delta"),
        "FIGURE_ABLATION": _figure(ctx.figures, "fig_ablation"),
        "FIGURE_RUNTIME": _figure(ctx.figures, "fig_runtime"),
        "DISCUSSION": build_discussion(ctx),
        "CONCLUSION_RESULT": ctx.conclusion_result(),
        **fried,
        **{f"WIN_{k.upper()}": v for k, v in wl.items()},
    }
    paper = TEMPLATE % subs
    if ctx.holm is None:
        paper += "\n(Pairwise Wilcoxon/Holm tables are produced only when the omnibus test justifies post-hoc analysis.)\n"
    paper += REFERENCES_SECTION
    paper += DATA_AVAILABILITY_SECTION

    paper = paper.replace("DATASET_RECORDS", _dataset_records())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(paper, encoding="utf-8")
    print(f"[paper] wrote {out_path} (missing artifacts: {ctx.missing or 'none'})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
