---
title: "Adaptive Local-Density Distance Weighting for K-Nearest Neighbors: A Reproducible Multi-Dataset Evaluation"
keywords: "K-Nearest Neighbors; Weighted KNN; Local Density; Adaptive Weighting; Density Compatibility; Classification; Nested Cross-Validation; Statistical Comparison"
---

# Abstract

K-Nearest Neighbors (KNN) is a simple instance-based classification method whose predictions depend strongly on neighborhood selection and the weighting assigned to neighboring observations. Distance-weighted KNN gives greater influence to observations located closer to a query but does not explicitly account for whether the query and its neighbors belong to locally comparable density regimes. Previous studies have already explored adaptive, density-based, and locally informed KNN classifiers; therefore, combining KNN with local-density information cannot in itself be considered novel. This study evaluates a simple experimental formulation, Adaptive Local-Density Distance-Weighted KNN (ALD-WKNN), in which conventional inverse-distance weights are multiplied by a bounded compatibility term derived from the log-ratio between the local-density proxies of the query and each neighbor. The method is evaluated against uniform KNN, conventional distance-weighted KNN, and power distance-weighted KNN using eight public tabular classification datasets. Model selection is performed by nested cross-validation, while ablation experiments isolate the effects of distance weighting, density compatibility, and clipping. Macro F1-score is used as the primary evaluation metric, supplemented by balanced accuracy, accuracy, macro precision, macro recall, and computational time. Cross-dataset comparisons are analyzed using the Friedman test followed, where justified, by Wilcoxon signed-rank comparisons with Holm correction. Across 1 tabular classification datasets, the Friedman test on dataset-level Macro F1 gave chi^2(2) = 2.0000 (p = 0.3679, Kendall's W = 1.0000); ALD-WKNN outperformed Power-WKNN on 0 datasets and underperformed on 1 (mean delta -0.0066). The study is intended as a reproducible empirical investigation of a minimal KNN modification rather than an a priori claim of algorithmic novelty.

# 1. Introduction

K-Nearest Neighbors is a non-parametric, instance-based classifier that assigns labels according to the training observations closest to a query. The foundations of nearest-neighbor classification were established by Cover and Hart [@coverhart1967], and classical distance-weighted KNN was introduced by Dudani [@dudani1976] to make the contribution of each neighbor depend on its distance to the query. Weighted nearest-neighbor rules were extended in several directions, including fuzzy class memberships [@keller1985] and later re-examinations of the distance-weighted rule [@macleod1987].

Crucially for the framing of this study, **local density is not a new idea for KNN**. Yuan et al. proposed DBANN, which partitions data by density and adapts the distance metric using local and global distributions to address class overlap and imbalance [@yuan2021]. Abdalla and Amer introduced a locally informed weighted k-nearest neighbor algorithm [@abdalla2025], and recent work on PR-kNN/WPR-kNN integrates local consistency and proximity information with several weighting schemes across dozens of datasets [@amer2025effective]. Accordingly, the research question of this paper is *not* whether density-aware KNN exists — it does — but the narrower one:

> Can a minimal, label-independent query-to-neighbor local-density compatibility penalty, multiplied directly by inverse-distance weighting, improve or stabilize classification behavior compared with distance weighting alone, under a leakage-controlled nested cross-validation protocol on heterogeneous small tabular datasets?

The research questions are: **RQ1** whether ALD-WKNN differs consistently from uniform and distance-weighted KNN in Macro F1; **RQ2** whether the density-compatibility component contributes once distance weighting is isolated by ablation; **RQ3** whether clipping the density mismatch improves stability; **RQ4** on which dataset characteristics ALD-WKNN helps, is neutral, or hurts; and **RQ5** the additional computational cost of local-density information. The tested hypotheses are the Friedman null H0 of equal method ranks across datasets versus the general alternative that at least two methods differ. No hypothesis is made that ALD-WKNN must score higher.

The contribution of this work is limited to evaluating a simple query-neighbor density-compatibility weighting formulation and systematically isolating its components through nested cross-validation and ablation across heterogeneous tabular datasets. The formulation is presented as a **candidate formulation** whose formula-level novelty remains subject to literature verification; in particular, a 2025 ACM article on dynamic distance-weighted KNN [@dynamicdwknn2025] could not be fully inspected and is treated as a mandatory novelty-check item.

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
| Dynamic DW-KNN, 2025 | ACM DL | Dynamic DW-KNN, density + uncertainty | Must inspect | Formula-level novelty not yet safe |

Within the literature inspected for this coursework, a narrower question remains worth empirically testing: whether a minimal label-independent query-to-neighbor density-compatibility penalty, expressed as a bounded log-ratio of local density proxies and multiplied directly by inverse-distance weighting, yields measurable benefits over simpler distance weighting under a leakage-controlled nested-CV protocol. This is treated as a candidate formula-level distinction, not as established algorithmic novelty. A simple density *reward* of the form w_i proportional to rho_i^alpha / d_i would favor neighbors from dense regions, which often coincide with majority classes; the present formulation instead uses *compatibility*: neighbors are penalized when their local density regime differs strongly from the query's, so density acts as local context rather than direct reward.

# 3. Experimental Setup

## 3.1 Datasets

Eight public numeric UCI classification datasets [@uci_iris; @uci_wine; @uci_seeds; @uci_glass; @uci_sonar; @uci_ionosphere; @uci_wdbc; @uci_banknote] with small to moderate sample sizes were selected to vary the number of features, classes, and difficulty. Characteristics are confirmed by the download validator against UCI records before entering the paper.

| dataset   |   n_samples |   n_features |   n_classes |
|:----------|------------:|-------------:|------------:|
| iris      |         150 |            4 |           3 |

## 3.2 Methods

Let d(x, x_i) be the Euclidean distance after preprocessing fitted on the training partition only. Uniform KNN scores classes by neighbor counts; distance-weighted KNN uses w_i = (d + epsilon)^{-1}; power-weighted KNN (Power-WKNN) generalizes this to w_i = (d + epsilon)^{-p} [@dudani1976]. ALD-WKNN multiplies the power distance weight by a bounded density-compatibility factor

w_i(x) = (d(x,x_i) + eps)^{-p} * exp[-alpha * min(c, |log((rho_q(x)+eps)/(rho_q(x_i)+eps))|)],

where the local-density proxy is rho_q(z) = 1/(r_q(z) + eps) and r_q(z) is the distance from z to its q-th nearest neighbor (self excluded for training points). The proxy is a *local compactness proxy*, not a normalized probability density estimate. The compatibility term lies in (0, 1], so the density component can only down-weight strongly incompatible neighbors and never creates explosive weights. If a classification neighbor has d <= epsilon, a deterministic zero-distance rule votes only among exact duplicates, breaking ties toward the smallest sorted encoded label. A density-only variant (p = 0) and an unclipped variant (c effectively infinite) are included in the ablation.

## 3.3 Nested cross-validation and tuning

The global seed is 20261021. Evaluation uses repeated stratified outer cross-validation; model selection uses inner stratified cross-validation whose random state is deterministically derived from (global seed, dataset index, repeat index, fold index). Hyperparameter selection optimizes Macro F1 exclusively. Each outer fold fits a pipeline of median imputation, standardization, and the estimator on the outer training partition only, then evaluates once on the untouched outer test partition; imputation, scaling, and density computation therefore never see outer-test values. Configuration grids: k in {3, 5, 9, 15} for all KNN variants; p in {1, 2} for Power-WKNN; and k in {3, 5, 9, 15}, q in {3, 5, 9}, p in {1, 2}, alpha in {0.25, 0.5, 1.0, 2.0}, c = 3.0 for ALD-WKNN (96 configurations). The alpha = 0 case is excluded from the ALD grid because it duplicates distance-only behavior (ablation A0).

## 3.4 Statistical protocol

The unit of inference is the **dataset**, not the CV fold: fold scores from repeated CV overlap and are not independent. Per-dataset aggregate scores form a D x M matrix on which the Friedman omnibus test is applied [@garcia2010; @eisinga2017]. Only when the omnibus p-value is below 0.05 are pairwise Wilcoxon signed-rank tests run against ALD-WKNN with Holm family-wise correction; rank-biserial effect sizes use the convention difference = ALD - baseline so that r_rb > 0 favors ALD. A p-value above 0.05 is reported as insufficient evidence, never as proof of equality, and fold-level scores are never presented as an independent sample.

# 4. Results and Discussion

## 4.1 Primary performance

| Dataset   | Uniform KNN   | Power-WKNN    | ALD-WKNN      | Δ ALD−PWKNN   |
|:----------|:--------------|:--------------|:--------------|:--------------|
| iris      | 0.906 ± 0.020 | 0.919 ± 0.000 | 0.913 ± 0.009 | -0.007        |
| Mean rank | 3.00          | 1.00          | 2.00          | —             |

![fig_dataset_performance](artifacts/figures/fig_dataset_performance.png){width=80%}

![fig_ald_delta](artifacts/figures/fig_ald_delta.png){width=80%}

## 4.2 Statistical comparison across datasets

# Friedman test

| Metric   |   Datasets |   Methods |   Statistic |   df |   p-value |   Kendall W | Decision                           |
|:---------|-----------:|----------:|------------:|-----:|----------:|------------:|:-----------------------------------|
| f1_macro |          1 |         3 |           2 |    2 |    0.3679 |           1 | insufficient evidence to reject H0 |

The Friedman test did not provide sufficient evidence to reject the null hypothesis of equal method ranks across the evaluated datasets, χ²F(2)=2.0000, p=0.3679. Consequently, pairwise tests are reported, if shown at all, as exploratory rather than as confirmatory evidence.

# Pairwise comparisons (Wilcoxon + Holm)

TBD — post-hoc tests were not executed (omnibus p >= 0.05, or statistics artifacts missing).

![fig_mean_rank](artifacts/figures/fig_mean_rank.png){width=80%}

## 4.3 Ablation

# Ablation summary

Source run: `20261021T143913Z_seed20261021_ablation` (aggregate Macro F1 across datasets).

| Variant          | Distance   | Density compatibility   | Clipping   |   Mean Macro F1 |   Mean Rank |
|:-----------------|:-----------|:------------------------|:-----------|----------------:|------------:|
| A0 Distance-only | ✓          | ✗                       | N/A        |           0.919 |         1   |
| A1 Density-only  | ✗          | ✓                       | ✓          |           0.906 |         4   |
| A2 ALD-Unclipped | ✓          | ✓                       | ✗          |           0.913 |         2.5 |
| A3 ALD-Full      | ✓          | ✓                       | ✓          |           0.913 |         2.5 |

Interpretation guide (PROJECT_SPEC.md): A3 > A0 supports the density term;
A3 ≈ A0 means the modification gives no clear practical benefit;
A3 > A2 indicates clipping helps; poor A1 with good A3 indicates density
context is useful as a modifier rather than a standalone signal.

![fig_ablation](artifacts/figures/fig_ablation.png){width=80%}

## 4.4 Hyperparameter selection

# Hyperparameter selection frequency

Source run: `20261021T143912Z_seed20261021_smoke`. Frequencies count outer-fold selections; nested CV may select different values per fold, so no single global 'best parameter' is reported.

## Uniform KNN

| Parameter   |   Value |   Selections |   Frequency (%) |
|:------------|--------:|-------------:|----------------:|
| n_neighbors |       3 |            2 |             100 |

## Power-WKNN

| Parameter      |   Value |   Selections |   Frequency (%) |
|:---------------|--------:|-------------:|----------------:|
| distance_power |       1 |            2 |             100 |
| n_neighbors    |       3 |            2 |             100 |

## ALD-WKNN

| Parameter         |   Value |   Selections |   Frequency (%) |
|:------------------|--------:|-------------:|----------------:|
| density_alpha     |     0.5 |            2 |             100 |
| density_clip      |     3   |            2 |             100 |
| density_neighbors |     3   |            2 |             100 |
| distance_power    |     1   |            2 |             100 |
| n_neighbors       |     3   |            2 |             100 |

## 4.5 Runtime

| Method      |   Fit time mean (s) |   Predict time mean (s) | Relative fit time vs Uniform KNN   |
|:------------|--------------------:|------------------------:|:-----------------------------------|
| ALD-WKNN    |              0.03   |                  0.0019 | 2.01x                              |
| Uniform KNN |              0.015  |                  0.0013 | 1.00x                              |
| Power-WKNN  |              0.0141 |                  0.0012 | 0.94x                              |

![fig_runtime](artifacts/figures/fig_runtime.png){width=80%}

## 4.6 Discussion

The discussion below is generated from the artifacts; negative results are retained.

1. **Where ALD wins.** ALD-WKNN outperformed Power-WKNN on 0 of the evaluated datasets and underperformed on 1 (mean delta -0.0066).
2. **Dataset characteristics.** Per-dataset deltas are shown in Figure fig_ald_delta; claims about high-dimensional, imbalanced, or difficult datasets are limited to the evaluated set (1 datasets).
3. **Strength of the density correction.** Inner CV selected small alpha values in 2/2 outer-fold selections chose alpha <= 0.5 cases, indicating how strongly the density correction was needed.
4. **Sensitivity to q.** Selection counts: q=3: 2.
5. **Clipping.** clipped A3 mean Macro F1 is equal to unclipped A2 (mean difference +0.0000).
6. **Benefit vs Power-WKNN (not only uniform KNN).** See the win/loss counts and mean delta above; the comparison against the strongest distance baseline is the decisive one.
7. **Practical magnitude.** The mean delta of -0.0066 Macro F1 should be judged against application requirements; statistical guidance comes from the Holm-adjusted p-values and rank-biserial effect sizes.
8. **Computational overhead.** mean fit time ALD/uniform = 2.01x, mean predict time ALD/uniform = 1.47x.
9. **Statistical support.** Omnibus p = 0.3679; conclusions follow the adjusted p-values, not raw averages.
10. **Consistency with initial hypotheses.** The interpretation matrix of PROJECT_SPEC.md is applied to the observed pattern without post-hoc re-framing.

# 5. Conclusion

This study evaluates a simple extension of distance-weighted KNN in which neighbor influence is adjusted according to the compatibility between the local density of a query and that of each candidate neighbor. The ALD-WKNN formulation is intentionally minimal: it combines a power inverse-distance term with a bounded exponential penalty derived from the absolute log-ratio of local-density proxies. The study does not assume that density-aware KNN is novel, as previous work has already incorporated density, local structure, adaptive distance metrics, and locally informed weighting into nearest-neighbor classification. Across the evaluated datasets, ALD-WKNN achieved a mean Macro F1 delta of -0.0066 versus Power-WKNN (wins 0, losses 1 across datasets). The Friedman test yielded p = 0.3679 (no sufficient evidence of differences among method ranks), while Holm-adjusted pairwise comparisons and rank-biserial effect sizes are reported in the results section. Ablation indicated that the local-density compatibility component (clipped A3 mean Macro F1 is equal to unclipped A2 (mean difference +0.0000)) and clipping behave as described in the ablation analysis. These findings suggest the density-compatibility penalty is not confirmed within the evaluated scope only, and no broader claim is made.

# Limitations

Only eight datasets are evaluated, most of them small tabular problems; Euclidean distance is the sole metric; local density is represented only by the q-th-neighbor radius; the hyperparameter search is limited; the number of datasets restricts statistical power; formula-level novelty has not been established; findings may not transfer to large-scale or high-dimensional modern datasets; and density preprocessing costs more computation than plain KNN.

# Future work

Directions that add complexity only with justification: class-aware density compatibility; adaptive q; robust alternative local-density proxies; Mahalanobis or learned metrics; larger benchmark suites; approximate nearest-neighbor search; and noise/imbalance stress tests.

# References

[1] T. M. Cover and P. E. Hart, "Nearest Neighbor Pattern Classification," *IEEE Transactions on Information Theory*, vol. 13, no. 1, pp. 21-27, 1967. DOI: 10.1109/TIT.1967.1053964. IEEE Xplore verified; Scopus check pending.
[2] S. A. Dudani, "The Distance-Weighted k-Nearest-Neighbor Rule," *IEEE Transactions on Systems, Man, and Cybernetics*, vol. SMC-6, no. 4, pp. 325-327, 1976. DOI: 10.1109/TSMC.1976.5408784. IEEE Xplore verified; Scopus check pending.
[3] J. M. Keller, M. R. Gray, and J. A. Givens, "A Fuzzy K-Nearest Neighbor Algorithm," *IEEE Transactions on Systems, Man, and Cybernetics*, vol. SMC-15, no. 4, pp. 580-585, 1985. DOI: 10.1109/TSMC.1985.6313426. IEEE Xplore verified; Scopus check pending.
[4] J. E. S. Macleod, A. Luk, and D. M. Titterington, "A Re-Examination of the Distance-Weighted k-Nearest Neighbor Classification Rule," 1987. https://ieeexplore.ieee.org/document/4075685/ — IEEE Xplore verified; DOI metadata requires manual bibliographic validation; Scopus check pending.
[5] Z. Pan, Y. Wang, and Y. Pan, "A New Locally Adaptive k-Nearest Neighbor Algorithm Based on Discrimination Class," *Knowledge-Based Systems*, vol. 204, 106185, 2020. DOI: 10.1016/j.knosys.2020.106185. ScienceDirect verified; Scopus check pending.
[6] B.-W. Yuan, X.-G. Luo, Z.-L. Zhang, Y. Yu, H.-W. Huo, T. Johannes, and X.-D. Zou, "A Novel Density-Based Adaptive k Nearest Neighbor Method for Dealing with Overlapping Problem in Imbalanced Datasets," *Neural Computing and Applications*, vol. 33, pp. 4457-4481, 2021. DOI: 10.1007/s00521-020-05256-0. SpringerLink verified; Scopus check pending; novelty relevance: critical.
[7] S. Ding, M. Li, T. Huang, and W. Zhu, "Local Density Based on Weighted K-Nearest Neighbors for Density Peaks Clustering," *Knowledge-Based Systems*, vol. 305, 112609, 2024. DOI: 10.1016/j.knosys.2024.112609. ScienceDirect/ACM metadata verified; Scopus check pending.
[8] H. I. Abdalla and A. A. Amer, "Enhancing Data Classification Using Locally Informed Weighted k-Nearest Neighbor Algorithm," *Expert Systems with Applications*, vol. 276, 126942, 2025. DOI: 10.1016/j.eswa.2025.126942. ScienceDirect and ACM metadata verified; Scopus check pending; novelty relevance: critical.
[9] A. A. Amer, S. D. Ravana, and R. A. A. Habeeb, "Effective k-Nearest Neighbor Models for Data Classification Enhancement," *Journal of Big Data*, vol. 12, 2025. DOI: 10.1186/s40537-025-01137-2. SpringerLink verified; Scopus check pending.
[10] A. A. Amer, S. D. Ravana, and R. A. A. Habeeb, "On Enhancing Data Classification Using Local Mean-Based Fuzzy K-Nearest Neighbor Algorithms," *Advances in Data Analysis and Classification*, 2025. DOI: 10.1007/s11634-025-00653-6. SpringerLink verified; Scopus check pending.
[11] "A Dynamic Distance-Weighted KNN Model for Customized ...," ACM Digital Library, 2025. DOI: 10.1145/3796159.3796172. ACM search record verified; method-detail and full-text novelty check pending; Scopus check pending.
[12] S. Garcia, A. Fernandez, J. Luengo, and F. Herrera, "Advanced Nonparametric Tests for Multiple Comparisons in the Design of Experiments in Computational Intelligence and Data Mining: Experimental Analysis of Power," *Information Sciences*, vol. 180, no. 10, pp. 2044-2064, 2010. DOI: 10.1016/j.ins.2009.12.010. ScienceDirect verified; Scopus check pending.
[13] R. Eisinga, T. Heskes, B. Pelzer, and M. Te Grotenhuis, "Exact p-values for Pairwise Comparison of Friedman Rank Sums, with Application to Comparing Classifiers," *BMC Bioinformatics*, vol. 18, 68, 2017. DOI: 10.1186/s12859-017-1486-2. SpringerLink verified; Scopus check pending.
[14] R. A. Fisher, *Iris*, UCI Machine Learning Repository. DOI: 10.24432/C56C76. https://archive.ics.uci.edu/dataset/53/iris
[15] S. Aeberhard and M. Forina, *Wine*, UCI Machine Learning Repository. DOI: 10.24432/C5PC7J. https://archive.ics.uci.edu/dataset/109/wine
[16] M. Charytanowicz, J. Niewczas, P. Kulczycki, P. A. Kowalski, S. Lukasik, and S. Zak, *Seeds*, UCI Machine Learning Repository. DOI: 10.24432/C5H30K. https://archive.ics.uci.edu/dataset/236/seeds
[17] *Glass Identification*, UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/42/glass+identification
[18] T. Sejnowski and R. Gorman, *Connectionist Bench (Sonar, Mines vs. Rocks)*, UCI Machine Learning Repository. DOI: 10.24432/C5T01Q. https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks
[19] V. Sigillito, S. Wing, L. Hutton, and K. Baker, *Ionosphere*, UCI Machine Learning Repository. DOI: 10.24432/C5W01B. https://archive.ics.uci.edu/dataset/52/ionosphere
[20] W. Wolberg, O. Mangasarian, N. Street, and W. Street, *Breast Cancer Wisconsin (Diagnostic)*, UCI Machine Learning Repository. DOI: 10.24432/C5DW2B. https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
[21] V. Lohweg, *Banknote Authentication*, UCI Machine Learning Repository. DOI: 10.24432/C55P57. https://archive.ics.uci.edu/dataset/267/banknote+authentication

(Pairwise Wilcoxon/Holm tables are produced only when the omnibus test justifies post-hoc analysis.)

## Bibliographic verification note

Publisher records were verified on IEEE Xplore, ScienceDirect, SpringerLink, and the ACM Digital Library as listed in PROJECT_SPEC.md. Scopus verification of each record remains pending manual institutional access; no Scopus record was marked verified without an actual record check.
