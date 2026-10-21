---
title: "Adaptive Local-Density Distance Weighting for K-Nearest Neighbors: A Reproducible Multi-Dataset Evaluation"
keywords: "K-Nearest Neighbors; Weighted KNN; Local Density; Adaptive Weighting; Density Compatibility; Classification; Nested Cross-Validation; Statistical Comparison"
---

# Abstract

K-Nearest Neighbors (KNN) is a simple instance-based classification method whose predictions depend strongly on neighborhood selection and the weighting assigned to neighboring observations. Distance-weighted KNN gives greater influence to observations located closer to a query but does not explicitly account for whether the query and its neighbors belong to locally comparable density regimes. Previous studies have already explored adaptive, density-based, and locally informed KNN classifiers; therefore, combining KNN with local-density information cannot in itself be considered novel. This study evaluates a simple experimental formulation, Adaptive Local-Density Distance-Weighted KNN (ALD-WKNN), in which conventional inverse-distance weights are multiplied by a bounded compatibility term derived from the log-ratio between the local-density proxies of the query and each neighbor. The method is evaluated against uniform KNN, conventional distance-weighted KNN, and power distance-weighted KNN using eight public tabular classification datasets. Model selection is performed by nested cross-validation, while ablation experiments isolate the effects of distance weighting, density compatibility, and clipping. Macro F1-score is used as the primary evaluation metric, supplemented by balanced accuracy, accuracy, macro precision, macro recall, and computational time. Cross-dataset comparisons are analyzed using the Friedman test followed, where justified, by Wilcoxon signed-rank comparisons with Holm correction. A 2025 ACM study introduces a dynamic distance-weighted KNN whose weight already contains an exponential absolute log-density-ratio gate between the query and each neighbor [@dynamicdwknn2025]; the present work deliberately reduces that published idea to its minimal two-factor form (distance power x clipped density-compatibility gate) and subjects it to a reproducible multi-dataset evaluation. Across 8 tabular classification datasets, the Friedman test on dataset-level Macro F1 gave chi^2(3) = 7.0000 (p = 0.0719, Kendall's W = 0.2917); ALD-WKNN was ahead of Power-WKNN on 5 datasets, behind on 1, and tied on 2 (mean delta +0.0120). The study is intended as a reproducible replication/extension-style empirical investigation of a minimal KNN modification, not as a claim of algorithmic novelty.

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

| dataset                            |   n_samples |   n_features |   n_classes |
|:-----------------------------------|------------:|-------------:|------------:|
| iris                               |         150 |            4 |           3 |
| wine                               |         178 |           13 |           3 |
| seeds                              |         210 |            7 |           3 |
| glass                              |         214 |            9 |           6 |
| sonar                              |         208 |           60 |           2 |
| ionosphere                         |         351 |           34 |           2 |
| breast_cancer_wisconsin_diagnostic |         569 |           30 |           2 |
| banknote                           |        1372 |            4 |           2 |

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

| Dataset                            | Uniform KNN   | Distance KNN   | Power-WKNN    | ALD-WKNN      | Δ ALD−PWKNN   |
|:-----------------------------------|:--------------|:---------------|:--------------|:--------------|:--------------|
| banknote                           | 0.998 ± 0.002 | 0.998 ± 0.002  | 0.998 ± 0.002 | 0.998 ± 0.002 | +0.000        |
| breast_cancer_wisconsin_diagnostic | 0.961 ± 0.016 | 0.961 ± 0.016  | 0.961 ± 0.018 | 0.964 ± 0.016 | +0.003        |
| glass                              | 0.614 ± 0.105 | 0.649 ± 0.096  | 0.645 ± 0.100 | 0.672 ± 0.089 | +0.026        |
| ionosphere                         | 0.809 ± 0.040 | 0.809 ± 0.045  | 0.809 ± 0.045 | 0.865 ± 0.043 | +0.056        |
| iris                               | 0.940 ± 0.052 | 0.949 ± 0.035  | 0.949 ± 0.035 | 0.949 ± 0.035 | +0.000        |
| seeds                              | 0.918 ± 0.047 | 0.918 ± 0.045  | 0.920 ± 0.046 | 0.928 ± 0.048 | +0.008        |
| sonar                              | 0.847 ± 0.045 | 0.847 ± 0.045  | 0.848 ± 0.039 | 0.842 ± 0.047 | -0.006        |
| wine                               | 0.963 ± 0.029 | 0.958 ± 0.029  | 0.961 ± 0.023 | 0.970 ± 0.027 | +0.009        |
| Mean rank                          | 2.81          | 2.94           | 2.69          | 1.56          | —             |

![fig_dataset_performance](artifacts/figures/fig_dataset_performance.png){width=80%}

![fig_ald_delta](artifacts/figures/fig_ald_delta.png){width=80%}

## 4.2 Statistical comparison across datasets

# Friedman test

| Metric   |   Datasets |   Methods |   Statistic |   df |   p-value |   Kendall W | Decision                           |
|:---------|-----------:|----------:|------------:|-----:|----------:|------------:|:-----------------------------------|
| f1_macro |          8 |         4 |           7 |    3 |    0.0719 |      0.2917 | insufficient evidence to reject H0 |

The Friedman test did not provide sufficient evidence to reject the null hypothesis of equal method ranks across the evaluated datasets, χ²F(3)=7.0000, p=0.0719. Consequently, pairwise tests are reported, if shown at all, as exploratory rather than as confirmatory evidence.

# Pairwise comparisons (Wilcoxon + Holm)

TBD — post-hoc tests were not executed (omnibus p >= 0.05, or statistics artifacts missing).

![fig_mean_rank](artifacts/figures/fig_mean_rank.png){width=80%}

## 4.3 Ablation

# Ablation summary

Source run: `20261021T145634Z_seed20261021_ablation` (aggregate Macro F1 across datasets).

| Variant          | Distance   | Density compatibility   | Clipping   |   Mean Macro F1 |   Mean Rank |
|:-----------------|:-----------|:------------------------|:-----------|----------------:|------------:|
| A0 Distance-only | ✓          | ✗                       | N/A        |           0.884 |        3.88 |
| A1 Density-only  | ✗          | ✓                       | ✓          |           0.893 |        3    |
| A2 ALD-Unclipped | ✓          | ✓                       | ✗          |           0.901 |        1.69 |
| A3 ALD-Full      | ✓          | ✓                       | ✓          |           0.901 |        1.44 |

Interpretation guide (PROJECT_SPEC.md): A3 > A0 supports the density term;
A3 ≈ A0 means the modification gives no clear practical benefit;
A3 > A2 indicates clipping helps; poor A1 with good A3 indicates density
context is useful as a modifier rather than a standalone signal.

![fig_ablation](artifacts/figures/fig_ablation.png){width=80%}

## 4.4 Hyperparameter selection

# Hyperparameter selection frequency

Source run: `20261021T145425Z_seed20261021_main`. Frequencies count outer-fold selections; nested CV may select different values per fold, so no single global 'best parameter' is reported.

## Uniform KNN

| Parameter   |   Value |   Selections |   Frequency (%) |
|:------------|--------:|-------------:|----------------:|
| n_neighbors |       3 |           62 |            51.7 |
| n_neighbors |       5 |           28 |            23.3 |
| n_neighbors |      15 |           16 |            13.3 |
| n_neighbors |       9 |           14 |            11.7 |

## Distance KNN

| Parameter   |   Value |   Selections |   Frequency (%) |
|:------------|--------:|-------------:|----------------:|
| n_neighbors |       3 |           55 |            45.8 |
| n_neighbors |       5 |           30 |            25   |
| n_neighbors |      15 |           25 |            20.8 |
| n_neighbors |       9 |           10 |             8.3 |

## Power-WKNN

| Parameter      |   Value |   Selections |   Frequency (%) |
|:---------------|--------:|-------------:|----------------:|
| distance_power |       1 |           73 |            60.8 |
| distance_power |       2 |           47 |            39.2 |
| n_neighbors    |       3 |           43 |            35.8 |
| n_neighbors    |       5 |           33 |            27.5 |
| n_neighbors    |      15 |           26 |            21.7 |
| n_neighbors    |       9 |           18 |            15   |

## ALD-WKNN

| Parameter         |   Value |   Selections |   Frequency (%) |
|:------------------|--------:|-------------:|----------------:|
| density_alpha     |   2     |           69 |            57.5 |
| density_alpha     |   0.25  |           38 |            31.7 |
| density_alpha     |   1     |            7 |             5.8 |
| density_alpha     |   0.5   |            6 |             5   |
| density_clip      |   3     |          120 |           100   |
| density_neighbors |   3     |           72 |            60   |
| density_neighbors |   9     |           27 |            22.5 |
| density_neighbors |   5     |           21 |            17.5 |
| distance_power    |   2     |           64 |            53.3 |
| distance_power    |   1     |           56 |            46.7 |
| epsilon           |   1e-12 |          120 |           100   |
| n_neighbors       |  15     |           43 |            35.8 |
| n_neighbors       |   3     |           29 |            24.2 |
| n_neighbors       |   5     |           27 |            22.5 |
| n_neighbors       |   9     |           21 |            17.5 |

## 4.5 Runtime

| Method       |   Fit time mean (s) |   Predict time mean (s) | Relative fit time vs Uniform KNN   |
|:-------------|--------------------:|------------------------:|:-----------------------------------|
| ALD-WKNN     |              0.7554 |                  0.0023 | 13.46x                             |
| Distance KNN |              0.0448 |                  0.0018 | 0.80x                              |
| Uniform KNN  |              0.0561 |                  0.0016 | 1.00x                              |
| Power-WKNN   |              0.0652 |                  0.0015 | 1.16x                              |

![fig_runtime](artifacts/figures/fig_runtime.png){width=80%}

## 4.6 Discussion

The discussion below is generated from the artifacts; negative results are retained.

1. **Where ALD wins.** Versus Power-WKNN, ALD-WKNN was ahead on 5 datasets, behind on 1, and tied on 2 (mean delta +0.0120).
2. **Dataset characteristics.** Per-dataset deltas are shown in Figure fig_ald_delta; claims about high-dimensional, imbalanced, or difficult datasets are limited to the evaluated set (8 datasets).
3. **Strength of the density correction.** inner-CV selection frequencies were alpha=0.25 (31.7%), alpha=0.5 (5.0%), alpha=1 (5.8%), alpha=2 (57.5%); alpha <= 0.5 was chosen in 44/120 folds (36.7%). The frequent selection of the largest alpha value suggests the density penalty was active rather than negligible, but selection frequencies alone do not prove benefit.
4. **Sensitivity to q.** Selection counts: q=3: 72, q=9: 27, q=5: 21.
5. **Clipping.** clipped A3 mean Macro F1 is higher than unclipped A2 (mean difference +0.0001).
6. **Benefit vs Power-WKNN (not only uniform KNN).** See the win/loss counts and mean delta above; the comparison against the strongest distance baseline is the decisive one.
7. **Practical magnitude.** The mean delta of +0.0120 Macro F1 should be judged against application requirements; statistical guidance comes from the Holm-adjusted p-values and rank-biserial effect sizes.
8. **Computational overhead.** mean fit time ALD/uniform = 13.46x, mean predict time ALD/uniform = 1.41x.
9. **Statistical support.** Omnibus p = 0.0719; conclusions follow the adjusted p-values, not raw averages.
10. **Consistency with initial hypotheses.** The interpretation matrix of PROJECT_SPEC.md is applied to the observed pattern without post-hoc re-framing.

# 5. Conclusion

This study evaluates a simple extension of distance-weighted KNN in which neighbor influence is adjusted according to the compatibility between the local density of a query and that of each candidate neighbor. The ALD-WKNN formulation is intentionally minimal: it combines a power inverse-distance term with a bounded exponential penalty derived from the absolute log-ratio of local-density proxies. The study does not assume that density-aware KNN is novel, as previous work has already incorporated density, local structure, adaptive distance metrics, and locally informed weighting into nearest-neighbor classification. Across the evaluated datasets, ALD-WKNN achieved a mean Macro F1 delta of +0.0120 versus Power-WKNN (wins 5, losses 1, ties 2 across datasets). The Friedman test yielded p = 0.0719 (no sufficient evidence of differences among method ranks), so pairwise comparisons are at most exploratory under this design. Ablation showed the full clipped formulation (A3) marginally above the unclipped variant (A2) and above distance-only (A0) in mean Macro F1, while density-only (A1) was the weakest standalone variant — consistent with density acting as a modifier rather than a sufficient signal by itself. These findings suggest the density-compatibility penalty is descriptively favorable on the evaluated benchmark but not statistically confirmed within this design, and no broader claim is made.

# Limitations

Only eight datasets are evaluated, most of them small tabular problems; Euclidean distance is the sole metric; local density is represented only by the q-th-neighbor radius; the hyperparameter search is limited; the number of datasets restricts statistical power; the core density-gate formula has a 2025 precedent (Han, ICDTE 2025) and only the minimal clipped variant is specific to this study; findings may not transfer to large-scale or high-dimensional modern datasets; and density preprocessing costs more computation than plain KNN.

# Future work

Directions that add complexity only with justification: class-aware density compatibility; adaptive q; robust alternative local-density proxies; Mahalanobis or learned metrics; larger benchmark suites; approximate nearest-neighbor search; and noise/imbalance stress tests.

# References

[1] T. M. Cover and P. E. Hart, "Nearest Neighbor Pattern Classification," *IEEE Transactions on Information Theory*, vol. 13, no. 1, pp. 21-27, 1967. DOI: 10.1109/TIT.1967.1053964. IEEE Xplore verified.
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


(Pairwise Wilcoxon/Holm tables are produced only when the omnibus test justifies post-hoc analysis.)

## Bibliographic verification note

Publisher records were verified on IEEE Xplore, ScienceDirect, SpringerLink, and the ACM Digital Library as listed in PROJECT_SPEC.md. Reference [11] (Han, ICDTE 2025) was additionally verified against its full text on 2026-10-21: its published weight (eqs. 5 and 9) contains an exponential absolute log-density-ratio density gate between query and neighbor, equivalent to the core of the formulation evaluated here; this overlap is disclosed in Sections 1-2 and the study is positioned as a minimal replication/extension-style variant. All numbered references [1]-[13] are sourced exclusively from the portals allowed for this coursework: IEEE Xplore [1]-[4], ScienceDirect [5], [7], [8], [12], SpringerLink [6], [9], [10], [13], and the ACM Digital Library [11]. The UCI dataset records are documented separately in the Data availability section; they are data-archive records, not literature references.

# Data availability

The following dataset records were used; each was validated after download against the characteristics published by the UCI Machine Learning Repository, and SHA-256 checksums are stored in the project artifacts. These are data-archive records, not literature references.

- [14] R. A. Fisher, *Iris*, UCI Machine Learning Repository. DOI: 10.24432/C56C76. https://archive.ics.uci.edu/dataset/53/iris
- [15] S. Aeberhard and M. Forina, *Wine*, UCI Machine Learning Repository. DOI: 10.24432/C5PC7J. https://archive.ics.uci.edu/dataset/109/wine
- [16] M. Charytanowicz, J. Niewczas, P. Kulczycki, P. A. Kowalski, S. Lukasik, and S. Zak, *Seeds*, UCI Machine Learning Repository. DOI: 10.24432/C5H30K. https://archive.ics.uci.edu/dataset/236/seeds
- [17] *Glass Identification*, UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/42/glass+identification
- [18] T. Sejnowski and R. Gorman, *Connectionist Bench (Sonar, Mines vs. Rocks)*, UCI Machine Learning Repository. DOI: 10.24432/C5T01Q. https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks
- [19] V. Sigillito, S. Wing, L. Hutton, and K. Baker, *Ionosphere*, UCI Machine Learning Repository. DOI: 10.24432/C5W01B. https://archive.ics.uci.edu/dataset/52/ionosphere
- [20] W. Wolberg, O. Mangasarian, N. Street, and W. Street, *Breast Cancer Wisconsin (Diagnostic)*, UCI Machine Learning Repository. DOI: 10.24432/C5DW2B. https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- [21] V. Lohweg, *Banknote Authentication*, UCI Machine Learning Repository. DOI: 10.24432/C55P57. https://archive.ics.uci.edu/dataset/267/banknote+authentication
