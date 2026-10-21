# Novelty Gate Literature Check — ALD-WKNN density-compatibility weighting

**Candidate formula under inspection (candidate, not a novelty claim):**

    w_i(x) = (d(x,x_i) + eps)^(-p) * exp(-alpha * min(c, |log((rho_q(x)+eps) / (rho_q(x_i)+eps))|))

where `rho_q(z) = 1/(r_q(z)+eps)` is a local-density proxy from the q-th neighbor radius (KNN classification).
Core idea: inverse-distance weighting multiplied by a **query-vs-neighbor local-density compatibility
penalty** — an absolute log-ratio of the two densities, clipped at `c` and passed through a negative
exponential. Note: `exp(-|log(a/b)|) = min(a,b)/max(a,b)`, so any equivalent form
`(min(rho_x, rho_i)/max(rho_x, rho_i))^alpha` counts as formula-level overlap.

**Search date:** 21 October 2026
**Screener:** automated web search (first result page per query), plus metadata APIs (Crossref, Semantic
Scholar, Unpaywall) and publisher pages where accessible.
**Honesty rule applied throughout:** a negative search result is NOT evidence of novelty; each record below
states exactly what could and could not be verified.

---

## Query 1

Search date: 21 October 2026
Database: WebSearch (web-wide, incl. ieeexplore.ieee.org, arxiv.org, link.springer.com)
Query: "local density" AND "weighted k-nearest neighbor" classification
Number of results screened: 10 (first result page)
Relevant papers:
- Amer, "On enhancing data classification using local mean-based fuzzy K-nearest neighbor algorithms" (Springer s11634-025-00653-6, 2025) — fuzzy local-mean KNN; local density mentioned as context, not as log-ratio weight.
- "DW-KNN: A Transparent Local Classifier Integrating Distance Consistency and Neighbor Reliability" (arXiv:2512.08956, 2025) — distance consistency + neighbor reliability; cites Kim, "Adaptive KNN based on local density for classification" (Expert Syst. Appl. 2021).
- Ding 2024, "Local density based on weighted K-nearest neighbors for density peaks clustering" (KNOSYS, doi:10.1016/j.knosys.2024.112609) — clustering (DPC), not classification weighting.
Potential formula overlap: None found. Adaptive-k-by-density and local-mean/fuzzy variants use density to choose k or shape membership, not as a multiplicative exp(-alpha*|log(rho_query/rho_neighbor)|) compatibility factor on top of d^-p.
Decision: no equivalent found in screened results — candidate formulation remains unverified, not claimed novel

## Query 2

Search date: 21 October 2026
Database: WebSearch (web-wide, incl. sciencedirect.com, jmlr.org)
Query: "density ratio" AND KNN AND classification
Number of results screened: 10 (first result page)
Relevant papers:
- Ye/Zhu et al. line of work on density-ratio based clustering (Pattern Recognition 2016) — clustering with varying densities, different task.
- "LOF weighted KNN regression ensemble" (Sādhanā 2023) — LOF outlier factor as weight in regression; multiplicative anomaly factor, but LOF is an outlier score, not a clipped symmetric log-density-ratio between query and neighbor.
- Portier & Delyon, "Nearest Neighbor Sampling for Covariate Shift Adaptation" (JMLR 2024) — density-ratio importance weighting between distributions, not per-neighbor voting weight in KNN classification.
Potential formula overlap: None found. "Density ratio" literature concerns distribution-level ratios (clustering, covariate shift, two-sample), not the query-vs-neighbor log-ratio compatibility penalty.
Decision: no equivalent found in screened results — candidate formulation remains unverified, not claimed novel

## Query 3

Search date: 21 October 2026
Database: WebSearch (web-wide, incl. ieeexplore.ieee.org, sciencedirect.com)
Query: "query density" AND "neighbor density" AND KNN
Number of results screened: 10 (first result page)
Relevant papers:
- Su & Guan, "Density and Distance Based KNN Approach to Classification" (Int. J. Applied Evol. Comput. 2016) — combines density and distance for non-uniform density distributions; density enters as region selection/vote adjustment, not as exp(-|log rho_q/rho_i|) factor.
- "Adaptive three-way KNN classifier using density-based granular balls" (Inf. Sci. 2024) — granular-ball (region) adaptation, different mechanism.
Potential formula overlap: None found. Joint use of query-side and neighbor-side density exists, but not in the clipped absolute-log-ratio exponential-penalty form.
Decision: no equivalent found in screened results — candidate formulation remains unverified, not claimed novel

## Query 4

Search date: 21 October 2026
Database: WebSearch (web-wide)
Query: "density compatibility" AND nearest neighbor
Number of results screened: 5 (first result page; low-recall query)
Relevant papers:
- Anders, parameter-free cluster detection in spatial data (ISPRS/ICA workshops) — "density compatibility" as a heuristic constraint between neighboring *clusters* in cartography; unrelated to KNN classification weights.
Potential formula overlap: None. The term "density compatibility" in the literature refers to spatial cluster merging heuristics, not a query-neighbor density agreement penalty.
Decision: no equivalent found in screened results — candidate formulation remains unverified, not claimed novel

## Query 5

Search date: 21 October 2026
Database: WebSearch (web-wide, incl. nih.gov, arxiv.org)
Query: "log density ratio" AND nearest neighbor classification
Number of results screened: 8 (first result page)
Relevant papers:
- "Density-Ratio Rescoring for Imbalanced Classification" (arXiv:2609.23926, 2026) — log density-ratio under a tilt model for global rescoring, not per-neighbor weights.
- Cheng & Xie, "Classification logit two-sample testing by neural networks" (PMC10558099) — log density-ratio as a two-sample witness function; hypothesis testing, not KNN voting.
Potential formula overlap: None found. Log density-ratios appear in two-sample testing, covariate shift, and global rescoring — never as a clipped per-neighbor compatibility penalty exp(-alpha*|log(rho_query/rho_i)|) inside a weighted KNN vote.
Decision: no equivalent found in screened results — candidate formulation remains unverified, not claimed novel

## Query 6

Search date: 21 October 2026
Database: WebSearch (web-wide)
Query: "adaptive distance weighted KNN" AND density
Number of results screened: 8 (first result page)
Relevant papers:
- Application-level/survey material (wafer defect classification survey arXiv:2310.10705; teaching/blog material) — describes adaptive-k-by-density and standard inverse-distance weighting as separate techniques.
- AKNN for collaborative filtering (Semantic Scholar) — adaptive metric via local density estimation; changes the *distance*, not a multiplicative vote-weight penalty.
Potential formula overlap: None found. "Adaptive distance weighted" variants either adapt k, adapt the metric, or weight purely by distance; no density-compatibility exponential factor on the weight.
Decision: no equivalent found in screened results — candidate formulation remains unverified, not claimed novel

## Query 7

Search date: 21 October 2026
Database: WebSearch (web-wide, incl. sciencedirect.com, arxiv.org)
Query: "local density aware" AND KNN
Number of results screened: 8 (first result page)
Relevant papers:
- "DW-KNN: A Transparent Local Classifier Integrating Distance Consistency and Neighbor Reliability" (arXiv:2512.08956, 2025) — reliability/consistency weighting, not density log-ratio.
- "Local Density-Aware Oversampling with Noise Resistance for Imbalanced Data Classification" (SOIC 2025/26) — data-level oversampling, upstream of the classifier.
- Xue, "Adaptive weighting and nearest neighbor-based area..." (ASOC 2025) — density-aware oversampling context.
Potential formula overlap: None found. Density-aware components are either preprocessing (oversampling) or reliability scores; no exp(-alpha*clip(|log(rho_q/rho_i)|)) vote factor.
Decision: no equivalent found in screened results — candidate formulation remains unverified, not claimed novel

## Query 8

Search date: 21 October 2026
Database: WebSearch (web-wide, incl. ieeexplore.ieee.org, springer.com, dl.acm.org)
Query: "density-based adaptive k nearest neighbor" classification
Number of results screened: 10 (first result page)
Relevant papers:
- Yuan et al., DBANN, "A novel density-based adaptive k nearest neighbor method for dealing with overlapping problem in imbalanced datasets" (Neural Comput. Appl. 33:4457-4481, 2021; doi:10.1007/s00521-020-05256-0) — density partitions + adaptive k for overlap/imbalance; reviewed in detail below.
- Amer, Ravana & Habeeb, "Effective k-nearest neighbor models for data classification enhancement" (J. Big Data 12:86, 2025; doi:10.1186/s40537-025-01137-2) — PR/WPR-kNN; full text inspected, details below.
- Pan, Wang & Pan, DC-LAKNN (KNOSYS 204:106185, 2020; doi:10.1016/j.knosys.2020.106185) — locally adaptive k by discrimination class; details below.
- IEEE "Multiview Adaptive K-Nearest Neighbor Classification" (2025) — multiview adaptive KNN, unrelated mechanism.
Potential formula overlap: None found among the screened adaptive/density-based KNN family; see per-paper assessment below.
Decision: no equivalent found in screened results — candidate formulation remains unverified, not claimed novel

---

## ACM 2025 Dynamic DW-KNN inspection

**Target:** DOI 10.1145/3796159.3796172, "A Dynamic Distance-Weighted KNN Model for Customized Mathematics
Education" (reported to use local density + uncertainty).

Verified from open metadata (Crossref, Semantic Scholar, Unpaywall, ACM search snippets):
- Title: "A Dynamic Distance-Weighted KNN Model for Customized Mathematics Education".
- Author: Dong-Kun Han (single author per Crossref record).
- Venue: Proceedings of the 9th International Conference on Digital Technology in Education (ICDTE 2025),
  pages 197-203; online 2025-12-21; indexed 2026-10-02. Type: proceedings-article, 31 references.
- License: CC BY 4.0 (hybrid OA per Unpaywall).
- Abstract (Semantic Scholar): DDW-KNN drives "eQuiz" (learner proficiency-level classification) and
  "eStudy" (personalized material delivery). The abstract describes a dynamic distance-weighted KNN but
  contains **no formula** and no mention of a density log-ratio.
- One search-engine snippet of the ACM full-text page contains the phrase "...local density and uncertainty
  criteria. Compared to a simple ...", i.e., neighbor selection/weighting is described as using local density
  and uncertainty criteria. The snippet is truncated; the sentence context and the equations cannot be seen.

Could NOT be verified:
- Full text and equations. dl.acm.org returns HTTP 403 / bot-verification to direct fetch, curl with browser
  UA, and a reader-proxy attempt; the Internet Archive has no snapshot of this DOI (CDX empty, very recent
  publication). The CC-BY PDF at dl.acm.org/doi/pdf/... is equally bot-blocked from this environment.
- Therefore it is **not possible from open sources to confirm or exclude** that DDW-KNN's weight contains an
  exp(-alpha * |log(rho_query/rho_neighbor)|)-type factor, nor what its density proxy is (KNN-radius,
  Gaussian kernel, count-in-radius, etc.).
- Assessment: an education-domain system paper whose "dynamic" weighting reportedly combines distance with
  local density and uncertainty criteria. Even under a charitable reading, nothing in the accessible
  metadata indicates the specific clipped absolute log-density-ratio compatibility penalty of the candidate
  formula — but this remains **unverified, not cleared**.

Update 2026-10-21 (full text obtained and inspected, all equations read):
- **FORMULA OVERLAP FOUND at the core-structure level.** Han (ICDTE 2025), eq. (5) and eq. (9):
  g_i = exp(-tau * |log rho~(x) - log rho~_i|) with w_i = (d_i + eps)^{-p(x)} * g_i * u_i * psi(z_i) * 1/n_{y_i}
  (plus a dual relative-distance rescaling, eq. (10)). The multiplicative pair
  (distance power weight) x (exponential absolute log-density-ratio gate between query and neighbor)
  is exactly the ALD-WKNN core. DDW-KNN adds: uncertainty reliability u_i = 1 - normalized entropy,
  Tukey biweight robustness, class-size normalization, adaptive exponent p(x), ReliefF feature weights.
  DDW-KNN has no clipping min(c, .); its density proxy is an inverse mean-distance over the
  classification neighborhood rather than a q-th-neighbor radius.
- The median normalization rho~ = rho / median cancels inside the log-ratio when one global
  training median is used, leaving the same |log(rho_x / rho_i)| quantity.
- **Decision: FORMULA OVERLAP FOUND — ALD-WKNN must be positioned as a minimal
  replication/extension-style variant of the published density-gate idea (independent density
  scale q, clipped mismatch, no auxiliary factors), explicitly citing Han 2025.** What remains
  distinct is the minimalism (two factors vs five), the clipping, the independent q scale, and the
  reproducible nested-CV multi-dataset evaluation — NOT the core concept.
- Prior-work concealment check: none. The overlap is disclosed in the paper's framing, related
  work matrix, and reference list (references.bib entry dynamicdwknn2025 completed with full metadata).

Action: obtain the full text via institutional access and check the DDW-KNN weight equation against
d^-p * exp(-alpha * min(c, |log(rho_q/rho_i)|)) before any positioning statement is made.

---

## Known-predecessor check (context confirmation, not novelty claims)

Question asked of each: does the method use a **penalty on query-vs-neighbor density compatibility of the
form |log(rho_query/rho_neighbor)|** (or its min/max-ratio equivalent), multiplying an inverse-distance
weight? (Rewarding density, adapting k, or adapting the metric does not count.)

1. DBANN — Yuan et al. 2021, doi:10.1007/s00521-020-05256-0 (Neural Comput. Appl. 33:4457-4481).
   Uses local density to partition the data and adapt the neighborhood/k for overlapping + imbalanced
   regions. Density is a region-level signal, not a symmetric clipped log-ratio vote factor.
   |log-ratio| compatibility penalty: NOT PRESENT (to the extent verifiable from abstracts/secondary
   descriptions; full text paywalled).

2. Abdalla & Amer 2025, doi:10.1016/j.eswa.2025.126942 (Expert Syst. Appl. 276, LIWkNN).
   Weighting adjusted by *class* density so minority classes receive adequate weight, plus a weight-update
   rule for repeatedly misclassified (outlier) points. Class-level density reward + error-driven updates —
   not a query-vs-neighbor density log-ratio penalty.
   |log-ratio| compatibility penalty: NOT PRESENT.

3. Amer, Ravana & Habeeb 2025, doi:10.1186/s40537-025-01137-2 (J. Big Data 12:86, PRkNN/EPRkNN/WPRkNN).
   Full text inspected (SpringerOpen PDF). Weights are: (i) proximal ratio PR = fraction of same-class
   points within a class radius (a class-consistency score, not a density), (ii) inverse distance 1/d,
   (iii) class frequency n_c/k, (iv) a Dudani-style linear distance weight. No exp/log density-ratio term
   anywhere in the methodology.
   |log-ratio| compatibility penalty: NOT PRESENT.

4. Pan, Wang & Pan 2020, doi:10.1016/j.knosys.2020.106185 (DC-LAKNN, KNOSYS 204:106185).
   Locally adaptive *k* chosen per query via a discrimination-class criterion. Adaptive neighborhood size —
   density is not used at all in the vote weighting, and no log-ratio term exists.
   |log-ratio| compatibility penalty: NOT PRESENT.

Conclusion of predecessor check: none of the four known predecessors uses a query-vs-neighbor density
compatibility penalty of the screened form; they use density/class-consistency as reward, as region signal
for k, or as outlier score.

---

## Overall conclusion

- Across the 8 queries (~67 result snippets screened on first pages), plus metadata-level inspection of the
  ACM DDW-KNN paper and formula-level review of the four known predecessors, **no formula-level equivalent**
  of `w_i = d^-p * exp(-alpha * min(c, |log(rho_q(x)/rho_q(x_i))|))` was found in any accessible source.
- No screened paper multiplies an inverse-distance KNN weight by a clipped absolute log-density-ratio
  compatibility factor between query and neighbor (nor by the equivalent min/max density ratio raised to a
  power).
- **Caveats that keep this from being a novelty certificate:** (a) the ACM 2025 DDW-KNN paper could not be
  full-text inspected from this environment (bot-blocked; no archive snapshot) and reportedly combines
  distance weighting with local-density and uncertainty criteria — its weight equation must be checked
  manually; (b) first-page screening only; (c) paywalled full texts (DBANN, ESWA LIWkNN, KNOSYS DC-LAKNN)
  were assessed from abstracts and secondary descriptions, not equation-by-equation.
- Gate decision: **no equivalent found in screened results — candidate formulation remains unverified, not
  claimed novel.** Any manuscript text must avoid novelty language for this weighting until the ACM DDW-KNN
  full text and, ideally, a deeper second-page/Scopus-style search pass are cleared.
