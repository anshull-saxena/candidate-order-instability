# Statistical Analysis Protocol

This document details the statistical inference framework, dependency handling, hypothesis family definitions, multiple-testing corrections, and interpretation standards.

---

## 1. Experimental Hierarchy and Independent Sampling Unit

A core flaw in naive order-sensitivity benchmarking is pseudo-replication: treating multiple permutations or pairwise permutation comparisons as independent observations.

Our statistical analysis strictly respects the following observational hierarchy:

```
Query ID (i = 1, ..., N)        <--- INDEPENDENT STATISTICAL SAMPLING UNIT
└── Candidate Set C_{i, K}       <--- Deterministically nested (K5 ⊂ ... ⊂ K77)
    └── Permutation (p = 1, ..., P)
        └── Pairwise Comparison (a < b)
```

- **Independent Unit:** The query $q_i \in \mathcal{D}_{\mathrm{test}}$ ($N=120$).
- All candidate set variations, permutations, and shift offsets are nested within each query.
- No inference test or confidence interval is ever computed by pooling permutation pairs across queries.

---

## 2. Clustered Percentile Bootstrap Confidence Intervals

For any metric $\theta$ (accuracy, ECE, $\mathrm{FR}_{\mathrm{top1}}$, Kendall distance, churn):
1. Compute the query-level summary statistic $x_i$ for each query $i \in \{1, \dots, N\}$.
2. Draw $B = 1,000$ bootstrap resamples of the query index set $\{1, \dots, N\}$ with replacement.
3. For each resample $b$, compute the resampled mean $\bar{x}^{*(b)} = \frac{1}{N}\sum_{i=1}^N x_{j_i}^{*(b)}$.
4. The 95% percentile confidence interval is:
   $$[\mathrm{CI}_{\mathrm{lo}}, \mathrm{CI}_{\mathrm{hi}}] = [\operatorname{Percentile}(\bar{x}^*, 2.5), \operatorname{Percentile}(\bar{x}^*, 97.5)]$$

---

## 3. Paired Hypothesis Testing for Accuracy

To test whether an intervention method $M$ changes classification accuracy relative to the uncontrolled baseline $\mathrm{B0\_Native}$:
1. Construct the $2 \times 2$ paired contingency table across the $N=120$ queries:
   - $n_{00}$: Both methods incorrect.
   - $n_{01}$: $\mathrm{B0\_Native}$ incorrect, Method $M$ correct (**Discordant Gain**).
   - $n_{10}$: $\mathrm{B0\_Native}$ correct, Method $M$ incorrect (**Discordant Loss**).
   - $n_{11}$: Both methods correct.
2. Under the null hypothesis $H_0: \mathbb{P}(\text{Gain}) = \mathbb{P}(\text{Loss}) = 0.5$, the discordant count $n_{01}$ follows a binomial distribution:
   $$n_{01} \sim \operatorname{Binomial}(n_{01} + n_{10}, 0.5)$$
3. Compute the two-sided exact binomial $p$-value:
   $$p = 2 \cdot \min\left( \sum_{k=0}^{n_{01}} \binom{n_{\mathrm{disc}}}{k} 0.5^{n_{\mathrm{disc}}}, \sum_{k=n_{01}}^{n_{\mathrm{disc}}} \binom{n_{\mathrm{disc}}}{k} 0.5^{n_{\mathrm{disc}}} \right)$$

---

## 4. Paired Testing for Continuous Instability Metrics

To test whether marginalization or cyclic shifting significantly reduces order instability:
1. Compute the paired per-query difference $d_i = \mathrm{FR}_i(\mathrm{Method}) - \mathrm{FR}_i(\mathrm{Baseline})$.
2. Apply the two-sided paired Wilcoxon signed-rank test on non-zero differences.
3. Compute the rank-biserial correlation effect size:
   $$r = \frac{W_+ - W_-}{W_+ + W_-}$$
   where $W_+$ and $W_-$ are the sums of positive and negative ranks.

---

## 5. Multiple-Testing Correction: Pre-Specified Hypothesis Family

### Pre-Specified Family Definition
To prevent post-hoc hypothesis cherry-picking, the comparison family is fixed *a priori* to include all paired accuracy comparisons of each intervention method against the baseline $\mathrm{B0\_Native}$ across all cardinalities:

$$\mathcal{H} = \left\{ \text{Method } m \text{ vs } \mathrm{B0\_Native} \mid K \in \{5, 10, 20, 40, 77\}, m \in \text{Methods} \setminus \{\mathrm{B0\_Native}\} \right\}$$

$$\text{Total Pre-Specified Hypotheses: } M = 5 \text{ cardinalities} \times 9 \text{ methods} = 45 \text{ hypotheses}$$

### Benjamini-Hochberg (BH) Procedure
To control the False Discovery Rate (FDR) at level $q = 0.05$:
1. Sort raw $p$-values in ascending order: $p_{(1)} \le p_{(2)} \le \dots \le p_{(M)}$.
2. Compute adjusted $p$-values:
   $$p_{(i)}^{\mathrm{adj}} = \min_{j \ge i} \left( \min\left(1.0, \frac{M \cdot p_{(j)}}{j}\right) \right)$$
3. Declare significance only if $p_{(i)}^{\mathrm{adj}} \le 0.05$.

---

## 6. Scientific Interpretation Rules

1. **Strict Terminology Gate:** The term *"statistically significant"* is strictly forbidden unless $p^{\mathrm{adj}} \le 0.05$. Numerical deltas with $p^{\mathrm{adj}} > 0.05$ must be described as descriptive or preliminary.
2. **Distinguishing Trends from Significance:** A monotonic trend across $K$ (e.g. logit variance increasing $8.40 \to 63.34$) is reported as descriptive evidence when individual cell tests are not parameterized as a single regression.
3. **Replication Transparency:** Exploratory findings from earlier underpowered runs (such as $N=40$ estimates) that fail to replicate under controlled conditions ($N=120$) must be explicitly reported as disconfirmed.
