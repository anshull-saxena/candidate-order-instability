# Mathematical Formulations and Metric Invariant Verification

This document provides formal mathematical definitions, estimands, boundary conditions, and dependency structures for all evaluation metrics used in the study.

---

## 1. Top-1 Decision Flip Rate ($\mathrm{FR}_{\mathrm{top1}}$)

### Estimand
The probability that two independent random candidate permutations of the exact same candidate set $\mathcal{C}$ yield different $\operatorname{argmax}$ class predictions for query $q$:

$$\mathrm{FR}_{\mathrm{top1}}(q, \mathcal{C}) = \mathbb{E}_{\pi_1, \pi_2 \sim \mathcal{S}_K} \left[ \mathbf{1}\left( \operatorname*{argmax}_{c \in \mathcal{C}} s(q, c; \pi_1) \neq \operatorname*{argmax}_{c \in \mathcal{C}} s(q, c; \pi_2) \right) \right]$$

### Empirical Estimator
Given a sample of $P$ independent permutations $\pi_1, \dots, \pi_P$ for query $q$:

$$\widehat{\mathrm{FR}}_{\mathrm{top1}}(q) = \frac{1}{\binom{P}{2}} \sum_{1 \le a < b \le P} \mathbf{1}\left( \operatorname*{argmax}_{c} s(q, c; \pi_a) \neq \operatorname*{argmax}_{c} s(q, c; \pi_b) \right)$$

### Statistical Dependency Structure
- For $P=10$, there are $\binom{10}{2} = 45$ pairwise comparisons.
- **Critical Dependency:** These 45 comparisons are **not independent Bernoulli trials**. Each individual permutation prediction $\hat{y}_a = \operatorname*{argmax}_c s(q, c; \pi_a)$ appears in $P - 1 = 9$ distinct pairs.
- Treating 45 pairs as 45 independent samples is a severe violation of statistical independence (analogous to pseudo-replication).
- **Correct Treatment:** The 45 comparisons form a one-sample $U$-statistic of degree 2 for query $q$. The independent statistical unit is strictly the **query ID** ($N=120$). Per-query mean flip rates are computed first, and bootstrap resampling is clustered at the query level.

### Boundary Conditions & Invariants
- **Identity Invariant:** $\mathrm{FR}(\pi_a, \pi_a) \equiv 0.0$.
- **Complete Inversion:** If $\hat{y}_a \neq \hat{y}_b$ for all pairs, $\widehat{\mathrm{FR}}_{\mathrm{top1}} = 1.0$.

---

## 2. Normalized Kendall's Tau Distance ($\tau_{\mathrm{norm}}$)

### Estimand
The fraction of discordant candidate pairs between two complete ranking vectors over $K$ candidates:

$$\tau_{\mathrm{norm}}(s_a, s_b) = \frac{1}{\binom{K}{2}} \sum_{1 \le i < j \le K} \mathbf{1}\left( (s_a[i] - s_a[j])(s_b[i] - s_b[j]) < 0 \right)$$

### Properties
- $\tau_{\mathrm{norm}} \in [0, 1]$.
- $\tau_{\mathrm{norm}}(s, s) = 0.0$ (identical rankings).
- $\tau_{\mathrm{norm}}(s, \mathrm{rev}(s)) = 1.0$ (strictly reversed rankings).
- For two uniform random independent permutations, $\mathbb{E}[\tau_{\mathrm{norm}}] = 0.50$.
- **Tie Handling:** In float32 log-space, exact ties are zero-probability events. However, when scores are rounded or softmax probabilities saturate near zero (common for distant distractors at $K \ge 40$), ties are treated as concordant ($< 0$ is false).

---

## 3. Top-3 Jaccard Churn ($\mathrm{Churn}_{\mathrm{top3}}$)

### Estimand
The complement of the Jaccard similarity between the top-3 candidate sets under two permutations:

$$\mathrm{Churn}_{\mathrm{top3}}(s_a, s_b) = 1 - \frac{|\operatorname{Top3}(s_a) \cap \operatorname{Top3}(s_b)|}{|\operatorname{Top3}(s_a) \cup \operatorname{Top3}(s_b)|}$$

where $\operatorname{Top3}(s) = \operatorname{argtopk}(s, k=3)$.

### Boundary Conditions
- If $\operatorname{Top3}(s_a) = \operatorname{Top3}(s_b)$, $\mathrm{Churn}_{\mathrm{top3}} = 0.0$.
- If $\operatorname{Top3}(s_a) \cap \operatorname{Top3}(s_b) = \emptyset$, $\mathrm{Churn}_{\mathrm{top3}} = 1.0$.
- At $K=5$, the maximum union size is 5. If top-3 sets share 1 candidate, $\mathrm{Churn} = 1 - 1/5 = 0.80$; if they share 2 candidates, $\mathrm{Churn} = 1 - 2/4 = 0.50$.
- If $K < 3$, churn is defined as 0.0.

---

## 4. Cross-Canonical Inconsistency Rate ($\mathrm{CrossCanonicalFlip}$)

### Rationale
Alphabetical sorting ($\mathrm{B1\_Alpha}$) produces zero run-to-run variance merely by repeating a deterministic sequence. Zero repeatability is **not** evidence of permutation invariance.

### Estimand
The disagreement rate between two opposite deterministic canonical orderings (Alphabetical $A \to Z$ vs Reverse Alphabetical $Z \to A$):

$$\mathrm{CrossCanonicalFlip}(q, \mathcal{C}) = \mathbf{1}\left( \operatorname*{argmax}_{c \in \mathcal{C}} s(q, c; \pi_{\alpha}) \neq \operatorname*{argmax}_{c \in \mathcal{C}} s(q, c; \pi_{\mathrm{rev\_}\alpha}) \right)$$

If an encoder has positional bias (e.g. primacy bias favoring index 0), reversing the alphabetical order reverses the positional assignments of all candidates. A model that is truly permutation-invariant will have $\mathrm{CrossCanonicalFlip} \equiv 0.0$.

---

## 5. Permutation-Level Logit Variance ($\sigma^2_{\mathrm{logit}}$)

### Estimand
For a fixed query $q$ and candidate set $\mathcal{C}$, let $\ell(q, c; \pi_p) = \log(p(q, c; \pi_p) + \epsilon)$ be the log-probability of candidate $c$ under permutation $\pi_p$.

The across-permutation variance for candidate $c$ is:

$$s^2(q, c) = \frac{1}{P - 1} \sum_{p=1}^P \left( \ell(q, c; \pi_p) - \bar{\ell}(q, c) \right)^2$$

where $\bar{\ell}(q, c) = \frac{1}{P} \sum_{p=1}^P \ell(q, c; \pi_p)$.

The query-level logit variance is the mean across all $K$ candidates:

$$\sigma^2_{\mathrm{logit}}(q) = \frac{1}{K} \sum_{c \in \mathcal{C}} s^2(q, c)$$

---

## 6. Residual Ensemble Flip Rate

### Estimand
To measure whether an ensemble of size $M$ has converged to order invariance, we partition independent permutations into two non-overlapping ensembles of size $M$:

$$\bar{p}^{(1)}(c) = \frac{1}{M} \sum_{m=1}^M p(q, c; \pi_m), \quad \bar{p}^{(2)}(c) = \frac{1}{M} \sum_{m=M+1}^{2M} p(q, c; \pi_m)$$

The residual flip rate is:

$$\mathrm{ResFlip}_M(q) = \mathbf{1}\left( \operatorname*{argmax}_c \bar{p}^{(1)}(c) \neq \operatorname*{argmax}_c \bar{p}^{(2)}(c) \right)$$

For cyclic marginalization, the two ensembles are constructed using two orthogonal cyclic shift phases:
- Phase 1: Shifts $s_m = (m \cdot \lfloor K / M \rfloor) \pmod K$
- Phase 2: Shifts $s'_m = ((m \cdot \lfloor K / M \rfloor) + \lfloor K / 2M \rfloor) \pmod K$

---

## 7. Expected Calibration Error (ECE)

### Estimand
Let confidence be the maximum probability assigned to the predicted class: $\hat{c} = \max_c p_c$.  
Data is partitioned into $B=15$ equally spaced confidence bins $I_b = (\frac{b-1}{B}, \frac{b}{B}]$.

$$\mathrm{ECE} = \sum_{b=1}^B \frac{|I_b|}{N} \left| \operatorname{acc}(I_b) - \operatorname{conf}(I_b) \right|$$

where $\operatorname{acc}(I_b) = \frac{1}{|I_b|}\sum_{i \in I_b} \mathbf{1}[\hat{y}_i = y_i^*]$ and $\operatorname{conf}(I_b) = \frac{1}{|I_b|}\sum_{i \in I_b} \hat{c}_i$.

### Critical Implementation Note
The calibrated model output probabilities from `Agent.system_one_batch` already reflect temperature scaling and softmax. Applying a redundant second softmax crushes true probabilities (e.g. $0.90 \to 0.54$) and severely invalidates ECE. Our implementation operates directly on calibrated probability vectors.
