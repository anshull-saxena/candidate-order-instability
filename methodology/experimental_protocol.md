# Experimental Protocol and Execution Standards

This document describes the exact experimental protocol required to reproduce the Candidate Cardinality Scaling and Inference-Time Order-Marginalization study.

---

## 1. Dataset and Query Sampling

- **Dataset:** `mteb/banking77` (Banking customer intent classification, test split).
- **Class Cardinality:** Total classes $K_{\mathrm{max}} = 77$.
- **Sample Size:** $N = 120$ unique queries.
- **Stratification Algorithm:**
  1. Group all 3,076 test queries by ground-truth intent class (77 classes).
  2. Sample exactly 1 query from each of the 77 classes (guaranteeing 100% label space coverage).
  3. Sample 1 additional query from 43 randomly selected classes to reach exactly $N = 120$ unique queries without replacement.
  4. Fix selection with random seed `GLOBAL_SEED = 42`.
  5. The identical list of 120 query indices is evaluated across all cardinalities $K$.

---

## 2. Strictly Nested Candidate Set Construction

To isolate candidate cardinality from random distractor composition, candidate sets are strictly nested:

For each query $q_i$ with ground truth class $y^*$:
1. Form candidate distractor pool $\mathcal{D}_{\mathrm{pool}} = \operatorname{sorted}(\text{All 77 classes} \setminus \{y^*\})$.
2. Generate a single master permutation of $\mathcal{D}_{\mathrm{pool}}$ using deterministic seed $S_i = \mathrm{GLOBAL\_SEED} + i \times 1000$:
   $$\mathcal{D}_{\mathrm{master}}^{(i)} = [d_1, d_2, \dots, d_{76}]$$
3. Define candidate sets strictly as slices of the master sequence:
   - $K=5$: $\mathcal{C}_5 = [y^*] + \mathcal{D}_{\mathrm{master}}[0:4]$
   - $K=10$: $\mathcal{C}_{10} = [y^*] + \mathcal{D}_{\mathrm{master}}[0:9]$
   - $K=20$: $\mathcal{C}_{20} = [y^*] + \mathcal{D}_{\mathrm{master}}[0:19]$
   - $K=40$: $\mathcal{C}_{40} = [y^*] + \mathcal{D}_{\mathrm{master}}[0:39]$
   - $K=77$: $\mathcal{C}_{77} = [y^*] + \mathcal{D}_{\mathrm{master}}[0:76]$
4. Programmatically assert:
   $$\operatorname{set}(\mathcal{C}_5) \subset \operatorname{set}(\mathcal{C}_{10}) \subset \operatorname{set}(\mathcal{C}_{20}) \subset \operatorname{set}(\mathcal{C}_{40}) \subset \operatorname{set}(\mathcal{C}_{77})$$
5. Compute and log the SHA-256 hash of the sorted candidate set for every query and cardinality.

---

## 3. Repeated Base Candidate Orderings at $K=77$

At $K=77$, all 77 classes are present, so distractor subsampling is not applicable.  
To prevent results from being an artifact of a single base ordering (e.g. alphabetical), we evaluate across $S=3$ independently seeded base candidate orderings:
- **Base Order 0:** Alphabetical order (`sorted(labels)`).
- **Base Order 1:** Pseudo-random shuffle initialized with seed `7701`.
- **Base Order 2:** Pseudo-random shuffle initialized with seed `7702`.

Per-query metrics at $K=77$ are computed by averaging across the 3 base orderings.

---

## 4. Evaluated Methods and Interventions

### Baselines
- **`B0_Native`:** Native insertion order ($\mathcal{C}_K = [y^*] + \mathcal{D}[0:K-1]$).
- **`B0_Random`:** Single pseudo-random permutation of $\mathcal{C}_K$.
- **`B1_Alpha`:** Deterministic alphabetical sorting ($A \to Z$).
- **`B1_ReverseAlpha`:** Deterministic reverse alphabetical sorting ($Z \to A$).
  - Evaluates cross-canonical inconsistency: $\mathbf{1}[y_{\alpha} \neq y_{\mathrm{rev\_}\alpha}]$.

### Random Order-Marginalization ($M \in \{2, 3, 5\}$)
- Sample $P=10$ independent random permutations of $\mathcal{C}_K$.
- Ensemble prediction is the average probability vector over the first $M$ permutations:
  $$\bar{p}_c = \frac{1}{M}\sum_{m=1}^M p(c; \pi_m)$$
- Residual flip rate is evaluated between two non-overlapping $M$-subsets (e.g. perms $[0:M]$ vs $[M:2M]$).

### Cyclic Orthogonal Marginalization ($M \in \{2, 3, 5\}$)
- For a base candidate sequence of length $K$, define shift step $s = \lfloor K / M \rfloor$.
- **Phase 1 Shifts:** $s_m = (m \cdot s) \pmod K$ for $m = 0, \dots, M-1$.
- **Phase 2 Shifts (Orthogonal Check):** $s'_m = (m \cdot s + \lfloor s / 2 \rfloor) \pmod K$.
- Prediction: Argmax of Phase 1 average.
- Residual Flip Rate: Disagreement between Phase 1 average and Phase 2 average.

---

## 5. Latency Measurement Protocol

- **Warmup:** Execute 1 complete batch forward pass prior to recording any latency numbers to allow PyTorch graph compilation and MPS shader allocation.
- **Chunking:** Questions are evaluated in memory-safe chunks of at most 4 concurrent questions per forward pass to prevent GPU unified memory thrashing.
- **Synchronization:** Call `torch.mps.synchronize()` (or `torch.cuda.synchronize()`) before and after timing blocks.
- **Reporting:** Report median (p50) and 95th percentile (p95) latency per query decision in milliseconds.
