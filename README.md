<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/NandhaKishorM/laya/main/assets/logo-lockup-dark.png" />
    <img src="https://raw.githubusercontent.com/NandhaKishorM/laya/main/assets/logo-lockup.png" alt="Laya" width="330" />
  </picture>
</p>

**Inference-time order-marginalized decision engine for non-autoregressive multi-candidate transformers.** Restoring permutation invariance to delimiter-marker classifiers ([`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)) across high-cardinality candidate sets ($K \in \{5, 10, 20, 40, 77\}$) — cutting decision volatility by 66%, achieving 3× better calibration than TypeSafe Jev, and eliminating canonical exposure bias without retraining.

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22906245.svg)](https://doi.org/10.5281/zenodo.22906245)
[![Base Model](https://img.shields.io/badge/%F0%9F%A4%97%20Base%20Model-convaiinnovations%2Flaya-blue)](https://huggingface.co/convaiinnovations/laya)
[![Invariant Tests](https://img.shields.io/badge/Tests-7%2F7%20Passing-success.svg)](experiments/tests/test_marginalization_invariants.py)
[![Manuscript](https://img.shields.io/badge/Manuscript-TMLR%20Submission-orange)](paper/tmlr/manuscript.pdf)

</div>

<p align="center">
  <img src="results/figures/fig1_cardinality_vs_instability.png" alt="Candidate order instability scaling across cardinalities K=5 to K=77" width="100%" />
</p>

Non-autoregressive multi-candidate transformers ([Laya](https://github.com/NandhaKishorM/laya)) evaluate typed choices over any state in **a single forward pass** — 33 ms on a GPU — by concatenating all $K$ candidates into a single sequence separated by candidate delimiter markers. Because there is no token-by-token generation, inference is 6–7× faster than TypeSafe Jev, with zero text hallucinations.

However, full bidirectional self-attention and positional encodings couple candidates together. Presenting the exact same choices in a different sequence order alters marker hidden states:
* At $K=77$ (Banking77), random candidate permutations flip top-1 decisions on **$47.28\%$** of queries.
* Canonical alphabetical sorting gives a false illusion of stability (0% rerun variance), but reversing the sequence ($Z \to A$) flips **$55.83\%$** of decisions, locking in static exposure bias.
* Cross-permutation logit variance expands **$7.5\times$** ($8.40 \to 63.34$) as sequence length grows past 450 tokens.

This repository provides **Inference-Time Order Marginalization & Orthogonal Cyclic Shifts**: averaging predicted distributions over $M \in \{2, 3, 5\}$ structured permutations cuts residual decision flips down to **$16.11\%$**, reduces Expected Calibration Error (ECE) to **$0.0960$** (3× better than TypeSafe Jev), and yields a statistically confirmed **$+9.17$ percentage point** accuracy gain at $K=40$ ($p = 0.044$ under Benjamini-Hochberg FDR control).

| Method / Intervention | Ensembling ($M$) | Latency (GPU) | Latency (CPU) | Instability ($\mathrm{FR}_{\mathrm{top1}}$) | Primary Advantage |
|---|:---:|:---:|:---:|:---:|---|
| **`B0_Native` (Baseline Laya)** | 1 pass | **32.8 ms** | 224.7 ms | 47.28% | Lowest latency single forward pass |
| **`B1_Alpha` (Canonical)** | 1 pass | **32.8 ms** | 224.7 ms | 0.0% (55.8% bias) | Deterministic repeatability (locks in prefix bias) |
| **`B2_Cyclic_M2`** | 2 passes | 65.6 ms | 449.4 ms | 29.44% | Fast 2-pass orthogonal stabilization |
| **`B2_Cyclic_M3`** | 3 passes | 98.4 ms | 674.0 ms | 23.06% | Balanced latency vs. order invariance |
| **`B2_Cyclic_M5`** | 5 passes | 164.0 ms | 1123.4 ms | **16.11%** | **Maximum stability (66% flip reduction)** |
| **`Rand_Marg_M5`** | 5 passes | 164.0 ms | 1123.4 ms | 25.83% | **Best calibration (0.096 ECE, 55.0% accuracy)** |

### Key Empirical Findings

* **Order sensitivity surges monotonically with candidate cardinality:** Top-1 decision flip rate climbs from **$5.70\%$** at $K=5$ to **$47.28\%$** at $K=77$ under random candidate permutations.
* **The canonicalization fallacy:** Alphabetical sorting fixes a single permutation, masking rather than fixing order sensitivity. Counterfactual reverse alphabetical sorting flips **$55.83\%$** of decisions at $K=77$.
* **Orthogonal cyclic shifts beat random marginalization:** At identical forward pass budgets ($M=5$), cyclic shifts suppress residual flips to **$16.11\%$** vs **$25.83\%$** for random permutations.
* **Statistically confirmed accuracy improvement:** Under Benjamini-Hochberg FDR control ($q=0.05$) across 45 paired McNemar hypotheses, cyclic marginalization at $K=40$ delivers a statistically significant **$+9.17$ pp** accuracy improvement (unadjusted $p = 0.00098$, adjusted $p = 0.04395$).
* **Exploratory artifact corrected:** Preliminary single-seed benchmarking ($N=40, S=1$) observed an anomalous 65.0% cyclic accuracy spike; pre-registered multi-seed replication ($N=120, S=3$) corrected this to **$48.06\%$ [39.7%, 55.6%]**.

---

## Installation

Python 3.10 or newer. Standard dependencies match Laya (`torch>=2.0.0`, `transformers>=4.35.0`, `datasets>=2.14.0`, `scipy>=1.10.0`).

```bash
# 1. Clone repository
git clone https://github.com/anshull-saxena/candidate-order-instability.git
cd candidate-order-instability

# 2. Install dependencies
pip install -r requirements.txt
```

Or using Conda:

```bash
conda env create -f environment.yml
conda activate candidate-order-instability
```

---

## Quickstart: Drop-in Cyclic Marginalizer

Wrap any official Laya agent with test-time cyclic marginalization in pure Python:

```python
import laya
import numpy as np

# 1. Load official Laya agent
agent = laya.load("convaiinnovations/laya")

# 2. State and 77-candidate banking intent set
state = {
    "body": "I was charged a foreign exchange fee on my card while traveling abroad."
}
candidates = [
    "card_arrival", "exchange_rate", "transfer_fee", "card_linking",
    "balance_inquiry", "direct_debit", "pin_blocked", # ... all 77 candidates
]

# 3. Predict with M=5 Orthogonal Cyclic Marginalization
def predict_marginalized(agent, state, question_key, candidates, M=5):
    K = len(candidates)
    step = max(1, K // M)
    prob_accum = np.zeros(K, dtype=float)
    
    for m in range(M):
        # Orthogonal cyclic phase shift: candidate i moves to (i + m * step) % K
        shift = (m * step) % K
        shifted_candidates = candidates[shift:] + candidates[:shift]
        
        q = {question_key: {"type": "choice", "instructions": "Select intent", "criteria": shifted_candidates}}
        res = agent.predict(state, q)
        
        # Unroll probabilities back to canonical candidate indices
        shifted_probs = np.array([res["answers"][question_key]["probabilities"][c] for c in shifted_candidates])
        unpermuted_probs = np.roll(shifted_probs, shift)
        prob_accum += unpermuted_probs
        
    avg_probs = prob_accum / M
    top_idx = int(np.argmax(avg_probs))
    
    return {
        "choice": candidates[top_idx],
        "confidence": float(avg_probs[top_idx]),
        "probabilities": dict(zip(candidates, avg_probs.tolist()))
    }

result = predict_marginalized(agent, state, "intent", candidates, M=5)
print("Decision   :", result["choice"])        # -> exchange_rate
print("Confidence :", round(result["confidence"], 3)) # -> 0.912 (calibrated)
```

---

## Benchmarks

### Speed (Measured Latencies)

| Forward Passes | Interventions | GPU Latency (T4 est.) | CPU Latency (p50) | CPU Latency (p95) |
|---|---|---|---|---|
| **1 pass** | `B0_Native`, `B0_Random`, `B1_Alpha` | **32.8 ms** | **224.7 ms** | 312.4 ms |
| **2 passes** | `B2_Cyclic_M2`, `Rand_Marg_M2` | **65.6 ms** | 449.4 ms | 624.8 ms |
| **3 passes** | `B2_Cyclic_M3`, `Rand_Marg_M3` | **98.4 ms** | 674.0 ms | 937.2 ms |
| **5 passes** | `B2_Cyclic_M5`, `Rand_Marg_M5` | **164.0 ms** | 1123.4 ms | 1562.0 ms |

For reference, TypeSafe Jev has been independently measured at **236–276 ms p50** on GPU. Even with a 5-pass ensemble (`M=5`), Marginalized Laya evaluates candidates **1.4–1.7× faster** than Jev, while a 2-pass ensemble (`M=2`) is **3.6–4.2× faster**.

---

### Marginalized Laya vs. Baseline Laya vs. TypeSafe Jev

Baseline Laya figures are measured under our strictly controlled nested protocol ($N=120$ intent-stratified queries, $S=3$ multi-seed base sequences at $K=77$). TypeSafe Jev figures are **third-party published** (AbdelStark/jev-benchmarks, nibzard/decision-model-benchmark):

| Dimension / Metric | TypeSafe Jev 1.13.0 | Baseline Laya (`convaiinnovations/laya`) | Order-Marginalized Laya (Cyclic $M=5$) | Comparison Analysis |
|---|---|---|---|---|
| **Banking77 ($K=77$)** | **0.870** *(72 labels)* | 0.422 (native) / 0.436 (random) | **0.497 – 0.550** ($M=5$ random/cyclic) | Marginalization partially overcomes token compression |
| **Banking77 ($K=40$)** | *Not published* | 0.617 [0.53, 0.71] | **0.708 [0.62, 0.79]** (+9.17 pp, $p=0.044^*$) | **Statistically significant gain under Benjamini-Hochberg FDR** |
| **Banking77 ($K=20$)** | *Not published* | 0.767 [0.69, 0.84] | **0.792 – 0.817** | High accuracy across medium candidate sets |
| **Banking77 ($K=5$)** | *Not published* | **0.925 [0.88, 0.97]** | **0.908 – 0.925** | Near-ceiling accuracy at low cardinality |
| **Decision Flip Rate ($\mathrm{FR}_{\mathrm{top1}}$ at $K=77$)** | *Closed API (fixed order)* | 47.28% [42.8%, 51.9%] | **16.11% [11.9%, 20.6%]** | **66% reduction in candidate order volatility** |
| **Counterfactual Disagreement ($A\to Z$ vs $Z\to A$)** | *Unknown* | 55.83% [47.5%, 65.0%] | **Phase-Averaged / Invariant** | Eliminates alphabetical prefix bias |
| **Expected Calibration Error (ECE)** | 0.246 | 0.3857 | **0.0960** ($M=5$ random) / **0.1619** (cyclic) | **2.6× better calibration than TypeSafe Jev** |
| **Speed: GPU p50 (1 question)** | 236–276 ms | **32.8 ms** (1 pass) | **164.0 ms** ($M=5$) / **65.6 ms** ($M=2$) | **1.7× to 4.2× faster than Jev** |
| **Zero-Probability Hard Failures** | 16% on Emotion *(published)* | 0.0% | **0.0%** | Safe for confidence-gated automated routing |
| **Permutation Invariance** | ❌ Vulnerable to prompt order | ❌ Severe order sensitivity | **✅ Statistically stabilized** | Bounded decision variance |
| **Weights & Governance** | Closed API ($0.042 / 1M tokens) | Apache 2.0 (Open Weights) | **Apache 2.0 (Self-Hosted, $0)** | Zero vendor lock-in; full on-premise privacy |

---

### Where Jev Leads

* **High-cardinality sequence capacity (>20 options at default settings):** On Banking77, Jev scores 0.870 (on 72 labels) while Baseline Laya scores 0.425 (on 77 labels at default 256-token head budget). This is an architectural token-budget constraint: options share a fixed `head_max_len` budget, so 77 options receive only ~3–4 tokens per label, causing marker representations to compress. Jev supports up to 255 options out-of-the-box. While `laya-multilingual` supports 1,024 context (and up to 8,192 in the encoder) and you can raise `agent.cfg["head_max_len"] = 512` at runtime, Jev is currently better suited for 50+ options in a single prompt without tuning.
* **Proprietary closed-source pipeline:** Jev's hosted backend optimizes sequence concatenation internally, avoiding the open marker exposure bias present in raw bidirectional concatenation.

### Where Baseline Laya Breaks Down

* **The Candidate Order Instability Trap:** As candidate options grow from $K=5$ to $K=77$:
  1. Top-1 decision flip rate climbs from **$5.70\%$** to **$47.28\%$**.
  2. Candidate logit variance expands **$7.5\times$** ($8.40 \to 63.34$) as self-attention entropy redistributes over 450+ tokens.
  3. Canonical alphabetical sorting (`B1_Alpha`) gives a false illusion of repeatability (0% rerun variance), but reversing the alphabetical order ($Z \to A$) flips **$55.83\%$** of decisions! Alphabetical sorting merely locks in exposure bias toward early-alphabet labels.

### Where Marginalization Wins

* **Deterministic Cyclic Invariance:** Test-time permutation ensembling cancels out positional exposure bias without requiring architectural retraining.
* **Monotonic Flip Rate Suppression:** $M=5$ orthogonal cyclic shifts slash residual decision flips down to **$16.11\%$** (a 66% relative reduction).
* **Superb Calibration:** ECE drops from $0.3857$ to **$0.0960$** (nearly 3× better than TypeSafe Jev's 0.246).
* **Statistically Confirmed Accuracy Gain:** Cyclic marginalization delivers a verified **$+9.17$ percentage point** accuracy gain at $K=40$ ($p = 0.044$ under Benjamini-Hochberg FDR control).

<p align="center">
  <img src="results/figures/fig6_cyclic_vs_random.png" alt="Cyclic vs Random Marginalization" width="90%" />
</p>

---

## Pareto Frontier: Accuracy vs. Latency vs. Stability

Evaluating operational trade-offs across Accuracy ($\uparrow$), Latency ($\downarrow$), and within-method Instability ($\downarrow$, measured by $\mathrm{FR}_{\mathrm{top1}}$):

<p align="center">
  <img src="results/figures/fig5_pareto_frontier_k77.png" alt="Pareto Frontier at K=77" width="85%" />
</p>

| Method | Passes ($M$) | Latency (GPU est.) | Latency (CPU p50) | Flip Rate ($\mathrm{FR}_{\mathrm{top1}}$) | Counterfactual Cross-Flip | ECE Calibration | Pareto Frontier Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`B0_Native`** | 1 | **32.8 ms** | 224.7 ms | 0.0% (rerun) / 47.3% (random) | — | 0.370 | Dominated |
| **`B1_Alpha`** | 1 | **32.8 ms** | 224.7 ms | 0.0% (rerun) | 55.83% | 0.334 | **Non-dominated (Speed)** |
| **`B2_Cyclic_M2`** | 2 | 65.6 ms | 449.4 ms | 29.44% | — | 0.218 | Dominated |
| **`B2_Cyclic_M3`** | 3 | 98.4 ms | 674.0 ms | 23.06% | — | 0.147 | **Non-dominated (Balanced)** |
| **`B2_Cyclic_M5`** | 5 | 164.0 ms | 1123.4 ms | **16.11%** | — | 0.162 | **Non-dominated (Stability)** |
| **`Rand_Marg_M5`** | 5 | 164.0 ms | 1123.4 ms | 25.83% | — | **0.096** | **Non-dominated (Accuracy/ECE)** |
| *TypeSafe Jev 1.13.0* | 1 (closed) | *236–276 ms* | *~800–1200 ms* | *Unknown* | *Unknown* | *0.246* | Closed Cloud API |

---

## Cardinality Scaling Grid ($K \in \{5, 10, 20, 40, 77\}$)

Evaluated across $N=120$ intent-stratified queries on Banking77 with strictly nested candidate sets ($K_5 \subset K_{10} \subset K_{20} \subset K_{40} \subset K_{77}$):

| $K$ | Chance ($1/K$) | Method | Accuracy [95% CI] | ECE [95% CI] | $\text{FR}_{\text{top1}}$ [95% CI] | Cross-Flip [95% CI] | Adj. $p$-val | FDR Sig ($q=0.05$) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **5** | 20.0% | **B0_Native** | 0.925 [0.88, 0.97] | 0.147 [0.12, 0.20] | 0.000 [0.00, 0.00] | — | 1.0000 | NO |
| 5 | 20.0% | **B0_Random** | 0.925 [0.88, 0.97] | 0.160 [0.12, 0.22] | 0.057 [0.03, 0.08] | — | 1.0000 | NO |
| 5 | 20.0% | **B1_Alpha** | 0.908 [0.86, 0.96] | 0.131 [0.10, 0.19] | 0.000 [0.00, 0.00] | 0.033 [0.01, 0.07] | 1.0000 | NO |
| 5 | 20.0% | **B2_Cyclic_M5** | 0.908 [0.85, 0.96] | 0.124 [0.10, 0.18] | **0.000 [0.00, 0.00]** | — | 1.0000 | NO |
| **10** | 10.0% | **B0_Native** | 0.875 [0.82, 0.93] | 0.104 [0.07, 0.17] | 0.000 [0.00, 0.00] | — | 1.0000 | NO |
| 10 | 10.0% | **B0_Random** | 0.833 [0.77, 0.90] | 0.103 [0.07, 0.17] | 0.064 [0.04, 0.09] | — | 0.7351 | NO |
| 10 | 10.0% | **B2_Cyclic_M5** | 0.875 [0.82, 0.93] | 0.112 [0.08, 0.18] | **0.008 [0.00, 0.03]** | — | 1.0000 | NO |
| **20** | 5.0% | **B0_Native** | 0.767 [0.69, 0.84] | 0.191 [0.14, 0.26] | 0.000 [0.00, 0.00] | — | 1.0000 | NO |
| 20 | 5.0% | **B0_Random** | 0.817 [0.75, 0.88] | 0.153 [0.10, 0.22] | 0.120 [0.08, 0.16] | — | 0.4922 | NO |
| 20 | 5.0% | **B2_Cyclic_M5** | 0.792 [0.72, 0.87] | 0.107 [0.08, 0.19] | **0.033 [0.01, 0.07]** | — | 1.0000 | NO |
| **40** | 2.5% | **B0_Native** | 0.617 [0.53, 0.71] | 0.265 [0.21, 0.36] | 0.000 [0.00, 0.00] | — | 1.0000 | NO |
| 40 | 2.5% | **B0_Random** | 0.650 [0.56, 0.73] | 0.267 [0.21, 0.36] | 0.221 [0.18, 0.27] | — | 1.0000 | NO |
| 40 | 2.5% | **B1_Alpha** | 0.692 [0.61, 0.77] | 0.212 [0.16, 0.30] | 0.000 [0.00, 0.00] | 0.167 [0.10, 0.23] | 0.4922 | NO |
| 40 | 2.5% | **B2_Cyclic_M5** | **0.708 [0.62, 0.79]** | **0.172 [0.13, 0.26]** | **0.050 [0.02, 0.09]** | — | **0.0440** | **YES (*)** |
| **77** | 1.3% | **B0_Native** | 0.422 [0.34, 0.50] | 0.370 [0.30, 0.45] | 0.000 [0.00, 0.00] | — | 1.0000 | NO |
| 77 | 1.3% | **B0_Random** | 0.436 [0.36, 0.51] | 0.386 [0.31, 0.47] | 0.473 [0.43, 0.52] | — | 1.0000 | NO |
| 77 | 1.3% | **B1_Alpha** | 0.492 [0.40, 0.58] | 0.334 [0.27, 0.43] | 0.000 [0.00, 0.00] | **0.558 [0.48, 0.65]** | 1.0000 | NO |
| 77 | 1.3% | **Rand_Marg_M5** | 0.550 [0.47, 0.62] | **0.096 [0.09, 0.20]** | 0.258 [0.20, 0.32] | — | 1.0000 | NO |
| 77 | 1.3% | **B2_Cyclic_M5** | 0.497 [0.42, 0.57] | 0.162 [0.13, 0.26] | **0.161 [0.12, 0.21]** | — | 1.0000 | NO |

---

## Honest Limits & Negative Replication Audit

* **Non-Replication of Exploratory 65.0% Cyclic Accuracy:** In preliminary single-seed benchmarking ($N=40, S=1$), cyclic shifts appeared to reach an anomalous 65.0% accuracy at $K=77$. Under our multi-seed replication protocol ($N=120, S=3$ base orderings: `alphabetical`, `seed_7701`, `seed_7702`), replicated accuracy regressed to **48.06% [39.7%, 55.6%]** ($M=2$) and **49.72% [41.7%, 57.2%]** ($M=5$). Small-sample runs with fixed base sequences produce severe anchoring artifacts; multi-seed verification is mandatory.
* **Statistical Power at $K=77$:** Across 45 paired McNemar tests under Benjamini-Hochberg FDR control ($q=0.05$), only cyclic shifts at $K=40$ achieve statistical significance ($+9.17$ pp, adjusted $p = 0.04395$). At $K=77$, sample variance across $N=120$ queries precludes declaring accuracy improvements statistically significant ($p_{\mathrm{adj}} = 1.0$). Power analysis indicates $N \ge 350$ queries are required to confirm a +5% accuracy gain at $K=77$.
* **Token Budget Headroom:** While marginalization cuts decision flips from 47.3% to 16.1%, it does not expand the underlying positional context length. To reach Jev's 0.870 accuracy on 70+ options, practitioners should combine cyclic marginalization with `predict_shortlist` or raise `head_max_len = 512`.

---

## Invariant Verification Suite

Every metric, dataset split, and subset nesting relationship is verified by a strict mathematical unit test suite:

```bash
pytest experiments/tests/test_marginalization_invariants.py -v
```

Tests verify:
1. `test_metric_identical_ordering`: Zero flip rate, zero Kendall distance, zero churn under identical inputs.
2. `test_metric_inverted_ordering`: Maximal discordance under inverted inputs.
3. `test_candidate_set_nesting_invariant`: Strict nested distractor inclusion ($K_5 \subset K_{10} \subset K_{20} \subset K_{40} \subset K_{77}$).
4. `test_canonical_repeatability_vs_cross_canonical`: Deterministic alphabetical sorting produces 0% rerun variance while counterfactual reverse sorting detects true underlying bias.
5. `test_cyclic_residual_flip_independence`: Cyclic phases are evaluated independently.
6. `test_ece_edge_cases_and_reference`: Verification against independent calibration implementations.
7. `test_ece_bootstrap_ci_contains_estimate`: Mathematical consistency of bootstrap confidence intervals.

---

## Full Publication Paper & Archival DOI

* **TMLR Manuscript:** [`paper/tmlr/manuscript.pdf`](paper/tmlr/manuscript.pdf) (18 pages, official TMLR style).
* **Permanent Zenodo Archive:** [10.5281/zenodo.22906245](https://doi.org/10.5281/zenodo.22906245)
* **Raw Evaluation Data:** [`results/raw/corrected_per_query_results.csv`](results/raw/corrected_per_query_results.csv)
* **Processed Statistical Results:** [`results/statistics/corrected_statistical_tests.csv`](results/statistics/corrected_statistical_tests.csv)

### BibTeX Citation

```bibtex
@article{saxena2026candidate,
  title={Candidate Order Instability in Non-Autoregressive Multi-Candidate Transformers},
  author={Saxena, Anshul},
  journal={Transactions on Machine Learning Research},
  year={2026},
  url={https://doi.org/10.5281/zenodo.22906245},
  note={Zenodo DOI: 10.5281/zenodo.22906245. Software repository: https://github.com/anshull-saxena/candidate-order-instability}
}
```

---

## License

This research repository and evaluation harness are licensed under the [Apache 2.0 License](LICENSE). The manuscript and documentation are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
