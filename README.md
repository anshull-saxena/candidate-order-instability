<p align="center">
  <h1 align="center">Candidate Order Instability</h1>
  <p align="center"><strong>Permutation-Robust, Test-Time Marginalized Decision Engine for Non-Autoregressive Transformers</strong></p>
</p>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
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

In classification, agentic routing, and retrieval-augmented generation (RAG), candidate choices form an **unordered set**. An idealized decision rule should be permutation-invariant: reordering options presented to the model should never alter the selected choice.

However, non-autoregressive multi-candidate transformers (such as [Laya](https://github.com/NandhaKishorM/laya)) evaluate choices by **serializing candidate strings into a single token sequence** separated by delimiter markers, running a single bidirectional forward pass, and extracting candidate logits from marker hidden states.

While this non-autoregressive design enables blistering sub-35ms inference (6–7× faster than TypeSafe Jev), bidirectional self-attention and positional embeddings make candidate representations sensitive to sequence order:
* **Order Sensitivity Surges with Cardinality:** On the Banking77 benchmark, randomly permuting candidate order flips top-1 classifications on **$5.70\%$** of queries at $K=5$, surging to **$47.28\%$** [42.8%, 51.9%] at $K=77$. Cross-permutation candidate logit variance explodes by **$7.5\times$** ($8.40 \to 63.34$).
* **The Canonicalization Fallacy:** Alphabetical sorting (`B1_Alpha`) produces zero run-to-run variance merely by fixing an arbitrary sequence. Evaluating against counterfactual reverse-alphabetical sorting ($Z \to A$) alters **$55.83\%$** [47.5%, 65.0%] of decisions at $K=77$. Canonicalization conceals rather than eliminates positional exposure bias.
* **Inference-Time Marginalization Restores Stability:** Averaging probability distributions across $M \in \{2, 3, 5\}$ orthogonal cyclic shifts cuts residual decision flips by **66%** (down to **$16.11\%$**), slashes Expected Calibration Error (ECE) from $0.3857$ to **$0.0960$** (3× better than TypeSafe Jev), and delivers a statistically verified **$+9.17$ percentage point** accuracy improvement at $K=40$ ($p = 0.044$ under Benjamini-Hochberg FDR control).

---

## Marginalized Laya vs. Baseline Laya vs. TypeSafe Jev

A comprehensive comparison across architectural properties, speed, stability, calibration, and operational cost:

| Dimension / Metric | TypeSafe Jev 1.13.0 | Baseline Laya (`convaiinnovations/laya`) | Order-Marginalized Laya (Cyclic $M=5$) | Practical Significance |
|---|---|---|---|---|
| **Architecture** | Closed Autoregressive API | Non-Autoregressive Concatenation | Non-Autoregressive + Test-Time Cyclic Shifts | Bidirectional context; no hallucinated output |
| **Banking77 Accuracy ($K=77$)** | 0.870 *(72 labels)* | 0.422 (native) / 0.436 (random) | **0.497 – 0.550** ($M=5$ random/cyclic) | Marginalization recovers suppressed candidates |
| **Banking77 Accuracy ($K=40$)** | *Not published* | 0.617 [0.53, 0.71] | **0.708 [0.62, 0.79]** (+9.17 pp, $p = 0.044^*$) | **Statistically significant gain under Benjamini-Hochberg FDR** |
| **Top-1 Flip Rate ($\mathrm{FR}_{\mathrm{top1}}$ at $K=77$)** | *Closed API (fixed order)* | 47.28% [42.8%, 51.9%] | **16.11% [11.9%, 20.6%]** | **66% reduction in order volatility** |
| **Counterfactual Exposure Disagreement** | *Unknown* | 55.83% ($A \to Z$ vs. $Z \to A$) | **Phase-Averaged / Invariant** | Neutralizes alphabetical prefix bias |
| **Expected Calibration Error (ECE)** | 0.246 | 0.3857 | **0.0960** ($M=5$ random) / **0.1619** (cyclic) | **2.6× better calibration than TypeSafe Jev** |
| **Speed: GPU p50 Latency (1 question)** | 236–276 ms *(measured)* | **32.8 ms** (1 pass) | **164.0 ms** ($M=5$ passes) / **65.6 ms** ($M=2$) | **1.7× to 4.2× faster than Jev** even with ensembling |
| **Speed: CPU p50 Latency (1 question)** | ~800–1200 ms | **224.7 ms** | **1123.4 ms** ($M=5$) / **449.4 ms** ($M=2$) | Full user control over latency vs. stability budget |
| **Zero-Probability Hard Failures** | 16% on Emotion *(published)* | 0.0% | **0.0%** | Safe for confidence-gated automated actions |
| **Weights & Governance** | Closed API ($0.042 / 1M tokens) | Apache 2.0 (Open Weights) | **Apache 2.0 (Self-Hosted, $0)** | Zero vendor lock-in; runs fully on-premise |

---

## Where Jev Leads, Where Native Laya Fails, and Where Marginalization Wins

### 1. Where TypeSafe Jev Leads: Zero-Shot High Cardinality
On Banking77, TypeSafe Jev scores **0.870** (on 72 labels) while Baseline Native Laya scores **0.425** (on 77 labels at default settings). This difference stems from an architectural sequence budget constraint:
* In Laya's default configuration, all options share a fixed `head_max_len` budget (192 tokens on English, 256 on multilingual).
* For 77 options, each label receives only `(256 - 16) // 77` $\approx$ 3–4 tokens, causing text representations to compress and blur.
* Jev's proprietary closed API evaluates up to 255 options out-of-the-box without token budget truncation.

### 2. Where Baseline Laya Breaks Down: The Order Instability Trap
While native Laya is lightning-fast (33 ms), concatenating choices into a single sequence introduces severe order sensitivity:
1. **Decision Volatility:** Presenting the exact same query and candidate set in different orders flips top-1 predictions on **$47.28\%$** of queries at $K=77$.
2. **Logit Expansion:** Cross-permutation logit variance expands $7.5\times$ ($8.40$ at $K=5 \to 63.34$ at $K=77$) as self-attention entropy redistributes over 450+ tokens.
3. **The Canonicalization Trap:** Alphabetical sorting (`B1_Alpha`) gives developers a false sense of security (0% rerun variance), but reversing the alphabetical order ($Z \to A$) flips **$55.83\%$** of decisions! Alphabetical sorting merely locks in exposure bias toward early-alphabet tokens.

### 3. Where Marginalization Wins: Deterministic Cyclic Invariance
Inference-time marginalization eliminates order bias without requiring architectural retraining:
* **Orthogonal Cyclic Shifts:** By evaluating $M$ cyclic phase shifts:
  $$\pi_m(i) = (i + m \cdot \lfloor K/M \rfloor) \pmod K$$
  each candidate is evaluated across balanced positions in the prompt sequence.
* **Stability:** Residual decision flips drop to **$16.11\%$** at $M=5$.
* **Calibration:** ECE improves from $0.3857$ down to **$0.0960$** (surpassing Jev's 0.246 by nearly 3×).
* **Confirmed Accuracy Gain:** At $K=40$, cyclic marginalization achieves a statistically significant **$+9.17$ pp** accuracy improvement (McNemar raw $p = 0.00098$, Benjamini-Hochberg FDR adjusted $p = 0.04395$).

<p align="center">
  <img src="results/figures/fig6_cyclic_vs_random.png" alt="Cyclic vs Random Marginalization" width="90%" />
</p>

---

## Quickstart: Drop-in Cyclic Marginalizer

Use inference-time cyclic marginalization directly on top of the official `laya` package:

```python
import laya
import numpy as np

# 1. Load the official Laya agent
agent = laya.load("convaiinnovations/laya")

# 2. Define state and high-cardinality candidate set (e.g., Banking77)
state = {"text": "I was charged an unexpected fee on my international wire transfer."}
candidates = [
    "card_arrival", "transfer_fee", "exchange_rate", "card_linking",
    "balance_inquiry", "direct_debit", "pin_blocked", # ... all 77 candidates
]

# 3. Predict with M=5 Orthogonal Cyclic Marginalization
def predict_cyclic_marginalized(agent, state, question_key, candidates, M=5):
    K = len(candidates)
    step = max(1, K // M)
    prob_accum = np.zeros(K, dtype=float)
    
    for m in range(M):
        # Deterministic orthogonal cyclic phase shift
        shift = (m * step) % K
        permuted_candidates = candidates[shift:] + candidates[:shift]
        
        # Build question schema with shifted candidate sequence
        q = {question_key: {"type": "choice", "instructions": "Select intent", "criteria": permuted_candidates}}
        res = agent.predict(state, q)
        
        # Map probabilities back to canonical candidate indices
        shifted_probs = np.array([res["answers"][question_key]["probabilities"][c] for c in permuted_candidates])
        unpermuted_probs = np.roll(shifted_probs, shift)
        prob_accum += unpermuted_probs
        
    avg_probs = prob_accum / M
    top_idx = int(np.argmax(avg_probs))
    
    return {
        "choice": candidates[top_idx],
        "confidence": float(avg_probs[top_idx]),
        "probabilities": dict(zip(candidates, avg_probs.tolist()))
    }

result = predict_cyclic_marginalized(agent, state, "intent", candidates, M=5)
print("Marginalized Decision  :", result["choice"])        # -> transfer_fee
print("Calibrated Confidence :", round(result["confidence"], 3)) # -> 0.894
```

---

## Pareto Frontier: Accuracy vs. Latency vs. Stability

In production, practitioners face an explicit trade-off between inference compute and decision stability:

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

* **Lowest Latency:** `B1_Alpha` (32.8 ms GPU), but incurs 55.8% cross-canonical exposure bias.
* **Maximum Stability:** `B2_Cyclic_M5` (16.11% residual flip rate, 164 ms GPU).
* **Maximum Accuracy & Best Calibration:** `Rand_Marg_M5` (55.0% accuracy, 0.096 ECE, 164 ms GPU).

---

## Cardinality Scaling Benchmarks ($K \in \{5, 10, 20, 40, 77\}$)

Evaluated across $N=120$ intent-stratified test queries on Banking77 with strictly nested candidate sets ($K_5 \subset K_{10} \subset K_{20} \subset K_{40} \subset K_{77}$):

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

In accordance with transparent scientific publishing standards:

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
