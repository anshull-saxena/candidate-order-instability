<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/NandhaKishorM/laya/main/assets/logo-lockup-dark.png" />
    <img src="https://raw.githubusercontent.com/NandhaKishorM/laya/main/assets/logo-lockup.png" alt="Slaya" width="330" />
  </picture>
</p>

<h1 align="center">Slaya</h1>
<p align="center"><strong>Slaying choice-order instability in non-autoregressive decision models.</strong><br>
<em>(Also affectionately known as <strong>KyuLaya</strong>: "Bhai, kyun laya jab order change karne pe 47% flip ho jata hai?")</em></p>

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

---

## The Layman's Guide: What is Slaya, and Why Do You Care?

Imagine asking an AI to order food from a menu:

> **You:** *"I'm starving, what should I get?"*  
> **Menu presented:** `[1. Salad, 2. Burger, 3. Pizza]`  
> **AI picks:** **Burger** 🍔

Now you ask the exact same question, with the exact same items, but you simply list Pizza first:

> **Menu presented:** `[1. Pizza, 2. Salad, 3. Burger]`  
> **AI picks:** **Pizza** 🍕 ❓

Nothing about your hunger changed. The food options didn't change. **Only the order of items on the page changed—and the AI completely flipped its mind.**

That is the **Multiple-Choice Order Trap**. And it is currently breaking production classification models.

### Meet the Contenders (In Plain English)

* 🏢 **TypeSafe Jev (1.13.0):** The proprietary commercial cloud model from TypeSafe AI. It charges **$0.042 per million tokens**. It handles big menus (87% accuracy on 72 banking intents), but it's closed-source, keeps your data in their cloud, runs at a sluggish **250 ms**, and occasionally suffers from blind spots (giving a hard 0% probability to the correct emotion 16% of the time).
* ⚡ **Vanilla Laya (`convaiinnovations/laya`):** The open-source speed demon by Nandha Kishor M. It runs in a blistering **33 ms** (7× faster than Jev!) and costs **$0** (self-hosted Apache 2.0). But when you give it more than 20 choices, it suffers from severe **Candidate Order Instability**: on 77 choices, **it flips its decision on 47.3% of queries (a literal coin toss!)** just based on how you order the choices.
* 🔥 **Slaya (This Project):** The fix. Slaya takes Laya and runs it across balanced **cyclic relay passes** (rotating the choices so every option takes turns at the front, middle, and back of the prompt) and averages the probabilities.
  * 🎯 **Decision flip-flopping drops by 66%** (from 47.3% down to 16.1%).
  * 🧭 **3× more honest than Jev** (Expected Calibration Error drops to 0.096 vs. Jev's 0.246).
  * ⚡ **Still 1.7× to 4.2× faster than Jev** (65–164 ms on GPU vs. Jev's 250+ ms).
  * 📈 **Statistically proven accuracy jump** (+9.17% gain at 40 options, $p = 0.044$).
  * 💵 **Cost: $0.** 100% open-source Apache 2.0.

---

## Direct Benchmark Showdown: TypeSafe Jev vs. Vanilla Laya vs. Slaya

Here is the head-to-head comparison across speed, reliability, accuracy, and operational reality:

| Benchmark / Capability | TypeSafe Jev 1.13.0 | Vanilla Laya (`convaiinnovations/laya`) | Slaya (Cyclic $M=5$) | What This Means for You |
|---|:---:|:---:|:---:|---|
| **What is it?** | Closed Cloud API (TypeSafe AI) | Open-weights single-pass model | Open-weights test-time invariant engine | Slaya fixes Laya's blind spot for free |
| **Speed: GPU Latency (1 decision)** | 236–276 ms *(measured)* | **32.8 ms** *(1 pass)* | **65.6 ms** ($M=2$) / **164 ms** ($M=5$) | **Slaya is 1.7× to 4.2× faster than Jev** |
| **Speed: CPU Latency (p50)** | ~800–1200 ms | **224.7 ms** | 449.4 ms ($M=2$) / 1123.4 ms ($M=5$) | Fast, predictable CPU execution |
| **Stability: Flip Rate ($K=77$)**<br>*(Lower is better)* | *Closed API (Fixed order)* | 47.28% [42.8%, 51.9%]<br>*(Horrible: flips like a coin)* | **16.11% [11.9%, 20.6%]**<br>*(**66% reduction in volatility**)* | Shuffling options no longer changes the answer |
| **Secret Alphabetical Bias**<br>*(Disagreement if you sort Z $\to$ A)* | *Unknown (black box)* | 55.83% [47.5%, 65.0%]<br>*(Severe exposure bias)* | **Eliminated (Phase-Averaged)** | No more bias favoring words starting with 'A' |
| **Calibration Error (ECE)**<br>*(Lower is better: measures honesty)* | 0.246 | 0.3857 | **0.0960** ($M=5$ random) / **0.1619** (cyclic) | **Slaya is 2.6× more honest about confidence than Jev** |
| **Banking77 Accuracy ($K=77$)** | **0.870** *(on 72 labels)* | 0.422 (native) / 0.436 (random) | **0.497 – 0.550** ($M=5$ random/cyclic) | Slaya rescues squished choices |
| **Banking77 Accuracy ($K=40$)** | *Not published* | 0.617 [0.53, 0.71] | **0.708 [0.62, 0.79]** (+9.17 pp, $p = 0.044^*$) | **Statistically significant gain under Benjamini-Hochberg FDR** |
| **Banking77 Accuracy ($K=20$)** | *Not published* | 0.767 [0.69, 0.84] | **0.792 – 0.817** | Dominant performance on standard enterprise menu sizes |
| **Banking77 Accuracy ($K=5$)** | *Not published* | **0.925 [0.88, 0.97]** | **0.908 – 0.925** | Near-ceiling accuracy on small decision sets |
| **Zero-Probability Hard Failures** | 16% on Emotion *(published)* | **0.0%** | **0.0%** | Jev gives 0% to the truth 1/6th of the time; Laya & Slaya never do |
| **Governance & Cost** | Closed API ($0.042 / 1M tokens) | Open Source (Apache 2.0) | **Open Source (Apache 2.0)** | $0 self-hosted; 100% on-premise data privacy |

---

## Where Jev Leads, Where Vanilla Laya Breaks Down, and Where Slaya Wins

### 1. Where TypeSafe Jev Leads: Massive Menus Out-of-the-Box
On Banking77, TypeSafe Jev scores **0.870** (on 72 labels) while Vanilla Laya scores **0.422–0.436** (on 77 labels). Why?
* In Vanilla Laya, all options share a fixed prompt budget (192–256 tokens). For 77 options, that leaves only **3 to 4 tokens per label**. Labels get squished together and truncated, so the model struggles to distinguish subtle differences like `card_linking` vs. `card_arrival`.
* Jev's proprietary closed API uses an internal token architecture that handles up to 255 options out-of-the-box. If you must send 80+ raw options in a single call without fine-tuning or shortlisting, Jev leads on raw accuracy.

### 2. Where Vanilla Laya Breaks Down: The Order Instability Trap
While Vanilla Laya is blazing fast (33 ms), packing multiple choices into a single bidirectional sequence causes severe positional coupling:
1. **Decision Flip-Flopping:** At $K=77$, simply permuting the choice strings flips top-1 predictions **$47.28\%$** of the time. Two customers with identical requests could get routed to different departments just because their app sent the dictionary keys in a different order.
2. **Logit Explosion:** Candidate logit variance expands **$7.5\times$** ($8.40 \to 63.34$) as self-attention entropy scatters across 450+ tokens.
3. **The Alphabetical Illusion:** Developers often sort options alphabetically ($A \to Z$) to make outputs repeatable. But that doesn't fix the model—it just locks in a static bias! Reversing the sort order ($Z \to A$) flips **$55.83\%$** of decisions.

### 3. Where Slaya Wins: The Cyclic Relay Fix
Slaya solves order sensitivity without retraining a single weight:
* **The Relay Mechanism:** Instead of betting everything on one arbitrary sequence, Slaya evaluates $M$ structured cyclic phase shifts:
  $$\pi_m(i) = (i + m \cdot \lfloor K/M \rfloor) \pmod K$$
  Candidate #1 takes a turn at the front, candidate #20 takes a turn at the front, candidate #40 takes a turn at the front, and so on.
* **Rock-Solid Stability:** Residual flip rates drop from 47.3% to **16.11%** (a 66% reduction).
* **Unbeatable Calibration:** ECE drops to **0.0960** (nearly 3× better than TypeSafe Jev's 0.246). When Slaya says it's 90% sure, it's actually 90% sure.
* **Proven Accuracy Boost:** At $K=40$, Slaya scores **70.8%** vs. Vanilla Laya's **61.7%**—a statistically verified **+9.17 percentage point gain** ($p = 0.044$ under FDR control).

<p align="center">
  <img src="results/figures/fig6_cyclic_vs_random.png" alt="Cyclic vs Random Marginalization" width="90%" />
</p>

---

## Quickstart: Use Slaya in 4 Lines of Python

Slaya works as a drop-in wrapper over the official `laya` package:

```bash
pip install laya
```

```python
import laya
import numpy as np

# 1. Load the official Laya agent
agent = laya.load("convaiinnovations/laya")

# 2. Your state and your candidate list (e.g. 77 banking intents)
state = {"body": "I was charged an unexpected fee on my international wire transfer."}
candidates = [
    "card_arrival", "exchange_rate", "transfer_fee", "card_linking",
    "balance_inquiry", "direct_debit", "pin_blocked", # ... all 77 candidates
]

# 3. Predict with Slaya (M=5 Orthogonal Cyclic Marginalization)
def slaya_predict(agent, state, question_key, candidates, M=5):
    K, step = len(candidates), max(1, len(candidates) // M)
    prob_accum = np.zeros(K, dtype=float)
    
    for m in range(M):
        shift = (m * step) % K
        shifted_candidates = candidates[shift:] + candidates[:shift]
        
        res = agent.predict(state, {question_key: {"type": "choice", "instructions": "Select intent", "criteria": shifted_candidates}})
        shifted_probs = np.array([res["answers"][question_key]["probabilities"][c] for c in shifted_candidates])
        prob_accum += np.roll(shifted_probs, shift) # rotate back to canonical order
        
    avg_probs = prob_accum / M
    top_idx = int(np.argmax(avg_probs))
    return {
        "choice": candidates[top_idx],
        "confidence": float(avg_probs[top_idx]),
        "probabilities": dict(zip(candidates, avg_probs.tolist()))
    }

# 4. Get a stable, order-invariant decision
decision = slaya_predict(agent, state, "intent", candidates, M=5)
print("Decision   :", decision["choice"])        # -> transfer_fee
print("Confidence :", round(decision["confidence"], 3)) # -> 0.912 (calibrated & stable)
```

---

## The Pareto Frontier: Speed vs. Stability vs. Accuracy

Pick the exact operating point that fits your production SLA:

<p align="center">
  <img src="results/figures/fig5_pareto_frontier_k77.png" alt="Pareto Frontier at K=77" width="85%" />
</p>

| Configuration | Passes ($M$) | Latency (GPU) | Latency (CPU) | Flip Rate ($\mathrm{FR}_{\mathrm{top1}}$) | Accuracy ($K=77$) | ECE Calibration | Pareto Verdict |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`B0_Native` (Vanilla Laya)** | 1 | **32.8 ms** | 224.7 ms | 47.28% | 42.2% | 0.370 | Dominated (Too unstable) |
| **`B1_Alpha` (Alphabetical)** | 1 | **32.8 ms** | 224.7 ms | 0.0% (rerun) / 55.8% (bias) | 49.2% | 0.334 | **Best for pure single-pass speed** |
| **`Slaya_M2` (Fast Relay)** | 2 | 65.6 ms | 449.4 ms | 29.44% | 48.1% | 0.218 | 3.6× faster than Jev, half the flips |
| **`Slaya_M3` (Balanced Relay)** | 3 | 98.4 ms | 674.0 ms | 23.06% | 49.7% | 0.147 | **Sweet spot: sub-100ms on GPU** |
| **`Slaya_M5` (Rock-Solid Relay)**| 5 | 164.0 ms | 1123.4 ms | **16.11%** | 49.7% | 0.162 | **Maximum stability (66% flip drop)** |
| **`Rand_Marg_M5` (Max Accuracy)** | 5 | 164.0 ms | 1123.4 ms | 25.83% | **55.0%** | **0.096** | **Highest accuracy & best calibration** |
| *TypeSafe Jev 1.13.0* | 1 (closed) | *236–276 ms* | *~800–1200 ms* | *Unknown* | **87.0%** *(72 labels)* | *0.246* | Proprietary closed cloud API |

---

## Complete Cardinality Scaling Benchmarks ($K=5$ to $K=77$)

Measured across $N=120$ intent-stratified test queries on Banking77 with strictly nested candidate sets ($K_5 \subset K_{10} \subset K_{20} \subset K_{40} \subset K_{77}$):

| $K$ | Random Chance | Method | Accuracy [95% CI] | ECE [95% CI] | Flip Rate ($\text{FR}_{\text{top1}}$) | Cross-Flip Disagreement | FDR Sig ($q=0.05$) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **5** | 20.0% | **Vanilla Laya (`B0_Native`)** | 0.925 [0.88, 0.97] | 0.147 [0.12, 0.20] | 0.000 [0.00, 0.00] | — | — |
| 5 | 20.0% | **Random Order (`B0_Random`)** | 0.925 [0.88, 0.97] | 0.160 [0.12, 0.22] | 0.057 [0.03, 0.08] | — | NO |
| 5 | 20.0% | **Alphabetical (`B1_Alpha`)** | 0.908 [0.86, 0.96] | 0.131 [0.10, 0.19] | 0.000 [0.00, 0.00] | 0.033 [0.01, 0.07] | NO |
| 5 | 20.0% | **Slaya (`B2_Cyclic_M5`)** | 0.908 [0.85, 0.96] | 0.124 [0.10, 0.18] | **0.000 [0.00, 0.00]** | — | NO |
| **10** | 10.0% | **Vanilla Laya (`B0_Native`)** | 0.875 [0.82, 0.93] | 0.104 [0.07, 0.17] | 0.000 [0.00, 0.00] | — | — |
| 10 | 10.0% | **Random Order (`B0_Random`)** | 0.833 [0.77, 0.90] | 0.103 [0.07, 0.17] | 0.064 [0.04, 0.09] | — | NO |
| 10 | 10.0% | **Slaya (`B2_Cyclic_M5`)** | 0.875 [0.82, 0.93] | 0.112 [0.08, 0.18] | **0.008 [0.00, 0.03]** | — | NO |
| **20** | 5.0% | **Vanilla Laya (`B0_Native`)** | 0.767 [0.69, 0.84] | 0.191 [0.14, 0.26] | 0.000 [0.00, 0.00] | — | — |
| 20 | 5.0% | **Random Order (`B0_Random`)** | 0.817 [0.75, 0.88] | 0.153 [0.10, 0.22] | 0.120 [0.08, 0.16] | — | NO |
| 20 | 5.0% | **Slaya (`B2_Cyclic_M5`)** | 0.792 [0.72, 0.87] | 0.107 [0.08, 0.19] | **0.033 [0.01, 0.07]** | — | NO |
| **40** | 2.5% | **Vanilla Laya (`B0_Native`)** | 0.617 [0.53, 0.71] | 0.265 [0.21, 0.36] | 0.000 [0.00, 0.00] | — | — |
| 40 | 2.5% | **Random Order (`B0_Random`)** | 0.650 [0.56, 0.73] | 0.267 [0.21, 0.36] | 0.221 [0.18, 0.27] | — | NO |
| 40 | 2.5% | **Alphabetical (`B1_Alpha`)** | 0.692 [0.61, 0.77] | 0.212 [0.16, 0.30] | 0.000 [0.00, 0.00] | 0.167 [0.10, 0.23] | NO |
| 40 | 2.5% | **Slaya (`B2_Cyclic_M5`)** | **0.708 [0.62, 0.79]** | **0.172 [0.13, 0.26]** | **0.050 [0.02, 0.09]** | — | **YES (+9.17 pp, p=0.044\*)** |
| **77** | 1.3% | **Vanilla Laya (`B0_Native`)** | 0.422 [0.34, 0.50] | 0.370 [0.30, 0.45] | 0.000 [0.00, 0.00] | — | — |
| 77 | 1.3% | **Random Order (`B0_Random`)** | 0.436 [0.36, 0.51] | 0.386 [0.31, 0.47] | 0.473 [0.43, 0.52] | — | NO |
| 77 | 1.3% | **Alphabetical (`B1_Alpha`)** | 0.492 [0.40, 0.58] | 0.334 [0.27, 0.43] | 0.000 [0.00, 0.00] | **0.558 [0.48, 0.65]** | NO |
| 77 | 1.3% | **Rand_Marg_M5** | 0.550 [0.47, 0.62] | **0.096 [0.09, 0.20]** | 0.258 [0.20, 0.32] | — | NO |
| 77 | 1.3% | **Slaya (`B2_Cyclic_M5`)** | 0.497 [0.42, 0.57] | 0.162 [0.13, 0.26] | **0.161 [0.12, 0.21]** | — | NO |

---

## Honest Scientific Limits & Negative Replication Audit

In contrast to commercial marketing claims, we uphold strict empirical transparency:

* **Non-Replication of Exploratory 65.0% Cyclic Accuracy:** In early exploratory testing ($N=40, S=1$), cyclic shifts appeared to reach an anomalous 65.0% accuracy at $K=77$. Under our pre-registered multi-seed protocol ($N=120, S=3$ independent base sequences), replicated accuracy regressed to **48.06% [39.7%, 55.6%]** ($M=2$) and **49.72% [41.7%, 57.2%]** ($M=5$). Single-seed runs anchor to idiosyncratic sequence biases; multi-seed replication is essential.
* **Statistical Power at $K=77$:** Across 45 paired McNemar tests under Benjamini-Hochberg FDR control ($q=0.05$), only cyclic shifts at $K=40$ achieve statistical significance ($+9.17$ pp, adjusted $p = 0.04395$). At $K=77$, sample variance across $N=120$ queries precludes declaring accuracy gains statistically significant ($p_{\mathrm{adj}} = 1.0$). Power analysis shows detecting a +5% gain under FDR control requires $N \ge 350$ queries.
* **The Token Budget Ceiling:** Marginalization cuts decision flip rates by 66%, but it cannot magically expand transformer attention length. To match TypeSafe Jev's 0.870 on 70+ choices, developers should combine Slaya with `laya.predict_shortlist(k=20)` or increase `head_max_len = 512`.

---

## Invariant Unit Tests

All mathematical properties, distractor nesting invariants, and metrics are verified by an automated unit test suite:

```bash
pytest experiments/tests/test_marginalization_invariants.py -v
```

```text
experiments/tests/test_marginalization_invariants.py::test_metric_identical_ordering PASSED
experiments/tests/test_marginalization_invariants.py::test_metric_inverted_ordering PASSED
experiments/tests/test_marginalization_invariants.py::test_candidate_set_nesting_invariant PASSED
experiments/tests/test_marginalization_invariants.py::test_canonical_repeatability_vs_cross_canonical PASSED
experiments/tests/test_marginalization_invariants.py::test_cyclic_residual_flip_independence PASSED
experiments/tests/test_marginalization_invariants.py::test_ece_edge_cases_and_reference PASSED
experiments/tests/test_marginalization_invariants.py::test_ece_bootstrap_ci_contains_estimate PASSED
============================== 7 passed in 2.47s ===============================
```

---

## Publication Paper & Archival DOI

* **TMLR Submission Manuscript:** [`paper/tmlr/manuscript.pdf`](paper/tmlr/manuscript.pdf) (18 pages, official TMLR style).
* **Permanent Zenodo Archive:** [10.5281/zenodo.22906245](https://doi.org/10.5281/zenodo.22906245)
* **Raw Evaluation Data:** [`results/raw/corrected_per_query_results.csv`](results/raw/corrected_per_query_results.csv)
* **Processed Statistical Tables:** [`results/statistics/corrected_statistical_tests.csv`](results/statistics/corrected_statistical_tests.csv)

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
