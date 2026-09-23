# Reviewer #2 Technical Audit & Vulnerability Assessment

**Paper Title:** Candidate Order Instability in Non-Autoregressive Multi-Candidate Transformers  
**Author:** Anshul Saxena  
**Auditor Persona:** Hostile, Highly Rigorous Senior Reviewer (NeurIPS / ICML / ACL Empirical ML Track)  
**Date:** September 2026  
**Artifact Baseline:** Frozen Release `v1.0.0` (commit `266bbb902871dc9974cbf516adea013e3c4eb084`, Zenodo DOI: `10.5281/zenodo.22906245`)

---

## Executive Summary

This manuscript addresses an important operational failure mode in non-autoregressive multi-candidate transformers: candidate serialization order acting as a hidden nuisance variable. The experimental design is unusually rigorous for an empirical study (strictly nested distractor subsets, query-clustered bootstrap CIs, family-wise Benjamini-Hochberg FDR control across 45 paired tests, and explicit documentation of a non-replicated exploratory finding). 

However, prior to final archival submission, the paper must withstand hostile scrutiny across ten critical axes. Below is the point-by-point adversarial audit, categorized by severity:
- **Blocker:** Fatal scientific or inferential flaw requiring structural correction.
- **Major:** Critical statistical, contextual, or framing vulnerability that would trigger an immediate rejection recommendation if unaddressed.
- **Minor:** Ambiguity, potential overclaim, or presentation deficit requiring tightening.
- **Cosmetic:** Polish, typography, or styling improvement.

---

## Section-by-Section Adversarial Audit

### A. Novelty & Prior Art Distinction

#### Finding A.1: Risk of Conflating with Autoregressive Prompt Order Sensitivity
- **Reviewer Critique:** *"We already know transformers are sensitive to prompt order. Lu et al. (ACL 2022) showed demonstration order matters; Zhao et al. (ICML 2021) showed calibration bias; Pezeshkpour & Hruschka (NAACL 2024) showed option order sensitivity in multiple choice. Why does candidate ordering in a marker-concatenation bidirectional encoder warrant an independent paper?"*
- **Severity:** Major
- **Audit Assessment:** The critique has teeth if the paper blurs the architectural distinction between autoregressive generative LLMs and non-autoregressive encoder-based discriminators. In few-shot prompting, order sensitivity arises from in-context demonstration sequences or next-token option letters (`A`, `B`, `C`). In the evaluated architecture (`convaiinnovations/laya`), all $K$ candidates are serialized into a *single bidirectional prompt* separated by structural markers, and candidate logits are read simultaneously from linear projections of marker hidden states.
- **Concrete Fix Applied:** 
  1. Section 3.3 explicitly constructs a 4-part taxonomy distinguishing demonstration ordering, token syntax, multiple-choice option letters, and non-autoregressive marker concatenation.
  2. Section 3.6 defines the exact unaddressed research gap: no prior study analyzed simultaneous marker-concatenated scoring, cardinality scaling $K \in [5, 77]$, or counterfactual canonical reversal.

#### Finding A.2: Relationship to Permutation-Invariant Set Architectures
- **Reviewer Critique:** *"If you want permutation invariance over candidates, Deep Sets (Zaheer et al., 2017) and Set Transformer (Lee et al., 2019) solved this years ago. Concatenating candidates into a sequence is just bad engineering."*
- **Severity:** Major
- **Audit Assessment:** Real-world agentic routing pipelines adopt marker concatenation because it allows immediate zero-shot candidate transfer leveraging massive pretrained bidirectional language models (RoBERTa), whereas Deep Sets requires training from scratch or separate instance encoding without cross-candidate attention.
- **Concrete Fix Applied:** Section 3.1 and 3.4 explicitly contrast the computational trade-offs: bi-encoders ($O(1)$, no cross-attention), cross-encoders ($O(K)$ passes, prohibitive latency), Set Transformers (require specialized pretraining), and marker concatenation ($O(1)$ pass with cross-attention, but introduces order coupling). The paper does not defend concatenation as theoretically optimal, but examines it as an empirical reality in production NLP.

---

### B. Experimental Validity & Protocol

#### Finding B.1: Cardinality vs. Distractor Composition Confounding
- **Reviewer Critique:** *"When you increase candidate cardinality from $K=5$ to $K=77$, how do you know instability increases because of cardinality rather than because you added more difficult or confusing distractor classes?"*
- **Severity:** Blocker (if unaddressed)
- **Audit Assessment:** This is the single most common flaw in cardinality scaling studies. If distractors are sampled randomly at each $K$, $K=40$ might happen to sample semantic near-duplicates that $K=5$ missed.
- **Verification:** The benchmark implements a **strictly nested candidate protocol**:
  $$\mathcal{C}_{q, 5} \subset \mathcal{C}_{q, 10} \subset \mathcal{C}_{q, 20} \subset \mathcal{C}_{q, 40} \subset \mathcal{C}_{q, 77}$$
  For every query $q$, a fixed deterministic master sequence of 76 distractors $D_q$ is generated from query seed $42 + q \times 1000$. Every candidate present in $K=5$ is guaranteed to be present in $K=10, 20, 40, 77$.
- **Concrete Fix Applied:** Section 5.2 explicitly highlights this nested subset property with programmatic SHA-256 validation, proving cardinality is isolated from sampling variance.

#### Finding B.2: Independent Statistical Unit
- **Reviewer Critique:** *"You evaluate 10 permutations per query across 120 queries, claiming 1,200 samples. Are you treating the permutation as the independent unit?"*
- **Severity:** Blocker (if violated)
- **Audit Assessment:** Permutations of the same query share identical query semantics and distractor sets; treating $(q, \pi)$ pairs as i.i.d. would catastrophically inflate degrees of freedom and induce false significance.
- **Verification:** The independent statistical sampling unit is strictly the **query ID** ($N=120$). Within-query permutation flip rate ($\mathrm{FR}_{\mathrm{top1}}(q)$) is computed as an internal degree-2 $U$-statistic across $\binom{10}{2}=45$ permutation pairs, collapsing to a single scalar per query. All bootstrap resamples cluster at the query level ($B=1,000$).
- **Concrete Fix Applied:** Section 6.1 and 6.2 state this mathematical definition with complete transparency.

#### Finding B.3: Cyclic Phase Shift Construction
- **Reviewer Critique:** *"Why evaluate cyclic shifts? Are cyclic shifts an arbitrary trick, and why should they work?"*
- **Severity:** Minor
- **Audit Assessment:** Cyclic shifts $s_m = (m \cdot \lfloor K/M \rfloor) \pmod K$ systematically maximize the minimum positional distance each candidate travels across $M$ passes. In contrast to random permutations that may inadvertently leave some candidates in adjacent or identical positions, cyclic phase shifts guarantee deterministic orthogonal dispersion.
- **Concrete Fix Applied:** Section 5.3 and 7.4 explain the phase-dispersion rationale while rigorously identifying cyclic shifts as a deterministic heuristic rather than a universal theoretical panacea.

---

### C. Statistical Validity & Multiple Testing

#### Finding C.1: False Discovery Rate Control Family
- **Reviewer Critique:** *"You ran many methods across 5 cardinalities. Did you cherry-pick the single significant result?"*
- **Severity:** Blocker (if uncorrected)
- **Audit Assessment:** Testing 9 interventions against `B0_Native` across 5 cardinalities yields $5 \times 9 = 45$ hypothesis tests. At $\alpha=0.05$, testing 45 null hypotheses without correction expects $\approx 2.25$ false positives by chance alone.
- **Verification:** The benchmark pre-specified the entire family of 45 hypotheses and applied Benjamini-Hochberg (BH) False Discovery Rate control at $q=0.05$. Under this control:
  - Raw $p$-value for `B2_Cyclic_M5` at $K=40$: $p = 0.0009765625$
  - BH critical threshold for rank 1: $1/45 \times 0.05 = 0.001111$
  - Since $0.0009766 \le 0.001111$, the adjusted $p$-value is $0.04395$.
  - Exactly **1 of 45 comparisons** survives FDR control.
- **Concrete Fix Applied:** Section 8 and Table 3 report the complete hypothesis table, explicitly confirming that only the $K=40$ cyclic comparison achieves formal significance, while all $K=77$ accuracy gains fail FDR control.

#### Finding C.2: Overlapping Confidence Intervals vs. Paired Testing
- **Reviewer Critique:** *"In Table 1, at $K=40$, B0_Native CI is [0.53, 0.71] and B2_Cyclic_M5 CI is [0.62, 0.79]. These CIs overlap substantially! How can you claim statistical significance?"*
- **Severity:** Major
- **Audit Assessment:** Marginal confidence intervals reflect total query-level variance. Because interventions are evaluated on the exact same queries in a paired design, query difficulty is a shared factor. Paired McNemar testing isolates the discordant pairs (11 gains, 0 losses, exact two-sided $p = 0.0009766$).
- **Concrete Fix Applied:** Section 6.2 and Section 8 explain that marginal CIs assess population mean uncertainty, while exact McNemar tests assess within-query discordant shifts.

---

### D. Accuracy Interpretation at $K=77$

#### Finding D.1: Apparent $+12.78\%$ Accuracy Gain at $K=77$
- **Reviewer Critique:** *"At $K=77$, Rand_Marg_M5 achieves $55.0\%$ accuracy versus $42.22\%$ for B0_Native (+12.78 percentage points). That looks like a massive improvement. Why is it not statistically significant?"*
- **Severity:** Major
- **Audit Assessment:** With $N=120$ binary outcomes, 27 discordant gains and 11 discordant losses yield an unadjusted two-sided McNemar $p = 0.0145$. While significant at an uncorrected $\alpha=0.05$, when adjusted across 45 family tests under Benjamini-Hochberg, its adjusted $p$-value is $1.0000$. Calling this "statistically significant" would be scientific malpractice.
- **Concrete Fix Applied:** Section 7.3, Section 8, and Section 11 explicitly report that this accuracy improvement is descriptive and preliminary, constrained by the statistical power of $N=120$.

---

### E. Calibration Interpretation

#### Finding E.1: Claiming Calibration "Significance"
- **Reviewer Critique:** *"Did you run a formal hypothesis test proving ECE reduction is statistically significant?"*
- **Severity:** Major
- **Audit Assessment:** ECE is a binned metric computed over grouped predictions, making paired hypothesis testing non-trivial. The study evaluates ECE via 1,000-sample clustered bootstrap distributions.
- **Concrete Fix Applied:** The manuscript strictly avoids claiming "statistically significant ECE reduction". Instead, it reports empirical point estimates and 95% bootstrap percentiles (e.g., ECE decreases from $0.3857$ [0.30, 0.45] to $0.0960$ [0.09, 0.20]), framing this as a descriptive calibration trend.

---

### F. Negative Replication Result

#### Finding F.1: Handling of the Exploratory $65.0\%$ Cyclic Result
- **Reviewer Critique:** *"Why did preliminary reports claim $65\%$ accuracy for cyclic shifts at $K=77$, and is the current paper sweeping this under the rug?"*
- **Severity:** Blocker (if concealed)
- **Audit Assessment:** In preliminary exploratory experiments with $N=40$ and $S=1$ base sequence, an anomalous $65.0\%$ accuracy was observed. When scaled to pre-specified multi-seed replication ($N=120, S=3$ base sequences), accuracy regressed to $48.06\%$ [39.7%, 55.6\%].
- **Concrete Fix Applied:** The paper dedicates an entire standalone section (Section 8: *Important Negative Replication Result*) and highlights this finding in the Abstract, Introduction, and Conclusion. Small sample size ($N=40$) combined with anchoring to a single base sequence produced an exploratory artifact that pre-specified replication successfully exposed.

---

### G. Scope and Generalization Limits

#### Finding G.1: Overgeneralization to All Transformers
- **Reviewer Critique:** *"The paper is titled 'Candidate Order Instability in Non-Autoregressive Multi-Candidate Transformers', but you only tested one model (Laya) on one dataset (Banking77)."*
- **Severity:** Major
- **Audit Assessment:** The title reflects the architectural class, but claims must be strictly bounded to the empirical evidence.
- **Concrete Fix Applied:** 
  1. Section 11 (*Limitations*) explicitly scopes findings to the evaluated checkpoint (`convaiinnovations/laya`) and Banking77 benchmark.
  2. The paper explicitly warns that while the concatenation pattern is general, quantitative instability rates will vary across backbones (e.g., DeBERTa, modern rotary architectures).
  3. Prohibited absolute generalizations ("all transformers", "universally", "always") are completely eliminated.

---

### H. Latency and Operational Trade-Offs

#### Finding H.1: The Compute Cost of Marginalization
- **Reviewer Critique:** *"Marginalization over $M=5$ runs requires $5\times$ more compute and latency. At $K=77$, latency jumps from 224.7 ms to 1123.4 ms. How is this practical for real-time routing?"*
- **Severity:** Minor
- **Audit Assessment:** Multi-pass marginalization directly converts candidate-order robustness into compute latency. This is an unavoidable trade-off that must be transparently quantified.
- **Concrete Fix Applied:** Section 7.5 and the Three-Objective Pareto Frontier (Figure 5, Table 2) explicitly evaluate Latency alongside Accuracy and Instability, showing that `B1_Alpha` is non-dominated for low-latency regimes, while marginalization is non-dominated for high-stability regimes.

---

### I. Figure Readability & Greyscale Legibility

#### Finding I.1: Figure Polish and Color Distinguishability
- **Reviewer Critique:** *"Are plots legible when printed in black and white? Are font sizes readable at standard column width?"*
- **Severity:** Cosmetic
- **Audit Assessment:** Figures 1 through 7 use distinct markers (`o`, `s`, `^`, `D`), high-contrast color palettes (tab10), filled confidence bands, and standalone readable axes (fontsize $\ge 11$pt).
- **Concrete Fix Applied:** Verified that all 7 figures render at 300 DPI with self-contained captions and visible error bounds.

---

### J. Scientific Reproducibility

#### Finding J.1: Verifiability from Repository
- **Reviewer Critique:** *"Can anyone reproduce these exact numbers without running the expensive GPU benchmark?"*
- **Severity:** Major
- **Audit Assessment:** The repository includes complete raw CSV records (`results/raw/corrected_per_query_results.csv`, 6,000 evaluations), processed tables, statistical test scripts, invariant test suites, and figure rendering code.
- **Verification:** Running `pytest` passes 7/7 tests in under 3 seconds; running `publication_sanity_check.py` passes 100% of consistency checks.
- **Concrete Fix Applied:** Section 12 (*Reproducibility and Artifact Availability*) provides the permanent Zenodo DOI, exact git commit SHA, and release tag.

---

## Audit Checklist & Status Summary

| ID | Issue Description | Severity | Resolution Status | Section in Manuscript |
|:---|:---|:---:|:---:|:---|
| **A.1** | Distinguish from prompt/demonstration order in LLMs | Major | Resolved | Section 3.3, 3.6 |
| **A.2** | Position relative to Deep Sets / Set Transformer | Major | Resolved | Section 3.1, 3.4 |
| **B.1** | Prove candidate cardinality is not confounded with distractor sampling | Blocker | Resolved (Strictly Nested Sets) | Section 5.2 |
| **B.2** | Verify query as independent statistical unit | Blocker | Resolved (Query-Clustered Unit) | Section 6.1, 6.2 |
| **B.3** | Justify cyclic phase shift heuristic | Minor | Resolved | Section 5.3, 7.4 |
| **C.1** | Correct family-wise FDR multiple testing control | Blocker | Resolved (45 pre-specified hypotheses, $q=0.05$) | Section 6.2, 8, Table 3 |
| **C.2** | Clarify overlapping marginal CIs vs paired McNemar tests | Major | Resolved | Section 6.2, 8 |
| **D.1** | Prevent overclaiming $K=77$ accuracy gains | Major | Resolved (Framed as descriptive) | Section 7.3, 8 |
| **E.1** | Prevent claiming ECE significance without paired test | Major | Resolved (Descriptive bootstrap CIs) | Section 3.5, 6.1, 7.5 |
| **F.1** | Transparent accounting of failed 65% replication | Blocker | Resolved (Dedicated Section 8) | Abstract, Section 8 |
| **G.1** | Prevent universal claims across all transformers | Major | Resolved (Strictly scoped) | Section 1, 10, 11 |
| **H.1** | Transparent reporting of $M\times$ latency trade-off | Minor | Resolved (Pareto Frontier) | Section 7.5, Table 2, Fig 5 |
| **I.1** | High-contrast markers and font readability | Cosmetic | Resolved | Figures 1–7 |
| **J.1** | Full archival verifiability without GPU reruns | Major | Resolved | Section 12, Zenodo DOI |

**Auditor Conclusion:** All potential reviewer objections have been preempted, methodologically insulated, and incorporated into the manuscript. The study is methodologically sound and ready for publication preparation.
