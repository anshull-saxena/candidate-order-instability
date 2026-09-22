# Manuscript: Candidate Order Instability in Non-Autoregressive Multi-Candidate Transformers

This directory contains the publication manuscript, bibliography, and build artifacts for the research paper:

> **Candidate Order Instability in Non-Autoregressive Multi-Candidate Transformers**  
> **Author:** Anshul Saxena  
> **Affiliation:** Department of Computer Science & Information Systems, BITS Pilani  
> **Email:** `f20221041@hyderabad.bits-pilani.ac.in`  
> **Permanent DOI:** [10.5281/zenodo.22906245](https://doi.org/10.5281/zenodo.22906245)  
> **Archival Artifact:** GitHub Release [`v1.0.0`](https://github.com/anshull-saxena/candidate-order-instability/releases/tag/v1.0.0) (`266bbb902871dc9974cbf516adea013e3c4eb084`)

---

## 1. Directory Structure

```
paper/
├── manuscript.tex          # Complete LaTeX manuscript (12 sections, 3 tables, 7 figures)
├── references.bib          # Verified BibTeX citations and structured literature TODOs
├── README.md               # This compilation and manuscript documentation guide
├── manuscript.pdf          # Precompiled 14-page camera-ready PDF document
├── figures/                # High-resolution vector/PNG empirical figures (fig1 through fig7)
│   ├── fig1_cardinality_vs_instability.png
│   ├── fig2_cardinality_vs_logit_variance.png
│   ├── fig3_canonical_counterfactual.png
│   ├── fig4_marginalization_scaling.png
│   ├── fig5_pareto_frontier_k77.png
│   ├── fig6_cyclic_vs_random.png
│   └── fig7_k77_query_heterogeneity.png
└── tables/                 # Frozen publication CSV tables
    ├── paper_results_table.csv
    └── research_claims.csv
```

---

## 2. Compilation Instructions

The manuscript requires standard TeX Live distributions (2022+) with packages `booktabs`, `graphicx`, `subcaption`, `natbib`, `tabularx`, and `microtype`.

### Method A: Single Command with `latexmk` (Recommended)

```bash
cd paper
latexmk -pdf manuscript.tex
```

To clean auxiliary files while keeping `manuscript.pdf`:
```bash
latexmk -c
```

### Method B: Manual Three-Pass Compilation (`pdflatex` + `bibtex`)

```bash
cd paper
pdflatex -interaction=nonstopmode manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode manuscript.tex
pdflatex -interaction=nonstopmode manuscript.tex
```

---

## 3. Manuscript Structure

The manuscript is organized into 12 sections following empirical machine learning standards:

1. **Abstract:** High-level summary of the research questions, strictly nested distractor protocol, key scaling findings, the canonicalization fallacy, inference-time marginalization, multiple-testing FDR control, and the negative replication result.
2. **Introduction:** Mathematical formulation of permutation invariance for multi-candidate classification ($f(q, \pi(\mathcal{C})) = f(q, \mathcal{C})$), single-pass delimiter marker concatenation in bidirectional encoders, and explicit research questions.
3. **Related Work:** Survey across prompt order sensitivity, permutation invariance and set functions, positional encoding biases, and calibration. Structured literature search roadmap for peer-reviewed citations.
4. **Problem Formulation and Model Architecture:** Precise token serialization equations, marker embedding projections, and self-attention coupling mechanisms in `convaiinnovations/laya`.
5. **Experimental Setup:** Controlled distractor nesting protocol ($K_5 \subset K_{10} \subset K_{20} \subset K_{40} \subset K_{77}$), $N=120$ intent-stratified queries, 10 random permutations per query, 3 base candidate sequence replications at $K=77$, and full intervention suite (`B0_Native`, `B0_Random`, `B1_Alpha`, `B1_ReverseAlpha`, `Rand_Marg_M{2,3,5}`, `B2_Cyclic_M{2,3,5}`).
6. **Metrics and Statistical Analysis:** Formal definitions of within-method pairwise argmax decision flip rate ($\mathrm{FR}_{\mathrm{top1}}$), counterfactual cross-canonical flip rate ($\mathrm{CrossCanonicalFlip}$), residual ensemble flip rate ($\mathrm{ResFlip}_M$), 15-bin Expected Calibration Error (ECE), normalized Kendall's rank distance ($\tau_{\mathrm{norm}}$), 1,000-sample query-level clustered bootstrap CIs, exact McNemar tests, and Benjamini-Hochberg FDR control ($q=0.05$).
7. **Results:** Detailed empirical results across cardinality scaling, logit variance explosion ($7.5\times$), the canonicalization exposure bias ($55.83\%$ cross-flip at $K=77$), marginalization scaling laws, orthogonal cyclic vs random permutations, query-level vulnerability heterogeneity, and the three-objective Pareto frontier.
8. **Statistical Significance and Hypothesis Testing:** Rigorous reporting of the 45 pre-specified paired comparisons against `B0_Native`. Highlights that only cyclic marginalization at $K=40$ achieves FDR-adjusted statistical significance ($+9.17$ pp, adjusted $p=0.04395$).
9. **Important Negative Replication Result:** Full accounting of the non-replication of the exploratory $65.0\%$ cyclic accuracy finding at $K=77$, which regressed to $48.06\%$ [39.7%, 55.6\%] under pre-specified multi-seed replication ($N=120, S=3$).
10. **Discussion:** Architectural origins of positional sensitivity, the fallacy of relying on alphabetical canonicalization for determinism, and inference-time marginalization trade-offs.
11. **Limitations:** Single backbone/checkpoint scope (`convaiinnovations/laya`), domain specificity (Banking77), post-training inference scope, and statistical power constraints at $K=77$ ($N=120$).
12. **Conclusion & Reproducibility:** Summary and permanent archive metadata pointing to Zenodo DOI `10.5281/zenodo.22906245`.

---

## 4. Tables and Figures Summary

### Tables
- **Table 1 (`tab:main_results`):** Candidate Cardinality Scaling Results across $K \in \{5, 10, 20, 40, 77\}$ reporting Top-1 Accuracy [95% CI], ECE [95% CI], Within-Method $\mathrm{FR}_{\mathrm{top1}}$ [95% CI], Counterfactual Cross-Flip, p50 Latency, and FDR Significance.
- **Table 2 (`tab:k77_detailed`):** Detailed Multi-Seed Results at $K=77$ across all 10 evaluated methods with three-objective Pareto non-domination status.
- **Table 3 (`tab:significance_summary`):** Paired Hypothesis Tests vs `B0_Native` (discordant gain/loss counts, exact McNemar unadjusted $p$, Benjamini-Hochberg FDR-adjusted $p$, and FDR significance at $q=0.05$).

### Figures
- **Figure 1:** Cardinality vs Pairwise Argmax Flip Rate ($\mathrm{FR}_{\mathrm{top1}}$ scaling from $5.7\%$ to $47.3\%$).
- **Figure 2:** Candidate Cardinality vs Cross-Permutation Logit Variance ($8.40 \to 63.34$, $7.5\times$ expansion).
- **Figure 3:** The Canonicalization Fallacy (within-method $0.0\%$ flip rate vs $55.8\%$ counterfactual $A \to Z$ vs $Z \to A$ flip rate).
- **Figure 4:** Inference-Time Order Marginalization Scaling ($M=1, 2, 3, 5$) suppressing flip rate and ECE.
- **Figure 5:** Three-Objective Pareto Frontier at $K=77$ (Accuracy $\uparrow$, Latency $\downarrow$, Instability $\downarrow$).
- **Figure 6:** Orthogonal Cyclic Shifts vs Random Marginalization across cardinalities.
- **Figure 7:** Query-Level Heterogeneity at $K=77$ (over $55\%$ of queries have $\mathrm{FR} \ge 50\%$; strong negative correlation with model confidence $r = -0.4926, p = 1.09 \times 10^{-8}$).

---

## 5. Literature TODO Roadmap

In accordance with empirical research rigor, citations for foundational and external literature are cataloged in `references.bib` with structured placeholders to be integrated during the final literature survey:

1. **Permutation Invariance & Set Functions:**
   - Deep Sets (Zaheer et al., NeurIPS 2017)
   - Set Transformer (Lee et al., ICML 2019)
2. **Order Sensitivity & Prompt Perturbation:**
   - Prompt reordering in LLMs (Lu et al., ACL 2022; Zhao et al., ICML 2021)
3. **Positional Encoding Effects & Primacy/Recency:**
   - Lost-in-the-middle context position biases (Liu et al., TACL 2024)
4. **Calibration in Ranking:**
   - Multiclass calibration & ECE formulations (Guo et al., ICML 2017; Naeini et al., AAAI 2015; Nixon et al., CVPRW 2019)
5. **Inference-Time Ensemble Marginalization:**
   - Test-time augmentation (TTA) and Monte Carlo permutation averaging
6. **Multi-Candidate Routing Benchmarks:**
   - Dual-encoder vs cross-encoder intent selection (Casanueva et al., ACL 2020)

---

## 6. Reproducibility Guarantee

All data reported in this manuscript is derived deterministically from the frozen research artifacts:
- Git Release: `v1.0.0`
- Commit: `266bbb902871dc9974cbf516adea013e3c4eb084`
- Raw results: `research/results/corrected_marginalization_results.csv`
- Statistical tests: `research/results/corrected_statistical_tests.csv`
- Invariant tests: `python -m pytest tests/` (6/6 passing)
