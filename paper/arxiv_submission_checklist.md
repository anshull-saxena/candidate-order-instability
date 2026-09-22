# arXiv Submission Checklist & Metadata

This checklist provides the submission metadata and pre-flight verification steps for:
**"Candidate Order Instability in Non-Autoregressive Multi-Candidate Transformers"**

---

## 1. Submission Metadata

- **Title:**  
  `Candidate Order Instability in Non-Autoregressive Multi-Candidate Transformers`

- **Author(s):**  
  `Anshul Saxena`  
  - Affiliation: *Department of Computer Science & Information Systems, BITS Pilani*  
  - Email: `f20221041@hyderabad.bits-pilani.ac.in`

- **Abstract (PlainText / TeX-free for arXiv metadata entry, 210 words):**  
  ```text
  Non-autoregressive multi-candidate transformers score multiple categorical choices simultaneously by concatenating candidate texts into a single sequence with delimiter tokens. While computationally efficient, this design couples candidate representations through bidirectional self-attention and positional embeddings. We present an empirical study of candidate order sensitivity in this architecture using the Banking77 benchmark on the convaiinnovations/laya checkpoint across candidate cardinalities K in {5, 10, 20, 40, 77} under a strictly nested distractor protocol with N=120 intent-stratified queries. Across candidate permutations, the pairwise argmax decision flip rate increases monotonically from 5.70% [3.1%, 8.4%] at K=5 to 47.28% [42.8%, 51.9%] at K=77, while cross-permutation logit variance expands by 7.5x. Canonical alphabetical sorting guarantees deterministic repeatability (0% rerun variance), but counterfactual reverse-alphabetical sorting flips top-1 decisions on 55.83% [47.5%, 65.0%] of queries at K=77, proving that canonicalization conceals rather than eliminates positional exposure bias. Averaging probabilities over M in {2, 3, 5} permutations systematically reduces residual instability, with orthogonal cyclic shifts achieving 16.11% residual flip rate at M=5. Across 45 pre-specified paired comparisons controlled under Benjamini-Hochberg False Discovery Rate (q=0.05), only cyclic marginalization at K=40 achieves statistically significant accuracy gains (+9.17 pp, adjusted p=0.04395). Finally, we document the non-replication of an exploratory 65.0% accuracy spike, which regressed to 48.06% under controlled multi-seed replication. All artifacts are permanently archived at https://doi.org/10.5281/zenodo.22906245.
  ```

- **Primary Category:**  
  `cs.CL` (Computation and Language)

- **Secondary Categories:**  
  - `cs.LG` (Machine Learning)  
  - `cs.AI` (Artificial Intelligence)

- **ACM / MSC Classifications:**  
  - `I.2.7 Computing Methodologies -> Artificial Intelligence -> Natural Language Processing`  
  - `G.3 Mathematics of Computing -> Probability and Statistics -> Nonparametric statistics`

- **Permanent Repository URL:**  
  `https://github.com/anshull-saxena/candidate-order-instability`

- **Permanent Zenodo DOI:**  
  `10.5281/zenodo.22906245` (`https://doi.org/10.5281/zenodo.22906245`)

- **Frozen Release Tag & Commit:**  
  - Release Tag: `v1.0.0`  
  - Git Commit SHA: `266bbb902871dc9974cbf516adea013e3c4eb084`

- **License:**  
  `arXiv.org perpetual, non-exclusive license to distribute this article` (Standard arXiv submission license)

---

## 2. Source Files Bundle

The standalone arXiv bundle is stored in `paper/arxiv/`:
```
paper/arxiv/
├── manuscript.tex          # Clean main LaTeX file (using article, xurl, hyperref, natbib)
├── references.bib          # 29 verified BibTeX entries (zero placeholders)
├── manuscript.bbl          # Precompiled bibliography (arXiv AutoTeX standard)
├── manuscript.pdf          # 19-page precompiled verification PDF
├── README.md               # Bundle documentation
└── figures/                # Complete figure set (PNG, 300 DPI)
    ├── fig1_cardinality_vs_instability.png
    ├── fig2_cardinality_vs_logit_variance.png
    ├── fig3_canonical_counterfactual.png
    ├── fig4_marginalization_scaling.png
    ├── fig5_pareto_frontier_k77.png
    ├── fig6_cyclic_vs_random.png
    └── fig7_k77_query_heterogeneity.png
```

---

## 3. Compilation Verification

- **Direct TeX Live Compilation Command:**  
  ```bash
  cd paper/arxiv
  pdflatex -interaction=nonstopmode manuscript.tex
  pdflatex -interaction=nonstopmode manuscript.tex
  ```
- **AutoTeX Invariant Check:**  
  - Zero compilation errors.
  - Zero undefined citations (`?`).
  - Zero undefined cross-references (`?`).
  - Zero overfull hboxes affecting text margins.
  - All 7 figures render correctly.

---

## 4. Human Pre-Submission Step (Manual Action)

> [!IMPORTANT]
> In accordance with scientific autonomy principles and arXiv policies, the actual submission to arXiv requires human action:
>
> 1. Log in to [arXiv.org](https://arxiv.org/login) using your registered author account (`f20221041@hyderabad.bits-pilani.ac.in` or primary arXiv identity).
> 2. Click **"Start New Submission"**.
> 3. Upload the bundled files from `paper/arxiv/` (or upload `tar -czvf arxiv_package.tar.gz paper/arxiv/*`).
> 4. Verify the AutoTeX preview on arXiv and confirm that the generated PDF matches `paper/manuscript.pdf`.
> 5. Enter the metadata provided in Section 1 above.
> 6. Approve the final submission.
