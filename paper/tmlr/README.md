# TMLR Submission Package

This directory contains the complete, submission-ready, double-blind manuscript package for Transactions on Machine Learning Research (TMLR):

> **Title:** *Candidate Order Instability in Non-Autoregressive Multi-Candidate Transformers*  
> **Target Venue:** Transactions on Machine Learning Research (TMLR)  
> **Submission Format:** Anonymous Double-Blind PDF & Supplementary Material  
> **Underlying Frozen Artifact:** GitHub Release `v1.0.0` (commit `266bbb902871dc9974cbf516adea013e3c4eb084`, Zenodo DOI: `10.5281/zenodo.22906245`)

---

## 1. Directory Structure

```text
paper/tmlr/
├── README.md                              # This build and navigation guide
├── tmlr_policy_check.md                   # Formal TMLR editorial and style policy audit
├── reproducibility_notes.md               # Empirical provenance and de-anonymization protocol
├── openreview_submission_checklist.md     # Pre-submission form checklist and AE guidance
├── arxiv_notes.md                         # Policy analysis and bundling guide for arXiv
│
├── manuscript.tex                         # Double-blind TMLR submission LaTeX source
├── references.bib                         # Curated, verified 28-entry BibTeX bibliography
├── math_commands.tex                      # Standard TMLR mathematical notation macros
├── tmlr.sty                               # Official, unmodified TMLR style file
├── tmlr.bst                               # Official, unmodified TMLR BibTeX style
├── fancyhdr.sty                           # Required header/footer formatting package
│
├── figures/                               # High-resolution PNG figures (lossless)
│   ├── fig1_cardinality_vs_instability.png
│   ├── fig2_cardinality_vs_logit_variance.png
│   ├── fig3_canonical_counterfactual.png
│   ├── fig4_marginalization_scaling.png
│   ├── fig5_pareto_frontier_k77.png
│   ├── fig6_cyclic_vs_random.png
│   └── fig7_k77_query_heterogeneity.png
│
├── manuscript.pdf                         # Compiled submission PDF (18 pages, zero errors)
├── supplementary/                         # Anonymized supplementary materials staging directory
├── supplementary.zip                     # Anonymized supplementary archive (<= 100 MB)
└── final_tmlr_audit.py                    # 16-point automated verification script
```

---

## 2. Compilation Instructions

The manuscript requires TeX Live 2023+ (or MacTeX) with `pdflatex` and `bibtex`.

To build the submission PDF cleanly from source:
```bash
cd paper/tmlr

# Step 1: Initial compilation pass
pdflatex -interaction=nonstopmode manuscript.tex

# Step 2: Build bibliography references
bibtex manuscript

# Step 3: Resolve cross-references and outlines
pdflatex -interaction=nonstopmode manuscript.tex
pdflatex -interaction=nonstopmode manuscript.tex
```

To clean auxiliary files (`.aux`, `.log`, `.out`, `.bbl`, `.blg`):
```bash
rm -f manuscript.aux manuscript.log manuscript.out manuscript.bbl manuscript.blg manuscript.fls manuscript.fdb_latexmk
```

---

## 3. Automated Quality Audit

To verify the package against all 16 scientific and double-blind constraints:
```bash
python3 paper/tmlr/final_tmlr_audit.py
```
This script validates:
1. Immutability of the frozen `v1.0.0` commit.
2. Complete double-blind author and affiliation suppression.
3. Absence of internal filesystem paths and sensitive credentials.
4. Exact numerical consistency across all tables, abstract, and text.
5. Invariant candidate set nesting and statistical significance reporting.
6. Size constraints on the supplementary material archive ($\le 100$ MB).
