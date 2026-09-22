# arXiv Submission Source Package

This directory contains the self-contained source bundle ready for upload to [arXiv.org](https://arxiv.org/):

- **Article Title:** Candidate Order Instability in Non-Autoregressive Multi-Candidate Transformers
- **Author:** Anshul Saxena
- **Affiliation:** Department of Computer Science & Information Systems, BITS Pilani
- **Permanent Zenodo DOI:** [10.5281/zenodo.22906245](https://doi.org/10.5281/zenodo.22906245)
- **Primary Subject Area:** Computation and Language (`cs.CL`)
- **Secondary Subject Area:** Machine Learning (`cs.LG`), Artificial Intelligence (`cs.AI`)

---

## Files in this Bundle

- `manuscript.tex`: Main LaTeX source file (uses standard `article` class with `xurl`, `hyperref`, `booktabs`, `subcaption`, `natbib`).
- `references.bib`: Complete BibTeX bibliography containing 29 verified citations with active DOIs/URLs.
- `manuscript.bbl`: Pre-generated bibliography auxiliary file (required by arXiv AutoTeX system).
- `manuscript.pdf`: Precompiled 19-page camera-ready reference PDF.
- `figures/`: High-resolution figures (`fig1` through `fig7`) referenced by `manuscript.tex`.

---

## Local Compilation Verification

To verify local compilation from this directory:

```bash
pdflatex -interaction=nonstopmode manuscript.tex
pdflatex -interaction=nonstopmode manuscript.tex
```

(Because `manuscript.bbl` is pre-compiled, BibTeX execution is optional).
