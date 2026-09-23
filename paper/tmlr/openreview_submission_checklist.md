# OpenReview Submission Checklist & Metadata Record

**Venue:** Transactions on Machine Learning Research (TMLR)  
**Portal:** `https://openreview.net/group?id=TMLR`  
**Date:** September 23, 2026  
**Status:** Ready for human submission

---

## 1. Submission Metadata

### 1.1 Paper Title
```text
Candidate Order Instability in Non-Autoregressive Multi-Candidate Transformers
```

### 1.2 Running Title (Header)
```text
Candidate Order Instability in Multi-Candidate Transformers
```

### 1.3 Primary Area & Keywords
- **Primary Area:** Natural Language Processing (Robustness, Transformers, Intent Classification)
- **Secondary Areas:** Machine Learning Reliability & Evaluation Methodology; Test-Time Ensembling
- **Keywords:**
  ```text
  multi-candidate transformers, candidate order instability, permutation invariance, non-autoregressive classification, exposure bias, calibration, test-time augmentation, false discovery rate
  ```

### 1.4 Plaintext Abstract (for OpenReview Form)
```text
Candidate choices in classification form an unordered set, yet transformer classifiers frequently serialize options into sequential text. We evaluate candidate presentation order in non-autoregressive multi-candidate transformers (convaiinnovations/laya), which score all choices in a single bidirectional pass via delimiter marker representations. Using the Banking77 benchmark under a strictly nested distractor protocol (K in {5, 10, 20, 40, 77}) across N=120 intent-stratified queries, we observe that pairwise argmax decision instability (FR_top1) increases from 5.70% [3.1%, 8.4%] at K=5 to 47.28% [42.8%, 51.9%] at K=77, accompanied by a 7.5x expansion in candidate logit variance. While canonical alphabetical sorting enforces deterministic repeatability, counterfactual reverse-alphabetical sorting reveals a 55.83% [47.5%, 65.0%] disagreement rate at K=77, demonstrating that canonicalization conceals rather than eliminates positional bias. Inference-time permutation marginalization suppresses residual flip rate to 25.83% (M=5 random) and 16.11% (M=5 cyclic shifts), while descriptively reducing Expected Calibration Error from 0.3857 to 0.0960. Crucially, under Benjamini-Hochberg False Discovery Rate control (q=0.05) across 45 pre-specified paired comparisons, only cyclic shifts at K=40 achieve statistically significant accuracy gains (+9.17 percentage points, adjusted p=0.04395); at K=77, accuracy gains fail FDR significance. Furthermore, we document the non-replication of an exploratory 65.0% cyclic accuracy finding, which regressed to 48.06% [39.7%, 55.6%] under multi-seed replication (N=120, S=3). All artifacts, data, and verification suites are openly archived.
```

---

## 2. Author Profile & Conflicts of Interest

### 2.1 Author Details
- **Full Name:** Anshul Saxena
- **Institutional Affiliation:** Birla Institute of Technology and Science (BITS) Pilani, Pilani Campus, Rajasthan, India
- **Institutional Email:** `f20221041@pilani.bits-pilani.ac.in`
- **Secondary Contact Email:** `anshul.saxena.work@gmail.com`
- **OpenReview Profile:** Ensure profile lists current affiliation, domain conflicts (`bits-pilani.ac.in`), and co-author history.

### 2.2 Institutional Conflicts of Interest (COI)
- Birla Institute of Technology and Science, Pilani (`bits-pilani.ac.in`, all campuses: Pilani, Goa, Hyderabad, Dubai).
- Any personal collaborators within the past 48 months.

---

## 3. Action Editor (AE) Selection Guidance

When selecting or ranking Action Editors on OpenReview, prioritize researchers with demonstrable expertise in:
1. **NLP Robustness & Perturbation Analysis:** Focus on sensitivity to prompt phrasing, token ordering, and position encoding artifacts in transformers.
2. **Transformer Architecture & Positional Bias:** Expertise in relative/absolute position encodings, self-attention dynamics, and sequence concatenation effects.
3. **Statistical Evaluation & Methodology:** Expertise in paired testing, multiple comparison corrections (FDR, Benjamini-Hochberg), calibration analysis (ECE), and empirical reproducibility in deep learning.

### Recommended Research Profiles / Keywords for AE Matching:
- *Keywords:* Natural Language Processing, Transformer Models, Robustness, Fairness & Bias, Model Calibration, Evaluation & Benchmarking.
- *Examples of relevant research areas:* Researchers working on sequence models, prompt sensitivity, set functions, or calibration in neural classifiers.

---

## 4. File Verification & Upload Checklist

- [x] **Primary Manuscript (`manuscript.pdf`):**
  - Generated using official, unmodified `tmlr.sty` and `tmlr.bst`.
  - Author and affiliation block suppressed (anonymous double-blind mode).
  - No personal GitHub URLs or identifying paths.
  - Page count: 18 pages (including all figures, tables, broader impact statement, and references).
  - Zero LaTeX compilation errors or undefined citations.
  - High-resolution figures embedded as vector/lossless PNGs (`figures/fig1` through `fig7`).

- [x] **Supplementary Material (`supplementary.zip`):**
  - Size $\approx 1.5\text{ MB}$ (strictly below 100 MB maximum threshold).
  - Anonymized code, manifests, tests, and raw CSV files included.
  - No secrets, personal tokens, or identifying absolute paths.
  - Contains invariant test suite with reproduction instructions.

- [x] **LLM Assistance Disclosure:**
  - Mandatory footnote placed on Page 1:
    > "During the preparation of this manuscript, the authors utilized large language model assistants solely for drafting suggestions, style refinements, literature query organization, and code formatting. All experimental design, empirical measurements, mathematical proofs, interpretations, and conclusions were independently conducted, rigorously verified, and finalized by the authors, who take full scholarly responsibility for the integrity of this work."
  - Corresponding checkbox ticked on OpenReview form.

- [x] **Broader Impact Statement:**
  - Explicit, unnumbered section (`\subsubsection*{Broader Impact Statement}`) placed immediately preceding References on page 14.
  - Covers deployment risks in mission-critical routing, exposure bias in automated selection, and mitigation protocols.

- [x] **Dual Submission Declaration:**
  - The manuscript represents original empirical work.
  - It is not currently under review at any other peer-reviewed, archival venue.

---

## 5. Post-Submission Workflow

1. **Submission Confirmation:** Note the OpenReview paper number (e.g., `#XXXX`) and record the submission timestamp.
2. **Review Period:** TMLR operates on a rolling review timeline. Reviewers are assigned by the Action Editor with a standard 4-week review turnaround.
3. **Camera-Ready Preparation:** Upon recommendation of acceptance by the Action Editor:
   - Toggle `\usepackage[accepted]{tmlr}` in `paper/tmlr/manuscript.tex`.
   - Restore author and affiliation header.
   - Insert the assigned OpenReview forum link into `\def\openreview{...}`.
   - De-anonymize the repository and permanent Zenodo DOI links.
