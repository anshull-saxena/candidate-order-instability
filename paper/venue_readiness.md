# Venue Readiness & Target Venue Analysis

This document evaluates realistic academic venues for the empirical study:
**"Candidate Order Instability in Non-Autoregressive Multi-Candidate Transformers"** (Anshul Saxena, 2026).

---

## 1. Paper Profile and Empirical Characteristics

- **Core Contribution:** Empirical identification and measurement of candidate order instability in non-autoregressive multi-candidate transformers; demonstration that canonical sorting conceals rather than eliminates positional exposure bias ($55.83\%$ cross-canonical flip rate); evaluation of inference-time permutation marginalization; rigorous multiple-testing control (Benjamini-Hochberg FDR); and a controlled negative replication result correcting an exploratory artifact ($65.0\% \to 48.06\%$).
- **Methodological Strengths:**
  - Strict distractor nesting protocol ($K_5 \subset K_{10} \subset K_{20} \subset K_{40} \subset K_{77}$).
  - Fully stratified evaluation on Banking77 ($N=120$ queries spanning all 77 intent classes).
  - Degree-2 $U$-statistic formulation for pairwise flip rates.
  - Clustered bootstrap confidence intervals ($B=1,000$).
  - Exact two-sided McNemar paired tests with FDR correction ($q=0.05$).
  - Permanent open-science archival artifact (Zenodo DOI: `10.5281/zenodo.22906245`, GitHub Release `v1.0.0`).
- **Primary Empirical Scope / Limitations:**
  - Evaluated on a single production-grade backbone checkpoint (`convaiinnovations/laya`, RoBERTa-based bidirectional encoder).
  - Evaluated on one fine-grained intent benchmark (Banking77).
  - Statistical power at $K=77$ with $N=120$ detects large effects but leaves modest accuracy differences descriptively suggestive rather than FDR-significant.

---

## 2. Target Venue Comparative Analysis

| Dimension | Venue 1: TMLR | Venue 2: ACL Rolling Review (ARR) | Venue 3: NeurIPS (Datasets & Benchmarks) | Venue 4: TACL |
| :--- | :--- | :--- | :--- | :--- |
| **Full Name** | *Transactions on Machine Learning Research* | *ACL Rolling Review* (EMNLP / ACL / Findings) | *NeurIPS Datasets and Benchmarks Track* | *Transactions of the Association for Computational Linguistics* |
| **Venue Type** | Open-Access Journal | Rolling Peer-Review System for Conferences | Annual Conference Track | Open-Access Journal |
| **Scope Fit** | **Exceptional** (Explicitly values empirical rigor & negative results) | **High** (Core NLP intent classification & ranking) | **High** (Evaluation methodology, benchmarking rigor) | **Moderate** (Linguistic/NLP depth required) |
| **Review Criteria** | (1) Accurate and convincing evidence; (2) Community relevance. No requirement for state-of-the-art breakthroughs. | Soundness, excitement, reproducibility, broad applicability across NLP. | Rigor of benchmarking, novelty of insights, dataset/evaluation hygiene, documentation. | Substantial empirical or conceptual contribution to computational linguistics. |
| **Review Model** | Double-blind via OpenReview, public reviews & responses. | Double-blind via OpenReview, structured meta-reviews. | Double-blind via OpenReview, rebuttal phase. | Double-blind via MIT Press / Softconf, rolling reviews. |
| **Submission Model** | Year-round rolling (submit anytime). | Bimonthly rolling cycles (typically 15th of even months). | Fixed annual deadline (typically mid-May). | Rolling monthly deadlines (1st of every month). |
| **Author Fees (APC)** | **$0** (Diamond Open Access, sponsored by ML community). | **$0** (No submission fees; conference registration if presented). | **$0** (No submission fees; conference registration if presented). | **$0** (Diamond Open Access by ACL). |
| **arXiv Compatibility** | **Fully compatible.** Preprints allowed before, during, and after submission. | **Compatible.** ACL preprinting policy applies (must not advertise on social media during review). | **Fully compatible.** Preprints allowed at any time. | **Compatible.** Follows ACL preprinting policy. |
| **Review Timeline** | ~2 months to initial decision; fast-track revisions. | ~2–2.5 months per cycle; commit to upcoming conference deadline. | Annual fixed schedule (~3.5 months review + rebuttal). | ~3 months to initial decision. |

---

## 3. In-Depth Venue Evaluations

### Venue A: Transactions on Machine Learning Research (TMLR) — *Recommended Primary Target*

- **Fit Rationale:**  
  TMLR is structurally designed for papers with rigorous experimental methodology, negative results, and careful empirical verification. TMLR's explicit review guidelines mandate that reviewers evaluate whether claims are supported by clear evidence, rather than whether the work achieves a "state-of-the-art" benchmark record or introduces a completely new neural architecture. Our paper directly answers an architectural question, establishes clear bounds, and documents a transparent negative replication result—making it an ideal fit.
- **Submission Requirements:**  
  - Format: TMLR LaTeX style template (`tmlr.sty`).
  - Page limit: None (typically 8–12 pages for main text, plus appendices).
  - OpenReview submission with abstract, author metadata, and PDF.
  - Mandatory reproducibility statement and conflict of interest disclosures.
- **Known Deadlines:**  
  - None. Year-round rolling submission.
- **Risks & Reviewer Friction Points:**  
  - *Risk:* Reviewers may request experiments across multiple model checkpoints or additional domains (e.g., CLINC150).  
  - *Defense:* The manuscript is carefully scoped around the non-autoregressive multi-candidate marker architecture in `laya`, and Section 10 ("Limitations") explicitly states that results characterize this architecture and provides power analysis for future extensions.
- **Exact Materials Needed:**  
  1. Main PDF compiled with `tmlr.sty`.  
  2. OpenReview metadata (title, abstract, keywords, author).  
  3. Zenodo DOI link (`10.5281/zenodo.22906245`) and GitHub repository link.

---

### Venue B: ACL Rolling Review (ARR) -> Target: EMNLP / ACL / Findings

- **Fit Rationale:**  
  Banking77 is a widely recognized conversational benchmark in the ACL community. The paper addresses candidate selection and intent routing, which are core topics in dialogue systems and language understanding.
- **Submission Requirements:**  
  - Format: ACL LaTeX style (`acl_natbib.sty`).
  - Page limit: Long paper (8 pages max for content before references) or Short paper (4 pages max). Current 19-page manuscript would require condensing to 8 pages + unlimited appendix.
  - Responsible NLP Research Checklist.
  - Software/Data artifacts disclosure.
- **Known Deadlines:**  
  - Bimonthly cycles (e.g., October 15, December 15, February 15, April 15, June 15, August 15).
- **Risks & Reviewer Friction Points:**  
  - *Risk:* Severe space constraints (8-page limit) would force substantial background literature and Pareto analysis into appendices.
  - *Risk:* Reviewers may ask why generative LLMs (e.g., LLaMA, GPT-4) were not benchmarked alongside encoder-based multi-candidate models.
  - *Defense:* Section 3 ("Related Work") clearly differentiates encoder-based non-autoregressive concatenation from LLM generative multiple-choice prompting.
- **Exact Materials Needed:**  
  1. Condensation into ACL 8-page double-column layout.  
  2. ARR Submission form on OpenReview.  
  3. Responsible NLP Checklist completion.

---

### Venue C: NeurIPS Datasets & Benchmarks Track

- **Fit Rationale:**  
  Focuses on benchmarking methodology, error analysis, dataset diagnostics, and replication studies. The paper's rigorously audited multi-seed replication protocol and distractor nesting design align with the track's standards.
- **Submission Requirements:**  
  - Format: NeurIPS stylesheet (`neurips_2026.sty`).
  - Page limit: 9 pages of content + unlimited references and appendix.
  - Mandatory NeurIPS Paper Checklist.
  - Data / benchmark hosting documentation (URL, persistent identifier, license, maintenance plan).
- **Known Deadlines:**  
  - Annual cycle (typically mid-May submission, September notifications).
- **Risks & Reviewer Friction Points:**  
  - *Risk:* Long turnaround time if not aligned with the May submission window.
  - *Risk:* Reviewers may expect a newly created dataset rather than a benchmark methodology critique on Banking77.
- **Exact Materials Needed:**  
  1. NeurIPS LaTeX template formatting.  
  2. NeurIPS Checklist and Datasheet/Documentation.  
  3. Zenodo archival link.

---

## 4. Factual Recommendation and Strategic Sequence

1. **Step 1: Release on arXiv (Immediate)**  
   - Establish timestamped scholarly priority with the complete 19-page standalone manuscript and Zenodo DOI archive.
   - Select primary category: `cs.CL` (Computation and Language) or `cs.LG` (Machine Learning).
2. **Step 2: Submit to TMLR (Primary Recommendation)**  
   - Format with `tmlr.sty`.
   - Submit via OpenReview on a rolling timeline.
   - Provides public review discourse and directly rewards the paper's negative replication findings and methodological rigor without arbitrary page constraints.
3. **Contingency / Alternative: ACL Rolling Review (ARR)**  
   - If a conference presentation at EMNLP or ACL is desired by the author, condense the main narrative into 8 double-column pages, moving methodological proofs and detailed Pareto tables into the appendix.
