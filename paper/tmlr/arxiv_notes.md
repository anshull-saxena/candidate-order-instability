# arXiv Preprint Strategy and TMLR Double-Blind Policy Analysis

## 1. TMLR Official Policy on Preprints

According to the official TMLR Editorial Policies (`https://jmlr.org/tmlr/editorial-policies.html`):
- **Preprint Compatibility:** TMLR explicitly permits authors to submit work that has previously appeared or currently appears on non-archival preprint servers such as arXiv.
- **Reviewer Obligation:** Reviewers are instructed not to actively search for preprint versions of submitted papers or attempt to de-anonymize authors.
- **Citation of Preprints:** The existence of an unrefereed preprint does not constitute prior publication and does not violate TMLR's dual submission policy.

---

## 2. Double-Blind Review Integrity Considerations

While preprints are officially permitted, posting to arXiv simultaneously with or prior to OpenReview submission involves a trade-off:

### Strategy A: Withhold arXiv until TMLR Decision (Recommended for Pure Double-Blind)
- **Advantage:** Guarantees 100% double-blind review integrity. No risk of accidental de-anonymization via automated search engine indexing, semantic scholar alerts, or social media discussion.
- **Recommendation:** Submit to TMLR first. Once the initial reviews are submitted or an acceptance decision is rendered by the Action Editor, upload the paper to arXiv.

### Strategy B: Immediate Concurrent arXiv Posting
- **Advantage:** Establishes public priority and immediate scientific timestamping on arXiv.
- **Risk:** Automated indexing (Google Scholar, Semantic Scholar) will associate the paper title with the author's name, potentially revealing identity if a reviewer searches for related citations.
- **Mitigation if Strategy B is chosen:** Do not share the arXiv link on social media during active review; ensure the OpenReview title matches exactly so that conflict detection systems function properly.

---

## 3. arXiv Submission Packaging Guide

When ready to submit to arXiv:

### 3.1 Template Configuration
In `paper/tmlr/manuscript.tex`, change line 13:
```latex
% For anonymous review:
\usepackage{tmlr}

% For arXiv preprint:
\usepackage[preprint]{tmlr}
```
Using `[preprint]` automatically:
1. Displays the complete author name and affiliation block on the title page.
2. Formats the running header as `Preprint. Under review at TMLR.` (or custom text).
3. Preserves all official TMLR layout, margin, and typography rules.

### 3.2 arXiv Upload Bundle
arXiv requires either full LaTeX source files or a compiled PDF. For optimal indexing and rendering on arXiv, submit LaTeX source:
1. `manuscript.tex` (with `\usepackage[preprint]{tmlr}`)
2. `manuscript.bbl` (pre-compiled bibliography file; do NOT upload `.bib` without `.bbl`)
3. `tmlr.sty`
4. `fancyhdr.sty`
5. `math_commands.tex`
6. `figures/` (all 7 PNG figures referenced in the manuscript)

### 3.3 Target arXiv Subject Classifications
- **Primary Category:** `cs.CL` (Computation and Language)
- **Secondary Category:** `cs.LG` (Learning / Machine Learning)
- **Cross-Lists (Optional):** `stat.ML` (Machine Learning)
