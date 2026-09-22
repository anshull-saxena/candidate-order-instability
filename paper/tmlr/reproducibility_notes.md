# Reproducibility Notes and Double-Blind Integrity Plan

## 1. Scientific Immutability & Provenance

The empirical findings reported in this manuscript are derived from a frozen, cryptographically verified research artifact:
- **Repository:** `candidate-order-instability`
- **Frozen Release Tag:** `v1.0.0`
- **Frozen Git Commit Hash:** `266bbb902871dc9974cbf516adea013e3c4eb084`
- **Archival Zenodo DOI:** `10.5281/zenodo.22906245`
- **License:** Apache 2.0 (code and benchmark harness) / CC BY 4.0 (paper and documentation)

This artifact is immutable. No benchmarks were re-run, no figures altered, and no statistical metrics modified during the preparation of this TMLR submission.

---

## 2. Double-Blind Review Protocol

To adhere to TMLR's mandatory double-blind review policy while preserving scientific reproducibility:
1. **Omission of Direct Self-Identifiers:** All direct references to author names, institutional affiliations, personal contact emails, and personal GitHub repository URLs have been removed from the submission manuscript (`manuscript.pdf`).
2. **Neutral Phrasing of Model Artifacts:** The target architecture is cited and referenced as an external open-source artifact (`convaiinnovations/laya`) using neutral, third-person descriptive language.
3. **Self-Contained Anonymized Supplementary Archive:** Reviewers and action editors are provided with `supplementary.zip` ($\le 100\text{ MB}$, actual size $\approx 1.5\text{ MB}$). This archive contains:
   - Complete benchmark evaluation harnesses (`order_instability_benchmark.py`).
   - Invariant testing verification suite (`test_invariants.py`).
   - Complete candidate configuration manifests (`data/manifests/k*.json`).
   - Query stratification splits (`data/queries_stratified_120.json`).
   - Raw per-query evaluation CSVs ($N=120$ intent queries across $K \in \{5, 10, 20, 40, 77\}$, multi-seed $S=3$ runs at $K=77$).
   - Python environment specification (`requirements.txt`, `environment.yml`).
   - All internal user paths (e.g., `/Users/anshul/...`) and author identifiers are stripped from the supplementary materials.

---

## 3. Post-Review De-Anonymization Plan

Upon notification of acceptance by the Action Editor:
1. **Stylefile Mode:** Change `\usepackage{tmlr}` to `\usepackage[accepted]{tmlr}` in `manuscript.tex`.
2. **Author and Affiliation Restoration:**
   - Author: Anshul Saxena
   - Affiliation: Birla Institute of Technology and Science (BITS) Pilani, Pilani Campus, Rajasthan, India
   - Correspondence: `f20221041@pilani.bits-pilani.ac.in` / `anshul.saxena.work@gmail.com`
3. **Repository and DOI Restoration:**
   - Restore the direct public GitHub repository hyperlink:  
     `https://github.com/anshull-saxena/candidate-order-instability`
   - Restore the permanent Zenodo DOI hyperlink:  
     `https://doi.org/10.5281/zenodo.22906245`
   - Update the Data & Code Availability section to link directly to the immutable v1.0.0 release tree.

---

## 4. Hardware, Software, and Deterministic Reproduction Environment

All empirical measurements reported in the paper were executed in the following controlled environment:

| Component | Specification |
| :--- | :--- |
| **Operating System** | macOS 15.0+ (Darwin arm64) |
| **Hardware** | Apple Silicon (M-series, MPS backend / deterministic fallback) |
| **Python Version** | Python 3.12.0+ |
| **PyTorch Version** | PyTorch 2.5.1 |
| **HuggingFace Transformers** | Transformers 4.48.0 |
| **Random Seeds** | Primary: `42`; Cardinality distractor sampling: deterministic query-hash; Multi-seed replication ($K=77$): `alphabetical`, `seed_7701`, `seed_7702` |

### Reproduction Commands
To verify the complete dataset and rerun the automated invariant test suite:
```bash
# 1. Unzip the supplementary archive
unzip supplementary.zip -d supplementary/
cd supplementary/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run invariant test suite (6/6 passing required)
pytest tests/test_invariants.py -v

# 4. Verify candidate set nesting invariants
python3 -c "
import json
for k in [5, 10, 20, 40]:
    with open(f'data/manifests/k{k}.json') as f1, open('data/manifests/k77.json') as f2:
        m1, m2 = json.load(f1), json.load(f2)
        for qid in m1:
            assert set(m1[qid]['candidates']).issubset(set(m2[qid]['candidates'])), f'Nesting violated for query {qid}'
print('Candidate nesting invariants verified 100% across all cardinalities.')
"
```

---

## 5. Statistical Rigor Protocol

- **Sample Unit:** Query ID ($N=120$ independent queries from the Banking77 benchmark).
- **Paired Comparisons:** All method-versus-baseline comparisons are paired by query ID.
- **Multiple Testing:** Two-tailed paired McNemar tests with exact binomial calculation for discordant pairs; $p$-values adjusted via Benjamini-Hochberg False Discovery Rate (FDR) control at $q = 0.05$ across all 45 pre-specified hypotheses.
- **Confidence Intervals:** $95\%$ empirical confidence intervals computed via 1,000 bias-corrected bootstrap resamples clustered by query ID.
- **Multi-Seed Replication:** At maximum cardinality ($K=77$), all metrics are averaged across $S=3$ independent candidate serialization seeds ($N \times S = 360$ evaluations per method) to eliminate exploratory sampling bias.
