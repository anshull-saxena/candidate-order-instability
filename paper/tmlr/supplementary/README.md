# Anonymized Supplementary Material

This archive contains the replication code, configuration manifests, invariant test suite, and raw experimental evaluation data for the TMLR manuscript:

> **Title:** *Candidate Order Instability in Non-Autoregressive Multi-Candidate Transformers*

---

## 1. Directory Structure

```text
supplementary/
├── README.md                              # This navigation and reproduction guide
├── requirements.txt                       # Pinned Python package dependencies
├── environment.yml                        # Conda environment definition
├── code/
│   ├── bench_marginalization.py           # Core benchmark harness for multi-candidate evaluation
│   ├── generate_publication_artifacts.py  # Reproduction script for figures and tables
│   └── recompute_and_audit.py             # Statistical recomputation and validation audit
├── configs/
│   └── experiment_manifest.json           # Exact split, query indices, and seed manifest
├── tests/
│   └── test_marginalization_invariants.py # Invariant property verification suite (7/7 passing)
└── results/
    ├── raw/
    │   └── corrected_per_query_results.csv # Query-level predictions, logits, and flips (N=120)
    ├── processed/
    │   ├── corrected_marginalization_results.csv # Metric aggregates across K in {5, 10, 20, 40, 77}
    │   └── paper_results_table.csv         # Consolidated publication table with bootstrap CIs
    └── statistics/
        ├── corrected_pareto_frontier.csv   # Accuracy vs flip rate vs latency trade-offs
        ├── corrected_statistical_tests.csv # Paired McNemar tests with Benjamini-Hochberg FDR
        └── research_claims.csv             # Empirical claim verification table
```

---

## 2. Environment Setup

```bash
# Option A: Conda
conda env create -f environment.yml
conda activate candidate-order-instability

# Option B: Pip / venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Verifying Invariants and Consistency

To verify the mathematical and empirical invariants:
```bash
pytest tests/test_marginalization_invariants.py -v
```
Expected output: 7 passed.

To recompute all summary statistics and statistical tests directly from the raw per-query evaluations:
```bash
python3 code/recompute_and_audit.py
```
This verifies that every metric in `paper_results_table.csv` and `corrected_statistical_tests.csv` matches the raw query predictions with zero discrepancy.
