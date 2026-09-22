# Reproduction Guide & Execution Protocol

This guide provides step-by-step instructions to reproduce all empirical findings, figures, statistical tables, and verification tests from scratch.

---

## Prerequisites

- Python 3.10+
- PyTorch 2.0+ (with MPS for Apple Silicon or CUDA for NVIDIA GPUs)
- ~2 GB free disk space for Hugging Face model weights and Banking77 dataset cache

---

## 1. Environment Setup

```bash
# Clone this repository
git clone https://github.com/candidate-order-instability/candidate-order-instability.git
cd candidate-order-instability

# Install required statistical and ML dependencies
pip install -r requirements.txt
```

### Install Base Model Engine
The benchmark evaluates the non-autoregressive decision model from the Laya repository:
```bash
# Clone Laya engine alongside
git clone https://github.com/NandhaKishorM/laya.git ../laya
pip install -e ../laya
```

---

## 2. Verify Metric Invariants (Unit Tests)

Before executing the full evaluation, run the invariant verification suite to confirm that metric calculations, nesting constraints, and calibration implementations match their formal specifications:

```bash
pytest -v experiments/tests/test_marginalization_invariants.py
```

*Expected output:*
```
experiments/tests/test_marginalization_invariants.py::test_metric_identical_ordering PASSED
experiments/tests/test_marginalization_invariants.py::test_metric_inverted_ordering PASSED
experiments/tests/test_marginalization_invariants.py::test_candidate_set_nesting_invariant PASSED
experiments/tests/test_marginalization_invariants.py::test_canonical_repeatability_vs_cross_canonical PASSED
experiments/tests/test_marginalization_invariants.py::test_cyclic_residual_flip_independence PASSED
experiments/tests/test_marginalization_invariants.py::test_ece_edge_cases_and_reference PASSED

============================== 6 passed in 1.16s ===============================
```

---

## 3. Run Fast Smoke Test (10 Queries)

To verify the end-to-end evaluation pipeline on a small subset before launching the full run:

```bash
python experiments/bench_marginalization.py --smoke-test 10
```

*Runtime:* ~1.5 minutes on Apple Silicon MPS.

---

## 4. Execute Full Benchmark ($N=120$)

Run the complete evaluation across all 5 candidate cardinalities ($K \in \{5, 10, 20, 40, 77\}$) using strictly nested candidate sets and stratified Banking77 queries:

```bash
python experiments/bench_marginalization.py
```

*Runtime:* ~60–75 minutes on Apple Silicon M-series GPU (MPS).  
*Generated Artifacts:*
- `results/raw/corrected_per_query_results.csv`
- `results/processed/corrected_marginalization_results.csv`
- `results/statistics/corrected_statistical_tests.csv`
- `results/statistics/corrected_pareto_frontier.csv`
- `experiments/configs/experiment_manifest.json`

---

## 5. Generate Publication Figures & Paper Tables

Re-render all publication figures (DPI 300) and structured tables:

```bash
python experiments/generate_publication_artifacts.py
```

*Generated Artifacts:*
- `paper/figures/fig1_cardinality_vs_instability.png`
- `paper/figures/fig2_cardinality_vs_logit_variance.png`
- `paper/figures/fig3_canonical_counterfactual.png`
- `paper/figures/fig4_marginalization_scaling.png`
- `paper/figures/fig5_pareto_frontier_k77.png`
- `paper/figures/fig6_cyclic_vs_random.png`
- `paper/figures/fig7_k77_query_heterogeneity.png`
- `paper/tables/paper_results_table.csv`
- `paper/tables/research_claims.csv`
