# Reproducibility Guide

This repository is designed for full, push-button reproduction of all experimental results, figures, tables, and statistical tests without proprietary dependencies.

---

## 1. Environment & Hardware Specifications

- **Operating System:** macOS (Sonoma / Sequoia on Apple Silicon) or Linux (Ubuntu 22.04+).
- **Python:** 3.10, 3.11, or 3.12.
- **Compute Accelerators:**
  - Tested on: Apple Silicon MPS (Metal Performance Shaders) with Unified Memory.
  - Compatible with: NVIDIA CUDA (11.8 / 12.1) and CPU (slower runtime).
- **Precision:** FP16 Autocast with standard float32 softmax accumulation.

---

## 2. Experiment Manifest

The exact runtime configuration is logged in [`experiments/configs/experiment_manifest.json`](../experiments/configs/experiment_manifest.json):
- **Base Model Checkpoint:** `convaiinnovations/laya` (Hugging Face Hub)
- **Dataset:** `mteb/banking77` (test split)
- **Git Commit Hash:** `573e5b62696ba441230cd6be71d593331b5d23af`
- **Global Seed:** `42`
- **Query Sample Indices:** Exactly specified list of 120 dataset indices.
- **$K=77$ Base Seeds:** `[0, 7701, 7702]`

---

## 3. Step-by-Step Reproduction Pipeline

### Step 1: Clone Repository & Install Dependencies
```bash
git clone https://github.com/candidate-order-instability/candidate-order-instability.git
cd candidate-order-instability
pip install -r requirements.txt
```

### Step 2: Install Base Laya Engine
The benchmark evaluates the non-autoregressive decision model from the Laya repository:
```bash
git clone https://github.com/NandhaKishorM/laya.git ../laya
pip install -e ../laya
```

### Step 3: Run Invariant Unit Tests
Before launching the benchmark, verify metric invariants, nesting assertions, and ECE reference calculations:
```bash
pytest -v experiments/tests/test_marginalization_invariants.py
```
*Expected: 6/6 tests passing in under 2 seconds.*

### Step 4: Run Smoke Test (Optional Fast Check)
Verify pipeline integration on 10 queries across all cardinalities:
```bash
python experiments/bench_marginalization.py --smoke-test 10
```

### Step 5: Execute Full Benchmark ($N=120$)
Run the complete evaluation across all 5 cardinalities:
```bash
python experiments/bench_marginalization.py
```
*Expected Runtime: ~60–75 minutes on Apple Silicon M-series GPU (MPS).*

### Step 6: Generate Publication Figures & Paper Tables
Recompute all publication figures (DPI 300) and structured tables:
```bash
python experiments/generate_publication_artifacts.py
```
Output artifacts will be saved in `results/figures/`, `paper/figures/`, and `paper/tables/`.
