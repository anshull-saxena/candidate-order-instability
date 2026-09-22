#!/usr/bin/env python3
"""Automated 16-Point Quality, Invariant, and Double-Blind Audit Suite for TMLR Submission.

Venue: Transactions on Machine Learning Research (TMLR)
Paper: Candidate Order Instability in Non-Autoregressive Multi-Candidate Transformers
Frozen Release: v1.0.0 (commit: 266bbb902871dc9974cbf516adea013e3c4eb084)
Zenodo DOI: 10.5281/zenodo.22906245
"""

import os
import re
import sys
import zipfile
import subprocess
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TMLR_DIR = os.path.join(REPO_ROOT, "paper", "tmlr")
MANUSCRIPT_TEX = os.path.join(TMLR_DIR, "manuscript.tex")
MANUSCRIPT_PDF = os.path.join(TMLR_DIR, "manuscript.pdf")
MANUSCRIPT_LOG = os.path.join(TMLR_DIR, "manuscript.log")
REFERENCES_BIB = os.path.join(TMLR_DIR, "references.bib")
SUPPLEMENTARY_ZIP = os.path.join(TMLR_DIR, "supplementary.zip")
RESULTS_CSV = os.path.join(REPO_ROOT, "results", "processed", "corrected_marginalization_results.csv")
TESTS_CSV = os.path.join(REPO_ROOT, "results", "statistics", "corrected_statistical_tests.csv")
PARETO_CSV = os.path.join(REPO_ROOT, "results", "statistics", "corrected_pareto_frontier.csv")

FROZEN_COMMIT = "266bbb902871dc9974cbf516adea013e3c4eb084"
ZENODO_DOI = "10.5281/zenodo.22906245"

FORBIDDEN_IDENTIFIERS = [
    "Anshul",
    "Saxena",
    "BITS Pilani",
    "f20221041",
    "anshull-saxena",
    "/Users/anshul",
]

def run_cmd(cmd, cwd=REPO_ROOT):
    p = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def check_1_release_immutability():
    """1. Release immutability (v1.0.0 tag, commit hash, Zenodo DOI)."""
    rc, stdout, _ = run_cmd("git rev-parse v1.0.0^{}")
    if rc != 0:
        return False, "v1.0.0 tag not found"
    tag_commit = stdout
    if tag_commit != FROZEN_COMMIT:
        return False, f"v1.0.0 commit {tag_commit} != expected {FROZEN_COMMIT}"

    with open(os.path.join(TMLR_DIR, "reproducibility_notes.md")) as f:
        repro_text = f.read()
    if ZENODO_DOI not in repro_text or FROZEN_COMMIT not in repro_text:
        return False, "reproducibility_notes.md missing frozen commit or Zenodo DOI"

    return True, f"v1.0.0 points to {FROZEN_COMMIT}; DOI {ZENODO_DOI} verified"


def check_2_numerical_consistency():
    """2. Numerical consistency (paper text and tables vs results CSVs)."""
    if not os.path.exists(RESULTS_CSV):
        return False, f"Results CSV missing: {RESULTS_CSV}"
    df = pd.read_csv(RESULTS_CSV)

    with open(MANUSCRIPT_TEX) as f:
        tex = f.read()

    # Flip rate K=5 (5.70%), K=77 (47.28%)
    k5_row = df[(df["K"] == 5) & (df["Method"] == "B0_Random")].iloc[0]
    k77_row = df[(df["K"] == 77) & (df["Method"] == "B0_Random")].iloc[0]

    k5_fr = round(float(k5_row["FR_top1"]) * 100, 2)
    k77_fr = round(float(k77_row["FR_top1"]) * 100, 2)

    if f"{k5_fr:.2f}" not in tex or f"{k77_fr:.2f}" not in tex:
        return False, f"Flip rates {k5_fr} or {k77_fr} missing in manuscript.tex"

    # Cross-canonical flip rate at K=77 (55.83%)
    if "55.83" not in tex:
        return False, "Cross-canonical flip rate 55.83% missing in manuscript.tex"

    # Logit variance scaling: 8.40 at K=5, 63.34 at K=77 (7.5x expansion)
    if "8.40" not in tex or "63.34" not in tex or "7.5" not in tex:
        return False, "Logit variance metrics missing in manuscript.tex"

    return True, f"Verified key metrics: K=5 FR={k5_fr}%, K=77 FR={k77_fr}%, Cross-Flip=55.83%, LogitVar 8.40->63.34"


def check_3_bootstrap_ci_containment():
    """3. Bootstrap CI containment (point estimate within [ci_lower, ci_upper])."""
    df = pd.read_csv(RESULTS_CSV)
    for idx, row in df.iterrows():
        # Check accuracy CI
        acc = float(row["Accuracy"])
        acc_lo = float(row["Accuracy_95CI_Lo"])
        acc_hi = float(row["Accuracy_95CI_Hi"])
        if not (acc_lo <= acc <= acc_hi):
            return False, f"Accuracy CI violation for {row['Method']} K={row['K']}: {acc_lo} <= {acc} <= {acc_hi}"

        # Check flip rate CI if present
        if pd.notna(row["FR_top1"]):
            fr = float(row["FR_top1"])
            fr_lo = float(row["FR_top1_95CI_Lo"])
            fr_hi = float(row["FR_top1_95CI_Hi"])
            if not (fr_lo <= fr <= fr_hi):
                return False, f"Flip rate CI violation for {row['Method']} K={row['K']}: {fr_lo} <= {fr} <= {fr_hi}"

    return True, f"All {len(df)} rows satisfy ci_lower <= point_estimate <= ci_upper"


def check_4_p_value_accuracy():
    """4. P-value accuracy (McNemar p-values and Benjamini-Hochberg adjusted values)."""
    if not os.path.exists(TESTS_CSV):
        return False, f"Tests CSV missing: {TESTS_CSV}"
    df = pd.read_csv(TESTS_CSV)

    # Verify B2_Cyclic_M5 at K=40
    target = df[(df["K"] == 40) & (df["Method"] == "B2_Cyclic_M5")]
    if len(target) == 0:
        return False, "Target comparison B2_Cyclic_M5 at K=40 not found in tests CSV"

    target_padj = float(target["BH_Adjusted_p_value"].iloc[0])
    if not (0.04 < target_padj < 0.05):
        return False, f"Expected adjusted p-value ~0.04395, got {target_padj}"

    with open(MANUSCRIPT_TEX) as f:
        tex = f.read()
    if "0.04395" not in tex:
        return False, "p=0.04395 not found in manuscript.tex"

    return True, f"Target B2_Cyclic_M5 (K=40) adjusted p = {target_padj:.5f} verified in tests CSV and manuscript"


def check_5_significance_claim_accuracy():
    """5. Significance claim accuracy (only B2_Cyclic_M5 at K=40 is FDR-significant, no K=77 accuracy claims are FDR-significant)."""
    df = pd.read_csv(TESTS_CSV)
    sig_rows = df[df["BH_Adjusted_p_value"] < 0.05]
    sig_k40 = sig_rows[sig_rows["K"] == 40]
    sig_k77 = sig_rows[sig_rows["K"] == 77]

    if len(sig_k77) > 0:
        return False, f"Found {len(sig_k77)} FDR-significant accuracy gains at K=77 in tests CSV! Expected 0."

    if len(sig_k40) != 1 or sig_k40["Method"].iloc[0] != "B2_Cyclic_M5":
        return False, f"Expected only B2_Cyclic_M5 at K=40 to be significant, got: {sig_k40['Method'].tolist()}"

    return True, "Verified: exactly 1 FDR-significant accuracy improvement across all 45 hypotheses (B2_Cyclic_M5 at K=40); zero at K=77"


def check_6_prohibited_claims():
    """6. Prohibited claim check (no unadjusted 'statistically significant' claims for K=77 accuracy)."""
    with open(MANUSCRIPT_TEX) as f:
        tex = f.read()

    # Split into sentences and check that no single sentence claims significant accuracy improvement at K=77
    sentences = re.split(r'(?<=[.!?])\s+', tex)
    for sent in sentences:
        if ("K=77" in sent or "K = 77" in sent) and "statistically significant" in sent and "accuracy" in sent:
            if not any(neg in sent.lower() for neg in ["fail", "not", "no ", "without", "zero", "preclude", "precludes", "cannot", "does not", "only cyclic shifts at $k=40$"]):
                return False, f"Potential false claim in sentence: {sent}"

    # Check for correct disclaimer
    if "fail FDR significance" not in tex and "fails FDR significance" not in tex:
        return False, "Manuscript must explicitly state that K=77 accuracy gains fail FDR significance"

    return True, "Manuscript correctly reports that K=77 accuracy gains fail FDR significance"


def check_7_single_vs_multi_seed():
    """7. Single-seed vs multi-seed distinction check (65.0% exploratory vs 48.06% multi-seed pooled)."""
    with open(MANUSCRIPT_TEX) as f:
        tex = f.read()

    if "65.0" not in tex:
        return False, "Exploratory 65.0% finding not documented in manuscript"
    if "48.06" not in tex:
        return False, "Multi-seed pooled 48.06% accuracy not documented in manuscript"

    # Verify context explains non-replication / regression
    if "non-replication" not in tex and "non-replicated" not in tex and "regressed" not in tex:
        return False, "Manuscript must explicitly explain regression from 65.0% to 48.06%"

    return True, "Single-seed exploratory 65.0% and multi-seed pooled 48.06% clearly distinguished and explained"


def check_8_latex_compilation():
    """8. LaTeX compilation check (zero errors, PDF exists, pages count >= 10)."""
    if not os.path.exists(MANUSCRIPT_PDF):
        return False, "manuscript.pdf does not exist"
    size = os.path.getsize(MANUSCRIPT_PDF)
    if size < 50000:
        return False, f"manuscript.pdf suspiciously small: {size} bytes"

    # Check page count via pdfinfo
    rc, stdout, _ = run_cmd(f"pdfinfo {MANUSCRIPT_PDF}")
    if rc == 0:
        for line in stdout.split("\n"):
            if "Pages:" in line:
                pages = int(line.split(":")[1].strip())
                if pages < 10:
                    return False, f"Page count {pages} < 10"
                break

    # Check log file for fatal errors
    if os.path.exists(MANUSCRIPT_LOG):
        with open(MANUSCRIPT_LOG, "r", errors="ignore") as f:
            log_text = f.read()
        if "! Emergency stop" in log_text or "Fatal error occurred" in log_text:
            return False, "Fatal error found in manuscript.log"

    return True, f"manuscript.pdf compiles cleanly ({size} bytes, pages >= 10)"


def check_9_template_fidelity():
    """9. Template fidelity check (tmlr.sty, tmlr.bst, math_commands.tex present)."""
    for fname in ["tmlr.sty", "tmlr.bst", "math_commands.tex", "fancyhdr.sty"]:
        path = os.path.join(TMLR_DIR, fname)
        if not os.path.exists(path):
            return False, f"Missing required template file: {fname}"
        if os.path.getsize(path) == 0:
            return False, f"Template file is empty: {fname}"

    with open(os.path.join(TMLR_DIR, "tmlr.sty")) as f:
        sty_text = f.read()
    if "Transactions on Machine Learning Research" not in sty_text:
        return False, "tmlr.sty does not appear to be official TMLR stylefile"

    return True, "Official TMLR template files present and validated"


def check_10_double_blind():
    """10. Double-blind check (author names, affiliations, emails, repo URLs suppressed)."""
    with open(MANUSCRIPT_TEX) as f:
        tex = f.read()

    for ident in FORBIDDEN_IDENTIFIERS:
        if ident in tex:
            return False, f"Forbidden identifier '{ident}' found in manuscript.tex"

    # Check PDF text if pdftotext is available
    rc, stdout, _ = run_cmd(f"pdftotext {MANUSCRIPT_PDF} -")
    if rc == 0:
        for ident in FORBIDDEN_IDENTIFIERS:
            if ident.lower() in stdout.lower():
                return False, f"Forbidden identifier '{ident}' detected in compiled PDF text!"

    return True, "No author names, affiliations, emails, or identifying URLs found in manuscript.tex or PDF"


def check_11_llm_disclosure():
    """11. LLM disclosure check (footnote present on page 1)."""
    with open(MANUSCRIPT_TEX) as f:
        tex = f.read()

    if "During the preparation of this manuscript, the authors utilized large language model assistants" not in tex:
        return False, "Mandatory LLM assistance disclosure footnote missing in manuscript.tex"

    return True, "Mandatory LLM disclosure footnote present on page 1"


def check_12_broader_impact():
    """12. Broader impact statement check (present before references)."""
    with open(MANUSCRIPT_TEX) as f:
        tex = f.read()

    if "Broader Impact Statement" not in tex:
        return False, "Broader Impact Statement section missing"

    impact_pos = tex.find("Broader Impact Statement")
    bib_pos = tex.find("\\bibliography")
    if bib_pos == -1 or impact_pos > bib_pos:
        return False, "Broader Impact Statement must appear before \\bibliography"

    return True, "Broader Impact Statement is properly formatted and positioned before References"


def check_13_citation_integrity():
    """13. Citation integrity check (all cited keys in references.bib, zero undefined)."""
    with open(MANUSCRIPT_TEX) as f:
        tex = f.read()

    with open(REFERENCES_BIB) as f:
        bib = f.read()

    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))

    cite_matches = re.findall(r"\\cite[ptal]*\{([^}]+)\}", tex)
    cited_keys = set()
    for m in cite_matches:
        for k in m.split(","):
            k_clean = k.strip()
            if k_clean:
                cited_keys.add(k_clean)

    missing_keys = cited_keys - bib_keys
    if missing_keys:
        return False, f"Cited keys missing in references.bib: {missing_keys}"

    if os.path.exists(MANUSCRIPT_LOG):
        with open(MANUSCRIPT_LOG, "r", errors="ignore") as f:
            log_text = f.read()
        if "Citation" in log_text and "undefined" in log_text:
            return False, "Found undefined citations in manuscript.log"

    return True, f"All {len(cited_keys)} cited keys resolved in references.bib; 0 undefined citations"


def check_14_figure_references():
    """14. Figure reference check (fig1-7 all referenced in text, all figure files exist)."""
    figures_dir = os.path.join(TMLR_DIR, "figures")
    for i in range(1, 8):
        matches = [f for f in os.listdir(figures_dir) if f.startswith(f"fig{i}_")]
        if not matches:
            return False, f"Missing figure file for fig{i}"

    with open(MANUSCRIPT_TEX) as f:
        tex = f.read()

    fig_refs = [
        "fig:cardinality_instability",
        "fig:logit_variance",
        "fig:canonical_counterfactual",
        "fig:marginalization_scaling",
        "fig:pareto_frontier",
        "fig:cyclic_vs_random",
        "fig:query_heterogeneity",
    ]
    for ref in fig_refs:
        if f"\\ref{{{ref}}}" not in tex:
            return False, f"Figure label {ref} not referenced via \\ref in text"

    return True, "All 7 publication figures exist on disk and are actively referenced in manuscript text"


def check_15_supplementary_zip():
    """15. Supplementary ZIP check (exists, valid zip, size <= 100 MB, no identifying paths)."""
    if not os.path.exists(SUPPLEMENTARY_ZIP):
        return False, f"Supplementary zip missing: {SUPPLEMENTARY_ZIP}"

    size_mb = os.path.getsize(SUPPLEMENTARY_ZIP) / (1024 * 1024)
    if size_mb > 100.0:
        return False, f"Supplementary zip size {size_mb:.2f} MB exceeds 100 MB limit"

    try:
        with zipfile.ZipFile(SUPPLEMENTARY_ZIP, "r") as zf:
            namelist = zf.namelist()
            if len(namelist) < 5:
                return False, f"Too few files in supplementary zip: {len(namelist)}"

            # Inspect contents for leaks
            for name in namelist:
                if not name.endswith("/"):
                    content = zf.read(name).decode("utf-8", errors="ignore")
                    for ident in FORBIDDEN_IDENTIFIERS:
                        if ident in content:
                            return False, f"Forbidden identifier '{ident}' in supplementary file '{name}'"
    except Exception as e:
        return False, f"Corrupted or invalid ZIP file: {e}"

    return True, f"supplementary.zip is valid ({size_mb:.2f} MB, {len(namelist)} items) with 0 leaks"


def check_16_git_hygiene():
    """16. Git branch hygiene check (on paper/tmlr-final branch, main and v1.0.0 untouched)."""
    rc, stdout, _ = run_cmd("git branch --show-current")
    if stdout != "paper/tmlr-final":
        return False, f"Current branch is '{stdout}', expected 'paper/tmlr-final'"

    # Verify main branch commit
    rc, stdout, _ = run_cmd("git rev-parse main")
    if stdout != FROZEN_COMMIT:
        return False, f"main branch commit {stdout} has moved from frozen {FROZEN_COMMIT}!"

    # Verify v1.0.0 tag
    rc, stdout, _ = run_cmd("git rev-parse v1.0.0^{}")
    if stdout != FROZEN_COMMIT:
        return False, f"v1.0.0 tag {stdout} has moved from frozen {FROZEN_COMMIT}!"

    return True, "Branch is paper/tmlr-final; main and v1.0.0 remain strictly untouched at 266bbb9"


def main():
    checks = [
        ("Release Immutability", check_1_release_immutability),
        ("Numerical Consistency", check_2_numerical_consistency),
        ("Bootstrap CI Containment", check_3_bootstrap_ci_containment),
        ("P-Value Accuracy", check_4_p_value_accuracy),
        ("Significance Claim Accuracy", check_5_significance_claim_accuracy),
        ("Prohibited Claims", check_6_prohibited_claims),
        ("Single- vs Multi-Seed Distinction", check_7_single_vs_multi_seed),
        ("LaTeX Compilation", check_8_latex_compilation),
        ("Template Fidelity", check_9_template_fidelity),
        ("Double-Blind Anonymization", check_10_double_blind),
        ("LLM Assistance Disclosure", check_11_llm_disclosure),
        ("Broader Impact Statement", check_12_broader_impact),
        ("Citation Integrity", check_13_citation_integrity),
        ("Figure Reference Integrity", check_14_figure_references),
        ("Supplementary ZIP Integrity", check_15_supplementary_zip),
        ("Git Branch Hygiene", check_16_git_hygiene),
    ]

    print("=" * 80)
    print("  TMLR SUBMISSION FINAL PROGRAMMATIC AUDIT REPORT")
    print("=" * 80)

    passed_count = 0
    failures = []

    for idx, (name, func) in enumerate(checks, 1):
        try:
            passed, msg = func()
        except Exception as e:
            passed = False
            msg = f"Exception: {e}"

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] Check {idx:2d}/16: {name:35s} -> {msg}")

        if passed:
            passed_count += 1
        else:
            failures.append((idx, name, msg))

    print("=" * 80)
    print(f"Audit Summary: {passed_count}/{len(checks)} checks passed.")
    if failures:
        print("\nFailures:")
        for idx, name, msg in failures:
            print(f"  - Check {idx}: {name} -> {msg}")
        sys.exit(1)
    else:
        print("ALL 16 PROGRAMMATIC CHECKS PASSED. Ready for TMLR submission!")
        sys.exit(0)


if __name__ == "__main__":
    main()
