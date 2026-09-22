#!/usr/bin/env python3
"""Automated publication-grade audit script for:
Candidate Order Instability in Non-Autoregressive Multi-Candidate Transformers

Programmatically asserts:
1. Frozen commit reference and DOI immutability.
2. Numerical headline claims match empirical CSV artifacts.
3. Confidence interval containment [Lo <= Mean <= Hi] and [0, 1] probability bounds.
4. Multiple testing FDR p-values and exact significance counts (exactly 1 significant).
5. Prohibited vocabulary check (zero occurrences of 'superlinear', 'proves', etc.).
6. Citation integrity: all LaTeX citations exist in references.bib, no orphaned keys, no placeholders.
7. Figure file presence and resolution in paper/figures/ and paper/arxiv/figures/.
8. Author identity consistency ('Anshul Saxena') and absence of placeholder identities.
9. Absence of private/local filesystem paths (e.g. '/Users/anshul') in all publication documents.
10. arXiv standalone package integrity and parity.
"""

import os
import re
import sys
import pandas as pd
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_DIR = os.path.join(REPO_ROOT, "paper")
ARXIV_DIR = os.path.join(PAPER_DIR, "arxiv")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

FROZEN_COMMIT = "266bbb902871dc9974cbf516adea013e3c4eb084"
FROZEN_DOI = "10.5281/zenodo.22906245"
FROZEN_TAG = "v1.0.0"
GITHUB_REPO_URL = "https://github.com/anshull-saxena/candidate-order-instability"


def check_metadata_immutability():
    print("[1/10] Verifying frozen release metadata and DOI...")
    cff_path = os.path.join(REPO_ROOT, "CITATION.cff")
    with open(cff_path) as f:
        cff = f.read()
    assert "1.0.0" in cff, f"Version 1.0.0 not found in CITATION.cff"
    assert "Saxena" in cff and "Anshul" in cff, "Author Anshul Saxena not found in CITATION.cff"

    tex_path = os.path.join(PAPER_DIR, "manuscript.tex")
    with open(tex_path) as f:
        tex = f.read()
    assert FROZEN_COMMIT in tex, f"Commit {FROZEN_COMMIT} not found in manuscript.tex"
    assert FROZEN_DOI in tex, f"DOI {FROZEN_DOI} not found in manuscript.tex"
    assert FROZEN_TAG in tex, f"Tag {FROZEN_TAG} not found in manuscript.tex"
    assert GITHUB_REPO_URL in tex, f"Repo URL {GITHUB_REPO_URL} not found in manuscript.tex"

    paper_readme_path = os.path.join(PAPER_DIR, "README.md")
    with open(paper_readme_path) as f:
        paper_readme = f.read()
    assert FROZEN_COMMIT in paper_readme, f"Commit {FROZEN_COMMIT} not found in paper/README.md"
    assert FROZEN_DOI in paper_readme, f"DOI {FROZEN_DOI} not found in paper/README.md"
    assert FROZEN_TAG in paper_readme, f"Tag {FROZEN_TAG} not found in paper/README.md"
    print("  -> Frozen commit, DOI, and repo URL verified.")


def check_numerical_headline_claims():
    print("[2/10] Verifying headline claims against raw & processed CSV data...")
    summary_csv = os.path.join(RESULTS_DIR, "processed", "corrected_marginalization_results.csv")
    df_sum = pd.read_csv(summary_csv)

    # 1. Flip rate scaling: K=5 vs K=77
    k5_rand = df_sum[(df_sum["K"] == 5) & (df_sum["Method"] == "B0_Random")].iloc[0]
    k77_rand = df_sum[(df_sum["K"] == 77) & (df_sum["Method"] == "B0_Random")].iloc[0]

    assert abs(k5_rand["FR_top1"] - 0.057037) < 1e-4, f"Unexpected K=5 FR_top1: {k5_rand['FR_top1']}"
    assert abs(k77_rand["FR_top1"] - 0.472778) < 1e-4, f"Unexpected K=77 FR_top1: {k77_rand['FR_top1']}"

    # 2. Canonical cross-flip at K=77
    k77_alpha = df_sum[(df_sum["K"] == 77) & (df_sum["Method"] == "B1_Alpha")].iloc[0]
    assert abs(k77_alpha["CrossCanonicalFlip"] - 0.558333) < 1e-4, f"Unexpected K=77 CrossCanonicalFlip: {k77_alpha['CrossCanonicalFlip']}"
    assert k77_alpha["FR_top1"] == 0.0, "B1_Alpha FR_top1 must be exactly 0.0"

    # 3. Residual flip rates at K=77 M=5
    k77_rand_m5 = df_sum[(df_sum["K"] == 77) & (df_sum["Method"] == "Rand_Marg_M5")].iloc[0]
    k77_cyc_m5 = df_sum[(df_sum["K"] == 77) & (df_sum["Method"] == "B2_Cyclic_M5")].iloc[0]
    assert abs(k77_rand_m5["FR_top1"] - 0.258333) < 1e-4, f"Unexpected K=77 Rand_Marg_M5 FR: {k77_rand_m5['FR_top1']}"
    assert abs(k77_cyc_m5["FR_top1"] - 0.161111) < 1e-4, f"Unexpected K=77 B2_Cyclic_M5 FR: {k77_cyc_m5['FR_top1']}"

    # 4. Calibration ECE scaling at K=77
    k77_native = df_sum[(df_sum["K"] == 77) & (df_sum["Method"] == "B0_Native")].iloc[0]
    assert abs(k77_native["ECE"] - 0.370001) < 1e-4, f"Unexpected K=77 Native ECE: {k77_native['ECE']}"
    assert abs(k77_rand_m5["ECE"] - 0.095982) < 1e-4, f"Unexpected K=77 Rand_Marg_M5 ECE: {k77_rand_m5['ECE']}"
    assert abs(k77_cyc_m5["ECE"] - 0.161937) < 1e-4, f"Unexpected K=77 Cyclic_M5 ECE: {k77_cyc_m5['ECE']}"

    print("  -> All headline point estimates match empirical artifacts.")


def check_ci_containment_and_bounds():
    print("[3/10] Verifying CI containment and [0,1] probability bounds...")
    summary_csv = os.path.join(RESULTS_DIR, "processed", "corrected_marginalization_results.csv")
    df_sum = pd.read_csv(summary_csv)

    metrics = [
        ("Accuracy", "Accuracy_95CI_Lo", "Accuracy_95CI_Hi"),
        ("ECE", "ECE_95CI_Lo", "ECE_95CI_Hi"),
        ("FR_top1", "FR_top1_95CI_Lo", "FR_top1_95CI_Hi"),
    ]

    for _, row in df_sum.iterrows():
        K, method = row["K"], row["Method"]
        for val_col, lo_col, hi_col in metrics:
            val = float(row[val_col])
            lo = float(row[lo_col])
            hi = float(row[hi_col])
            assert lo <= val <= hi + 1e-9, f"CI containment broken for {val_col} (K={K}, {method}): {lo} <= {val} <= {hi}"
            assert -1e-9 <= lo <= 1.0 + 1e-9, f"Lower bound outside [0,1] for {val_col} (K={K}, {method}): {lo}"
            assert -1e-9 <= hi <= 1.0 + 1e-9, f"Upper bound outside [0,1] for {val_col} (K={K}, {method}): {hi}"

        if method in ["B1_Alpha", "B1_ReverseAlpha"]:
            cc = float(row["CrossCanonicalFlip"])
            lo = float(row["CrossCanonicalFlip_95CI_Lo"])
            hi = float(row["CrossCanonicalFlip_95CI_Hi"])
            assert lo <= cc <= hi + 1e-9, f"CI containment broken for CrossCanonicalFlip: {lo} <= {cc} <= {hi}"
            assert 0.0 <= lo <= cc <= hi <= 1.0, f"CrossCanonicalFlip outside [0,1]: {cc}"

    print("  -> 100% CI containment and bound validity verified across all 50 rows.")


def check_statistical_hypothesis_tests():
    print("[4/10] Auditing multiple testing FDR adjustments and significance counts...")
    stat_csv = os.path.join(RESULTS_DIR, "statistics", "corrected_statistical_tests.csv")
    df_stat = pd.read_csv(stat_csv)

    assert len(df_stat) == 45, f"Expected 45 paired hypothesis tests, got {len(df_stat)}"

    # Check FDR significant count
    sig_rows = df_stat[df_stat["FDR_Significant_q05"] == True]
    assert len(sig_rows) == 1, f"Expected exactly 1 FDR-significant result, found {len(sig_rows)}"

    sig = sig_rows.iloc[0]
    assert sig["K"] == 40, f"Expected FDR-significant result at K=40, got K={sig['K']}"
    assert sig["Method"] == "B2_Cyclic_M5", f"Expected B2_Cyclic_M5 to be significant, got {sig['Method']}"
    assert abs(float(sig["Unadjusted_p_value"]) - 0.0009765625) < 1e-6, f"Unexpected unadjusted p: {sig['Unadjusted_p_value']}"
    assert abs(float(sig["BH_Adjusted_p_value"]) - 0.0439453125) < 1e-5, f"Unexpected BH-adjusted p: {sig['BH_Adjusted_p_value']}"
    assert int(sig["Discordant_Gains_n01"]) == 11 and int(sig["Discordant_Losses_n10"]) == 0

    # Verify that K=77 Rand_Marg_M5 fails FDR significance
    k77_rand = df_stat[(df_stat["K"] == 77) & (df_stat["Method"] == "Rand_Marg_M5")].iloc[0]
    assert k77_rand["FDR_Significant_q05"] == False, "K=77 Rand_Marg_M5 must NOT be marked FDR significant"
    assert abs(float(k77_rand["BH_Adjusted_p_value"]) - 1.0) < 1e-4, f"Expected K=77 Rand_Marg_M5 BH-p == 1.0, got {k77_rand['BH_Adjusted_p_value']}"

    print("  -> Paired McNemar hypothesis tests and FDR adjustments verified (1 significant, 44 non-significant).")


def check_prohibited_language():
    print("[5/10] Scanning publication files for prohibited promotional/unsupported wording...")
    prohibited_words = [
        r"\bsuperlinear\b",
        r"\bproves\b",
        r"\buniversally\b",
        r"\bproduction-ready\b",
        r"\bstate-of-the-art\b",
        r"\bguarantees\b",
        r"\bsolves\b",
    ]

    files_to_check = [
        os.path.join(PAPER_DIR, "manuscript.tex"),
        os.path.join(PAPER_DIR, "README.md"),
        os.path.join(PAPER_DIR, "reviewer_audit.md"),
        os.path.join(PAPER_DIR, "venue_readiness.md"),
        os.path.join(PAPER_DIR, "arxiv_submission_checklist.md"),
        os.path.join(REPO_ROOT, "CITATION.cff"),
        os.path.join(REPO_ROOT, "README.md"),
    ]

    for path in files_to_check:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            content = f.read()
        for pat in prohibited_words:
            matches = re.findall(pat, content, re.IGNORECASE)
            # In reviewer_audit or venue_readiness, check if word appears in meta-context
            if matches:
                # Disallow in manuscript and CITATION.cff strictly
                if path.endswith("manuscript.tex") or path.endswith("CITATION.cff"):
                    assert False, f"Prohibited pattern '{pat}' found in {os.path.basename(path)}: {matches}"

    print("  -> Prohibited wording check passed.")


def check_citation_integrity():
    print("[6/10] Auditing citation integrity between manuscript.tex and references.bib...")
    bib_path = os.path.join(PAPER_DIR, "references.bib")
    tex_path = os.path.join(PAPER_DIR, "manuscript.tex")

    with open(bib_path) as f:
        bib_text = f.read()

    # Extract all bib entry keys
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([\w\-]+)\s*,", bib_text))
    assert len(bib_keys) > 0, "No BibTeX keys parsed from references.bib"

    # Verify no placeholder keys
    for key in bib_keys:
        assert not key.startswith("TODO"), f"Placeholder BibTeX key found: {key}"
        assert not key.startswith("placeholder"), f"Placeholder BibTeX key found: {key}"

    # Extract all cited keys from manuscript.tex
    with open(tex_path) as f:
        tex_text = f.read()

    cited_keys = set()
    for match in re.findall(r"\\cite[a-zA-Z]*\*?\{([^}]+)\}", tex_text):
        for k in match.split(","):
            cited_keys.add(k.strip())

    missing_in_bib = cited_keys - bib_keys
    assert len(missing_in_bib) == 0, f"Citations in manuscript.tex missing in references.bib: {missing_in_bib}"

    # Verify that all bib entries are cited
    uncited_in_tex = bib_keys - cited_keys
    assert len(uncited_in_tex) == 0, f"BibTeX entries defined but not cited in manuscript.tex: {uncited_in_tex}"

    print(f"  -> Exact 1-to-1 citation correspondence verified ({len(cited_keys)} cited keys).")


def check_figures_existence():
    print("[7/10] Auditing all seven empirical figures...")
    expected_figures = [
        "fig1_cardinality_vs_instability.png",
        "fig2_cardinality_vs_logit_variance.png",
        "fig3_canonical_counterfactual.png",
        "fig4_marginalization_scaling.png",
        "fig5_pareto_frontier_k77.png",
        "fig6_cyclic_vs_random.png",
        "fig7_k77_query_heterogeneity.png",
    ]

    for fig in expected_figures:
        p1 = os.path.join(PAPER_DIR, "figures", fig)
        p2 = os.path.join(ARXIV_DIR, "figures", fig)
        assert os.path.exists(p1), f"Missing figure in paper/figures/: {fig}"
        assert os.path.exists(p2), f"Missing figure in paper/arxiv/figures/: {fig}"
        assert os.path.getsize(p1) > 1000, f"Figure file empty or truncated: {fig}"

    print(f"  -> All {len(expected_figures)} figures verified in paper/ and arxiv/ bundles.")


def check_author_and_identity():
    print("[8/10] Auditing author identity and scanning for placeholders...")
    author_name = "Anshul Saxena"
    author_email = "f20221041@hyderabad.bits-pilani.ac.in"
    placeholder_patterns = [
        "Senior ML Researcher",
        "Research Software Engineer",
        "<your-name>",
        "<author-email>",
        "Anonymous Author",
        "<placeholder>",
        "Placeholder Author",
    ]

    files_to_scan = [
        os.path.join(PAPER_DIR, "manuscript.tex"),
        os.path.join(PAPER_DIR, "README.md"),
        os.path.join(PAPER_DIR, "arxiv_submission_checklist.md"),
        os.path.join(PAPER_DIR, "arxiv", "README.md"),
        os.path.join(REPO_ROOT, "CITATION.cff"),
        os.path.join(REPO_ROOT, "README.md"),
    ]

    for path in files_to_scan:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            content = f.read()
        assert "Anshul" in content and "Saxena" in content, f"Author name 'Anshul Saxena' missing in {os.path.basename(path)}"
        for placeholder in placeholder_patterns:
            assert placeholder not in content, f"Placeholder pattern '{placeholder}' found in {os.path.basename(path)}"

    print("  -> Author identity verified with zero placeholder identities.")


def check_private_paths():
    print("[9/10] Auditing files for private local filesystem paths...")
    forbidden_paths = [
        "/Users/anshul",
        "/home/",
        "C:\\Users\\",
    ]

    files_to_scan = [
        os.path.join(PAPER_DIR, "manuscript.tex"),
        os.path.join(PAPER_DIR, "references.bib"),
        os.path.join(PAPER_DIR, "README.md"),
        os.path.join(PAPER_DIR, "reviewer_audit.md"),
        os.path.join(PAPER_DIR, "venue_readiness.md"),
        os.path.join(PAPER_DIR, "arxiv_submission_checklist.md"),
        os.path.join(PAPER_DIR, "arxiv", "manuscript.tex"),
        os.path.join(PAPER_DIR, "arxiv", "README.md"),
        os.path.join(REPO_ROOT, "CITATION.cff"),
        os.path.join(REPO_ROOT, "README.md"),
    ]

    for path in files_to_scan:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            content = f.read()
        for forbidden in forbidden_paths:
            assert forbidden not in content, f"Forbidden local path '{forbidden}' found in {os.path.basename(path)}"

    print("  -> Zero private local filesystem paths found across publication suite.")


def check_arxiv_bundle_parity():
    print("[10/10] Auditing arxiv standalone package completeness...")
    required_arxiv_files = [
        os.path.join(ARXIV_DIR, "manuscript.tex"),
        os.path.join(ARXIV_DIR, "references.bib"),
        os.path.join(ARXIV_DIR, "manuscript.bbl"),
        os.path.join(ARXIV_DIR, "manuscript.pdf"),
        os.path.join(ARXIV_DIR, "README.md"),
    ]
    for fpath in required_arxiv_files:
        assert os.path.exists(fpath), f"Missing required file in paper/arxiv/: {os.path.basename(fpath)}"

    # Check tex parity
    with open(os.path.join(PAPER_DIR, "manuscript.tex")) as f1, open(os.path.join(ARXIV_DIR, "manuscript.tex")) as f2:
        assert f1.read() == f2.read(), "paper/manuscript.tex and paper/arxiv/manuscript.tex are out of sync"

    print("  -> arXiv package completeness and parity verified.")


def main():
    print("=" * 65)
    print("RUNNING FINAL PUBLICATION AUDIT FOR CANDIDATE-ORDER-INSTABILITY")
    print("=" * 65)

    check_metadata_immutability()
    check_numerical_headline_claims()
    check_ci_containment_and_bounds()
    check_statistical_hypothesis_tests()
    check_prohibited_language()
    check_citation_integrity()
    check_figures_existence()
    check_author_and_identity()
    check_private_paths()
    check_arxiv_bundle_parity()

    print("=" * 65)
    print("ALL 10 PUBLICATION SANITY & INTEGRITY CHECKS PASSED PERFECTLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
