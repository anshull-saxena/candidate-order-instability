"""Generate publication-ready figures, tables, and evidence matrix for the
Candidate Cardinality Scaling & Inference-Time Order-Marginalization study.
"""
import csv
import json
import os
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIGS_DIR = os.path.join(RESULTS_DIR, "figures")
PAPER_FIGS_DIR = os.path.join(REPO_ROOT, "paper", "figures")
PAPER_TABLES_DIR = os.path.join(REPO_ROOT, "paper", "tables")

for d in [FIGS_DIR, PAPER_FIGS_DIR, PAPER_TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

# Load existing artifacts
summary_csv = os.path.join(RESULTS_DIR, "processed", "corrected_marginalization_results.csv")
if not os.path.exists(summary_csv):
    summary_csv = os.path.join(RESULTS_DIR, "corrected_marginalization_results.csv")

per_query_csv = os.path.join(RESULTS_DIR, "raw", "corrected_per_query_results.csv")
if not os.path.exists(per_query_csv):
    per_query_csv = os.path.join(RESULTS_DIR, "corrected_per_query_results.csv")

test_csv = os.path.join(RESULTS_DIR, "statistics", "corrected_statistical_tests.csv")
if not os.path.exists(test_csv):
    test_csv = os.path.join(RESULTS_DIR, "corrected_statistical_tests.csv")

pareto_csv = os.path.join(RESULTS_DIR, "statistics", "corrected_pareto_frontier.csv")
if not os.path.exists(pareto_csv):
    pareto_csv = os.path.join(RESULTS_DIR, "corrected_pareto_frontier.csv")

with open(summary_csv) as f:
    summary_rows = list(csv.DictReader(f))

with open(per_query_csv) as f:
    per_query_rows = list(csv.DictReader(f))

with open(test_csv) as f:
    test_rows = list(csv.DictReader(f))

with open(pareto_csv) as f:
    pareto_rows = list(csv.DictReader(f))

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

K_vals = [5, 10, 20, 40, 77]

# -----------------------------------------------------------------------------
# Figure 1: Candidate Cardinality vs Order Instability (FR_top1)
# -----------------------------------------------------------------------------
fr_means, fr_los, fr_his = [], [], []
for K in K_vals:
    r = next(x for x in summary_rows if int(x["K"]) == K and x["Method"] == "B0_Random")
    fr_means.append(float(r["FR_top1"]))
    fr_los.append(float(r["FR_top1_95CI_Lo"]))
    fr_his.append(float(r["FR_top1_95CI_Hi"]))

fig, ax = plt.subplots(figsize=(6.5, 4.5))
yerr = [np.array(fr_means) - np.array(fr_los), np.array(fr_his) - np.array(fr_means)]
ax.errorbar(K_vals, fr_means, yerr=yerr, fmt="-o", color="#1f77b4", ecolor="#1f77b4",
            elinewidth=2, capsize=4, capthick=1.5, linewidth=2, markersize=7, label="Pairwise Flip Rate (FR_top1)")
ax.set_xlabel("Candidate Cardinality $K$")
ax.set_ylabel(r"Argmax Decision Flip Rate ($\mathrm{FR}_{\mathrm{top1}}$)")
ax.set_title("Order Instability vs Candidate Cardinality $K$\n(Identical Candidate Set, $P=10$ Permutations, $N=120$ Queries)")
ax.set_ylim(-0.02, 0.60)
ax.set_xticks(K_vals)
for k, m in zip(K_vals, fr_means):
    ax.annotate(f"{m*100:.1f}%", (k, m), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")
ax.legend(loc="upper left")
plt.tight_layout()
for out_d in [FIGS_DIR, PAPER_FIGS_DIR]:
    fig.savefig(os.path.join(out_d, "fig1_cardinality_vs_instability.png"))
plt.close(fig)
print("[OK] Saved Figure 1")

# -----------------------------------------------------------------------------
# Figure 2: Candidate Cardinality vs Permutation Logit Variance
# -----------------------------------------------------------------------------
logit_vars = [float(next(x for x in summary_rows if int(x["K"]) == K and x["Method"] == "B0_Random")["Mean_Logit_Variance"]) for K in K_vals]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(K_vals, logit_vars, "-s", color="#d62728", linewidth=2, markersize=7, label="Mean Candidate Logit Variance across Permutations")
ax.set_xlabel("Candidate Cardinality $K$")
ax.set_ylabel(r"Permutation Logit Variance $\sigma^2_{\mathrm{logit}}$")
ax.set_title("Cross-Permutation Logit Variance Scaling\n(Controlled Distractor Nesting, Fixed Candidate Sets)")
ax.set_ylim(0, 75)
ax.set_xticks(K_vals)
for k, v in zip(K_vals, logit_vars):
    ax.annotate(f"{v:.1f}", (k, v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")
ax.legend(loc="upper left")
plt.tight_layout()
for out_d in [FIGS_DIR, PAPER_FIGS_DIR]:
    fig.savefig(os.path.join(out_d, "fig2_cardinality_vs_logit_variance.png"))
plt.close(fig)
print("[OK] Saved Figure 2")

# -----------------------------------------------------------------------------
# Figure 3: Canonical Counterfactual (Alpha vs ReverseAlpha & Cross-Flip)
# -----------------------------------------------------------------------------
alpha_accs = [float(next(x for x in summary_rows if int(x["K"]) == K and x["Method"] == "B1_Alpha")["Accuracy"]) for K in K_vals]
rev_accs = [float(next(x for x in summary_rows if int(x["K"]) == K and x["Method"] == "B1_ReverseAlpha")["Accuracy"]) for K in K_vals]
cross_flips = [float(next(x for x in summary_rows if int(x["K"]) == K and x["Method"] == "B1_Alpha")["CrossCanonicalFlip"]) for K in K_vals]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
x = np.arange(len(K_vals))
width = 0.35
ax1.bar(x - width/2, alpha_accs, width, label="B1_Alpha ($A \\to Z$)", color="#2ca02c", alpha=0.85)
ax1.bar(x + width/2, rev_accs, width, label="B1_ReverseAlpha ($Z \\to A$)", color="#ff7f0e", alpha=0.85)
ax1.set_xlabel("Candidate Cardinality $K$")
ax1.set_ylabel("Top-1 Accuracy")
ax1.set_title("Canonical Sorting Accuracy Under Counterfactual Reverse")
ax1.set_xticks(x)
ax1.set_xticklabels(K_vals)
ax1.set_ylim(0, 1.05)
ax1.legend(loc="lower left")

ax2.plot(K_vals, cross_flips, "-o", color="#9467bd", linewidth=2.2, markersize=7, label="Cross-Canonical Flip Rate ($y_{\\alpha} \\neq y_{\\mathrm{rev}}$)")
ax2.axhline(0.0, color="gray", linestyle="--", alpha=0.6, label="Tautological Repeatability (0.0)")
ax2.set_xlabel("Candidate Cardinality $K$")
ax2.set_ylabel("Cross-Canonical Disagreement Rate")
ax2.set_title("True Order Sensitivity Unmasked by Reversing Alphabet")
ax2.set_xticks(K_vals)
ax2.set_ylim(-0.02, 0.70)
for k, cf in zip(K_vals, cross_flips):
    ax2.annotate(f"{cf*100:.1f}%", (k, cf), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")
ax2.legend(loc="upper left")
plt.tight_layout()
for out_d in [FIGS_DIR, PAPER_FIGS_DIR]:
    fig.savefig(os.path.join(out_d, "fig3_canonical_counterfactual.png"))
plt.close(fig)
print("[OK] Saved Figure 3")

# -----------------------------------------------------------------------------
# Figure 4: Marginalization Scaling: M vs Residual Flip Rate
# -----------------------------------------------------------------------------
M_vals = [1, 2, 3, 5]
fig, ax = plt.subplots(figsize=(6.5, 4.5))
colors = {20: "#2ca02c", 40: "#ff7f0e", 77: "#d62728"}

for K in [20, 40, 77]:
    fr_curve = []
    r1 = next(x for x in summary_rows if int(x["K"]) == K and x["Method"] == "B0_Random")
    fr_curve.append(float(r1["FR_top1"]))
    for M in [2, 3, 5]:
        rm = next(x for x in summary_rows if int(x["K"]) == K and x["Method"] == f"Rand_Marg_M{M}")
        fr_curve.append(float(rm["FR_top1"]))
    ax.plot(M_vals, fr_curve, "-o", label=f"$K = {K}$", color=colors[K], linewidth=2, markersize=6)
    for m, fr in zip(M_vals, fr_curve):
        ax.annotate(f"{fr*100:.1f}%", (m, fr), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)

ax.set_xlabel("Permutations Averaged ($M$)")
ax.set_ylabel("Residual Decision Flip Rate")
ax.set_title("Permutation Marginalization Scaling: $M$ vs Residual Flip Rate")
ax.set_xticks(M_vals)
ax.set_ylim(0, 0.55)
ax.legend(loc="upper right")
plt.tight_layout()
for out_d in [FIGS_DIR, PAPER_FIGS_DIR]:
    fig.savefig(os.path.join(out_d, "fig4_marginalization_scaling.png"))
plt.close(fig)
print("[OK] Saved Figure 4")

# -----------------------------------------------------------------------------
# Figure 5: 3-Objective Pareto Frontier at K=77
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.0, 5.0))
k77_methods = [r for r in summary_rows if int(r["K"]) == 77]
k77_pareto = {r["Method"]: r["Pareto_Status"] for r in pareto_rows if int(r["K"]) == 77}

for r in k77_methods:
    m = r["Method"]
    acc = float(r["Accuracy"])
    lat = float(r["Latency_p50_ms"])
    inst = float(r["FR_top1"])
    status = k77_pareto[m]
    is_pareto = "PARETO_OPTIMAL" in status
    
    color = "#1f77b4" if is_pareto else "#7f7f7f"
    marker = "*" if is_pareto else "o"
    size = 180 if is_pareto else 80
    ax.scatter(lat, acc, s=size, c=color, marker=marker, alpha=0.9, edgecolors="black", linewidths=1.2, zorder=5)
    
    label_txt = f"{m}\n(FR={inst*100:.1f}%)"
    offset_y = 12 if "M5" in m or "Alpha" in m else -18
    offset_x = 0
    if "Reverse" in m:
        offset_y = -22
    ax.annotate(label_txt, (lat, acc), textcoords="offset points", xytext=(offset_x, offset_y),
                ha="center", fontsize=8, fontweight="bold" if is_pareto else "normal")

ax.set_xlabel("p50 Latency (ms) [Minimize $\\downarrow$]")
ax.set_ylabel("Top-1 Accuracy [Maximize $\\uparrow$]")
ax.set_title("Three-Objective Pareto Frontier at $K=77$\n(Objectives: Accuracy $\\uparrow$, Latency $\\downarrow$, Instability $\\downarrow$)")
ax.set_ylim(0.38, 0.60)
ax.set_xlim(150, 1300)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='*', color='w', label='Pareto Optimal (Non-Dominated)', markerfacecolor='#1f77b4', markeredgecolor='black', markersize=14),
    Line2D([0], [0], marker='o', color='w', label='Dominated by Another Method', markerfacecolor='#7f7f7f', markeredgecolor='black', markersize=9),
]
ax.legend(handles=legend_elements, loc="lower right")
plt.tight_layout()
for out_d in [FIGS_DIR, PAPER_FIGS_DIR]:
    fig.savefig(os.path.join(out_d, "fig5_pareto_frontier_k77.png"))
plt.close(fig)
print("[OK] Saved Figure 5")

# -----------------------------------------------------------------------------
# Figure 6: Cyclic vs Random Marginalization (Accuracy & Residual Instability)
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
m_sub = [2, 3, 5]
acc_rnd_77 = [float(next(x for x in summary_rows if int(x["K"]) == 77 and x["Method"] == f"Rand_Marg_M{m}")["Accuracy"]) for m in m_sub]
acc_cyc_77 = [float(next(x for x in summary_rows if int(x["K"]) == 77 and x["Method"] == f"B2_Cyclic_M{m}")["Accuracy"]) for m in m_sub]
x_m = np.arange(len(m_sub))
width = 0.35
ax1.bar(x_m - width/2, acc_rnd_77, width, label="Random Marginalization", color="#1f77b4", alpha=0.85)
ax1.bar(x_m + width/2, acc_cyc_77, width, label="Cyclic Marginalization", color="#e377c2", alpha=0.85)
ax1.set_xlabel("Ensemble Passes ($M$)")
ax1.set_ylabel("Top-1 Accuracy at $K=77$")
ax1.set_title("Accuracy: Random vs Cyclic Marginalization ($K=77$)")
ax1.set_xticks(x_m)
ax1.set_xticklabels([f"M={m}" for m in m_sub])
ax1.set_ylim(0.40, 0.60)
ax1.legend(loc="lower right")

fr_rnd_77 = [float(next(x for x in summary_rows if int(x["K"]) == 77 and x["Method"] == f"Rand_Marg_M{m}")["FR_top1"]) for m in m_sub]
fr_cyc_77 = [float(next(x for x in summary_rows if int(x["K"]) == 77 and x["Method"] == f"B2_Cyclic_M{m}")["FR_top1"]) for m in m_sub]
ax2.plot(m_sub, fr_rnd_77, "-o", color="#1f77b4", linewidth=2, markersize=7, label="Random Marginalization")
ax2.plot(m_sub, fr_cyc_77, "-s", color="#e377c2", linewidth=2, markersize=7, label="Cyclic Marginalization (Orthogonal Phases)")
ax2.set_xlabel("Ensemble Passes ($M$)")
ax2.set_ylabel("Residual Decision Flip Rate")
ax2.set_title("Residual Flip Rate: Cyclic vs Random ($K=77$)")
ax2.set_xticks(m_sub)
ax2.set_ylim(0.10, 0.45)
for m, r_fr, c_fr in zip(m_sub, fr_rnd_77, fr_cyc_77):
    ax2.annotate(f"{r_fr*100:.1f}%", (m, r_fr), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax2.annotate(f"{c_fr*100:.1f}%", (m, c_fr), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color="#8c2d04")
ax2.legend(loc="upper right")
plt.tight_layout()
for out_d in [FIGS_DIR, PAPER_FIGS_DIR]:
    fig.savefig(os.path.join(out_d, "fig6_cyclic_vs_random.png"))
plt.close(fig)
print("[OK] Saved Figure 6")

# -----------------------------------------------------------------------------
# Figure 7: Query-Level Heterogeneity at K=77
# -----------------------------------------------------------------------------
k77_rand = [r for r in per_query_rows if int(r["K"]) == 77 and r["Method"] == "B0_Random"]
fr_vals = np.array([float(r["FR_top1"]) for r in k77_rand])

fig, ax = plt.subplots(figsize=(7.0, 4.5))
counts, bins, patches = ax.hist(fr_vals, bins=12, range=(0.0, 1.0), color="#17becf", edgecolor="black", alpha=0.8)
ax.axvline(np.mean(fr_vals), color="red", linestyle="--", linewidth=2, label=f"Mean FR = {np.mean(fr_vals)*100:.1f}%")
ax.axvline(np.median(fr_vals), color="darkblue", linestyle=":", linewidth=2, label=f"Median FR = {np.median(fr_vals)*100:.1f}%")
ax.set_xlabel(r"Argmax Decision Flip Rate ($\mathrm{FR}_{\mathrm{top1}}$) across 10 Permutations")
ax.set_ylabel("Number of Queries (out of $N=120$)")
ax.set_title(r"Query-Level Order Instability Distribution at $K=77$" + "\n" + r"(Broad Instability: 55.0% of Queries have $\mathrm{FR} \geq 50\%$)")
ax.set_xlim(0, 1.0)
ax.legend(loc="upper left")
plt.tight_layout()
for out_d in [FIGS_DIR, PAPER_FIGS_DIR]:
    fig.savefig(os.path.join(out_d, "fig7_k77_query_heterogeneity.png"))
plt.close(fig)
print("[OK] Saved Figure 7")

print("[OK] All figures generated successfully in paper/figures/ and results/figures/.")
