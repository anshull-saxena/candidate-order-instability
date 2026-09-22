"""Candidate Cardinality Scaling & Inference-Time Order-Marginalization Benchmark.

Evaluation of non-autoregressive multi-candidate transformers under candidate order perturbations.
Evaluates:
  - Baseline B0_Native: Insertion order
  - Baseline B0_Random: Single random permutation
  - Baseline B1_Alpha: Alphabetical sorting
  - Baseline B1_ReverseAlpha: Reverse alphabetical sorting (counterfactual cross-order check)
  - Random Marginalization: M in {2, 3, 5}
  - Cyclic Marginalization: M in {2, 3, 5} with orthogonal cyclic phases
"""
import argparse
import csv
import hashlib
import json
import itertools
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from scipy import stats

# Path resolution
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
PROCESSED_DIR = os.path.join(RESULTS_DIR, "processed")
STATS_DIR = os.path.join(RESULTS_DIR, "statistics")
CONFIGS_DIR = os.path.join(REPO_ROOT, "experiments", "configs")

for d in [RAW_DIR, PROCESSED_DIR, STATS_DIR, CONFIGS_DIR]:
    os.makedirs(d, exist_ok=True)

OUT_SUMMARY_CSV = os.path.join(PROCESSED_DIR, "corrected_marginalization_results.csv")
OUT_PER_QUERY_CSV = os.path.join(RAW_DIR, "corrected_per_query_results.csv")
OUT_TESTS_CSV = os.path.join(STATS_DIR, "corrected_statistical_tests.csv")
OUT_PARETO_CSV = os.path.join(STATS_DIR, "corrected_pareto_frontier.csv")
OUT_MANIFEST_JSON = os.path.join(CONFIGS_DIR, "experiment_manifest.json")

GLOBAL_SEED = 42

# Ensure laya is importable
try:
    import laya
    from laya.common import ece_score
except ImportError:
    # Attempt sibling / parent path resolution if cloned alongside
    parent_dir = os.path.dirname(REPO_ROOT)
    if os.path.exists(os.path.join(parent_dir, "laya")):
        sys.path.insert(0, os.path.join(parent_dir, "laya"))
        import laya
        from laya.common import ece_score
    else:
        raise ImportError(
            "The 'laya' package is required to execute this benchmark. "
            "Please install laya or clone https://github.com/NandhaKishorM/laya into your Python environment."
        )


def kendall_tau_dist(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Normalized Kendall's Tau Distance between two scoring vectors over K candidates."""
    K = len(scores_a)
    if K < 2:
        return 0.0
    diff_a = scores_a[:, None] - scores_a[None, :]
    diff_b = scores_b[:, None] - scores_b[None, :]
    tri_i, tri_j = np.triu_indices(K, k=1)
    sign_a = np.sign(diff_a[tri_i, tri_j])
    sign_b = np.sign(diff_b[tri_i, tri_j])
    discordant = np.sum((sign_a * sign_b) < 0)
    total_pairs = len(tri_i)
    return float(discordant / total_pairs)


def top3_jaccard_churn(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Top-3 Jaccard Churn: 1.0 - |Top3_A ∩ Top3_B| / |Top3_A ∪ Top3_B|."""
    K = len(scores_a)
    if K < 3:
        return 0.0
    k_eval = min(3, K)
    top_a = set(np.argsort(-scores_a)[:k_eval])
    top_b = set(np.argsort(-scores_b)[:k_eval])
    intersection = len(top_a & top_b)
    union = len(top_a | top_b)
    if union == 0:
        return 0.0
    return float(1.0 - (intersection / union))


def compute_pairwise_instability(all_scores: List[np.ndarray]) -> Tuple[float, float, float]:
    """Compute mean Argmax FR, Kendall Tau distance, and Top-3 Churn across P permutations."""
    P = len(all_scores)
    if P < 2:
        return 0.0, 0.0, 0.0
    flip_counts = 0
    kendall_dists = []
    churns = []
    pairs = list(itertools.combinations(range(P), 2))
    for a, b in pairs:
        sa, sb = all_scores[a], all_scores[b]
        if int(np.argmax(sa)) != int(np.argmax(sb)):
            flip_counts += 1
        kendall_dists.append(kendall_tau_dist(sa, sb))
        churns.append(top3_jaccard_churn(sa, sb))
    n_pairs = len(pairs)
    return float(flip_counts / n_pairs), float(np.mean(kendall_dists)), float(np.mean(churns))


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return "unknown"


def paired_mcnemar_test(corr_a: np.ndarray, corr_b: np.ndarray) -> Tuple[int, int, float]:
    """Exact paired McNemar test on discordant pairs."""
    n01 = int(np.sum((corr_a == 0) & (corr_b == 1)))
    n10 = int(np.sum((corr_a == 1) & (corr_b == 0)))
    total_disc = n01 + n10
    if total_disc == 0:
        return 0, 0, 1.0
    p_val = stats.binomtest(n01, total_disc, 0.5, alternative="two-sided").pvalue
    return n01, n10, float(p_val)


def bootstrap_ci_query_level(query_values: List[float], n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    """Query-level percentile bootstrap confidence interval."""
    arr = np.asarray(query_values, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    rng = np.random.RandomState(42)
    boot_means = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means.append(np.mean(arr[idx]))
    lo = float(np.percentile(boot_means, 100 * (alpha / 2)))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """Benjamini-Hochberg FDR correction."""
    m = len(p_values)
    if m == 0:
        return []
    sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
    adj_p = [0.0] * m
    min_adj = 1.0
    for rank_idx in range(m - 1, -1, -1):
        orig_idx, p = sorted_pairs[rank_idx]
        rank = rank_idx + 1
        adj = (p * m) / rank
        min_adj = min(min_adj, adj)
        adj_p[orig_idx] = min(1.0, max(0.0, min_adj))
    return [(adj_p[i], adj_p[i] <= alpha) for i in range(m)]


def run_benchmark(sample_size: int = 120):
    print("=" * 90)
    print("CANDIDATE CARDINALITY SCALING & ORDER-MARGINALIZATION BENCHMARK")
    print(f"Sample Size: N={sample_size} unique test queries (Banking77)")
    print("=" * 90)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading Laya model on device: {device}...")
    agent = laya.load("convaiinnovations/laya", device=device)
    print(f"Agent loaded. Model device: {agent.device}, dtype: {agent.dtype}")

    print("Loading mteb/banking77 test split...")
    d = load_dataset("mteb/banking77", split="test")
    all_label_names = sorted(set(d["label_text"]))
    formatted_labels = [x.replace("_", " ") for x in all_label_names]
    K_max = len(formatted_labels)
    assert K_max == 77, f"Expected 77 Banking77 classes, got {K_max}"

    by_label = defaultdict(list)
    for idx, ex in enumerate(d):
        by_label[ex["label"]].append(idx)
    labels = sorted(by_label.keys())

    rng_strat = random.Random(GLOBAL_SEED)
    selected_indices = []
    for l in labels:
        selected_indices.append(rng_strat.choice(by_label[l]))

    rem_classes = rng_strat.sample(labels, sample_size - len(labels)) if sample_size > len(labels) else []
    for l in rem_classes:
        rem_candidates = [i for i in by_label[l] if i not in selected_indices]
        selected_indices.append(rng_strat.choice(rem_candidates))

    if sample_size <= len(labels):
        selected_indices = selected_indices[:sample_size]

    test_queries = [d[i] for i in selected_indices]
    print(f"[OK] Selected {len(test_queries)} unique test queries covering {len(set(q['label'] for q in test_queries))} classes.")

    print("Warming up execution graph...")
    warm_state = {"message": "How do I activate my new card?"}
    warm_qs = {"w0": {"type": "choice", "instructions": "Warmup", "criteria": {c: None for c in formatted_labels[:5]}}}
    agent.system_one_batch([warm_state], warm_qs, batch_size=1)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    print("Warmup complete.")

    cardinalities = [5, 10, 20, 40, 77]
    P = 10
    MAX_QS_CHUNK = 4

    method_names = [
        "B0_Native",
        "B0_Random",
        "B1_Alpha",
        "B1_ReverseAlpha",
        "Rand_Marg_M2",
        "Rand_Marg_M3",
        "Rand_Marg_M5",
        "B2_Cyclic_M2",
        "B2_Cyclic_M3",
        "B2_Cyclic_M5",
    ]

    k77_base_seeds = [0, 7701, 7702]
    per_query_records = []
    summary_by_k_method = {}
    candidate_set_hashes = defaultdict(dict)

    t_start = time.perf_counter()

    for K in cardinalities:
        print(f"\nEvaluating Cardinality K = {K} (Random Guessing Baseline = {1.0/K*100:.2f}%)")
        n_base_orders = len(k77_base_seeds) if K == K_max else 1

        k_query_correct = {m: [] for m in method_names}
        k_query_confs = {m: [] for m in method_names}
        k_query_latencies = {m: [] for m in method_names}
        k_query_fr_top1 = {m: [] for m in method_names}
        k_query_kendall = {m: [] for m in method_names}
        k_query_churn = {m: [] for m in method_names}
        k_query_logit_var = []
        k_cross_canonical_flips = []

        for q_idx, row in enumerate(test_queries):
            true_label_text = row["label_text"].replace("_", " ")
            state = {"message": row["text"]}

            d_pool = sorted([c for c in formatted_labels if c != true_label_text])
            rng_master = random.Random(GLOBAL_SEED + q_idx * 1000)
            master_distractors = list(d_pool)
            rng_master.shuffle(master_distractors)

            c5 = [true_label_text] + master_distractors[:4]
            c10 = [true_label_text] + master_distractors[:9]
            c20 = [true_label_text] + master_distractors[:19]
            c40 = [true_label_text] + master_distractors[:39]
            c77 = [true_label_text] + master_distractors[:76]

            assert set(c5).issubset(set(c10))
            assert set(c10).issubset(set(c20))
            assert set(c20).issubset(set(c40))
            assert set(c40).issubset(set(c77))

            cand_map = {5: c5, 10: c10, 20: c20, 40: c40, 77: c77}
            candidate_set = cand_map[K]
            cand_hash = hashlib.sha256(",".join(sorted(candidate_set)).encode()).hexdigest()[:12]
            candidate_set_hashes[q_idx][K] = cand_hash
            true_cand_idx = candidate_set.index(true_label_text)

            base_correct = {m: [] for m in method_names}
            base_confs = {m: [] for m in method_names}
            base_latencies = {m: [] for m in method_names}
            base_fr_top1 = {m: [] for m in method_names}
            base_kendall = {m: [] for m in method_names}
            base_churn = {m: [] for m in method_names}
            base_logit_vars = []
            base_cross_canonical = []

            for base_order_idx in range(n_base_orders):
                if K == K_max and base_order_idx > 0:
                    rng_base = random.Random(k77_base_seeds[base_order_idx] + q_idx)
                    active_candidate_set = list(candidate_set)
                    rng_base.shuffle(active_candidate_set)
                else:
                    active_candidate_set = list(candidate_set)

                active_true_idx = active_candidate_set.index(true_label_text)

                native_cands = list(active_candidate_set)
                alpha_cands = sorted(active_candidate_set)
                rev_alpha_cands = sorted(active_candidate_set, reverse=True)

                rand_perms = []
                for p_idx in range(P):
                    p_rng = random.Random(20000 * K + 1000 * q_idx + 100 * base_order_idx + p_idx)
                    p = list(range(K))
                    p_rng.shuffle(p)
                    rand_perms.append(p)

                cyclic_phases = {}
                for M_cyc in [2, 3, 5]:
                    step = max(1, K // M_cyc)
                    phase_offset = max(1, step // 2)
                    p1_shifts = [(m * step) % K for m in range(M_cyc)]
                    p2_shifts = [((m * step) + phase_offset) % K for m in range(M_cyc)]
                    p1_perms = [[(i + s) % K for i in range(K)] for s in p1_shifts]
                    p2_perms = [[(i + s) % K for i in range(K)] for s in p2_shifts]
                    cyclic_phases[M_cyc] = (p1_perms, p2_perms)

                questions_dict = {}
                questions_dict["q_native"] = {
                    "type": "choice",
                    "instructions": "Which banking intent does `message` express?",
                    "criteria": {c: None for c in native_cands},
                }
                questions_dict["q_alpha"] = {
                    "type": "choice",
                    "instructions": "Which banking intent does `message` express?",
                    "criteria": {c: None for c in alpha_cands},
                }
                questions_dict["q_rev_alpha"] = {
                    "type": "choice",
                    "instructions": "Which banking intent does `message` express?",
                    "criteria": {c: None for c in rev_alpha_cands},
                }
                for p_idx in range(P):
                    p_cands = [active_candidate_set[i] for i in rand_perms[p_idx]]
                    questions_dict[f"q_rand_{p_idx}"] = {
                        "type": "choice",
                        "instructions": "Which banking intent does `message` express?",
                        "criteria": {c: None for c in p_cands},
                    }
                for M_cyc in [2, 3, 5]:
                    p1_perms, p2_perms = cyclic_phases[M_cyc]
                    for m_idx, c_perm in enumerate(p1_perms):
                        c_cands = [active_candidate_set[i] for i in c_perm]
                        questions_dict[f"q_cyc_{M_cyc}_p1_{m_idx}"] = {
                            "type": "choice",
                            "instructions": "Which banking intent does `message` express?",
                            "criteria": {c: None for c in c_cands},
                        }
                    for m_idx, c_perm in enumerate(p2_perms):
                        c_cands = [active_candidate_set[i] for i in c_perm]
                        questions_dict[f"q_cyc_{M_cyc}_p2_{m_idx}"] = {
                            "type": "choice",
                            "instructions": "Which banking intent does `message` express?",
                            "criteria": {c: None for c in c_cands},
                        }

                answers = {}
                total_ms = 0.0
                q_items = list(questions_dict.items())
                for chunk_start in range(0, len(q_items), MAX_QS_CHUNK):
                    chunk_qs = dict(q_items[chunk_start : chunk_start + MAX_QS_CHUNK])
                    t0 = time.perf_counter()
                    chunk_res = agent.system_one_batch([state], chunk_qs, batch_size=1)[0]
                    if torch.backends.mps.is_available():
                        torch.mps.synchronize()
                    total_ms += (time.perf_counter() - t0) * 1000
                    answers.update(chunk_res["answers"])

                per_q_latency = total_ms / len(questions_dict)

                prob_native = np.zeros(K, dtype=np.float32)
                p_dict = answers["q_native"]["probabilities"]
                for i, c in enumerate(native_cands):
                    prob_native[active_candidate_set.index(c)] = p_dict[c]

                prob_alpha = np.zeros(K, dtype=np.float32)
                p_dict = answers["q_alpha"]["probabilities"]
                for i, c in enumerate(alpha_cands):
                    prob_alpha[active_candidate_set.index(c)] = p_dict[c]

                prob_rev_alpha = np.zeros(K, dtype=np.float32)
                p_dict = answers["q_rev_alpha"]["probabilities"]
                for i, c in enumerate(rev_alpha_cands):
                    prob_rev_alpha[active_candidate_set.index(c)] = p_dict[c]

                rand_probs_all = []
                for p_idx in range(P):
                    p_dict = answers[f"q_rand_{p_idx}"]["probabilities"]
                    s_vec = np.zeros(K, dtype=np.float32)
                    perm = rand_perms[p_idx]
                    for pos, orig_idx in enumerate(perm):
                        c = active_candidate_set[orig_idx]
                        s_vec[orig_idx] = p_dict[c]
                    rand_probs_all.append(s_vec)

                cyc_p1_probs = {}
                cyc_p2_probs = {}
                for M_cyc in [2, 3, 5]:
                    p1_perms, p2_perms = cyclic_phases[M_cyc]
                    p1_list = []
                    for m_idx, c_perm in enumerate(p1_perms):
                        p_dict = answers[f"q_cyc_{M_cyc}_p1_{m_idx}"]["probabilities"]
                        s_vec = np.zeros(K, dtype=np.float32)
                        for orig_idx in c_perm:
                            c = active_candidate_set[orig_idx]
                            s_vec[orig_idx] = p_dict[c]
                        p1_list.append(s_vec)
                    cyc_p1_probs[M_cyc] = p1_list

                    p2_list = []
                    for m_idx, c_perm in enumerate(p2_perms):
                        p_dict = answers[f"q_cyc_{M_cyc}_p2_{m_idx}"]["probabilities"]
                        s_vec = np.zeros(K, dtype=np.float32)
                        for orig_idx in c_perm:
                            c = active_candidate_set[orig_idx]
                            s_vec[orig_idx] = p_dict[c]
                        p2_list.append(s_vec)
                    cyc_p2_probs[M_cyc] = p2_list

                log_probs_mat = np.log(np.clip(np.array(rand_probs_all), 1e-12, 1.0))
                cand_logit_vars = np.var(log_probs_mat, axis=0, ddof=1)
                mean_logit_var = float(np.mean(cand_logit_vars))
                base_logit_vars.append(mean_logit_var)

                fr_t1_rand, fr_rk_rand, ch_t3_rand = compute_pairwise_instability(rand_probs_all)

                pred_alpha = int(np.argmax(prob_alpha))
                pred_rev_alpha = int(np.argmax(prob_rev_alpha))
                cross_flip = 1 if pred_alpha != pred_rev_alpha else 0
                base_cross_canonical.append(cross_flip)

                # Assign per method
                pred_native = int(np.argmax(prob_native))
                base_correct["B0_Native"].append(1 if pred_native == active_true_idx else 0)
                base_confs["B0_Native"].append(float(np.max(prob_native)))
                base_latencies["B0_Native"].append(per_q_latency)
                base_fr_top1["B0_Native"].append(0.0)
                base_kendall["B0_Native"].append(0.0)
                base_churn["B0_Native"].append(0.0)

                pred_rand0 = int(np.argmax(rand_probs_all[0]))
                base_correct["B0_Random"].append(1 if pred_rand0 == active_true_idx else 0)
                base_confs["B0_Random"].append(float(np.max(rand_probs_all[0])))
                base_latencies["B0_Random"].append(per_q_latency)
                base_fr_top1["B0_Random"].append(fr_t1_rand)
                base_kendall["B0_Random"].append(fr_rk_rand)
                base_churn["B0_Random"].append(ch_t3_rand)

                base_correct["B1_Alpha"].append(1 if pred_alpha == active_true_idx else 0)
                base_confs["B1_Alpha"].append(float(np.max(prob_alpha)))
                base_latencies["B1_Alpha"].append(per_q_latency)
                base_fr_top1["B1_Alpha"].append(float(cross_flip))
                base_kendall["B1_Alpha"].append(kendall_tau_dist(prob_alpha, prob_rev_alpha))
                base_churn["B1_Alpha"].append(top3_jaccard_churn(prob_alpha, prob_rev_alpha))

                base_correct["B1_ReverseAlpha"].append(1 if pred_rev_alpha == active_true_idx else 0)
                base_confs["B1_ReverseAlpha"].append(float(np.max(prob_rev_alpha)))
                base_latencies["B1_ReverseAlpha"].append(per_q_latency)
                base_fr_top1["B1_ReverseAlpha"].append(float(cross_flip))
                base_kendall["B1_ReverseAlpha"].append(kendall_tau_dist(prob_alpha, prob_rev_alpha))
                base_churn["B1_ReverseAlpha"].append(top3_jaccard_churn(prob_alpha, prob_rev_alpha))

                for M_rand in [2, 3, 5]:
                    m_key = f"Rand_Marg_M{M_rand}"
                    ens1_p = np.mean(rand_probs_all[:M_rand], axis=0)
                    ens2_p = np.mean(rand_probs_all[M_rand : 2 * M_rand], axis=0)
                    pred1 = int(np.argmax(ens1_p))
                    pred2 = int(np.argmax(ens2_p))
                    res_flip = 1 if pred1 != pred2 else 0

                    base_correct[m_key].append(1 if pred1 == active_true_idx else 0)
                    base_confs[m_key].append(float(np.max(ens1_p)))
                    base_latencies[m_key].append(per_q_latency * M_rand)
                    base_fr_top1[m_key].append(float(res_flip))
                    base_kendall[m_key].append(kendall_tau_dist(ens1_p, ens2_p))
                    base_churn[m_key].append(top3_jaccard_churn(ens1_p, ens2_p))

                for M_cyc in [2, 3, 5]:
                    m_key = f"B2_Cyclic_M{M_cyc}"
                    ens1_p = np.mean(cyc_p1_probs[M_cyc], axis=0)
                    ens2_p = np.mean(cyc_p2_probs[M_cyc], axis=0)
                    pred1 = int(np.argmax(ens1_p))
                    pred2 = int(np.argmax(ens2_p))
                    res_flip = 1 if pred1 != pred2 else 0

                    base_correct[m_key].append(1 if pred1 == active_true_idx else 0)
                    base_confs[m_key].append(float(np.max(ens1_p)))
                    base_latencies[m_key].append(per_q_latency * M_cyc)
                    base_fr_top1[m_key].append(float(res_flip))
                    base_kendall[m_key].append(kendall_tau_dist(ens1_p, ens2_p))
                    base_churn[m_key].append(top3_jaccard_churn(ens1_p, ens2_p))

            for m in method_names:
                q_acc = float(np.mean(base_correct[m]))
                q_conf = float(np.mean(base_confs[m]))
                q_lat = float(np.mean(base_latencies[m]))
                q_fr = float(np.mean(base_fr_top1[m]))
                q_kd = float(np.mean(base_kendall[m]))
                q_ch = float(np.mean(base_churn[m]))

                k_query_correct[m].append(q_acc)
                k_query_confs[m].append(q_conf)
                k_query_latencies[m].append(q_lat)
                k_query_fr_top1[m].append(q_fr)
                k_query_kendall[m].append(q_kd)
                k_query_churn[m].append(q_ch)

                per_query_records.append({
                    "Query_ID": q_idx,
                    "Original_Dataset_Index": selected_indices[q_idx],
                    "K": K,
                    "Candidate_Set_Hash": cand_hash,
                    "Method": m,
                    "Correct": q_acc,
                    "Confidence": round(q_conf, 4),
                    "FR_top1": round(q_fr, 4),
                    "Kendall_Dist": round(q_kd, 4),
                    "Top3_Churn": round(q_ch, 4),
                    "Latency_ms": round(q_lat, 2),
                })

            k_query_logit_var.append(float(np.mean(base_logit_vars)))
            k_cross_canonical_flips.append(float(np.mean(base_cross_canonical)))

            if (q_idx + 1) % 20 == 0 or (q_idx + 1) == len(test_queries):
                print(f"  Processed {q_idx + 1:3d}/{len(test_queries)} queries (K={K})...", flush=True)

        summary_by_k_method[K] = {}
        for m in method_names:
            acc_arr = np.array(k_query_correct[m])
            conf_arr = np.array(k_query_confs[m])
            fr_arr = np.array(k_query_fr_top1[m])
            kd_arr = np.array(k_query_kendall[m])
            ch_arr = np.array(k_query_churn[m])
            lat_arr = np.array(k_query_latencies[m])

            ece_val = float(ece_score(conf_arr, (acc_arr >= 0.5).astype(float), bins=15))

            summary_by_k_method[K][m] = {
                "K": K,
                "Random_Guess_Baseline": round(1.0 / K, 4),
                "Method": m,
                "Accuracy": round(float(np.mean(acc_arr)), 4),
                "Accuracy_CI": bootstrap_ci_query_level(list(acc_arr)),
                "ECE": round(ece_val, 4),
                "ECE_CI": bootstrap_ci_query_level([abs(c - a) for c, a in zip(conf_arr, acc_arr)]),
                "FR_top1": round(float(np.mean(fr_arr)), 4),
                "FR_top1_CI": bootstrap_ci_query_level(list(fr_arr)),
                "Kendall_Dist": round(float(np.mean(kd_arr)), 4),
                "Top3_Churn": round(float(np.mean(ch_arr)), 4),
                "Mean_Logit_Variance": round(float(np.mean(k_query_logit_var)), 4),
                "Latency_p50_ms": round(float(np.percentile(lat_arr, 50)), 2),
                "Latency_p95_ms": round(float(np.percentile(lat_arr, 95)), 2),
                "raw_acc": acc_arr,
                "raw_fr": fr_arr,
            }

    # Paired Statistical Tests & FDR Correction
    hypotheses = []
    for K in cardinalities:
        base_acc = summary_by_k_method[K]["B0_Native"]["raw_acc"]
        for m in method_names:
            if m == "B0_Native":
                continue
            m_acc = summary_by_k_method[K][m]["raw_acc"]
            n01, n10, p_val = paired_mcnemar_test(base_acc, m_acc)
            acc_delta = float(np.mean(m_acc) - np.mean(base_acc))
            hypotheses.append({
                "K": K,
                "Method": m,
                "Comparison": f"{m} vs B0_Native",
                "Delta_Accuracy": round(acc_delta, 4),
                "Delta_CI": bootstrap_ci_query_level(list(m_acc - base_acc)),
                "Discordant_Gains_n01": n01,
                "Discordant_Losses_n10": n10,
                "Unadjusted_p_value": p_val,
            })

    raw_p_vals = [h["Unadjusted_p_value"] for h in hypotheses]
    bh_results = benjamini_hochberg(raw_p_vals, alpha=0.05)

    test_records = []
    for h, (adj_p, sig) in zip(hypotheses, bh_results):
        h_rec = {
            **h,
            "BH_Adjusted_p_value": round(adj_p, 5),
            "FDR_Significant_q05": sig,
            "Significance_Status": "SUPPORTED" if sig else "NOT_SIGNIFICANT",
        }
        test_records.append(h_rec)
        summary_by_k_method[h["K"]][h["Method"]]["Adjusted_p_value"] = round(adj_p, 5)
        summary_by_k_method[h["K"]][h["Method"]]["Significant_vs_B0"] = sig

    summary_by_k_method[K]["B0_Native"]["Adjusted_p_value"] = 1.0
    summary_by_k_method[K]["B0_Native"]["Significant_vs_B0"] = False

    # 3-Objective Pareto Frontier
    pareto_records = []
    for K in cardinalities:
        k_methods = [summary_by_k_method[K][m] for m in method_names]
        for m_curr in k_methods:
            acc_curr = m_curr["Accuracy"]
            lat_curr = m_curr["Latency_p50_ms"]
            inst_curr = m_curr["FR_top1"]

            is_dominated = False
            dominating_methods = []

            for m_other in k_methods:
                if m_other["Method"] == m_curr["Method"]:
                    continue
                acc_other = m_other["Accuracy"]
                lat_other = m_other["Latency_p50_ms"]
                inst_other = m_other["FR_top1"]

                no_worse = (acc_other >= acc_curr) and (lat_other <= lat_curr) and (inst_other <= inst_curr)
                strictly_better = (acc_other > acc_curr) or (lat_other < lat_curr) or (inst_other < inst_curr)

                if no_worse and strictly_better:
                    is_dominated = True
                    dominating_methods.append(m_other["Method"])

            pareto_status = "Dominated by: " + ",".join(dominating_methods) if is_dominated else "PARETO_OPTIMAL"
            pareto_records.append({
                "K": K,
                "Method": m_curr["Method"],
                "Accuracy": acc_curr,
                "Latency_p50_ms": lat_curr,
                "Instability_FR": inst_curr,
                "Pareto_Status": pareto_status,
            })

    # Save summary CSV
    summary_flat = []
    for K in cardinalities:
        for m in method_names:
            r = summary_by_k_method[K][m]
            summary_flat.append({
                "K": r["K"],
                "Random_Guess_Baseline": r["Random_Guess_Baseline"],
                "Method": r["Method"],
                "Accuracy": r["Accuracy"],
                "Accuracy_95CI_Lo": round(r["Accuracy_CI"][0], 4),
                "Accuracy_95CI_Hi": round(r["Accuracy_CI"][1], 4),
                "ECE": r["ECE"],
                "ECE_95CI_Lo": round(r["ECE_CI"][0], 4),
                "ECE_95CI_Hi": round(r["ECE_CI"][1], 4),
                "FR_top1": r["FR_top1"],
                "FR_top1_95CI_Lo": round(r["FR_top1_CI"][0], 4),
                "FR_top1_95CI_Hi": round(r["FR_top1_CI"][1], 4),
                "Kendall_Dist": r["Kendall_Dist"],
                "Top3_Churn": r["Top3_Churn"],
                "Mean_Logit_Variance": r["Mean_Logit_Variance"],
                "Latency_p50_ms": r["Latency_p50_ms"],
                "Latency_p95_ms": r["Latency_p95_ms"],
                "Adjusted_p_value": r.get("Adjusted_p_value", 1.0),
                "Significant_vs_B0": r.get("Significant_vs_B0", False),
            })

    with open(OUT_SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_flat[0].keys()))
        writer.writeheader()
        writer.writerows(summary_flat)

    with open(OUT_PER_QUERY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_query_records[0].keys()))
        writer.writeheader()
        writer.writerows(per_query_records)

    with open(OUT_TESTS_CSV, "w", newline="") as f:
        fields = ["K", "Method", "Comparison", "Delta_Accuracy", "Delta_CI", "Discordant_Gains_n01", "Discordant_Losses_n10", "Unadjusted_p_value", "BH_Adjusted_p_value", "FDR_Significant_q05", "Significance_Status"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(test_records)

    with open(OUT_PARETO_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pareto_records[0].keys()))
        writer.writeheader()
        writer.writerows(pareto_records)

    manifest = {
        "git_commit": get_git_commit(),
        "dataset": "mteb/banking77",
        "split": "test",
        "sample_size": sample_size,
        "query_indices": selected_indices,
        "global_seed": GLOBAL_SEED,
        "cardinalities": cardinalities,
        "permutations_P": P,
        "k77_base_seeds": k77_base_seeds,
        "model_checkpoint": "convaiinnovations/laya",
        "pytorch_version": torch.__version__,
        "hardware_device": device,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_experiment_duration_sec": round(time.perf_counter() - t_start, 2),
    }
    with open(OUT_MANIFEST_JSON, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[OK] Benchmark completed. Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", type=int, default=None, help="Run smoke test on N queries")
    args = parser.parse_args()
    n_queries = args.smoke_test if args.smoke_test is not None else 120
    run_benchmark(sample_size=n_queries)
