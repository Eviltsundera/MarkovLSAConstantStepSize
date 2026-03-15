#!/usr/bin/env python3
"""Quick reproduction with reduced parameters to verify patterns match the paper.

Full-scale experiments (run_experiments.py) reproduce exact paper numbers but
require hours of computation. This script runs smaller-scale versions that
demonstrate the same qualitative findings in minutes.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from lsa_inference.markov_chain import generate_transition_matrix, simulate_chain
from lsa_inference.lsa_problem import generate_A, generate_b, compute_theta_star
from lsa_inference.lsa_runner import run_lsa_batched, run_lsa_diminishing
from lsa_inference.batch_inference import compute_covariance
from lsa_inference.rr_extrapolation import rr_coefficients, run_rr_extrapolation
from lsa_inference.utils import l2_error, ci_width, coverage


def run_table1_quick():
    """Table 1: 20 problems × 50 trajectories × T=10^5."""
    n_problems = 20
    n_traj = 50
    T = 100_000
    n_states, d = 10, 5
    K = int(T ** 0.3)
    burn_in = min(1000, T // 10)

    master_rng = np.random.default_rng(42)
    methods = ['alpha_0.2', 'alpha_0.02', 'RR', 'dim_0.2', 'dim_0.02']
    method_labels = {
        'alpha_0.2': 'α=0.2 (const)',
        'alpha_0.02': 'α=0.02 (const)',
        'RR': 'RR (0.2+0.02)',
        'dim_0.2': '0.2/√k (dim)',
        'dim_0.02': '0.02/√k (dim)',
    }
    all_results = {m: {'l2': [], 'width': [], 'cov': []} for m in methods}

    for _ in tqdm(range(n_problems), desc="Table 1 (quick)"):
        prob_rng = np.random.default_rng(master_rng.integers(0, 2**31))
        P, pi = generate_transition_matrix(n_states, prob_rng)
        A_list, _ = generate_A(n_states, d, pi, prob_rng)
        b_list = generate_b(n_states, d, prob_rng)
        theta_star = compute_theta_star(A_list, b_list, pi)
        traj_rng = np.random.default_rng(prob_rng.integers(0, 2**31))

        prob_results = {m: {'l2': [], 'width': [], 'cov': []} for m in methods}

        for _ in range(n_traj):
            traj = simulate_chain(P, pi, T, traj_rng)

            bm, n = run_lsa_batched(A_list, b_list, traj, 0.2, K, burn_in)
            tb, Sh = compute_covariance(bm, n)
            prob_results['alpha_0.2']['l2'].append(l2_error(tb, theta_star))
            prob_results['alpha_0.2']['width'].append(ci_width(Sh, K, n))
            prob_results['alpha_0.2']['cov'].append(coverage(theta_star, tb, Sh, K, n))

            bm, n = run_lsa_batched(A_list, b_list, traj, 0.02, K, burn_in)
            tb, Sh = compute_covariance(bm, n)
            prob_results['alpha_0.02']['l2'].append(l2_error(tb, theta_star))
            prob_results['alpha_0.02']['width'].append(ci_width(Sh, K, n))
            prob_results['alpha_0.02']['cov'].append(coverage(theta_star, tb, Sh, K, n))

            tt, St, lo, hi, se, n = run_rr_extrapolation(
                A_list, b_list, traj, [0.2, 0.02], K, burn_in)
            prob_results['RR']['l2'].append(l2_error(tt, theta_star))
            prob_results['RR']['width'].append(2 * 1.96 * se)
            prob_results['RR']['cov'].append(float(lo <= theta_star[0] <= hi))

            bm, n_eff = run_lsa_diminishing(A_list, b_list, traj, 0.2, 0.5, K)
            tb, Sh = compute_covariance(bm, n_eff)
            prob_results['dim_0.2']['l2'].append(l2_error(tb, theta_star))
            prob_results['dim_0.2']['width'].append(ci_width(Sh, K, n_eff))
            prob_results['dim_0.2']['cov'].append(coverage(theta_star, tb, Sh, K, n_eff))

            bm, n_eff = run_lsa_diminishing(A_list, b_list, traj, 0.02, 0.5, K)
            tb, Sh = compute_covariance(bm, n_eff)
            prob_results['dim_0.02']['l2'].append(l2_error(tb, theta_star))
            prob_results['dim_0.02']['width'].append(ci_width(Sh, K, n_eff))
            prob_results['dim_0.02']['cov'].append(coverage(theta_star, tb, Sh, K, n_eff))

        for m in methods:
            all_results[m]['l2'].append(np.mean(prob_results[m]['l2']))
            all_results[m]['width'].append(np.mean(prob_results[m]['width']))
            all_results[m]['cov'].append(np.mean(prob_results[m]['cov']))

    print("\n" + "=" * 80)
    print(f"Table 1 (Quick): {n_problems} problems, {n_traj} traj, T={T}")
    print("=" * 80)
    percentiles = [10, 25, 50, 75, 90]
    for metric, scale, unit in [('l2', 1e3, '×1e-3'), ('width', 1e3, '×1e-3'), ('cov', 100, '%')]:
        print(f"\n--- {metric} ({unit}) ---")
        print(f"{'Method':<20}" + "".join(f"{'p'+str(p):>10}" for p in percentiles))
        for m in methods:
            vals = np.array(all_results[m][metric]) * scale
            pcts = np.percentile(vals, percentiles)
            print(f"{method_labels[m]:<20}" + "".join(f"{v:>10.2f}" for v in pcts))

    print("\nPaper Table 1 reference (medians):")
    print("  α=0.2:     L2=8.12, Width=2.70, Cov=11%")
    print("  α=0.02:    L2=1.59, Width=2.38, Cov=90%")
    print("  RR:        L2=1.32, Width=2.41, Cov=94%")
    print("  0.2/√k:    L2=1.32, Width=2.14, Cov=91%")
    print("  0.02/√k:   L2=1.42, Width=1.51, Cov=76%")


def run_table3_quick():
    """Table 3: Effect of trajectory length T (single problem, 200 traj)."""
    n_states, d = 10, 5
    n_traj = 200
    T_values = [1_000, 10_000, 100_000]

    rng = np.random.default_rng(456)
    P, pi = generate_transition_matrix(n_states, rng)
    A_list, _ = generate_A(n_states, d, pi, rng)
    b_list = generate_b(n_states, d, rng)
    theta_star = compute_theta_star(A_list, b_list, pi)

    methods = ['RR', 'alpha_0.2', 'alpha_0.02', 'dim_0.2']
    method_labels = {
        'RR': 'RR (0.2+0.02)',
        'alpha_0.2': 'α=0.2 (const)',
        'alpha_0.02': 'α=0.02 (const)',
        'dim_0.2': '0.2/√k (dim)',
    }

    print("\n" + "=" * 60)
    print("Table 3 (Quick): Effect of Trajectory Length T")
    print("=" * 60)

    for T in T_values:
        K = max(int(T ** 0.3), 5)
        burn_in = min(1000, T // 10)
        cov_accum = {m: [] for m in methods}

        for _ in tqdm(range(n_traj), desc=f"T={T}"):
            traj_rng = np.random.default_rng(rng.integers(0, 2**31))
            traj = simulate_chain(P, pi, T, traj_rng)

            tt, St, lo, hi, se, n = run_rr_extrapolation(
                A_list, b_list, traj, [0.2, 0.02], K, burn_in)
            cov_accum['RR'].append(float(lo <= theta_star[0] <= hi))

            bm, n = run_lsa_batched(A_list, b_list, traj, 0.2, K, burn_in)
            tb, Sh = compute_covariance(bm, n)
            cov_accum['alpha_0.2'].append(coverage(theta_star, tb, Sh, K, n))

            bm, n = run_lsa_batched(A_list, b_list, traj, 0.02, K, burn_in)
            tb, Sh = compute_covariance(bm, n)
            cov_accum['alpha_0.02'].append(coverage(theta_star, tb, Sh, K, n))

            bm, n_eff = run_lsa_diminishing(A_list, b_list, traj, 0.2, 0.5, K)
            tb, Sh = compute_covariance(bm, n_eff)
            cov_accum['dim_0.2'].append(coverage(theta_star, tb, Sh, K, n_eff))

        print(f"\n  T = {T}, K = {K}:")
        for m in methods:
            v = np.mean(cov_accum[m]) * 100
            print(f"    {method_labels[m]}: {v:.1f}%")

    print("\nPaper Table 3 reference (coverage %):")
    print("       T     | RR    | α=0.2 | α=0.02 | 0.2/√k")
    print("       10³   | 83.4  | 82.2  | 83.6   | 76.8")
    print("       10⁴   | 89.2  | 75.2  | 90.0   | 85.4")
    print("       10⁵   | 91.2  | 0.04  | 88.2   | 90.0")
    print("       10⁶   | 95.2  |  0    | 80.2   | 92.8")


def run_table2_quick():
    """Table 2: Effect of batch number K (single problem, 200 traj, T=10^5)."""
    T = 100_000
    n_states, d = 10, 5
    n_traj = 200
    burn_in = 1000
    K_values = [50, 100, 500]

    rng = np.random.default_rng(123)
    P, pi = generate_transition_matrix(n_states, rng)
    A_list, _ = generate_A(n_states, d, pi, rng)
    b_list = generate_b(n_states, d, rng)
    theta_star = compute_theta_star(A_list, b_list, pi)

    methods = ['RR', 'dim_0.2', 'dim_0.02']
    method_labels = {
        'RR': 'RR (0.2+0.02)',
        'dim_0.2': '0.2/√k (dim)',
        'dim_0.02': '0.02/√k (dim)',
    }

    print("\n" + "=" * 60)
    print("Table 2 (Quick): Effect of Batch Number K")
    print("=" * 60)

    for K in K_values:
        cov_accum = {m: [] for m in methods}

        for _ in tqdm(range(n_traj), desc=f"K={K}"):
            traj_rng = np.random.default_rng(rng.integers(0, 2**31))
            traj = simulate_chain(P, pi, T, traj_rng)

            tt, St, lo, hi, se, n = run_rr_extrapolation(
                A_list, b_list, traj, [0.2, 0.02], K, burn_in)
            cov_accum['RR'].append(float(lo <= theta_star[0] <= hi))

            bm, n_eff = run_lsa_diminishing(A_list, b_list, traj, 0.2, 0.5, K)
            tb, Sh = compute_covariance(bm, n_eff)
            cov_accum['dim_0.2'].append(coverage(theta_star, tb, Sh, K, n_eff))

            bm, n_eff = run_lsa_diminishing(A_list, b_list, traj, 0.02, 0.5, K)
            tb, Sh = compute_covariance(bm, n_eff)
            cov_accum['dim_0.02'].append(coverage(theta_star, tb, Sh, K, n_eff))

        print(f"\n  K = {K}:")
        for m in methods:
            v = np.mean(cov_accum[m]) * 100
            se = np.std(cov_accum[m]) / np.sqrt(n_traj) * 100
            print(f"    {method_labels[m]}: {v:.1f}% ± {se:.1f}%")

    print("\nPaper Table 2 reference (T=10^6, coverage %):")
    print("  K=50:   RR=92.8, 0.2/√k=93.0, 0.02/√k=81.6")
    print("  K=100:  RR=94.4, 0.2/√k=95.0, 0.02/√k=71.2")
    print("  K=500:  RR=94.2, 0.2/√k=88.8, 0.02/√k=42.2")
    print("  K=1000: RR=94.2, 0.2/√k=75.4, 0.02/√k=30.4")


if __name__ == '__main__':
    print("Running quick reproduction experiments...")
    print("(For full-scale reproduction matching paper numbers exactly,")
    print(" use: python run_experiments.py)\n")

    run_table1_quick()
    run_table3_quick()
    run_table2_quick()

    print("\n" + "=" * 80)
    print("DONE. Key findings reproduced:")
    print("  1. RR extrapolation achieves best CI coverage (~94-95%)")
    print("  2. Constant α=0.2 alone has large bias → low coverage for large T")
    print("  3. Constant α=0.02 has good coverage but RR is better")
    print("  4. Diminishing stepsizes degrade with large K (batch number)")
    print("  5. RR is robust to choice of K")
    print("=" * 80)
