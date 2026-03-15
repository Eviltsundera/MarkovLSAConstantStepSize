"""Table 1: Main comparison — 100 random Markovian problems, T=10^5.

Methods: constant alpha=0.2, constant alpha=0.02, RR(0.2+0.02),
         diminishing 0.2/sqrt(k), diminishing 0.02/sqrt(k).

Metrics: L2 error, CI width, coverage (percentiles across 100 problems).
Each problem: 100 independent trajectories.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from lsa_inference.markov_chain import generate_transition_matrix, simulate_chain
from lsa_inference.lsa_problem import generate_A, generate_b, compute_theta_star
from lsa_inference.lsa_runner import run_lsa_batched, run_lsa_diminishing
from lsa_inference.batch_inference import compute_covariance
from lsa_inference.rr_extrapolation import run_rr_extrapolation
from lsa_inference.utils import l2_error, ci_width, coverage


def run_single_problem(A_list, b_list, P, pi, theta_star, T, K, n_traj,
                       burn_in, rng):
    """Run all methods on one problem instance across n_traj trajectories."""
    d = len(theta_star)
    methods = ['alpha_0.2', 'alpha_0.02', 'RR', 'dim_0.2', 'dim_0.02']
    results = {m: {'l2': [], 'width': [], 'cov': []} for m in methods}

    for _ in range(n_traj):
        traj = simulate_chain(P, pi, T, rng)

        # --- Constant alpha=0.2 ---
        bm, n = run_lsa_batched(A_list, b_list, traj, 0.2, K, burn_in)
        tb, Sh = compute_covariance(bm, n)
        results['alpha_0.2']['l2'].append(l2_error(tb, theta_star))
        results['alpha_0.2']['width'].append(ci_width(Sh, K, n))
        results['alpha_0.2']['cov'].append(coverage(theta_star, tb, Sh, K, n))

        # --- Constant alpha=0.02 ---
        bm, n = run_lsa_batched(A_list, b_list, traj, 0.02, K, burn_in)
        tb, Sh = compute_covariance(bm, n)
        results['alpha_0.02']['l2'].append(l2_error(tb, theta_star))
        results['alpha_0.02']['width'].append(ci_width(Sh, K, n))
        results['alpha_0.02']['cov'].append(coverage(theta_star, tb, Sh, K, n))

        # --- RR (0.2 + 0.02) ---
        tt, St, lo, hi, se, n = run_rr_extrapolation(
            A_list, b_list, traj, [0.2, 0.02], K, burn_in)
        results['RR']['l2'].append(l2_error(tt, theta_star))
        results['RR']['width'].append(2 * 1.96 * se)
        results['RR']['cov'].append(float(lo <= theta_star[0] <= hi))

        # --- Diminishing 0.2/sqrt(k) ---
        bm, n_eff = run_lsa_diminishing(A_list, b_list, traj, 0.2, 0.5, K)
        tb, Sh = compute_covariance(bm, n_eff)
        results['dim_0.2']['l2'].append(l2_error(tb, theta_star))
        results['dim_0.2']['width'].append(ci_width(Sh, K, n_eff))
        results['dim_0.2']['cov'].append(coverage(theta_star, tb, Sh, K, n_eff))

        # --- Diminishing 0.02/sqrt(k) ---
        bm, n_eff = run_lsa_diminishing(A_list, b_list, traj, 0.02, 0.5, K)
        tb, Sh = compute_covariance(bm, n_eff)
        results['dim_0.02']['l2'].append(l2_error(tb, theta_star))
        results['dim_0.02']['width'].append(ci_width(Sh, K, n_eff))
        results['dim_0.02']['cov'].append(coverage(theta_star, tb, Sh, K, n_eff))

    # Average over trajectories for this problem
    summary = {}
    for m in methods:
        summary[m] = {
            'l2': np.mean(results[m]['l2']),
            'width': np.mean(results[m]['width']),
            'cov': np.mean(results[m]['cov']),
        }
    return summary


def main(n_problems=100, n_traj=100, T=100_000):
    n_states = 10
    d = 5
    K = int(T ** 0.3)  # ~16
    burn_in = min(1000, T // 10)

    master_rng = np.random.default_rng(42)

    methods = ['alpha_0.2', 'alpha_0.02', 'RR', 'dim_0.2', 'dim_0.02']
    all_results = {m: {'l2': [], 'width': [], 'cov': []} for m in methods}

    for prob_idx in tqdm(range(n_problems), desc="Problems"):
        prob_seed = master_rng.integers(0, 2**31)
        prob_rng = np.random.default_rng(prob_seed)

        P, pi = generate_transition_matrix(n_states, prob_rng)
        A_list, A_bar = generate_A(n_states, d, pi, prob_rng)
        b_list = generate_b(n_states, d, prob_rng)
        theta_star = compute_theta_star(A_list, b_list, pi)

        traj_rng = np.random.default_rng(prob_rng.integers(0, 2**31))
        summary = run_single_problem(A_list, b_list, P, pi, theta_star,
                                     T, K, n_traj, burn_in, traj_rng)

        for m in methods:
            all_results[m]['l2'].append(summary[m]['l2'])
            all_results[m]['width'].append(summary[m]['width'])
            all_results[m]['cov'].append(summary[m]['cov'])

    # Report percentiles
    percentiles = [10, 25, 50, 75, 90]
    print("\n" + "=" * 80)
    print(f"Table 1: Main Comparison ({n_problems} problems, T={T})")
    print("=" * 80)

    method_labels = {
        'alpha_0.2': 'α=0.2 (const)',
        'alpha_0.02': 'α=0.02 (const)',
        'RR': 'RR (0.2+0.02)',
        'dim_0.2': '0.2/√k (dim)',
        'dim_0.02': '0.02/√k (dim)',
    }

    for metric_name, scale, unit in [('l2', 1e3, '×1e-3'), ('width', 1e3, '×1e-3'), ('cov', 100, '%')]:
        print(f"\n--- {metric_name} ({unit}) ---")
        header = f"{'Method':<20}" + "".join(f"{'p'+str(p):>10}" for p in percentiles)
        print(header)
        for m in methods:
            vals = np.array(all_results[m][metric_name]) * scale
            pcts = np.percentile(vals, percentiles)
            row = f"{method_labels[m]:<20}" + "".join(f"{v:>10.2f}" for v in pcts)
            print(row)

    # Save results
    rows = []
    for m in methods:
        for metric in ['l2', 'width', 'cov']:
            vals = np.array(all_results[m][metric])
            rows.append({
                'method': method_labels[m],
                'metric': metric,
                'p10': np.percentile(vals, 10),
                'p25': np.percentile(vals, 25),
                'p50': np.percentile(vals, 50),
                'p75': np.percentile(vals, 75),
                'p90': np.percentile(vals, 90),
                'mean': np.mean(vals),
            })
    df = pd.DataFrame(rows)
    df.to_csv('results_table1.csv', index=False)
    print("\nResults saved to results_table1.csv")


if __name__ == '__main__':
    main()
