"""Table 3: Effect of trajectory length T (single problem, 500 trajectories).

Methods: RR(0.2+0.02), constant alpha=0.2, constant alpha=0.02,
         diminishing 0.2/sqrt(k).
T values: 10^3, 10^4, 10^5, 10^6.
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


def main():
    n_states = 10
    d = 5
    n_traj = 500
    T_values = [1_000, 10_000, 100_000, 1_000_000]

    rng = np.random.default_rng(456)
    P, pi = generate_transition_matrix(n_states, rng)
    A_list, A_bar = generate_A(n_states, d, pi, rng)
    b_list = generate_b(n_states, d, rng)
    theta_star = compute_theta_star(A_list, b_list, pi)

    methods = ['RR', 'alpha_0.2', 'alpha_0.02', 'dim_0.2']
    method_labels = {
        'RR': 'RR (0.2+0.02)',
        'alpha_0.2': 'α=0.2 (const)',
        'alpha_0.02': 'α=0.02 (const)',
        'dim_0.2': '0.2/√k (dim)',
    }

    results = []

    for T in T_values:
        K = max(int(T ** 0.3), 5)
        burn_in = min(1000, T // 10)
        print(f"\n--- T = {T}, K = {K} ---")
        cov_accum = {m: [] for m in methods}

        for _ in tqdm(range(n_traj), desc=f"T={T}"):
            traj_rng = np.random.default_rng(rng.integers(0, 2**31))
            traj = simulate_chain(P, pi, T, traj_rng)

            # RR
            tt, St, lo, hi, se, n = run_rr_extrapolation(
                A_list, b_list, traj, [0.2, 0.02], K, burn_in)
            cov_accum['RR'].append(float(lo <= theta_star[0] <= hi))

            # Constant alpha=0.2
            bm, n = run_lsa_batched(A_list, b_list, traj, 0.2, K, burn_in)
            tb, Sh = compute_covariance(bm, n)
            cov_accum['alpha_0.2'].append(
                coverage(theta_star, tb, Sh, K, n))

            # Constant alpha=0.02
            bm, n = run_lsa_batched(A_list, b_list, traj, 0.02, K, burn_in)
            tb, Sh = compute_covariance(bm, n)
            cov_accum['alpha_0.02'].append(
                coverage(theta_star, tb, Sh, K, n))

            # Diminishing 0.2/sqrt(k)
            bm, n_eff = run_lsa_diminishing(A_list, b_list, traj, 0.2, 0.5, K)
            tb, Sh = compute_covariance(bm, n_eff)
            cov_accum['dim_0.2'].append(
                coverage(theta_star, tb, Sh, K, n_eff))

        for m in methods:
            vals = np.array(cov_accum[m])
            mean_cov = np.mean(vals) * 100
            se_cov = np.std(vals) / np.sqrt(n_traj) * 100
            print(f"  {method_labels[m]}: {mean_cov:.1f}% ± {se_cov:.1f}%")
            results.append({
                'T': T,
                'method': method_labels[m],
                'coverage_pct': mean_cov,
                'se_pct': se_cov,
            })

    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("Table 3: Effect of Trajectory Length T")
    print("=" * 60)
    pivot = df.pivot(index='T', columns='method', values='coverage_pct')
    print(pivot.to_string())

    df.to_csv('results_table3.csv', index=False)
    print("\nResults saved to results_table3.csv")


if __name__ == '__main__':
    main()
