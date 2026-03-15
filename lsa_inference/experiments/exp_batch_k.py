"""Table 2: Effect of batch number K (T=10^6, single problem, 500 trajectories).

Methods: RR(0.2+0.02), diminishing 0.2/sqrt(k), diminishing 0.02/sqrt(k).
K values: 50, 100, 500, 1000.
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
    T = 1_000_000
    n_states = 10
    d = 5
    n_traj = 500
    burn_in = min(1000, T // 10)
    K_values = [50, 100, 500, 1000]

    rng = np.random.default_rng(123)
    P, pi = generate_transition_matrix(n_states, rng)
    A_list, A_bar = generate_A(n_states, d, pi, rng)
    b_list = generate_b(n_states, d, rng)
    theta_star = compute_theta_star(A_list, b_list, pi)

    methods = ['RR', 'dim_0.2', 'dim_0.02']
    method_labels = {
        'RR': 'RR (0.2+0.02)',
        'dim_0.2': '0.2/√k (dim)',
        'dim_0.02': '0.02/√k (dim)',
    }

    results = []

    for K in K_values:
        print(f"\n--- K = {K} ---")
        cov_accum = {m: [] for m in methods}

        for traj_idx in tqdm(range(n_traj), desc=f"K={K}"):
            traj_rng = np.random.default_rng(rng.integers(0, 2**31))
            traj = simulate_chain(P, pi, T, traj_rng)

            # RR
            tt, St, lo, hi, se, n = run_rr_extrapolation(
                A_list, b_list, traj, [0.2, 0.02], K, burn_in)
            cov_accum['RR'].append(float(lo <= theta_star[0] <= hi))

            # Diminishing 0.2/sqrt(k)
            bm, n_eff = run_lsa_diminishing(A_list, b_list, traj, 0.2, 0.5, K)
            tb, Sh = compute_covariance(bm, n_eff)
            cov_accum['dim_0.2'].append(
                coverage(theta_star, tb, Sh, K, n_eff))

            # Diminishing 0.02/sqrt(k)
            bm, n_eff = run_lsa_diminishing(A_list, b_list, traj, 0.02, 0.5, K)
            tb, Sh = compute_covariance(bm, n_eff)
            cov_accum['dim_0.02'].append(
                coverage(theta_star, tb, Sh, K, n_eff))

        for m in methods:
            vals = np.array(cov_accum[m])
            mean_cov = np.mean(vals) * 100
            se_cov = np.std(vals) / np.sqrt(n_traj) * 100
            print(f"  {method_labels[m]}: {mean_cov:.1f}% ± {se_cov:.1f}%")
            results.append({
                'K': K,
                'method': method_labels[m],
                'coverage_pct': mean_cov,
                'se_pct': se_cov,
            })

    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("Table 2: Effect of Batch Number K (T=10^6)")
    print("=" * 60)
    pivot = df.pivot(index='K', columns='method', values='coverage_pct')
    print(pivot.to_string())

    df.to_csv('results_table2.csv', index=False)
    print("\nResults saved to results_table2.csv")


if __name__ == '__main__':
    main()
