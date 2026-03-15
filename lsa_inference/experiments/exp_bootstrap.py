"""Table 4 (bootstrap comparison): Constant + RR vs Bootstrapping (T=10^6).

Metrics: coverage, L2 error, CI width.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from lsa_inference.markov_chain import generate_transition_matrix, simulate_chain
from lsa_inference.lsa_problem import generate_A, generate_b, compute_theta_star
from lsa_inference.lsa_runner import run_lsa_batched
from lsa_inference.batch_inference import compute_covariance
from lsa_inference.rr_extrapolation import run_rr_extrapolation
from lsa_inference.utils import l2_error, ci_width, coverage


def bootstrap_lsa(A_list, b_list, trajectory, alpha, n_bootstrap=200,
                  burn_in=100, q=0.05, coord=0):
    """Bootstrap inference for LSA: resample trajectory blocks and re-run LSA.

    Uses block bootstrap with block size proportional to mixing time.
    """
    T = len(trajectory)
    d = len(b_list[0])

    # Run original LSA to get point estimate
    theta = np.zeros(d)
    iterates = np.zeros((T, d))
    for t in range(T):
        x = trajectory[t]
        theta = theta + alpha * (A_list[x] @ theta + b_list[x])
        iterates[t] = theta

    theta_bar = np.mean(iterates[burn_in:], axis=0)

    # Block bootstrap
    block_size = max(int(np.log(T) / alpha), 100)
    n_blocks = (T - burn_in) // block_size
    post_iterates = iterates[burn_in:]

    bootstrap_means = np.zeros((n_bootstrap, d))
    rng = np.random.default_rng(999)

    for b in range(n_bootstrap):
        # Resample blocks
        block_indices = rng.choice(n_blocks, size=n_blocks, replace=True)
        resampled = np.concatenate([
            post_iterates[i * block_size:(i + 1) * block_size]
            for i in block_indices
        ], axis=0)
        bootstrap_means[b] = np.mean(resampled, axis=0)

    # Bootstrap CI (percentile method)
    lo = np.percentile(bootstrap_means[:, coord], 100 * q / 2)
    hi = np.percentile(bootstrap_means[:, coord], 100 * (1 - q / 2))
    width = hi - lo

    return theta_bar, lo, hi, width


def main():
    T = 1_000_000
    n_states = 10
    d = 5
    n_traj = 500
    burn_in = 1000
    K = int(T ** 0.3)

    rng = np.random.default_rng(789)
    P, pi = generate_transition_matrix(n_states, rng)
    A_list, A_bar = generate_A(n_states, d, pi, rng)
    b_list = generate_b(n_states, d, rng)
    theta_star = compute_theta_star(A_list, b_list, pi)

    results = {'RR': {'cov': [], 'l2': [], 'width': []},
               'Bootstrap': {'cov': [], 'l2': [], 'width': []}}

    for _ in tqdm(range(n_traj), desc="Trajectories"):
        traj_rng = np.random.default_rng(rng.integers(0, 2**31))
        traj = simulate_chain(P, pi, T, traj_rng)

        # RR
        tt, St, lo, hi, se, n = run_rr_extrapolation(
            A_list, b_list, traj, [0.2, 0.02], K, burn_in)
        results['RR']['cov'].append(float(lo <= theta_star[0] <= hi))
        results['RR']['l2'].append(l2_error(tt, theta_star))
        results['RR']['width'].append(2 * 1.96 * se)

        # Bootstrap
        tb, lo_b, hi_b, width_b = bootstrap_lsa(
            A_list, b_list, traj, 0.02, n_bootstrap=200, burn_in=burn_in)
        results['Bootstrap']['cov'].append(
            float(lo_b <= theta_star[0] <= hi_b))
        results['Bootstrap']['l2'].append(l2_error(tb, theta_star))
        results['Bootstrap']['width'].append(width_b)

    print("\n" + "=" * 60)
    print("Table: RR vs Bootstrap Comparison (T=10^6)")
    print("=" * 60)

    for method in ['RR', 'Bootstrap']:
        cov_mean = np.mean(results[method]['cov']) * 100
        l2_mean = np.mean(results[method]['l2'])
        w_mean = np.mean(results[method]['width'])
        print(f"{method:>12}: Coverage={cov_mean:.1f}%, "
              f"L2={l2_mean:.2e}, CI Width={w_mean:.5f}")

    rows = []
    for method in ['RR', 'Bootstrap']:
        rows.append({
            'method': method,
            'coverage_pct': np.mean(results[method]['cov']) * 100,
            'l2_error': np.mean(results[method]['l2']),
            'ci_width': np.mean(results[method]['width']),
        })
    df = pd.DataFrame(rows)
    df.to_csv('results_bootstrap.csv', index=False)
    print("\nResults saved to results_bootstrap.csv")


if __name__ == '__main__':
    main()
