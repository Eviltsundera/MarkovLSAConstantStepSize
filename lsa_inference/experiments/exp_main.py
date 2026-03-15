"""Table 1: Main comparison — 100 random Markovian problems, T=10^5.

Methods: constant alpha=0.2, constant alpha=0.02, RR(0.2+0.02),
         diminishing 0.2/sqrt(k), diminishing 0.02/sqrt(k).

Metrics: L2 error, CI width, coverage (percentiles across 100 problems).
Each problem: 100 independent trajectories.
"""

import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from lsa_inference.markov_chain import generate_transition_matrix, simulate_chain
from lsa_inference.lsa_problem import generate_A, generate_b, compute_theta_star
from lsa_inference.lsa_runner import run_lsa_batched, run_lsa_diminishing
from lsa_inference.batch_inference import compute_covariance
from lsa_inference.rr_extrapolation import run_rr_extrapolation
from lsa_inference.utils import l2_error, ci_width, coverage


def run_single_problem(A_list, b_list, P, pi, theta_star, T, K, n_traj,
                       burn_in, rng, prob_idx):
    """Run all methods on one problem instance across n_traj trajectories."""
    d = len(theta_star)
    methods = ['alpha_0.2', 'alpha_0.02', 'RR', 'dim_0.2', 'dim_0.02']
    results = {m: {'l2': [], 'width': [], 'cov': []} for m in methods}

    for traj_idx in tqdm(range(n_traj), desc=f"  Problem {prob_idx} trajectories",
                         leave=False):
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

    print(f"[Config] n_problems={n_problems}, n_traj={n_traj}, T={T:,}, "
          f"n_states={n_states}, d={d}, K={K}, burn_in={burn_in}")

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

    t_start_all = time.time()

    for prob_idx in range(n_problems):
        t_prob_start = time.time()
        prob_seed = master_rng.integers(0, 2**31)
        prob_rng = np.random.default_rng(prob_seed)

        P, pi = generate_transition_matrix(n_states, prob_rng)
        A_list, A_bar = generate_A(n_states, d, pi, prob_rng)
        b_list = generate_b(n_states, d, prob_rng)
        theta_star = compute_theta_star(A_list, b_list, pi)

        evals = np.linalg.eigvals(A_bar)
        max_real = np.max(np.real(evals))
        print(f"\n[Problem {prob_idx+1}/{n_problems}] seed={prob_seed}, "
              f"||θ*||={np.linalg.norm(theta_star):.4f}, "
              f"max_re(λ(Ā))={max_real:.4f}")

        traj_rng = np.random.default_rng(prob_rng.integers(0, 2**31))
        summary = run_single_problem(A_list, b_list, P, pi, theta_star,
                                     T, K, n_traj, burn_in, traj_rng, prob_idx + 1)

        for m in methods:
            all_results[m]['l2'].append(summary[m]['l2'])
            all_results[m]['width'].append(summary[m]['width'])
            all_results[m]['cov'].append(summary[m]['cov'])

        t_prob = time.time() - t_prob_start
        t_elapsed = time.time() - t_start_all
        t_per_prob = t_elapsed / (prob_idx + 1)
        t_remaining = t_per_prob * (n_problems - prob_idx - 1)

        print(f"  Results: "
              f"RR cov={summary['RR']['cov']*100:.0f}%, "
              f"α=0.02 cov={summary['alpha_0.02']['cov']*100:.0f}%, "
              f"α=0.2 cov={summary['alpha_0.2']['cov']*100:.0f}%, "
              f"dim0.2 cov={summary['dim_0.2']['cov']*100:.0f}%, "
              f"dim0.02 cov={summary['dim_0.02']['cov']*100:.0f}%")
        print(f"  L2 (×1e-3): "
              f"RR={summary['RR']['l2']*1e3:.2f}, "
              f"α=0.02={summary['alpha_0.02']['l2']*1e3:.2f}, "
              f"α=0.2={summary['alpha_0.2']['l2']*1e3:.2f}")
        print(f"  Time: {t_prob:.1f}s this problem | "
              f"{t_elapsed:.0f}s elapsed | "
              f"~{t_remaining:.0f}s remaining ({t_remaining/60:.1f}min)")

        # Print running medians every 10 problems
        if (prob_idx + 1) % 10 == 0:
            n_done = prob_idx + 1
            print(f"\n  --- Running medians after {n_done} problems ---")
            for m in methods:
                med_cov = np.median(all_results[m]['cov'][:n_done]) * 100
                med_l2 = np.median(all_results[m]['l2'][:n_done]) * 1e3
                print(f"    {method_labels[m]:<20}: cov={med_cov:.1f}%, L2={med_l2:.2f}×1e-3")

    # Report percentiles
    t_total = time.time() - t_start_all
    percentiles = [10, 25, 50, 75, 90]
    print("\n" + "=" * 80)
    print(f"Table 1: Main Comparison ({n_problems} problems, T={T})")
    print(f"Total time: {t_total:.0f}s ({t_total/60:.1f}min)")
    print("=" * 80)

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
