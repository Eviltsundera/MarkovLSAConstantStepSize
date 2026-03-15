"""Table 1: Main comparison — 100 random Markovian problems, T=10^5.

Methods: constant alpha=0.2, constant alpha=0.02, RR(0.2+0.02),
         diminishing 0.2/sqrt(k), diminishing 0.02/sqrt(k).

Metrics: L2 error, CI width, coverage (percentiles across 100 problems).
Each problem: 100 independent trajectories.

Parallelized via multiprocessing across problems.
Vectorized: all trajectories for a single problem run simultaneously.
"""

import argparse
import time
import multiprocessing as mp

import numpy as np
import pandas as pd

from lsa_inference.markov_chain import generate_transition_matrix, simulate_chains_batch
from lsa_inference.lsa_problem import generate_A, generate_b, compute_theta_star
from lsa_inference.vectorized import (
    _prepare_arrays, run_lsa_batched_vec, run_lsa_diminishing_vec,
    compute_metrics_vec, run_rr_vec,
)
from lsa_inference.logging_utils import setup_logger


def _solve_one_problem(args):
    """Worker function: solve one problem instance (all trajectories vectorized)."""
    prob_idx, prob_seed, n_traj, T, K, burn_in, n_states, d = args

    prob_rng = np.random.default_rng(prob_seed)
    P, pi = generate_transition_matrix(n_states, prob_rng)
    A_list, A_bar = generate_A(n_states, d, pi, prob_rng)
    b_list = generate_b(n_states, d, prob_rng)
    theta_star = compute_theta_star(A_list, b_list, pi)
    A_arr, b_arr = _prepare_arrays(A_list, b_list)

    traj_rng = np.random.default_rng(prob_rng.integers(0, 2**31))
    trajs = simulate_chains_batch(P, pi, T, n_traj, traj_rng)

    evals = np.linalg.eigvals(A_bar)
    max_re = float(np.max(np.real(evals)))
    theta_norm = float(np.linalg.norm(theta_star))

    results = {}

    # Constant alpha=0.2
    bm, n = run_lsa_batched_vec(A_arr, b_arr, trajs, 0.2, K, burn_in)
    l2, w, c = compute_metrics_vec(bm, n, theta_star)
    results['alpha_0.2'] = {'l2': float(np.nanmean(l2)), 'width': float(np.nanmean(w)),
                            'cov': float(np.nanmean(c))}

    # Constant alpha=0.02
    bm, n = run_lsa_batched_vec(A_arr, b_arr, trajs, 0.02, K, burn_in)
    l2, w, c = compute_metrics_vec(bm, n, theta_star)
    results['alpha_0.02'] = {'l2': float(np.nanmean(l2)), 'width': float(np.nanmean(w)),
                             'cov': float(np.nanmean(c))}

    # RR (0.2 + 0.02)
    l2, w, c = run_rr_vec(A_arr, b_arr, trajs, [0.2, 0.02], K, burn_in,
                           theta_star=theta_star)
    results['RR'] = {'l2': float(np.nanmean(l2)), 'width': float(np.nanmean(w)),
                     'cov': float(np.nanmean(c))}

    # Diminishing 0.2/sqrt(k)
    bm, n_eff = run_lsa_diminishing_vec(A_arr, b_arr, trajs, 0.2, 0.5, K)
    l2, w, c = compute_metrics_vec(bm, n_eff, theta_star)
    results['dim_0.2'] = {'l2': float(np.nanmean(l2)), 'width': float(np.nanmean(w)),
                          'cov': float(np.nanmean(c))}

    # Diminishing 0.02/sqrt(k)
    bm, n_eff = run_lsa_diminishing_vec(A_arr, b_arr, trajs, 0.02, 0.5, K)
    l2, w, c = compute_metrics_vec(bm, n_eff, theta_star)
    results['dim_0.02'] = {'l2': float(np.nanmean(l2)), 'width': float(np.nanmean(w)),
                           'cov': float(np.nanmean(c))}

    return prob_idx, results, theta_norm, max_re


def main(n_problems=100, n_traj=100, T=100_000, n_workers=None):
    logger, log_path = setup_logger("table1")

    n_states = 10
    d = 5
    K = int(T ** 0.3)
    burn_in = min(1000, T // 10)

    if n_workers is None:
        n_workers = min(mp.cpu_count(), n_problems)

    logger.info(f"[Config] n_problems={n_problems}, n_traj={n_traj}, T={T:,}, "
                f"n_states={n_states}, d={d}, K={K}, burn_in={burn_in}")
    logger.info(f"[Config] n_workers={n_workers}, log_file={log_path}")

    master_rng = np.random.default_rng(42)
    seeds = [int(master_rng.integers(0, 2**31)) for _ in range(n_problems)]

    methods = ['alpha_0.2', 'alpha_0.02', 'RR', 'dim_0.2', 'dim_0.02']
    method_labels = {
        'alpha_0.2': 'α=0.2 (const)',
        'alpha_0.02': 'α=0.02 (const)',
        'RR': 'RR (0.2+0.02)',
        'dim_0.2': '0.2/√k (dim)',
        'dim_0.02': '0.02/√k (dim)',
    }
    all_results = {m: {'l2': [], 'width': [], 'cov': []} for m in methods}

    task_args = [
        (i, seeds[i], n_traj, T, K, burn_in, n_states, d)
        for i in range(n_problems)
    ]

    t_start = time.time()
    completed = 0

    with mp.Pool(n_workers) as pool:
        for prob_idx, results, theta_norm, max_re in pool.imap_unordered(
                _solve_one_problem, task_args):
            completed += 1
            t_elapsed = time.time() - t_start
            t_per = t_elapsed / completed
            t_remaining = t_per * (n_problems - completed)

            for m in methods:
                all_results[m]['l2'].append(results[m]['l2'])
                all_results[m]['width'].append(results[m]['width'])
                all_results[m]['cov'].append(results[m]['cov'])

            logger.info(
                f"[Problem {completed}/{n_problems}] "
                f"||θ*||={theta_norm:.4f}, max_re(λ)={max_re:.4f} | "
                f"RR cov={results['RR']['cov']*100:.0f}%, "
                f"α0.02={results['alpha_0.02']['cov']*100:.0f}%, "
                f"α0.2={results['alpha_0.2']['cov']*100:.0f}% | "
                f"{t_elapsed:.0f}s elapsed, ~{t_remaining:.0f}s left"
            )
            logger.debug(
                f"  L2(×1e-3): RR={results['RR']['l2']*1e3:.2f}, "
                f"α0.02={results['alpha_0.02']['l2']*1e3:.2f}, "
                f"α0.2={results['alpha_0.2']['l2']*1e3:.2f}, "
                f"dim0.2={results['dim_0.2']['l2']*1e3:.2f}, "
                f"dim0.02={results['dim_0.02']['l2']*1e3:.2f}"
            )

            if completed % 10 == 0:
                logger.info(f"  --- Running medians after {completed} problems ---")
                for m in methods:
                    med_cov = np.median(all_results[m]['cov']) * 100
                    med_l2 = np.median(all_results[m]['l2']) * 1e3
                    logger.info(f"    {method_labels[m]:<20}: "
                                f"cov={med_cov:.1f}%, L2={med_l2:.2f}×1e-3")

    t_total = time.time() - t_start

    # Final report
    percentiles = [10, 25, 50, 75, 90]
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"Table 1: Main Comparison ({n_problems} problems, T={T:,})")
    logger.info(f"Total time: {t_total:.0f}s ({t_total/60:.1f}min)")
    logger.info("=" * 80)

    for metric_name, scale, unit in [('l2', 1e3, '×1e-3'), ('width', 1e3, '×1e-3'), ('cov', 100, '%')]:
        logger.info(f"\n--- {metric_name} ({unit}) ---")
        header = f"{'Method':<20}" + "".join(f"{'p'+str(p):>10}" for p in percentiles)
        logger.info(header)
        for m in methods:
            vals = np.array(all_results[m][metric_name]) * scale
            pcts = np.percentile(vals, percentiles)
            row = f"{method_labels[m]:<20}" + "".join(f"{v:>10.2f}" for v in pcts)
            logger.info(row)

    # Save CSV
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
    logger.info(f"\nResults saved to results_table1.csv")
    logger.info(f"Full log saved to {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Table 1: Main comparison across random problems")
    parser.add_argument("--n-problems", type=int, default=100, help="Number of random problems (default: 100)")
    parser.add_argument("--n-traj", type=int, default=100, help="Trajectories per problem (default: 100)")
    parser.add_argument("-T", type=int, default=100_000, help="Trajectory length (default: 100000)")
    parser.add_argument("--n-workers", type=int, default=None, help="Multiprocessing workers (default: min(cpu_count, n_problems))")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(n_problems=args.n_problems, n_traj=args.n_traj, T=args.T, n_workers=args.n_workers)
