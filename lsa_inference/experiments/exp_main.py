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

from lsa_inference.logging_utils import setup_logger
from lsa_inference.experiments.common import (
    METHODS_ALL, METHOD_LABELS, solve_problem_worker, print_percentile_table,
)


def main(n_problems=100, n_traj=100, T=100_000, n_workers=None):
    logger, log_path = setup_logger("table1")

    n_states, d = 10, 5
    K = int(T ** 0.3)
    burn_in = min(1000, T // 10)
    if n_workers is None:
        n_workers = min(mp.cpu_count(), n_problems)

    logger.info(f"[Config] n_problems={n_problems}, n_traj={n_traj}, T={T:,}, "
                f"n_states={n_states}, d={d}, K={K}, burn_in={burn_in}")
    logger.info(f"[Config] n_workers={n_workers}, log_file={log_path}")

    master_rng = np.random.default_rng(42)
    seeds = [int(master_rng.integers(0, 2**31)) for _ in range(n_problems)]

    methods = METHODS_ALL
    all_results = {m: {'l2': [], 'width': [], 'cov': []} for m in methods}

    task_args = [
        (i, seeds[i], n_traj, T, K, burn_in, n_states, d)
        for i in range(n_problems)
    ]

    t_start = time.time()
    completed = 0

    with mp.Pool(n_workers) as pool:
        for prob_idx, results, theta_norm, max_re in pool.imap_unordered(
                solve_problem_worker, task_args):
            completed += 1
            t_elapsed = time.time() - t_start
            t_remaining = t_elapsed / completed * (n_problems - completed)

            for m in methods:
                for metric in ('l2', 'width', 'cov'):
                    all_results[m][metric].append(results[m][metric])

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
                    logger.info(f"    {METHOD_LABELS[m]:<20}: "
                                f"cov={med_cov:.1f}%, L2={med_l2:.2f}×1e-3")

    t_total = time.time() - t_start

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"Table 1: Main Comparison ({n_problems} problems, T={T:,})")
    logger.info(f"Total time: {t_total:.0f}s ({t_total/60:.1f}min)")
    logger.info("=" * 80)

    print_percentile_table(logger, all_results, methods)

    # Save CSV
    rows = []
    for m in methods:
        for metric in ('l2', 'width', 'cov'):
            vals = np.array(all_results[m][metric])
            rows.append({
                'method': METHOD_LABELS[m], 'metric': metric,
                **{f'p{p}': np.percentile(vals, p) for p in [10, 25, 50, 75, 90]},
                'mean': np.mean(vals),
            })
    pd.DataFrame(rows).to_csv('results_table1.csv', index=False)
    logger.info(f"\nResults saved to results_table1.csv")
    logger.info(f"Full log saved to {log_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Table 1: Main comparison across random problems")
    parser.add_argument("--n-problems", type=int, default=100)
    parser.add_argument("--n-traj", type=int, default=100)
    parser.add_argument("-T", type=int, default=100_000)
    parser.add_argument("--n-workers", type=int, default=None)
    args = parser.parse_args()
    main(n_problems=args.n_problems, n_traj=args.n_traj, T=args.T, n_workers=args.n_workers)
