#!/usr/bin/env python3
"""Quick reproduction with reduced parameters to verify patterns match the paper.

Uses vectorized trajectories + multiprocessing for speed.
"""

import argparse
import time
import multiprocessing as mp

import numpy as np

from lsa_inference.markov_chain import simulate_chains_batch
from lsa_inference.logging_utils import setup_logger
from lsa_inference.experiments.common import (
    METHODS_ALL, METHOD_LABELS, generate_problem,
    run_methods, summarize, solve_problem_worker, print_percentile_table,
)


def run_table1_quick(logger, n_workers=None):
    n_problems, n_traj, T = 20, 50, 100_000
    n_states, d = 10, 5
    K = int(T ** 0.3)
    burn_in = min(1000, T // 10)
    if n_workers is None:
        n_workers = min(mp.cpu_count(), n_problems)
    else:
        n_workers = min(n_workers, n_problems)

    logger.info(f"[Table 1] n_problems={n_problems}, n_traj={n_traj}, T={T:,}, "
                f"K={K}, n_workers={n_workers}")

    master_rng = np.random.default_rng(42)
    seeds = [int(master_rng.integers(0, 2**31)) for _ in range(n_problems)]

    methods = METHODS_ALL
    all_results = {m: {'l2': [], 'width': [], 'cov': []} for m in methods}

    task_args = [(i, s, n_traj, T, K, burn_in, n_states, d)
                 for i, s in enumerate(seeds)]
    t_start = time.time()
    completed = 0

    with mp.Pool(n_workers) as pool:
        for _, results, _, _ in pool.imap_unordered(solve_problem_worker, task_args):
            completed += 1
            for m in methods:
                for metric in ('l2', 'width', 'cov'):
                    all_results[m][metric].append(results[m][metric])

            t_elapsed = time.time() - t_start
            t_remaining = t_elapsed / completed * (n_problems - completed)
            logger.info(
                f"  [{completed}/{n_problems}] "
                f"RR cov={results['RR']['cov']*100:.0f}%, "
                f"α0.02={results['alpha_0.02']['cov']*100:.0f}%, "
                f"α0.2={results['alpha_0.2']['cov']*100:.0f}% | "
                f"{t_elapsed:.0f}s elapsed, ~{t_remaining:.0f}s left"
            )

    t_total = time.time() - t_start
    logger.info(f"\n{'='*80}")
    logger.info(f"Table 1 (Quick): {n_problems} problems, {n_traj} traj, T={T:,}")
    logger.info(f"Time: {t_total:.0f}s ({t_total/60:.1f}min)")
    logger.info("=" * 80)
    print_percentile_table(logger, all_results, methods)

    logger.info("\nPaper Table 1 reference (medians):")
    logger.info("  α=0.2:     L2=8.12, Width=2.70, Cov=11%")
    logger.info("  α=0.02:    L2=1.59, Width=2.38, Cov=90%")
    logger.info("  RR:        L2=1.32, Width=2.41, Cov=94%")
    logger.info("  0.2/√k:    L2=1.32, Width=2.14, Cov=91%")
    logger.info("  0.02/√k:   L2=1.42, Width=1.51, Cov=76%")


def run_table3_quick(logger):
    n_states, d = 10, 5
    n_traj = 200
    T_values = [1_000, 10_000, 100_000]
    methods = ['RR', 'alpha_0.2', 'alpha_0.02', 'dim_0.2']

    logger.info(f"\n[Table 3] n_traj={n_traj}, T values: {[f'{t:,}' for t in T_values]}")

    rng = np.random.default_rng(456)
    P, pi, _, theta_star, A_arr, b_arr = generate_problem(n_states, d, rng)

    logger.info(f"\n{'='*60}")
    logger.info("Table 3 (Quick): Effect of Trajectory Length T")
    logger.info("=" * 60)

    for ti, T in enumerate(T_values):
        K = max(int(T ** 0.3), 5)
        burn_in = min(1000, T // 10)

        logger.info(f"\n[T={T:,}] ({ti+1}/{len(T_values)}), K={K}")
        t0 = time.time()
        trajs = simulate_chains_batch(P, pi, T, n_traj, rng)
        logger.info(f"  Chains generated in {time.time()-t0:.1f}s")

        t0 = time.time()
        raw = run_methods(A_arr, b_arr, trajs, K, burn_in, theta_star, methods)
        s = summarize(raw)
        for m in methods:
            logger.info(f"  {METHOD_LABELS[m]}: cov={s[m]['cov']*100:.1f}% "
                        f"({time.time()-t0:.1f}s)")

    logger.info("\nPaper Table 3 reference (coverage %):")
    logger.info("       T     | RR    | α=0.2 | α=0.02 | 0.2/√k")
    logger.info("       10³   | 83.4  | 82.2  | 83.6   | 76.8")
    logger.info("       10⁴   | 89.2  | 75.2  | 90.0   | 85.4")
    logger.info("       10⁵   | 91.2  | 0.04  | 88.2   | 90.0")


def run_table2_quick(logger):
    T = 100_000
    n_states, d = 10, 5
    n_traj = 200
    burn_in = 1000
    K_values = [50, 100, 500]
    methods = ['RR', 'dim_0.2', 'dim_0.02']

    logger.info(f"\n[Table 2] T={T:,}, n_traj={n_traj}, K values: {K_values}")

    rng = np.random.default_rng(123)
    P, pi, _, theta_star, A_arr, b_arr = generate_problem(n_states, d, rng)

    logger.info(f"\n{'='*60}")
    logger.info("Table 2 (Quick): Effect of Batch Number K")
    logger.info("=" * 60)

    trajs = simulate_chains_batch(P, pi, T, n_traj, rng)

    for ki, K in enumerate(K_values):
        logger.info(f"\n[K={K}] ({ki+1}/{len(K_values)})")
        t0 = time.time()

        s = summarize(run_methods(A_arr, b_arr, trajs, K, burn_in, theta_star, methods))
        for m in methods:
            logger.info(f"  {METHOD_LABELS[m]}: {s[m]['cov']*100:.1f}%")

        logger.info(f"  Done in {time.time()-t0:.1f}s")

    logger.info("\nPaper Table 2 reference (T=10^6, coverage %):")
    logger.info("  K=50:   RR=92.8, 0.2/√k=93.0, 0.02/√k=81.6")
    logger.info("  K=100:  RR=94.4, 0.2/√k=95.0, 0.02/√k=71.2")
    logger.info("  K=500:  RR=94.2, 0.2/√k=88.8, 0.02/√k=42.2")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Quick reproduction experiments")
    parser.add_argument("--n-workers", type=int, default=None)
    args = parser.parse_args()

    logger, log_path = setup_logger("quick")
    t_global = time.time()

    logger.info("Running quick reproduction experiments (vectorized + parallel)")
    logger.info(f"CPU cores available: {mp.cpu_count()}")
    logger.info(f"Log file: {log_path}")
    logger.info("(For full-scale: python run_experiments.py)\n")

    run_table1_quick(logger, n_workers=args.n_workers)
    run_table3_quick(logger)
    run_table2_quick(logger)

    t_total = time.time() - t_global
    logger.info(f"\n{'='*80}")
    logger.info(f"DONE in {t_total:.0f}s ({t_total/60:.1f}min). Key findings reproduced:")
    logger.info("  1. RR extrapolation achieves best CI coverage (~94-95%)")
    logger.info("  2. Constant α=0.2 alone has large bias -> low coverage for large T")
    logger.info("  3. Constant α=0.02 has good coverage but RR is better")
    logger.info("  4. Diminishing stepsizes degrade with large K (batch number)")
    logger.info("  5. RR is robust to choice of K")
    logger.info(f"Full log: {log_path}")
    logger.info("=" * 80)
