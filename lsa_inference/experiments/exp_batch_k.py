"""Table 2: Effect of batch number K (T=10^6, single problem, 500 trajectories).

Methods: RR(0.2+0.02), diminishing 0.2/sqrt(k), diminishing 0.02/sqrt(k).
K values: 50, 100, 500, 1000.

Vectorized: all trajectories run simultaneously.
"""

import argparse
import time

import numpy as np
import pandas as pd

from lsa_inference.markov_chain import simulate_chains_batch
from lsa_inference.logging_utils import setup_logger
from lsa_inference.experiments.common import (
    METHOD_LABELS, generate_problem, run_methods,
)


def main(T=1_000_000, n_traj=500):
    logger, log_path = setup_logger("table2")

    n_states, d = 10, 5
    burn_in = min(1000, T // 10)
    K_values = [50, 100, 500, 1000]
    methods = ['RR', 'dim_0.2', 'dim_0.02']

    logger.info(f"[Config] T={T:,}, n_traj={n_traj}, n_states={n_states}, d={d}, "
                f"burn_in={burn_in}")
    logger.info(f"[Config] K values: {K_values}")
    logger.info(f"[Config] log_file={log_path}")

    rng = np.random.default_rng(123)
    P, pi, A_bar, theta_star, A_arr, b_arr = generate_problem(n_states, d, rng)

    evals = np.linalg.eigvals(A_bar)
    logger.info(f"[Problem] ||θ*||={np.linalg.norm(theta_star):.4f}, "
                f"max_re(λ(Ā))={np.max(np.real(evals)):.4f}")

    logger.info(f"\nGenerating {n_traj} trajectories of length T={T:,}...")
    t_gen = time.time()
    trajs = simulate_chains_batch(P, pi, T, n_traj, rng)
    logger.info(f"Chain generation done in {time.time()-t_gen:.1f}s")

    rows = []
    t_start_all = time.time()

    for ki, K in enumerate(K_values):
        logger.info(f"\n{'='*60}")
        logger.info(f"[K={K}] Starting ({ki+1}/{len(K_values)})")
        logger.info(f"{'='*60}")
        t_k = time.time()

        raw = run_methods(A_arr, b_arr, trajs, K, burn_in, theta_star, methods)

        for m in methods:
            c = raw[m]['cov']
            cov_mean = float(np.nanmean(c)) * 100
            cov_se = float(np.std(c) / np.sqrt(n_traj)) * 100
            logger.info(f"  {METHOD_LABELS[m]} done: {cov_mean:.1f}% ± {cov_se:.1f}%")
            rows.append({'K': K, 'method': METHOD_LABELS[m],
                         'coverage_pct': cov_mean, 'se_pct': cov_se})

        logger.info(f"[K={K}] Done in {time.time()-t_k:.1f}s ({(time.time()-t_k)/60:.1f}min)")

    t_total = time.time() - t_start_all
    df = pd.DataFrame(rows)
    logger.info(f"\n{'='*60}")
    logger.info(f"Table 2: Effect of Batch Number K (T={T:,})")
    logger.info(f"Total time: {t_total:.0f}s ({t_total/60:.1f}min)")
    logger.info("=" * 60)
    pivot = df.pivot(index='K', columns='method', values='coverage_pct')
    logger.info(f"\n{pivot.to_string()}")

    df.to_csv('results_table2.csv', index=False)
    logger.info(f"\nResults saved to results_table2.csv")
    logger.info(f"Full log saved to {log_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Table 2: Effect of batch number K")
    parser.add_argument("-T", type=int, default=1_000_000)
    parser.add_argument("--n-traj", type=int, default=500)
    args = parser.parse_args()
    main(T=args.T, n_traj=args.n_traj)
