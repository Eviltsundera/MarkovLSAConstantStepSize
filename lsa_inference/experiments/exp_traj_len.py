"""Table 3: Effect of trajectory length T (single problem, 500 trajectories).

Methods: RR(0.2+0.02), constant alpha=0.2, constant alpha=0.02,
         diminishing 0.2/sqrt(k).
T values: 10^3, 10^4, 10^5, 10^6.

Vectorized: all trajectories run simultaneously.
"""

import argparse
import time

import numpy as np
import pandas as pd

from lsa_inference.markov_chain import simulate_chains_batch
from lsa_inference.logging_utils import setup_logger
from lsa_inference.experiments.common import (
    METHOD_LABELS, generate_problem, run_methods, summarize,
)


def main(n_traj=500):
    logger, log_path = setup_logger("table3")

    n_states, d = 10, 5
    T_values = [1_000, 10_000, 100_000, 1_000_000]
    methods = ['RR', 'alpha_0.2', 'alpha_0.02', 'dim_0.2']

    logger.info(f"[Config] n_traj={n_traj}, n_states={n_states}, d={d}")
    logger.info(f"[Config] T values: {[f'{t:,}' for t in T_values]}")
    logger.info(f"[Config] log_file={log_path}")

    rng = np.random.default_rng(456)
    P, pi, A_bar, theta_star, A_arr, b_arr = generate_problem(n_states, d, rng)

    evals = np.linalg.eigvals(A_bar)
    logger.info(f"[Problem] ||θ*||={np.linalg.norm(theta_star):.4f}, "
                f"max_re(λ(Ā))={np.max(np.real(evals)):.4f}")

    rows = []
    t_start_all = time.time()

    for ti, T in enumerate(T_values):
        K = max(int(T ** 0.3), 5)
        burn_in = min(1000, T // 10)

        logger.info(f"\n{'='*60}")
        logger.info(f"[T={T:,}] Starting ({ti+1}/{len(T_values)}), K={K}, burn_in={burn_in}")
        logger.info(f"{'='*60}")

        t_gen = time.time()
        trajs = simulate_chains_batch(P, pi, T, n_traj, rng)
        logger.info(f"  Chain generation: {time.time()-t_gen:.1f}s")

        t0 = time.time()
        raw = run_methods(A_arr, b_arr, trajs, K, burn_in, theta_star, methods)
        summary = summarize(raw)

        for m in methods:
            cov = summary[m]['cov'] * 100
            l2 = summary[m]['l2']
            logger.info(f"  {METHOD_LABELS[m]}: cov={cov:.1f}%, L2={l2:.2e} "
                        f"({time.time()-t0:.1f}s)")

            rows.append({'T': T, 'method': METHOD_LABELS[m],
                         'coverage_pct': cov, 'l2_mean': l2})

        logger.info(f"[T={T:,}] All methods done in {time.time()-t0:.1f}s "
                    f"({(time.time()-t0)/60:.1f}min)")

    t_total = time.time() - t_start_all
    df = pd.DataFrame(rows)
    logger.info(f"\n{'='*60}")
    logger.info(f"Table 3: Effect of Trajectory Length T")
    logger.info(f"Total time: {t_total:.0f}s ({t_total/60:.1f}min)")
    logger.info("=" * 60)
    pivot = df.pivot(index='T', columns='method', values='coverage_pct')
    logger.info(f"\n{pivot.to_string()}")

    df.to_csv('results_table3.csv', index=False)
    logger.info(f"\nResults saved to results_table3.csv")
    logger.info(f"Full log saved to {log_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Table 3: Effect of trajectory length T")
    parser.add_argument("--n-traj", type=int, default=500)
    args = parser.parse_args()
    main(n_traj=args.n_traj)
