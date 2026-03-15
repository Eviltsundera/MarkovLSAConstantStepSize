"""Table 2: Effect of batch number K (T=10^6, single problem, 500 trajectories).

Methods: RR(0.2+0.02), diminishing 0.2/sqrt(k), diminishing 0.02/sqrt(k).
K values: 50, 100, 500, 1000.

Vectorized: all trajectories run simultaneously.
"""

import time
import numpy as np
import pandas as pd

from lsa_inference.markov_chain import generate_transition_matrix, simulate_chains_batch
from lsa_inference.lsa_problem import generate_A, generate_b, compute_theta_star
from lsa_inference.vectorized import (
    _prepare_arrays, run_lsa_diminishing_vec, compute_metrics_vec, run_rr_vec,
)
from lsa_inference.logging_utils import setup_logger


def main():
    logger, log_path = setup_logger("table2")

    T = 1_000_000
    n_states = 10
    d = 5
    n_traj = 500
    burn_in = min(1000, T // 10)
    K_values = [50, 100, 500, 1000]

    logger.info(f"[Config] T={T:,}, n_traj={n_traj}, n_states={n_states}, d={d}, "
                f"burn_in={burn_in}")
    logger.info(f"[Config] K values: {K_values}")
    logger.info(f"[Config] log_file={log_path}")

    rng = np.random.default_rng(123)
    P, pi = generate_transition_matrix(n_states, rng)
    A_list, A_bar = generate_A(n_states, d, pi, rng)
    b_list = generate_b(n_states, d, rng)
    theta_star = compute_theta_star(A_list, b_list, pi)
    A_arr, b_arr = _prepare_arrays(A_list, b_list)

    evals = np.linalg.eigvals(A_bar)
    logger.info(f"[Problem] ||θ*||={np.linalg.norm(theta_star):.4f}, "
                f"max_re(λ(Ā))={np.max(np.real(evals)):.4f}")

    methods = ['RR', 'dim_0.2', 'dim_0.02']
    method_labels = {
        'RR': 'RR (0.2+0.02)',
        'dim_0.2': '0.2/√k (dim)',
        'dim_0.02': '0.02/√k (dim)',
    }

    logger.info(f"\nGenerating {n_traj} trajectories of length T={T:,}...")
    t_gen = time.time()
    trajs = simulate_chains_batch(P, pi, T, n_traj, rng)
    logger.info(f"Chain generation done in {time.time()-t_gen:.1f}s")

    results = []
    t_start_all = time.time()

    for ki, K in enumerate(K_values):
        logger.info(f"\n{'='*60}")
        logger.info(f"[K={K}] Starting ({ki+1}/{len(K_values)})")
        logger.info(f"{'='*60}")
        t_k = time.time()

        # RR
        l2, w, c = run_rr_vec(A_arr, b_arr, trajs, [0.2, 0.02], K, burn_in,
                               theta_star=theta_star)
        rr_cov = float(np.nanmean(c)) * 100
        rr_se = float(np.std(c) / np.sqrt(n_traj)) * 100
        logger.info(f"  RR done: {rr_cov:.1f}% ± {rr_se:.1f}%")

        # Diminishing 0.2/sqrt(k)
        bm, n_eff = run_lsa_diminishing_vec(A_arr, b_arr, trajs, 0.2, 0.5, K)
        l2, w, c = compute_metrics_vec(bm, n_eff, theta_star)
        dim02_cov = float(np.nanmean(c)) * 100
        dim02_se = float(np.std(c) / np.sqrt(n_traj)) * 100
        logger.info(f"  dim 0.2/√k done: {dim02_cov:.1f}% ± {dim02_se:.1f}%")

        # Diminishing 0.02/sqrt(k)
        bm, n_eff = run_lsa_diminishing_vec(A_arr, b_arr, trajs, 0.02, 0.5, K)
        l2, w, c = compute_metrics_vec(bm, n_eff, theta_star)
        dim002_cov = float(np.nanmean(c)) * 100
        dim002_se = float(np.std(c) / np.sqrt(n_traj)) * 100
        logger.info(f"  dim 0.02/√k done: {dim002_cov:.1f}% ± {dim002_se:.1f}%")

        t_k_elapsed = time.time() - t_k
        logger.info(f"[K={K}] Done in {t_k_elapsed:.1f}s ({t_k_elapsed/60:.1f}min)")

        results.append({'K': K, 'method': method_labels['RR'],
                        'coverage_pct': rr_cov, 'se_pct': rr_se})
        results.append({'K': K, 'method': method_labels['dim_0.2'],
                        'coverage_pct': dim02_cov, 'se_pct': dim02_se})
        results.append({'K': K, 'method': method_labels['dim_0.02'],
                        'coverage_pct': dim002_cov, 'se_pct': dim002_se})

    t_total = time.time() - t_start_all
    df = pd.DataFrame(results)
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
    main()
