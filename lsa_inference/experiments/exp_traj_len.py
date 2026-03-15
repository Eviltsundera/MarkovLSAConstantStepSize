"""Table 3: Effect of trajectory length T (single problem, 500 trajectories).

Methods: RR(0.2+0.02), constant alpha=0.2, constant alpha=0.02,
         diminishing 0.2/sqrt(k).
T values: 10^3, 10^4, 10^5, 10^6.

Vectorized: all trajectories run simultaneously.
"""

import time
import numpy as np
import pandas as pd

from lsa_inference.markov_chain import generate_transition_matrix, simulate_chains_batch
from lsa_inference.lsa_problem import generate_A, generate_b, compute_theta_star
from lsa_inference.vectorized import (
    _prepare_arrays, run_lsa_batched_vec, run_lsa_diminishing_vec,
    compute_metrics_vec, run_rr_vec,
)
from lsa_inference.logging_utils import setup_logger


def main():
    logger, log_path = setup_logger("table3")

    n_states = 10
    d = 5
    n_traj = 500
    T_values = [1_000, 10_000, 100_000, 1_000_000]

    logger.info(f"[Config] n_traj={n_traj}, n_states={n_states}, d={d}")
    logger.info(f"[Config] T values: {[f'{t:,}' for t in T_values]}")
    logger.info(f"[Config] log_file={log_path}")

    rng = np.random.default_rng(456)
    P, pi = generate_transition_matrix(n_states, rng)
    A_list, A_bar = generate_A(n_states, d, pi, rng)
    b_list = generate_b(n_states, d, rng)
    theta_star = compute_theta_star(A_list, b_list, pi)
    A_arr, b_arr = _prepare_arrays(A_list, b_list)

    evals = np.linalg.eigvals(A_bar)
    logger.info(f"[Problem] ||θ*||={np.linalg.norm(theta_star):.4f}, "
                f"max_re(λ(Ā))={np.max(np.real(evals)):.4f}")

    methods = ['RR', 'alpha_0.2', 'alpha_0.02', 'dim_0.2']
    method_labels = {
        'RR': 'RR (0.2+0.02)',
        'alpha_0.2': 'α=0.2 (const)',
        'alpha_0.02': 'α=0.02 (const)',
        'dim_0.2': '0.2/√k (dim)',
    }

    results = []
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

        t_T = time.time()

        # RR
        t0 = time.time()
        l2, w, c = run_rr_vec(A_arr, b_arr, trajs, [0.2, 0.02], K, burn_in,
                               theta_star=theta_star)
        rr_cov = float(np.nanmean(c)) * 100
        rr_l2 = float(np.nanmean(l2))
        logger.info(f"  RR: cov={rr_cov:.1f}%, L2={rr_l2:.2e} ({time.time()-t0:.1f}s)")

        # Constant alpha=0.2
        t0 = time.time()
        bm, n = run_lsa_batched_vec(A_arr, b_arr, trajs, 0.2, K, burn_in)
        l2, w, c = compute_metrics_vec(bm, n, theta_star)
        a02_cov = float(np.nanmean(c)) * 100
        a02_l2 = float(np.nanmean(l2))
        logger.info(f"  α=0.2: cov={a02_cov:.1f}%, L2={a02_l2:.2e} ({time.time()-t0:.1f}s)")

        # Constant alpha=0.02
        t0 = time.time()
        bm, n = run_lsa_batched_vec(A_arr, b_arr, trajs, 0.02, K, burn_in)
        l2, w, c = compute_metrics_vec(bm, n, theta_star)
        a002_cov = float(np.nanmean(c)) * 100
        a002_l2 = float(np.nanmean(l2))
        logger.info(f"  α=0.02: cov={a002_cov:.1f}%, L2={a002_l2:.2e} ({time.time()-t0:.1f}s)")

        # Diminishing 0.2/sqrt(k)
        t0 = time.time()
        bm, n_eff = run_lsa_diminishing_vec(A_arr, b_arr, trajs, 0.2, 0.5, K)
        l2, w, c = compute_metrics_vec(bm, n_eff, theta_star)
        dim_cov = float(np.nanmean(c)) * 100
        dim_l2 = float(np.nanmean(l2))
        logger.info(f"  dim 0.2/√k: cov={dim_cov:.1f}%, L2={dim_l2:.2e} ({time.time()-t0:.1f}s)")

        t_T_elapsed = time.time() - t_T
        logger.info(f"[T={T:,}] All methods done in {t_T_elapsed:.1f}s ({t_T_elapsed/60:.1f}min)")

        for m, cov_val, l2_val in [
            ('RR', rr_cov, rr_l2), ('alpha_0.2', a02_cov, a02_l2),
            ('alpha_0.02', a002_cov, a002_l2), ('dim_0.2', dim_cov, dim_l2)
        ]:
            se_val = 0.0  # would need per-traj data for SE
            results.append({
                'T': T, 'method': method_labels[m],
                'coverage_pct': cov_val, 'l2_mean': l2_val,
            })

    t_total = time.time() - t_start_all
    df = pd.DataFrame(results)
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
    main()
