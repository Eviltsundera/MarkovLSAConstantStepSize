"""Table 4 (bootstrap comparison): Constant + RR vs Bootstrapping (T=10^6).

Metrics: coverage, L2 error, CI width.

RR is vectorized. Bootstrap uses multiprocessing across trajectories.
"""

import argparse
import time
import multiprocessing as mp

import numpy as np
import pandas as pd

from lsa_inference.markov_chain import simulate_chains_batch
from lsa_inference.logging_utils import setup_logger
from lsa_inference.experiments.common import generate_problem, run_methods


def _bootstrap_one(args):
    """Worker: run LSA + block bootstrap for a single trajectory."""
    traj, A_arr, b_arr, theta_star, alpha, burn_in, n_bootstrap, coord = args
    T = len(traj)
    d = b_arr.shape[1]

    # Run LSA iteration and store all iterates
    theta = np.zeros(d)
    iterates = np.zeros((T, d))
    for t in range(T):
        x = traj[t]
        theta = theta + alpha * (A_arr[x] @ theta + b_arr[x])
        iterates[t] = theta

    theta_bar = np.mean(iterates[burn_in:], axis=0)

    # Block bootstrap
    block_size = max(int(np.log(T) / alpha), 100)
    n_blocks = (T - burn_in) // block_size
    post_iterates = iterates[burn_in:]

    bootstrap_means = np.zeros((n_bootstrap, d))
    rng = np.random.default_rng(999)
    for b in range(n_bootstrap):
        idx = rng.choice(n_blocks, size=n_blocks, replace=True)
        resampled = np.concatenate(
            [post_iterates[i * block_size:(i + 1) * block_size] for i in idx],
            axis=0,
        )
        bootstrap_means[b] = np.mean(resampled, axis=0)

    lo = np.percentile(bootstrap_means[:, coord], 2.5)
    hi = np.percentile(bootstrap_means[:, coord], 97.5)
    return (
        float(np.linalg.norm(theta_bar - theta_star)),  # l2
        float(hi - lo),                                  # width
        float(lo <= theta_star[coord] <= hi),             # cov
    )


def main(T=1_000_000, n_traj=500, n_workers=None):
    logger, log_path = setup_logger("bootstrap")

    n_states, d = 10, 5
    burn_in = 1000
    K = int(T ** 0.3)
    if n_workers is None:
        n_workers = mp.cpu_count()

    logger.info(f"[Config] T={T:,}, n_traj={n_traj}, n_states={n_states}, d={d}, "
                f"K={K}, burn_in={burn_in}")
    logger.info(f"[Config] n_workers={n_workers} (for bootstrap)")
    logger.info(f"[Config] log_file={log_path}")

    rng = np.random.default_rng(789)
    P, pi, A_bar, theta_star, A_arr, b_arr = generate_problem(n_states, d, rng)

    evals = np.linalg.eigvals(A_bar)
    logger.info(f"[Problem] ||θ*||={np.linalg.norm(theta_star):.4f}, "
                f"max_re(λ(Ā))={np.max(np.real(evals)):.4f}")

    # Generate trajectories
    logger.info(f"\nGenerating {n_traj} trajectories of length T={T:,}...")
    t_gen = time.time()
    trajs = simulate_chains_batch(P, pi, T, n_traj, rng)
    logger.info(f"Chain generation done in {time.time()-t_gen:.1f}s")

    t_start = time.time()

    # RR — vectorized
    logger.info("\nRunning RR (vectorized)...")
    t0 = time.time()
    raw = run_methods(A_arr, b_arr, trajs, K, burn_in, theta_star, ['RR'])
    rr_l2, rr_w, rr_c = raw['RR']['l2'], raw['RR']['width'], raw['RR']['cov']
    logger.info(f"RR done in {time.time()-t0:.1f}s: "
                f"cov={np.nanmean(rr_c)*100:.1f}%, L2={np.nanmean(rr_l2):.2e}")

    # Bootstrap — multiprocessing
    logger.info(f"\nRunning Bootstrap ({n_traj} trajectories, {n_workers} workers)...")
    t0 = time.time()
    boot_args = [
        (trajs[i], A_arr, b_arr, theta_star, 0.02, burn_in, 200, 0)
        for i in range(n_traj)
    ]

    boot_l2, boot_w, boot_c = [], [], []
    done = 0
    log_interval = max(1, n_traj // 10)

    with mp.Pool(n_workers) as pool:
        for l2, width, cov in pool.imap_unordered(_bootstrap_one, boot_args):
            boot_l2.append(l2)
            boot_w.append(width)
            boot_c.append(cov)
            done += 1
            if done % log_interval == 0:
                t_elapsed = time.time() - t0
                t_remaining = t_elapsed / done * (n_traj - done)
                logger.info(f"  Bootstrap [{done}/{n_traj}]: "
                            f"cov={np.mean(boot_c)*100:.1f}%, "
                            f"L2={np.mean(boot_l2):.2e} | "
                            f"{t_elapsed:.0f}s elapsed, ~{t_remaining:.0f}s left")

    logger.info(f"Bootstrap done in {time.time()-t0:.1f}s")

    t_total = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"Table: RR vs Bootstrap Comparison (T={T:,})")
    logger.info(f"Total time: {t_total:.0f}s ({t_total/60:.1f}min)")
    logger.info("=" * 60)

    for name, l2_arr, w_arr, c_arr in [
        ('RR', rr_l2, rr_w, rr_c),
        ('Bootstrap', boot_l2, boot_w, boot_c),
    ]:
        c_arr = np.array(c_arr)
        cov_mean = np.mean(c_arr) * 100
        cov_se = np.std(c_arr) / np.sqrt(n_traj) * 100
        logger.info(f"  {name:>12}: Coverage={cov_mean:.1f}% ± {cov_se:.1f}%, "
                    f"L2={np.mean(l2_arr):.2e}, CI Width={np.mean(w_arr):.5f}")

    rows = []
    for name, l2_arr, w_arr, c_arr in [
        ('RR', rr_l2, rr_w, rr_c),
        ('Bootstrap', boot_l2, boot_w, boot_c),
    ]:
        rows.append({
            'method': name,
            'coverage_pct': float(np.mean(c_arr)) * 100,
            'l2_error': float(np.mean(l2_arr)),
            'ci_width': float(np.mean(w_arr)),
        })
    pd.DataFrame(rows).to_csv('results_bootstrap.csv', index=False)
    logger.info(f"\nResults saved to results_bootstrap.csv")
    logger.info(f"Full log saved to {log_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Table 4: RR vs Bootstrap comparison")
    parser.add_argument("-T", type=int, default=1_000_000)
    parser.add_argument("--n-traj", type=int, default=500)
    parser.add_argument("--n-workers", type=int, default=None)
    args = parser.parse_args()
    main(T=args.T, n_traj=args.n_traj, n_workers=args.n_workers)
