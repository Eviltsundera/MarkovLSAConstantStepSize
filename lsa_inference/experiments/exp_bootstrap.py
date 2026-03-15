"""Table 4 (bootstrap comparison): Constant + RR vs Bootstrapping (T=10^6).

Metrics: coverage, L2 error, CI width.

RR is vectorized. Bootstrap uses multiprocessing across trajectories.
"""

import time
import multiprocessing as mp

import numpy as np
import pandas as pd

from lsa_inference.markov_chain import generate_transition_matrix, simulate_chains_batch
from lsa_inference.lsa_problem import generate_A, generate_b, compute_theta_star
from lsa_inference.vectorized import _prepare_arrays, run_rr_vec
from lsa_inference.logging_utils import setup_logger


def _bootstrap_one(args):
    """Worker: run bootstrap for a single trajectory."""
    traj, A_arr, b_arr, theta_star, alpha, burn_in, n_bootstrap, coord = args
    T = len(traj)
    d = b_arr.shape[1]

    theta = np.zeros(d)
    iterates = np.zeros((T, d))
    for t in range(T):
        x = traj[t]
        theta = theta + alpha * (A_arr[x] @ theta + b_arr[x])
        iterates[t] = theta

    theta_bar = np.mean(iterates[burn_in:], axis=0)

    block_size = max(int(np.log(T) / alpha), 100)
    n_blocks = (T - burn_in) // block_size
    post_iterates = iterates[burn_in:]

    bootstrap_means = np.zeros((n_bootstrap, d))
    rng = np.random.default_rng(999)
    for b in range(n_bootstrap):
        block_indices = rng.choice(n_blocks, size=n_blocks, replace=True)
        resampled = np.concatenate([
            post_iterates[i * block_size:(i + 1) * block_size]
            for i in block_indices
        ], axis=0)
        bootstrap_means[b] = np.mean(resampled, axis=0)

    lo = np.percentile(bootstrap_means[:, coord], 2.5)
    hi = np.percentile(bootstrap_means[:, coord], 97.5)
    width = hi - lo
    l2 = float(np.linalg.norm(theta_bar - theta_star))
    cov = float(lo <= theta_star[coord] <= hi)

    return l2, width, cov


def main():
    logger, log_path = setup_logger("bootstrap")

    T = 1_000_000
    n_states = 10
    d = 5
    n_traj = 500
    burn_in = 1000
    K = int(T ** 0.3)
    n_workers = mp.cpu_count()

    logger.info(f"[Config] T={T:,}, n_traj={n_traj}, n_states={n_states}, d={d}, "
                f"K={K}, burn_in={burn_in}")
    logger.info(f"[Config] n_workers={n_workers} (for bootstrap)")
    logger.info(f"[Config] log_file={log_path}")

    rng = np.random.default_rng(789)
    P, pi = generate_transition_matrix(n_states, rng)
    A_list, A_bar = generate_A(n_states, d, pi, rng)
    b_list = generate_b(n_states, d, rng)
    theta_star = compute_theta_star(A_list, b_list, pi)
    A_arr, b_arr = _prepare_arrays(A_list, b_list)

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
    rr_l2, rr_w, rr_c = run_rr_vec(A_arr, b_arr, trajs, [0.2, 0.02], K,
                                     burn_in, theta_star=theta_star)
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
    df = pd.DataFrame(rows)
    df.to_csv('results_bootstrap.csv', index=False)
    logger.info(f"\nResults saved to results_bootstrap.csv")
    logger.info(f"Full log saved to {log_path}")


if __name__ == '__main__':
    main()
