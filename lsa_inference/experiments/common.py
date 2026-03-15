"""Shared utilities for experiment scripts.

Centralizes problem generation, method execution, result summarization,
and display formatting used across all experiment tables.
"""

import numpy as np

from lsa_inference.markov_chain import generate_transition_matrix, simulate_chains_batch
from lsa_inference.lsa_problem import generate_A, generate_b, compute_theta_star
from lsa_inference.vectorized import (
    _prepare_arrays, run_lsa_batched_vec, run_lsa_diminishing_vec,
    compute_metrics_vec, run_rr_vec,
)

# Canonical method keys and display labels.
METHODS_ALL = ['alpha_0.2', 'alpha_0.02', 'RR', 'dim_0.2', 'dim_0.02']
METHOD_LABELS = {
    'alpha_0.2': 'α=0.2 (const)',
    'alpha_0.02': 'α=0.02 (const)',
    'RR': 'RR (0.2+0.02)',
    'dim_0.2': '0.2/√k (dim)',
    'dim_0.02': '0.02/√k (dim)',
}


def generate_problem(n_states, d, rng):
    """Generate a complete LSA problem instance.

    Returns:
        P: (n_states, n_states) transition matrix.
        pi: (n_states,) stationary distribution.
        A_bar: (d, d) mean matrix.
        theta_star: (d,) true solution.
        A_arr: (n_states, d, d) per-state matrices (numpy array).
        b_arr: (n_states, d) per-state vectors (numpy array).
    """
    P, pi = generate_transition_matrix(n_states, rng)
    A_list, A_bar = generate_A(n_states, d, pi, rng)
    b_list = generate_b(n_states, d, rng)
    theta_star = compute_theta_star(A_list, b_list, pi)
    A_arr, b_arr = _prepare_arrays(A_list, b_list)
    return P, pi, A_bar, theta_star, A_arr, b_arr


def run_methods(A_arr, b_arr, trajs, K, burn_in, theta_star, methods=None):
    """Run requested LSA methods and return per-trajectory metric arrays.

    Args:
        methods: list of method keys (default: METHODS_ALL).

    Returns:
        dict {method_key: {'l2': ndarray, 'width': ndarray, 'cov': ndarray}}
        where each array has shape (n_traj,).
    """
    if methods is None:
        methods = METHODS_ALL

    _runners = {
        'alpha_0.2':  lambda: _run_const(A_arr, b_arr, trajs, 0.2, K, burn_in, theta_star),
        'alpha_0.02': lambda: _run_const(A_arr, b_arr, trajs, 0.02, K, burn_in, theta_star),
        'RR':         lambda: run_rr_vec(A_arr, b_arr, trajs, [0.2, 0.02], K,
                                         burn_in, theta_star=theta_star),
        'dim_0.2':    lambda: _run_dim(A_arr, b_arr, trajs, 0.2, K, theta_star),
        'dim_0.02':   lambda: _run_dim(A_arr, b_arr, trajs, 0.02, K, theta_star),
    }

    results = {}
    for m in methods:
        l2, w, c = _runners[m]()
        results[m] = {'l2': l2, 'width': w, 'cov': c}
    return results


def _run_const(A_arr, b_arr, trajs, alpha, K, burn_in, theta_star):
    bm, n = run_lsa_batched_vec(A_arr, b_arr, trajs, alpha, K, burn_in)
    return compute_metrics_vec(bm, n, theta_star)


def _run_dim(A_arr, b_arr, trajs, alpha0, K, theta_star):
    bm, n_eff = run_lsa_diminishing_vec(A_arr, b_arr, trajs, alpha0, 0.5, K)
    return compute_metrics_vec(bm, n_eff, theta_star)


def summarize(raw_results):
    """Convert per-trajectory arrays to scalar means.

    Returns:
        dict {method_key: {'l2': float, 'width': float, 'cov': float}}
    """
    return {
        m: {k: float(np.nanmean(v)) for k, v in metrics.items()}
        for m, metrics in raw_results.items()
    }


def solve_problem_worker(args):
    """Multiprocessing worker: generate one problem, run all methods.

    Args (tuple):
        prob_idx, seed, n_traj, T, K, burn_in, n_states, d

    Returns:
        (prob_idx, summary_dict, theta_norm, max_re_eigenvalue)
    """
    prob_idx, seed, n_traj, T, K, burn_in, n_states, d = args

    rng = np.random.default_rng(seed)
    P, pi, A_bar, theta_star, A_arr, b_arr = generate_problem(n_states, d, rng)

    traj_rng = np.random.default_rng(rng.integers(0, 2**31))
    trajs = simulate_chains_batch(P, pi, T, n_traj, traj_rng)

    results = summarize(run_methods(A_arr, b_arr, trajs, K, burn_in, theta_star))

    evals = np.linalg.eigvals(A_bar)
    max_re = float(np.max(np.real(evals)))
    theta_norm = float(np.linalg.norm(theta_star))

    return prob_idx, results, theta_norm, max_re


def print_percentile_table(logger, all_results, methods,
                           percentiles=(10, 25, 50, 75, 90)):
    """Log a percentile table for L2, width, and coverage."""
    for metric, scale, unit in [('l2', 1e3, '×1e-3'),
                                 ('width', 1e3, '×1e-3'),
                                 ('cov', 100, '%')]:
        logger.info(f"\n--- {metric} ({unit}) ---")
        header = f"{'Method':<20}" + "".join(f"{'p'+str(p):>10}" for p in percentiles)
        logger.info(header)
        for m in methods:
            vals = np.array(all_results[m][metric]) * scale
            pcts = np.percentile(vals, percentiles)
            logger.info(f"{METHOD_LABELS[m]:<20}" + "".join(f"{v:>10.2f}" for v in pcts))
