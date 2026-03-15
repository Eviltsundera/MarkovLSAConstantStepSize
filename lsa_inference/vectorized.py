"""Vectorized LSA: run all trajectories simultaneously via numpy broadcasting.

Instead of looping over n_traj trajectories one at a time, this module processes
all of them in a single time loop. For d=5, n_traj=100, this eliminates ~100x
Python loop overhead.

Shapes throughout:
    thetas:  (n_traj, d)
    A_arr:   (n_states, d, d)
    b_arr:   (n_states, d)
    trajs:   (n_traj, T)
"""

import numpy as np
from scipy import stats


def _prepare_arrays(A_list, b_list):
    """Stack A_list and b_list into contiguous numpy arrays."""
    A_arr = np.array(A_list)   # (n_states, d, d)
    b_arr = np.array(b_list)   # (n_states, d)
    return A_arr, b_arr


def run_lsa_batched_vec(A_arr, b_arr, trajs, alpha, K, burn_in=100, n0=0):
    """Run constant-stepsize LSA for all trajectories simultaneously.

    Args:
        A_arr: (n_states, d, d) array of state matrices.
        b_arr: (n_states, d) array of state vectors.
        trajs: (n_traj, T) int array of state indices.
        alpha: Constant stepsize.
        K: Number of batches.
        burn_in: Initial iterates to discard.
        n0: Intra-batch discard.

    Returns:
        batch_means: (n_traj, K, d) array of batch means.
        n: Batch size.
    """
    n_traj, T = trajs.shape
    d = b_arr.shape[1]
    usable_T = T - burn_in
    n = usable_T // K

    thetas = np.zeros((n_traj, d))
    # Accumulate batch sums on the fly: (n_traj, K, d)
    batch_sums = np.zeros((n_traj, K, d))

    current_batch = 0
    batch_count = 0

    for t in range(T):
        x_t = trajs[:, t]                         # (n_traj,)
        A_t = A_arr[x_t]                           # (n_traj, d, d)
        b_t = b_arr[x_t]                           # (n_traj, d)
        # thetas += alpha * (A_t @ thetas + b_t)
        # einsum is faster than bmm for small d
        thetas += alpha * (np.einsum('nij,nj->ni', A_t, thetas) + b_t)

        if t < burn_in:
            continue
        if current_batch >= K:
            break

        if batch_count >= n0:
            batch_sums[:, current_batch, :] += thetas

        batch_count += 1
        if batch_count == n:
            current_batch += 1
            batch_count = 0

    effective = n - n0
    if effective > 0:
        batch_means = batch_sums / effective
    else:
        batch_means = batch_sums

    # Mark diverged trajectories: replace inf/nan and unreasonably large values
    # with nan.  Threshold 1e10 is generous (true theta* is order 1) while
    # catching transient divergence that stays below float64 inf.
    batch_means = np.where(
        np.isfinite(batch_means) & (np.abs(batch_means) < 1e10),
        batch_means, np.nan
    )

    return batch_means, n


def run_lsa_diminishing_vec(A_arr, b_arr, trajs, alpha0, alpha_exp=0.5,
                            K=50):
    """Run diminishing-stepsize LSA for all trajectories simultaneously.

    Args:
        A_arr: (n_states, d, d).
        b_arr: (n_states, d).
        trajs: (n_traj, T) int array.
        alpha0: Initial stepsize.
        alpha_exp: Decay exponent.
        K: Number of batches.

    Returns:
        batch_means: (n_traj, K, d) array.
        n_eff: Effective average batch size.
    """
    n_traj, T = trajs.shape
    d = b_arr.shape[1]

    # CLTZ20 batch endpoints
    r = T ** (1 - alpha_exp) / (K + 1)
    endpoints = [0]
    for k in range(1, K + 1):
        e_k = int((k * r) ** (1 / (1 - alpha_exp)))
        e_k = min(e_k, T)
        endpoints.append(e_k)
    endpoints.append(T)

    thetas = np.zeros((n_traj, d))
    batch_sums = np.zeros((n_traj, K, d))
    batch_counts = np.zeros(K, dtype=np.int64)

    # Precompute which batch each timestep belongs to
    batch_for_t = np.full(T, -1, dtype=np.int32)
    for k in range(K):
        batch_for_t[endpoints[k]:endpoints[k + 1]] = k
        batch_counts[k] = endpoints[k + 1] - endpoints[k]

    for t in range(T):
        x_t = trajs[:, t]
        A_t = A_arr[x_t]
        b_t = b_arr[x_t]
        step = alpha0 / (t + 1) ** alpha_exp
        thetas += step * (np.einsum('nij,nj->ni', A_t, thetas) + b_t)

        k = batch_for_t[t]
        if k >= 0:
            batch_sums[:, k, :] += thetas

    # Compute means
    batch_means = np.zeros((n_traj, K, d))
    total_used = 0
    for k in range(K):
        if batch_counts[k] > 0:
            batch_means[:, k, :] = batch_sums[:, k, :] / batch_counts[k]
            total_used += batch_counts[k]

    batch_means = np.where(
        np.isfinite(batch_means) & (np.abs(batch_means) < 1e10),
        batch_means, np.nan
    )

    n_eff = total_used // K if K > 0 else T
    return batch_means, n_eff


def compute_metrics_vec(batch_means, n, theta_star, n0=0, q=0.05, coord=0):
    """Compute L2 error, CI width, and coverage for all trajectories at once.

    Args:
        batch_means: (n_traj, K, d) array.
        n: Batch size.
        theta_star: (d,) true solution.
        n0: Intra-batch discard.
        q: CI error level.
        coord: Coordinate for CI/coverage.

    Returns:
        l2_errors: (n_traj,) L2 errors.
        ci_widths: (n_traj,) CI widths.
        coverages: (n_traj,) coverage indicators (0 or 1).
    """
    z = stats.norm.ppf(1 - q / 2)
    n_traj, K, d = batch_means.shape

    # Grand means: (n_traj, d) — use nanmean to handle diverged trajectories
    theta_bars = np.nanmean(batch_means, axis=1)

    # L2 errors
    l2_errors = np.linalg.norm(theta_bars - theta_star, axis=1)

    # Batch-mean covariance for coordinate `coord`
    # diffs: (n_traj, K)
    diffs = batch_means[:, :, coord] - theta_bars[:, coord:coord + 1]
    # Variance of coordinate: (n_traj,)
    var_coord = (n - n0) / K * np.nansum(diffs ** 2, axis=1)

    se = np.sqrt(var_coord / (K * (n - n0)))
    ci_widths = 2 * z * se

    lo = theta_bars[:, coord] - z * se
    hi = theta_bars[:, coord] + z * se
    coverages = ((lo <= theta_star[coord]) & (theta_star[coord] <= hi)).astype(float)

    # Mark trajectories with NaN as non-covering, NaN L2/width
    has_nan = np.any(np.isnan(theta_bars), axis=1)
    l2_errors[has_nan] = np.nan
    ci_widths[has_nan] = np.nan
    coverages[has_nan] = 0.0

    return l2_errors, ci_widths, coverages


def rr_coefficients(alphas):
    """Compute RR extrapolation weights via Lagrange interpolation.

    Returns h such that sum(h) = 1 and sum(h_m * alpha_m^l) = 0
    for l = 1, ..., M-1.
    """
    alphas = np.asarray(alphas, dtype=float)
    M = len(alphas)
    h = np.ones(M)
    for m in range(M):
        for l in range(M):
            if l != m:
                h[m] *= alphas[l] / (alphas[l] - alphas[m])
    return h


def run_rr_vec(A_arr, b_arr, trajs, alphas, K, burn_in=100, n0=0,
               q=0.05, coord=0, theta_star=None):
    """Run RR extrapolation for all trajectories simultaneously.

    Args:
        A_arr: (n_states, d, d).
        b_arr: (n_states, d).
        trajs: (n_traj, T) int array — SAME trajectories for all stepsizes.
        alphas: list of M stepsizes.
        K: Number of batches.
        burn_in: Initial burn-in.
        n0: Intra-batch discard.
        q: CI error level.
        coord: Coordinate for CI.
        theta_star: (d,) true solution (needed for metrics).

    Returns:
        l2_errors: (n_traj,) L2 errors.
        ci_widths: (n_traj,) CI widths.
        coverages: (n_traj,) coverage indicators.
    """
    h = rr_coefficients(alphas)
    M = len(alphas)

    # Run LSA for each stepsize
    all_batch_means = []
    n = None
    for m in range(M):
        bm, n_m = run_lsa_batched_vec(A_arr, b_arr, trajs, alphas[m], K,
                                      burn_in, n0)
        all_batch_means.append(bm)
        if n is None:
            n = n_m

    # Combine: (n_traj, K, d)
    rr_batch_means = sum(h[m] * all_batch_means[m] for m in range(M))

    return compute_metrics_vec(rr_batch_means, n, theta_star, n0, q, coord)
