"""Algorithm 2: Richardson-Romberg extrapolation for bias removal."""

import numpy as np
from .lsa_runner import run_lsa_batched
from .batch_inference import compute_covariance, confidence_interval


def rr_coefficients(alphas):
    """Compute RR extrapolation coefficients via Lagrange interpolation.

    Args:
        alphas: list/array of M distinct stepsizes.

    Returns:
        h: (M,) array of RR weights satisfying sum(h) = 1 and
           sum(h_m * alpha_m^l) = 0 for l = 1, ..., M-1.
    """
    M = len(alphas)
    h = np.ones(M)
    for m in range(M):
        for l in range(M):
            if l != m:
                h[m] *= alphas[l] / (alphas[l] - alphas[m])
    return h


def run_rr_extrapolation(A_list, b_list, trajectory, alphas, K,
                         burn_in=100, n0=0, q=0.05, coord=0):
    """Run RR extrapolation: M LSA runs with different stepsizes on same data.

    Args:
        A_list: list of (d, d) matrices.
        b_list: list of (d,) vectors.
        trajectory: list of state indices (shared across all stepsizes).
        alphas: list of M stepsizes.
        K: Number of batches.
        burn_in: Initial burn-in.
        n0: Intra-batch discard.
        q: CI error level.
        coord: Coordinate for CI.

    Returns:
        theta_tilde: (d,) RR point estimate.
        Sigma_tilde: (d, d) RR covariance estimate.
        lo, hi: CI bounds for coordinate `coord`.
        se: Standard error.
        n: Batch size used.
    """
    h = rr_coefficients(alphas)
    M = len(alphas)

    # Run LSA for each stepsize on same trajectory
    all_batch_means = []
    n = None
    for m in range(M):
        bm, n_m = run_lsa_batched(A_list, b_list, trajectory, alphas[m],
                                  K, burn_in, n0)
        all_batch_means.append(bm)
        if n is None:
            n = n_m

    # Combine via RR weights
    rr_batch_means = np.zeros_like(all_batch_means[0])
    for m in range(M):
        rr_batch_means += h[m] * all_batch_means[m]

    # Covariance and CI
    theta_tilde, Sigma_tilde = compute_covariance(rr_batch_means, n, n0)
    lo, hi, se = confidence_interval(theta_tilde, Sigma_tilde, K, n, n0, q, coord)

    return theta_tilde, Sigma_tilde, lo, hi, se, n
