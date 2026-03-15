"""Evaluation metrics: L2 error, CI width, coverage."""

import numpy as np
from scipy import stats


def l2_error(theta_bar, theta_star):
    """L2 norm of point estimate error."""
    return np.linalg.norm(theta_bar - theta_star)


def ci_width(Sigma_hat, K, n, n0=0, q=0.05, coord=0):
    """Width of the (1-q)*100% CI for coordinate `coord`."""
    z = stats.norm.ppf(1 - q / 2)
    se = np.sqrt(Sigma_hat[coord, coord] / (K * (n - n0)))
    return 2 * z * se


def coverage(theta_star, theta_bar, Sigma_hat, K, n, n0=0, q=0.05, coord=0):
    """Check if theta_star[coord] is inside the CI. Returns 0 or 1."""
    z = stats.norm.ppf(1 - q / 2)
    se = np.sqrt(Sigma_hat[coord, coord] / (K * (n - n0)))
    lo = theta_bar[coord] - z * se
    hi = theta_bar[coord] + z * se
    return float(lo <= theta_star[coord] <= hi)
