"""Algorithm 1 Steps 3-4: Batch-mean covariance estimation and confidence intervals."""

import numpy as np
from scipy import stats


def compute_covariance(batch_means, n, n0=0):
    """Compute the batch-mean covariance estimator.

    Args:
        batch_means: (K, d) array of batch means.
        n: Batch size (total before intra-batch discard).
        n0: Intra-batch discard.

    Returns:
        theta_bar: (d,) grand mean.
        Sigma_hat: (d, d) covariance estimator.
    """
    K = batch_means.shape[0]
    theta_bar = np.mean(batch_means, axis=0)
    diffs = batch_means - theta_bar
    Sigma_hat = (n - n0) / K * (diffs.T @ diffs)
    return theta_bar, Sigma_hat


def confidence_interval(theta_bar, Sigma_hat, K, n, n0=0, q=0.05, coord=0):
    """Compute a (1-q)*100% confidence interval for a single coordinate.

    Args:
        theta_bar: (d,) point estimate.
        Sigma_hat: (d, d) covariance estimator.
        K: Number of batches.
        n: Batch size.
        n0: Intra-batch discard.
        q: Error level (0.05 for 95% CI).
        coord: Coordinate index.

    Returns:
        lo, hi: Lower and upper CI bounds.
        se: Standard error.
    """
    z = stats.norm.ppf(1 - q / 2)
    se = np.sqrt(Sigma_hat[coord, coord] / (K * (n - n0)))
    lo = theta_bar[coord] - z * se
    hi = theta_bar[coord] + z * se
    return lo, hi, se
