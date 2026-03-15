"""Algorithm 1: Constant-stepsize LSA runner with batch-mean collection.

Memory-efficient: maintains running sums within each batch instead of storing all iterates.
"""

import numpy as np


def run_lsa_batched(A_list, b_list, trajectory, alpha, K, burn_in=100, n0=0):
    """Run constant-stepsize LSA and collect batch means (memory-efficient).

    Args:
        A_list: list of (d, d) matrices, one per state.
        b_list: list of (d,) vectors, one per state.
        trajectory: list of state indices of length T.
        alpha: Constant stepsize.
        K: Number of batches.
        burn_in: Number of initial iterates to discard.
        n0: Intra-batch discard (iterates discarded at start of each batch).

    Returns:
        batch_means: (K, d) array of batch means.
        n: Batch size (total iterates per batch before intra-batch discard).
    """
    T = len(trajectory)
    d = len(b_list[0])
    usable_T = T - burn_in
    n = usable_T // K  # batch size

    theta = np.zeros(d)
    batch_means = np.zeros((K, d))

    # Current position in the post-burn-in sequence
    post_burn_idx = 0
    current_batch = 0
    batch_count = 0  # count within current batch

    for t in range(T):
        x = trajectory[t]
        theta = theta + alpha * (A_list[x] @ theta + b_list[x])

        if t < burn_in:
            continue

        if current_batch >= K:
            break

        if batch_count >= n0:  # past intra-batch discard
            batch_means[current_batch] += theta

        batch_count += 1
        if batch_count == n:
            # Finalize this batch mean
            effective = n - n0
            if effective > 0:
                batch_means[current_batch] /= effective
            current_batch += 1
            batch_count = 0

        post_burn_idx += 1

    return batch_means, n


def run_lsa_diminishing(A_list, b_list, trajectory, alpha0, alpha_exp=0.5,
                        K=50, burn_in=0):
    """Run diminishing-stepsize LSA: alpha_t = alpha0 / (t+1)^alpha_exp.

    Uses the CLTZ20 batching scheme for diminishing stepsizes.

    Args:
        A_list: list of (d, d) matrices, one per state.
        b_list: list of (d,) vectors, one per state.
        trajectory: list of state indices of length T.
        alpha0: Initial stepsize.
        alpha_exp: Decay exponent (0.5 for 1/sqrt(t)).
        K: Number of batches.
        burn_in: Initial burn-in (not used in diminishing, kept for interface).

    Returns:
        batch_means: (K, d) array of batch means.
        n_eff: Effective average batch size.
    """
    T = len(trajectory)
    d = len(b_list[0])

    # CLTZ20 batch endpoints
    r = T ** (1 - alpha_exp) / (K + 1)
    endpoints = [0]
    for k in range(1, K + 1):
        e_k = int((k * r) ** (1 / (1 - alpha_exp)))
        e_k = min(e_k, T)
        endpoints.append(e_k)
    endpoints.append(T)

    # Run LSA
    theta = np.zeros(d)
    iterates = np.zeros((T, d))
    for t in range(T):
        x = trajectory[t]
        step = alpha0 / (t + 1) ** alpha_exp
        theta = theta + step * (A_list[x] @ theta + b_list[x])
        iterates[t] = theta

    # Compute batch means using CLTZ20 endpoints
    batch_means = np.zeros((K, d))
    total_used = 0
    for k in range(K):
        start = endpoints[k]
        end = endpoints[k + 1]
        if end > start:
            batch_means[k] = np.mean(iterates[start:end], axis=0)
            total_used += (end - start)

    n_eff = total_used // K if K > 0 else T
    return batch_means, n_eff
