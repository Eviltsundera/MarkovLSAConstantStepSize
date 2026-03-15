"""Markov chain generation and simulation."""

import numpy as np
from numpy.linalg import eig


def generate_transition_matrix(n_states, rng):
    """Generate a random irreducible aperiodic transition matrix and its stationary distribution.

    Args:
        n_states: Number of states in the Markov chain.
        rng: numpy random Generator.

    Returns:
        P: (n_states, n_states) row-stochastic transition matrix.
        pi: (n_states,) stationary distribution.
    """
    while True:
        M = rng.uniform(0, 1, (n_states, n_states))
        P = M / M.sum(axis=1, keepdims=True)
        vals, vecs = eig(P.T)
        idx = np.argmin(np.abs(vals - 1.0))
        pi = np.real(vecs[:, idx])
        pi /= pi.sum()
        if np.all(pi > 0):
            return P, pi



def simulate_chains_batch(P, pi, T, n_traj, rng):
    """Simulate n_traj independent Markov chain trajectories simultaneously.

    Uses vectorized cumulative-probability sampling for speed.

    Args:
        P: (n_states, n_states) transition matrix.
        pi: (n_states,) stationary distribution.
        T: Trajectory length.
        n_traj: Number of independent trajectories.
        rng: numpy random Generator.

    Returns:
        trajs: (n_traj, T) int array of state indices.
    """
    n_states = len(pi)
    cum_P = np.cumsum(P, axis=1)  # (n_states, n_states)
    cum_pi = np.cumsum(pi)

    trajs = np.empty((n_traj, T), dtype=np.int32)

    # Initialize from stationary distribution
    u = rng.uniform(size=n_traj)
    trajs[:, 0] = np.searchsorted(cum_pi, u)

    # Vectorized transitions
    for t in range(1, T):
        u = rng.uniform(size=n_traj)
        prev = trajs[:, t - 1]
        # cum_P[prev] has shape (n_traj, n_states)
        # For each trajectory, find the state where cumulative prob exceeds u
        trajs[:, t] = (u[:, None] < cum_P[prev]).argmax(axis=1)

    return trajs
