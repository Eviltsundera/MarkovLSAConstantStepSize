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


def simulate_chain(P, pi, T, rng):
    """Simulate a Markov chain trajectory of length T.

    Args:
        P: (n_states, n_states) transition matrix.
        pi: (n_states,) stationary distribution (used for initialization).
        T: Trajectory length.
        rng: numpy random Generator.

    Returns:
        trajectory: list of length T with state indices.
    """
    n_states = len(pi)
    x = rng.choice(n_states, p=pi)
    trajectory = [x]
    for _ in range(T - 1):
        x = rng.choice(n_states, p=P[x])
        trajectory.append(x)
    return trajectory
