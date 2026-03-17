"""LSA problem generation: A(x), b(x), theta_star."""

import numpy as np


def generate_A(n_states, d, pi, rng):
    """Generate state-dependent matrices A(x) with Hurwitz mean.

    Follows the paper's setup (Section 6.1): random Hurwitz A_bar with
    per-state noise whose pi-weighted mean is exactly A_bar.

    Args:
        n_states: Number of Markov chain states.
        d: Dimension of theta.
        pi: (n_states,) stationary distribution.
        rng: numpy random Generator.

    Returns:
        A_list: list of (d, d) matrices, one per state.
        A_bar: (d, d) mean matrix E_pi[A(x)].
    """
    M_A = rng.standard_normal((d, d))
    evals = np.linalg.eigvals(M_A)
    max_real = np.max(np.real(evals))
    if max_real >= 0:
        M_A -= 2 * max_real * np.eye(d)

    A_bar = M_A.copy()

    # Per-state noise: first n_states-1 get i.i.d. noise, last state absorbs
    # the correction so that E_pi[A(x)] = A_bar exactly.
    E = rng.uniform(-1, 1, (n_states - 1, d, d))
    A_list = [A_bar + E[x] for x in range(n_states - 1)]
    A_last = A_bar - sum(pi[x] * E[x] for x in range(n_states - 1)) / pi[-1]
    A_list.append(A_last)

    return A_list, A_bar


def generate_b(n_states, d, rng):
    """Generate state-dependent vectors b(x).

    Args:
        n_states: Number of Markov chain states.
        d: Dimension of theta.
        rng: numpy random Generator.

    Returns:
        b_list: list of (d,) vectors, one per state.
    """
    return [rng.uniform(-1, 1, d) for _ in range(n_states)]


def compute_theta_star(A_list, b_list, pi):
    """Compute the true solution theta* = -A_bar^{-1} b_bar.

    Args:
        A_list: list of (d, d) matrices.
        b_list: list of (d,) vectors.
        pi: (n_states,) stationary distribution.

    Returns:
        theta_star: (d,) true solution vector.
    """
    n_states = len(A_list)
    A_bar = sum(pi[x] * A_list[x] for x in range(n_states))
    b_bar = sum(pi[x] * b_list[x] for x in range(n_states))
    theta_star = np.linalg.solve(A_bar, -b_bar)
    return theta_star
