"""LSA problem generation: A(x), b(x), theta_star."""

import numpy as np


def generate_A(n_states, d, pi, rng):
    """Generate state-dependent matrices A(x) with Hurwitz mean.

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
        M_A -= (max_real + 0.5) * np.eye(d)

    # Ensure rho(I + alpha_max * A_bar) < 1 for alpha_max = 0.2.
    # For eigenvalue lambda = a + bi, |1 + alpha*s*lambda|^2 < 1 requires
    # s < 2|a| / (alpha * |lambda|^2).  Scale M_A if needed.
    alpha_max = 0.2
    evals = np.linalg.eigvals(M_A)
    sr = np.max(np.abs(1 + alpha_max * evals))
    if sr >= 1.0:
        s_limits = []
        for lam in evals:
            a = np.real(lam)
            mag_sq = np.abs(lam) ** 2
            if mag_sq > 0 and a < 0:
                s_limits.append(2 * abs(a) / (alpha_max * mag_sq))
        if s_limits:
            M_A *= min(s_limits) * 0.95

    A_bar = M_A.copy()

    # Generate per-state noise with centering to enforce E_pi[A(x)] = A_bar.
    # Previous approach divided by pi[-1] which amplifies noise when pi[-1] is
    # small.  Centering distributes the correction across all states evenly.
    E = rng.uniform(-1, 1, (n_states, d, d))
    E_bar = np.einsum('x,xij->ij', pi, E)
    A_list = [A_bar + (E[x] - E_bar) for x in range(n_states)]

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
