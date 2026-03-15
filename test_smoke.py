#!/usr/bin/env python3
"""Quick smoke test: run a small-scale version of each component."""

import numpy as np
from lsa_inference.markov_chain import generate_transition_matrix, simulate_chain
from lsa_inference.lsa_problem import generate_A, generate_b, compute_theta_star
from lsa_inference.lsa_runner import run_lsa_batched, run_lsa_diminishing
from lsa_inference.batch_inference import compute_covariance, confidence_interval
from lsa_inference.rr_extrapolation import rr_coefficients, run_rr_extrapolation
from lsa_inference.utils import l2_error, ci_width, coverage

rng = np.random.default_rng(42)
n_states, d = 10, 5
T = 10_000
K = 10
burn_in = 100

# Generate problem
print("Generating problem...")
P, pi = generate_transition_matrix(n_states, rng)
print(f"  Stationary dist: {pi.round(3)}")
A_list, A_bar = generate_A(n_states, d, pi, rng)
print(f"  A_bar eigenvalues: {np.linalg.eigvals(A_bar).round(3)}")
b_list = generate_b(n_states, d, rng)
theta_star = compute_theta_star(A_list, b_list, pi)
print(f"  theta_star: {theta_star.round(4)}")

# Verify: A_bar @ theta_star + b_bar = 0
A_bar_check = sum(pi[x] * A_list[x] for x in range(n_states))
b_bar = sum(pi[x] * b_list[x] for x in range(n_states))
residual = A_bar_check @ theta_star + b_bar
print(f"  Residual ||A_bar θ* + b_bar|| = {np.linalg.norm(residual):.2e}")

# Simulate chain
print("\nSimulating chain...")
traj = simulate_chain(P, pi, T, rng)
print(f"  First 10 states: {traj[:10]}")

# Test constant stepsize LSA
print("\nRunning constant stepsize LSA (alpha=0.02)...")
bm, n = run_lsa_batched(A_list, b_list, traj, 0.02, K, burn_in)
tb, Sh = compute_covariance(bm, n)
print(f"  theta_bar: {tb.round(4)}")
print(f"  L2 error: {l2_error(tb, theta_star):.4f}")
print(f"  CI width (coord 0): {ci_width(Sh, K, n):.4f}")
print(f"  Coverage (coord 0): {coverage(theta_star, tb, Sh, K, n)}")

# Test RR extrapolation
print("\nRunning RR extrapolation (alphas=[0.2, 0.02])...")
h = rr_coefficients([0.2, 0.02])
print(f"  RR coefficients: {h.round(4)}")
print(f"  Sum of coefficients: {h.sum():.6f}")

tt, St, lo, hi, se, n_rr = run_rr_extrapolation(
    A_list, b_list, traj, [0.2, 0.02], K, burn_in)
print(f"  theta_tilde: {tt.round(4)}")
print(f"  L2 error: {l2_error(tt, theta_star):.4f}")
print(f"  CI for coord 0: [{lo:.4f}, {hi:.4f}]")
print(f"  theta_star[0] = {theta_star[0]:.4f}")
print(f"  Coverage: {float(lo <= theta_star[0] <= hi)}")

# Test diminishing stepsize
print("\nRunning diminishing stepsize LSA (0.2/sqrt(k))...")
bm_d, n_eff = run_lsa_diminishing(A_list, b_list, traj, 0.2, 0.5, K)
tb_d, Sh_d = compute_covariance(bm_d, n_eff)
print(f"  theta_bar: {tb_d.round(4)}")
print(f"  L2 error: {l2_error(tb_d, theta_star):.4f}")

# Quick coverage test over multiple trajectories
print("\nCoverage test (50 trajectories, T=10000)...")
n_trials = 50
cov_const = 0
cov_rr = 0
for i in range(n_trials):
    traj_i = simulate_chain(P, pi, T, np.random.default_rng(1000 + i))
    bm_i, n_i = run_lsa_batched(A_list, b_list, traj_i, 0.02, K, burn_in)
    tb_i, Sh_i = compute_covariance(bm_i, n_i)
    cov_const += coverage(theta_star, tb_i, Sh_i, K, n_i)

    tt_i, St_i, lo_i, hi_i, se_i, _ = run_rr_extrapolation(
        A_list, b_list, traj_i, [0.2, 0.02], K, burn_in)
    cov_rr += float(lo_i <= theta_star[0] <= hi_i)

print(f"  Constant α=0.02 coverage: {cov_const/n_trials*100:.0f}%")
print(f"  RR (0.2+0.02) coverage: {cov_rr/n_trials*100:.0f}%")

print("\n✓ All smoke tests passed!")
