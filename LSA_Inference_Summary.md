# Paper Summary & Reproduction Guide

**Effectiveness of Constant Stepsize in Markovian LSA and Statistical Inference**  
Huo, Chen, Xie — arXiv:2312.10894v1 [stat.ML] 18 Dec 2023

---

## Table of Contents

1. [Paper Overview](#1-paper-overview)
2. [Main Contributions](#2-main-contributions)
3. [Algorithms — Full Specification](#3-algorithms--full-specification-for-reproduction)
   - 3.1 [Algorithm 1: Constant-Stepsize LSA with Batch-Mean Inference](#31-algorithm-1-constant-stepsize-lsa-with-batch-mean-inference)
   - 3.2 [Algorithm 2: Richardson-Romberg Extrapolation](#32-algorithm-2-richardson-romberg-rr-extrapolation)
   - 3.3 [Stepsize Schedules for RR](#33-stepsize-schedules-for-rr-extrapolation)
   - 3.4 [Zero-Bias Special Cases](#34-zero-bias-special-cases-no-rr-needed)
   - 3.5 [Experiment Setup — LSA Problem Generation](#35-experiment-setup--lsa-problem-generation)
4. [Key Theoretical Results](#4-key-theoretical-results)
5. [Key Numerical Results](#5-key-numerical-results-summary)
6. [Step-by-Step Reproduction Guide](#6-step-by-step-reproduction-guide)
7. [Notation Glossary](#7-notation-glossary)
8. [Common Pitfalls & Debugging Tips](#8-common-pitfalls--debugging-tips)

---

## 1. Paper Overview

**Goal:** Demonstrate that constant-stepsize Linear Stochastic Approximation (LSA) with Markovian data is effective for statistical inference — specifically for building confidence intervals (CIs) around the true solution θ\*.

> **Key insight:** Constant stepsize is simpler to tune, converges faster initially, and produces better CI coverage than diminishing stepsize, especially when data is limited — provided Richardson-Romberg (RR) extrapolation is used to correct the asymptotic bias.

### 1.1 Problem Setting

The LSA update is:

```
θ_{t+1} = θ_t + α * ( A(x_t) @ θ_t + b(x_t) )
```

- **Data:** `(x_t)` is a uniformly ergodic Markov chain on state space `X` with stationary distribution `π`
- **Target:** `θ*` solves `E_{x~π}[A(x)] θ* + E_{x~π}[b(x)] = 0`
- **Stepsize:** constant `α`, independent of total iterations `T` (unlike diminishing schedules)
- **Assumptions:**
  1. Uniform ergodicity of the chain
  2. `A` bounded (`‖A(x)‖ ≤ 1`), `b` bounded, `Ā := E[A(x)]` is **Hurwitz** (all eigenvalues have strictly negative real parts)

### 1.2 Why Constant Stepsize Is Tricky

Constant stepsize creates an **asymptotic bias**: the iterates converge in distribution to `θ_∞ ≠ θ*`. Specifically:

```
E[θ_∞^(α)] = θ* + α B^(1) + α² B^(2) + ...   (bias expansion)
```

The bias is `O(α)`, so a large stepsize (e.g., 0.2) gives large bias. The fix is Richardson-Romberg extrapolation.

---

## 2. Main Contributions

- **CLT (Theorem 3.1):** `√T (θ̄_T − E[θ_∞]) → N(0, Σ*)` as `T → ∞`. The averaged iterates are asymptotically normal around `E[θ_∞]`, not `θ*`.
- **Inference Procedure:** Batch-mean estimator for covariance `Σ*`, combined with RR extrapolation for bias removal.
- **RR Stepsize Theory:** Variance bounds for geometric vs equidistant stepsize schedules in RR extrapolation.
- **Zero-Bias Cases:** Three practically important settings where bias = 0 even without extrapolation.
- **Experiments:** Comprehensive comparison vs diminishing stepsize and bootstrapping.

---

## 3. Algorithms — Full Specification for Reproduction

> ⚙️ This section provides complete pseudocode, all hyperparameters, and data structures needed to implement the paper's inference procedure from scratch.

---

### 3.1 Algorithm 1: Constant-Stepsize LSA with Batch-Mean Inference

#### Inputs

| Parameter | Description | Paper values |
|-----------|-------------|-------------|
| `(x_0,...,x_{T-1})` | Markov chain trajectory | — |
| `α` | Constant stepsize | 0.2, 0.02 |
| `b` | Burn-in length (initial iterates discarded) | 100 (QQ experiment) |
| `K` | Number of batches | `T^0.3`, or 50/100/500/1000 |
| `n_0` | Intra-batch discard (reduces inter-batch correlation ∝ `exp(-α n_0)`) | 0 (QQ experiment) |
| `n` | Batch size: `floor((T - b) / K)` | derived |
| `q` | Error level for `(1-q)×100%` CIs | 0.05 (95% CIs) |

#### Step 1 — Run LSA, collect iterates

```python
# Initialize
theta = np.zeros(d)   # or any initialization

for t in range(T):
    theta = theta + alpha * (A(x[t]) @ theta + b(x[t]))
    store theta_t

# Discard burn-in
iterates = [theta_b, theta_{b+1}, ..., theta_{T-1}]
```

#### Step 2 — Split into K batches, compute batch means

```python
batch_means = []
for k in range(K):
    batch   = iterates[k*n : (k+1)*n]
    usable  = batch[n_0:]           # discard first n_0 of each batch
    theta_bar_k = np.mean(usable, axis=0)   # shape: (d,)
    batch_means.append(theta_bar_k)
```

#### Step 3 — Point estimate and variance

```python
theta_bar = np.mean(batch_means, axis=0)   # grand mean, shape (d,)

# Batch-mean covariance estimator (consistent for Σ* as n, K → ∞)
diffs = np.array(batch_means) - theta_bar  # shape (K, d)
Sigma_hat = (n - n_0) / K * (diffs.T @ diffs)   # shape (d, d)
```

#### Step 4 — Confidence interval for coordinate i

```python
# Standard error for coordinate i
se_i = np.sqrt(Sigma_hat[i, i] / (K * (n - n_0)))

# 95% CI  (z_{0.975} ≈ 1.96)
CI_i = [theta_bar[i] - 1.96 * se_i,
        theta_bar[i] + 1.96 * se_i]

# IMPORTANT: This CI covers E[θ_∞], NOT θ* directly.
# Use RR extrapolation (Algorithm 2) to target θ*.
```

---

### 3.2 Algorithm 2: Richardson-Romberg (RR) Extrapolation

#### Intuition

Run Algorithm 1 with `M` different stepsizes **simultaneously**, using the **same data stream**. Then take a weighted combination that cancels the bias terms up to order `α^M`.

#### Inputs

| Parameter | Description | Paper default |
|-----------|-------------|--------------|
| `alphas = [α_1,...,α_M]` | M distinct stepsizes, all in `(0, α_0)` | `[0.2, 0.02]` (M=2) |
| `(x_t)` | **Same** chain trajectory for all M runs | — |
| `K, n, n_0, b` | Same as Algorithm 1 | — |

#### Step 1 — Compute RR coefficients via Vandermonde system

Solve the linear system:
```
V * h = e_1,  where V[l, m] = α_m^l  for l=0..M-1, m=1..M
```

Constraints satisfied: `Σ h_m = 1` and `Σ h_m α_m^l = 0` for `l = 1,...,M-1`.

Explicit closed-form (Lagrange interpolation):
```
h_m = ∏_{l ≠ m}  α_l / (α_l − α_m)
```

```python
def rr_coefficients(alphas):
    M = len(alphas)
    h = np.ones(M)
    for m in range(M):
        for l in range(M):
            if l != m:
                h[m] *= alphas[l] / (alphas[l] - alphas[m])
    return h   # shape: (M,)

# Example: alphas = [0.2, 0.02]
# h = [0.02/(0.02-0.2),  0.2/(0.2-0.02)]
#   = [−1/9,  10/9]
```

#### Step 2 — Run M LSA trajectories in parallel (same data!)

```python
# ALL M trajectories use SAME x_t sequence
theta_all = {a: np.zeros(d) for a in alphas}
iterates_all = {a: [] for a in alphas}

for t in range(T):
    for a in alphas:
        theta_all[a] = theta_all[a] + a * (A(x[t]) @ theta_all[a] + b(x[t]))
        iterates_all[a].append(theta_all[a].copy())
```

#### Step 3 — Batch means per stepsize

```python
batch_means_all = {}
for a in alphas:
    iterates = iterates_all[a][burn_in:]    # remove burn-in
    batch_means_all[a] = []
    for k in range(K):
        batch  = iterates[k*n : (k+1)*n]
        usable = batch[n_0:]
        batch_means_all[a].append(np.mean(usable, axis=0))
```

#### Step 4 — Combine via RR weights and construct CIs

```python
h = rr_coefficients(alphas)   # shape (M,)

# RR batch means (bias-reduced)
rr_batch_means = []
for k in range(K):
    theta_tilde_k = sum(h[m] * batch_means_all[alphas[m]][k]
                        for m in range(len(alphas)))
    rr_batch_means.append(theta_tilde_k)

# Grand mean and covariance — same formulas as Algorithm 1 Steps 3-4
theta_tilde   = np.mean(rr_batch_means, axis=0)
diffs         = np.array(rr_batch_means) - theta_tilde
Sigma_tilde   = (n - n_0) / K * (diffs.T @ diffs)
se_i          = np.sqrt(Sigma_tilde[i, i] / (K * (n - n_0)))
CI_i_RR       = [theta_tilde[i] - 1.96 * se_i,
                 theta_tilde[i] + 1.96 * se_i]

# This CI now targets θ* directly (bias ≈ 0 after extrapolation).
```

---

### 3.3 Stepsize Schedules for RR Extrapolation

The paper studies two schedules for choosing `{α_m}` when `M > 2`:

#### Geometric Decay ✅ Recommended

```python
# α_m = α_1 / c^(m-1),   c ≥ 2
# Example: α_1=0.2, c=2  →  [0.2, 0.1, 0.05, 0.025, ...]
# Variance bound: O(c · exp(16 c^{-1/2}))  — INDEPENDENT of M (safe)

alphas = [alpha1 / c**m for m in range(M)]
```

#### Equidistant Decay ⚠️ Use with small M only

```python
# α_m = (a+b) − b*(m−1)/(M−1),   a+b < 1
# Example: a=0.02, b=0.18, M=5  →  [0.20, 0.155, 0.11, 0.065, 0.02]
# Variance bound: O((2M/b)^{2M})  — blows up quickly with M

alphas = [(a + b) - b * (m / (M - 1)) for m in range(M)]
```

> **Practical guidance:** For most uses `M=2` (geometric with `c=10`: `α₁=0.2`, `α₂=0.02`) suffices. Benefits plateau around `M=5`. For equidistant, keep spacing `b/(M-1) ≥ 0.03`.

---

### 3.4 Zero-Bias Special Cases (no RR needed)

The paper proves `E[θ_∞] = θ*` in three important settings — if your problem matches one, **skip RR extrapolation**.

| Setting | Structure | Bias |
|---------|-----------|------|
| Independent multiplicative noise | `A(x_t) = Ā + ξ_t`, `ξ_t` i.i.d. mean-zero; `b(x_t) = b(s_t)` Markovian | **Zero** |
| Linear regression (additive ε) | `y_t = s_t^T w* + ε_t`, `ε_t` i.i.d. mean-zero, SGD on squared loss | **Zero** |
| Realizable linear-TD (semi-simulator) | TD learning where `V(s) = φ(s)^T θ` exactly; next state `s_next` sampled fresh | **Zero** |

**Sufficient condition for zero bias (Corollary A.5):**
```
E[ A(x_t) θ* + b(x_t) | x_{t+1} = x ] = 0   for all x ∈ X
```

---

### 3.5 Experiment Setup — LSA Problem Generation

#### Transition Matrix P (paper: 10 states)

```python
import numpy as np
from numpy.linalg import eig

def generate_P(n_states, rng):
    while True:
        M = rng.uniform(0, 1, (n_states, n_states))
        P = M / M.sum(axis=1, keepdims=True)   # row-normalize
        # Check irreducible + aperiodic (both needed for uniform ergodicity)
        # For random dense matrices this almost always holds; add explicit check
        vals, vecs = eig(P.T)
        idx = np.argmin(np.abs(vals - 1.0))
        pi  = np.real(vecs[:, idx])
        pi /= pi.sum()
        if np.all(pi > 0):
            return P, pi
```

#### Matrix A(x) (paper: d=5, Hurwitz mean Ā)

```python
def generate_A(n_states, d, pi, rng):
    # 1. Generate Hurwitz mean matrix Ā
    M_A = rng.standard_normal((d, d))
    evals = np.linalg.eigvals(M_A)
    if np.max(np.real(evals)) >= 0:
        M_A -= 2 * np.max(np.real(evals)) * np.eye(d)
    A_bar = M_A

    # 2. Add noise to each state (noise averages to 0 under π)
    E = rng.uniform(-1, 1, (n_states - 1, d, d))
    A_list = [A_bar + E[x] for x in range(n_states - 1)]

    # 3. Last state enforces E_π[A(x)] = Ā exactly
    A_last = A_bar - sum(pi[x] * E[x] for x in range(n_states - 1)) / pi[-1]
    A_list.append(A_last)

    return A_list, A_bar
```

#### Vector b(x)

```python
def generate_b(n_states, d, rng):
    # No mean constraint needed on b
    return [rng.uniform(-1, 1, d) for _ in range(n_states)]
```

#### True solution θ\*

```python
def compute_theta_star(A_list, b_list, pi):
    A_bar = sum(pi[x] * A_list[x] for x in range(len(A_list)))
    b_bar = sum(pi[x] * b_list[x] for x in range(len(b_list)))
    theta_star = np.linalg.solve(A_bar, -b_bar)
    # Verify: np.allclose(A_bar @ theta_star + b_bar, 0)
    return theta_star
```

#### Markov chain simulation

```python
def simulate_chain(P, pi, T, rng):
    x = rng.choice(len(pi), p=pi)   # start from stationary dist
    trajectory = [x]
    for _ in range(T - 1):
        x = rng.choice(len(pi), p=P[x])
        trajectory.append(x)
    return trajectory
```

---

## 4. Key Theoretical Results

### 4.1 Central Limit Theorem (Theorem 3.1)

```
√T (θ̄_T − E[θ_∞]) → N(0, Σ*)   as T → ∞

where:  θ̄_T = (1/T) Σ_{t=0}^{T-1} θ_t
        Σ*   = lim_{T→∞} E_μ[(θ̄_T − E[θ_∞])(θ̄_T − E[θ_∞])^T]

Requires: α < α_0  (sufficiently small constant)
```

**Key challenge:** `(θ_t)` alone is not Markov when data is Markovian. The proof works with the joint process `(x_t, θ_t)` and applies the Maxwell-Woodroofe CLT condition.

### 4.2 Asymptotic Bias (Theorem A.4)

```
E[θ_∞^(α)] − θ* = α B^(1) + α² B^(2) + ...   (α < α_1)

RR with M stepsizes reduces bias from O(α) to O(max_m α_m)^M
```

### 4.3 Variance Bounds (Propositions 5.1 & 5.2)

| Schedule | Formula | Variance Upper Bound | Risk as M grows |
|----------|---------|---------------------|-----------------|
| Geometric | `α_m = α_1 / c^{m-1}`, `c≥2` | `O(c · exp(16 c^{-1/2}))` | **Bounded — safe** |
| Equidistant | `α_m = (a+b) − b(m-1)/(M-1)` | `O((2M/b)^{2M})` | **Blows up — use small M** |

### 4.4 Weak Convergence (Theorem A.3, from HCX23)

```
W̄²₂(L(x_t, θ_t), μ̄) = O((1 - cα)^t)   for t ≥ τ_α

Tr(Var(θ_∞)) = O(α τ_α) = O(1)
```

---

## 5. Key Numerical Results Summary

### 5.1 Main Comparison: 100 Random Markovian Problems (T=10⁵)

Metrics: ℓ₂ error (×10⁻³), CI width (×10⁻³), coverage probability. **Median percentile shown.**

| Method | ℓ₂ error | CI Width | Coverage |
|--------|---------|---------|---------|
| α = 0.2 (constant) | 8.12 | 2.70 | 11% |
| α = 0.02 (constant) | 1.59 | 2.38 | 90% |
| **RR (0.2 + 0.02)** | **1.32** | **2.41** | **94% ✓** |
| α = 0.2/√k (diminishing) | 1.32 | 2.14 | 91% |
| α = 0.02/√k (diminishing) | 1.42 | 1.51 | 76% |

> RR extrapolation achieves the best coverage (closest to nominal 95%) while maintaining competitive ℓ₂ error.

### 5.2 Effect of Batch Number K (T=10⁶)

Constant stepsize is robust to K; diminishing stepsize degrades sharply when K exceeds the recommended level.

| K (batches) | RR Coverage | 0.2/√k Coverage | 0.02/√k Coverage |
|------------|------------|----------------|-----------------|
| 50 | 92.8% | 93.0% | 81.6% |
| 100 | 94.4% | 95.0% | 71.2% |
| 500 | 94.2% | 88.8% | 42.2% |
| 1000 | 94.2% | 75.4% | 30.4% |

### 5.3 Effect of Trajectory Length T

| T | RR Coverage | α=0.2/√k | α=0.02 | α=0.2 |
|---|------------|---------|--------|-------|
| 10³ | 83.4% | 76.8% | 83.6% | 82.2% |
| 10⁴ | 89.2% | 85.4% | 90.0% | 75.2% |
| 10⁵ | 91.2% | 90.0% | 88.2% | 0.04% |
| 10⁶ | 95.2% | 92.8% | 80.2% | 0% |

> At small T (10³–10⁴), constant stepsize + RR is competitive with or better than diminishing stepsizes, which converge too slowly.

### 5.4 Comparison vs Bootstrapping (T=10⁶)

| Method | Coverage | ℓ₂ Error | CI Width | Memory |
|--------|---------|---------|---------|--------|
| Constant + RR | 95.2% | 1.0×10⁻⁵ | 0.00134 | O(d) |
| Bootstrapping | 93.4% | 2.0×10⁻³ | 0.00411 | O(n·d) |

---

## 6. Step-by-Step Reproduction Guide

### 6.1 Dependencies

```bash
pip install numpy scipy matplotlib pandas tqdm
```

### 6.2 Suggested Module Structure

```
lsa_inference/
├── markov_chain.py       # generate P, simulate chain
├── lsa_problem.py        # generate A(x), b(x), compute θ*
├── lsa_runner.py         # Algorithm 1: run LSA, collect iterates
├── batch_inference.py    # Algorithm 1 Steps 3-4: Σ̂, CIs
├── rr_extrapolation.py   # Algorithm 2: RR weights + combined CIs
├── experiments/
│   ├── exp_main.py       # Table 1: 100 random problems, T=10^5
│   ├── exp_batch_k.py    # Table 2: vary K
│   ├── exp_traj_len.py   # Table 3: vary T
│   └── exp_bootstrap.py  # Section 6.1.5
└── utils.py              # metrics: l2_error, ci_width, coverage
```

### 6.3 Evaluation Metrics

```python
def l2_error(theta_bar, theta_star):
    return np.linalg.norm(theta_bar - theta_star)

def ci_width(Sigma_hat, K, n, n0, z=1.96, coord=0):
    se = np.sqrt(Sigma_hat[coord, coord] / (K * (n - n0)))
    return 2 * z * se

def coverage(theta_star, theta_bar, Sigma_hat, K, n, n0, z=1.96, coord=0):
    se = np.sqrt(Sigma_hat[coord, coord] / (K * (n - n0)))
    lo = theta_bar[coord] - z * se
    hi = theta_bar[coord] + z * se
    return float(lo <= theta_star[coord] <= hi)
```

### 6.4 Critical Implementation Details

1. **Same random seed for data:** All `M` LSA runs in RR **must** use the same `(x_t)` trajectory.

2. **Burn-in length `b`:** The paper uses `b=100` in the QQ experiment. For main tables, `b` is not stated explicitly — use a reasonable fraction (e.g., `min(1000, T//10)`).

3. **Intra-batch discard `n_0`:** Paper uses `n_0=0` in QQ plots. For Markovian data, `n_0 > 0` (e.g., proportional to mixing time `τ_α = O(log(1/α))`) helps decorrelate batches. Effect scales as `exp(-α n_0)`.

4. **Batch number K:** Use `K = int(T**0.3)` for trajectory length experiments (as recommended by CLTZ20). For the main comparison table, use `K = 50–100`.

5. **Diminishing stepsize batching:** Batch endpoints follow the CLTZ20 scheme:
   ```python
   r = T**(1 - alpha_exp) / (K + 1)
   e_k = int(((k + 1) * r) ** (1 / (1 - alpha_exp)))
   ```
   where `alpha_exp` is the exponent in `α_k = α_0 k^{-alpha_exp}` (paper uses `alpha_exp=0.5`).

6. **Coverage computation:** For each of 100 problems, simulate 100 independent trajectories. For each trajectory, check if `θ*[0]` (first coordinate only) is inside the CI. Average over 100 trajectories = coverage for that problem. Report percentiles (10/25/50/75/90) across the 100 problems.

7. **ℓ₂ error:** `‖θ̄ − θ*‖₂` (point estimate vs true solution, **not** vs `E[θ_∞]`).

8. **Variance estimator scaling:** Note the factor `(n - n_0)` in `Σ̂` — this ensures consistency as `n, K → ∞` and matches the scaling in the CI formula denominator `K(n - n_0)`.

### 6.5 Experiment Table Mapping

| Paper Table | T | K | # Problems | # Trajectories/problem | Metrics |
|-------------|---|---|------------|----------------------|---------|
| Table 1 (main) | 10⁵ | ~T^0.3 ≈ 16 | 100 | 100 | Percentiles of ℓ₂, CI, Cov |
| Table 2 (batch K) | 10⁶ | 50/100/500/1000 | 1 | 500 | Mean ± std err |
| Table 3 (traj len) | 10³–10⁶ | T^0.3 | 1 | 500 | Mean ± std err |
| Table 4 (nonlinear) | 10³–10⁶ | T^0.3 | 1 (logistic reg.) | 500 | Mean ± std err |
| Table 5 (i.i.d.) | 10⁵ | 50 | 100 | 100 | Percentiles |

### 6.6 Nonlinear SA Extension (Table 4 / Section 6.1.6)

The paper tests robustness on logistic regression with Markovian data:

```python
# Data: x_t from 2D Gaussian AR(1),  y_t ~ Bernoulli(sigmoid(w* @ x_t))
# Update: SGD on logistic loss (nonlinear SA, non-Hurwitz, unbounded chain)

def ar1_step(x_prev, rho=0.5, rng=None):
    noise = rng.standard_normal(2)
    return rho * x_prev + np.sqrt(1 - rho**2) * noise

def sgd_logistic_update(w, x_t, y_t, alpha):
    p = 1 / (1 + np.exp(-w @ x_t))
    grad = (p - y_t) * x_t
    return w - alpha * grad
```

---

## 7. Notation Glossary

| Symbol | Meaning |
|--------|---------|
| `θ*` | True target: `Ā θ* + b̄ = 0` |
| `θ_t^(α)` | LSA iterate at time `t` with stepsize `α` |
| `θ̄_T` | Time-averaged iterate: `(1/T) Σ θ_t` |
| `θ_∞^(α)` | Stationary distribution limit of `θ_t` |
| `E[θ_∞^(α)]` | Mean of limit distribution — biased w.r.t. `θ*` |
| `θ̃_k` | RR-extrapolated batch mean for batch `k` |
| `Ā` | `E_{x~π}[A(x)]` — mean matrix |
| `b̄` | `E_{x~π}[b(x)]` — mean vector |
| `α`, `α_m` | Constant stepsize(s) |
| `T` | Total trajectory length |
| `K` | Number of batches |
| `n` | Batch size: `(T - b) / K` |
| `n_0` | Intra-batch burn-in (iterates discarded at start of each batch) |
| `b` | Initial burn-in length |
| `Σ*` | True asymptotic covariance matrix |
| `Σ̂` | Batch-mean estimator of `Σ*` |
| `τ_α` | α-mixing time of the Markov chain: `O(K log(1/α))` |
| `μ̄` | Stationary distribution of joint process `(x_t, θ_t)` |
| `π` | Stationary distribution of data chain `(x_t)` |
| `M` | Number of stepsizes in RR extrapolation |
| `h_m` | RR coefficients: `Σ h_m = 1`, `Σ h_m α_m^l = 0` for `l=1..M-1` |
| `W̄₂` | Wasserstein-2 distance w.r.t. metric `d̄((x,θ),(x',θ')) = √(d₀(x,x') + ‖θ-θ'‖²)` |

---

## 8. Common Pitfalls & Debugging Tips

- **Coverage ≈ 0 for large T with α=0.2:** Expected — large stepsize creates large bias. The iterate converges to `E[θ_∞] ≠ θ*`. Use RR or reduce `α`.

- **Coverage degrades with large K for diminishing stepsize:** Diminishing-stepsize iterates become highly correlated as `α→0`; variance is underestimated. Don't increase K too aggressively beyond the CLTZ20 recommendation.

- **RR coefficients blow up:** For equidistant schedule, ensure spacing `b/(M-1)` is not too small (e.g., ≥ 0.03). Prefer geometric schedule for large M.

- **ℓ₂ error non-decreasing with T for large α:** Correct — the iterate converges to `E[θ_∞] ≠ θ*`, so error plateaus at `‖E[θ_∞] − θ*‖ = O(α)`.

- **Batch-mean Σ̂ is singular:** Happens when `K < d`. Ensure `K >> d` (paper uses `d=5`, `K ≥ 50`).

- **Markov chain not mixing:** Verify `P` is irreducible and aperiodic. For generated random dense P, this almost always holds, but add an explicit check.

- **Memory for large T:** You don't need to store all iterates — maintain a running sum within each batch and reset at batch boundaries. This reduces memory from `O(T·d)` to `O(K·d)`.

- **Numerical instability in RR:** If `|h_m|` are very large (check `np.max(np.abs(h))`), the combined estimator will have high variance. Increase spacing between `α_m` values.

---

*Summary of: Huo, Chen, Xie (2023). "Effectiveness of Constant Stepsize in Markovian LSA and Statistical Inference." arXiv:2312.10894*
