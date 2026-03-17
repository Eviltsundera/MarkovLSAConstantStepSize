# Reproducing: Effectiveness of Constant Stepsize in Markovian LSA and Statistical Inference

Reproduction of the numerical experiments from:

> **Huo, Chen, Xie (2023).** "Effectiveness of Constant Stepsize in Markovian LSA and Statistical Inference." [arXiv:2312.10894](https://arxiv.org/abs/2312.10894)

## Key Findings

The paper shows that constant-stepsize Linear Stochastic Approximation (LSA) with Richardson-Romberg (RR) extrapolation achieves:
- **~94% CI coverage** (closest to nominal 95%) vs diminishing stepsize methods
- **Lowest L2 error** among all methods
- **Robustness** to batch number K, unlike diminishing stepsizes which degrade

## Setup

```bash
uv sync
```

Or with pip:
```bash
pip install -r requirements.txt
```

## Usage

### Quick run (~30 min)

Reduced-scale experiments that demonstrate the same qualitative patterns:

```bash
uv run python run_quick.py
```

### Full reproduction (~hours)

Exact paper parameters (100 problems × 100 trajectories, T up to 10^6):

```bash
# All experiments
uv run python run_experiments.py

# Individual tables
uv run python run_experiments.py table1      # Table 1: Main comparison (100 problems, T=10^5)
uv run python run_experiments.py table2      # Table 2: Effect of batch number K (T=10^6)
uv run python run_experiments.py table3      # Table 3: Effect of trajectory length T
uv run python run_experiments.py bootstrap   # Bootstrap comparison (T=10^6)
```

## Experiments

| Script | Paper Table | Description | Parameters |
|--------|------------|-------------|------------|
| `exp_main.py` | Table 1 | Main comparison: 5 methods | 100 problems, 100 traj, T=10^5 |
| `exp_batch_k.py` | Table 2 | Effect of batch number K | K=50/100/500/1000, T=10^6 |
| `exp_traj_len.py` | Table 3 | Effect of trajectory length | T=10^3 to 10^6 |
| `exp_bootstrap.py` | Section 6.1.5 | RR vs bootstrapping | T=10^6, 500 traj |

### Methods compared

| Method | Description |
|--------|-------------|
| α=0.2 (const) | Constant stepsize, large |
| α=0.02 (const) | Constant stepsize, small |
| **RR (0.2+0.02)** | Richardson-Romberg extrapolation (recommended) |
| 0.2/√k (dim) | Diminishing stepsize |
| 0.02/√k (dim) | Diminishing stepsize, small initial |

## Project Structure

```
lsa_inference/
├── markov_chain.py       # Transition matrix generation, chain simulation
├── lsa_problem.py        # A(x), b(x) generation with Hurwitz mean, θ* computation
├── vectorized.py         # Core vectorized LSA engine (all trajectories at once)
├── logging_utils.py      # Dual console+file logger setup
└── experiments/
    ├── common.py         # Shared utilities: problem generation, method dispatch
    ├── exp_main.py       # Table 1
    ├── exp_batch_k.py    # Table 2
    ├── exp_traj_len.py   # Table 3
    └── exp_bootstrap.py  # Bootstrap comparison

run_quick.py              # Quick validation (~30 min)
run_experiments.py        # Full-scale reproduction
docs/
├── LSA_Inference_Summary.md  # Detailed paper summary & algorithm specs
└── architecture.md           # Module layout & parallelization strategy
```

## Expected Results (Paper Reference)

### Table 1: Main comparison (T=10^5, median percentile)

| Method | L2 error (×10⁻³) | CI Width (×10⁻³) | Coverage |
|--------|:-:|:-:|:-:|
| α=0.2 (const) | 8.12 | 2.70 | 11% |
| α=0.02 (const) | 1.59 | 2.38 | 90% |
| **RR (0.2+0.02)** | **1.32** | **2.41** | **94%** |
| 0.2/√k (dim) | 1.32 | 2.14 | 91% |
| 0.02/√k (dim) | 1.42 | 1.51 | 76% |

### Table 3: Coverage vs trajectory length

| T | RR | α=0.2 | α=0.02 | 0.2/√k |
|---|:--:|:-----:|:------:|:------:|
| 10³ | 83.4% | 82.2% | 83.6% | 76.8% |
| 10⁴ | 89.2% | 75.2% | 90.0% | 85.4% |
| 10⁵ | 91.2% | 0.04% | 88.2% | 90.0% |
| 10⁶ | 95.2% | 0% | 80.2% | 92.8% |
