# Architecture

Reproduction of "Effectiveness of Constant Stepsize in Markovian LSA and Statistical Inference" (Huo, Chen, Xie 2023, arXiv:2312.10894).

## Module Layout

```
lsa_inference/
├── markov_chain.py      # Transition matrix generation, chain simulation
├── lsa_problem.py       # A(x), b(x) generation with Hurwitz mean, θ* computation
├── vectorized.py        # Core vectorized LSA engine (all trajectories at once)
├── logging_utils.py     # Dual console+file logger setup
└── experiments/
    ├── common.py        # Shared utilities: problem generation, method dispatch
    ├── exp_main.py      # Table 1: 100 problems comparison (multiprocessing)
    ├── exp_batch_k.py   # Table 2: Effect of batch number K
    ├── exp_traj_len.py  # Table 3: Effect of trajectory length T
    └── exp_bootstrap.py # Table 4: RR vs bootstrap (multiprocessing)
```

## Vectorized Engine

The `vectorized.py` module is the primary compute engine. It processes all `n_traj`
trajectories simultaneously using numpy broadcasting:

```
theta update: thetas += alpha * (einsum('nij,nj->ni', A_t, thetas) + b_t)
shapes:       (n_traj, d)  += alpha * ((n_traj, d, d) @ (n_traj, d) + (n_traj, d))
```

It contains constant-stepsize LSA, diminishing-stepsize LSA, RR extrapolation,
and metric computation (L2 error, CI width, coverage) — all vectorized across trajectories.

## Parallelization Strategy

- **Within a problem**: All trajectories are vectorized via numpy (no Python loops
  over trajectories). This is handled by `vectorized.py`.
- **Across problems**: `exp_main.py` and `run_quick.py` use `multiprocessing.Pool`
  with `imap_unordered` to distribute independent problems across CPU cores.
- **Bootstrap**: `exp_bootstrap.py` uses `multiprocessing.Pool` for per-trajectory
  bootstrap resampling (the inner loop is not vectorizable due to random block selection).

## Key Parameters (Paper Defaults)

| Parameter | Value | Notes |
|-----------|-------|-------|
| n_states  | 10    | Markov chain states |
| d         | 5     | LSA dimension |
| T         | 10^5 (Table 1), 10^6 (Tables 2-4) | Trajectory length |
| K         | T^0.3 | Number of batches |
| burn_in   | min(1000, T/10) | Initial iterates discarded |
| alphas    | [0.2, 0.02] | Constant stepsizes for RR |

## CLI Usage

All scripts accept `--help` for options. Key flags:

```bash
# Full experiments with worker control
uv run python run_experiments.py table1 --n-workers 64
uv run python run_experiments.py bootstrap --n-workers 32 --n-traj 200

# Quick validation
uv run python run_quick.py --n-workers 16

# Individual experiments
uv run python -m lsa_inference.experiments.exp_main --n-workers 64 --n-problems 50
uv run python -m lsa_inference.experiments.exp_bootstrap --n-workers 32
```
