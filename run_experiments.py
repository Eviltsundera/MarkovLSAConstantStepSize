#!/usr/bin/env python3
"""Main entry point to run all or individual experiments.

Usage:
    python run_experiments.py                          # run all experiments
    python run_experiments.py table1                   # Table 1 only
    python run_experiments.py table1 --n-workers 64    # Table 1 with 64 workers
    python run_experiments.py bootstrap --n-workers 32 # Bootstrap with 32 workers
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Run reproduction experiments for Markovian LSA paper"
    )
    parser.add_argument(
        "experiments", nargs="*",
        default=["table1", "table2", "table3", "bootstrap"],
        choices=["table1", "table2", "table3", "bootstrap"],
        help="Experiments to run (default: all)"
    )
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Multiprocessing workers (for table1, bootstrap)")
    parser.add_argument("--n-traj", type=int, default=None,
                        help="Override number of trajectories")
    parser.add_argument("--n-problems", type=int, default=None,
                        help="Override number of problems (table1 only)")
    parser.add_argument("-T", type=int, default=None,
                        help="Override trajectory length")
    args = parser.parse_args()

    for exp in args.experiments:
        if exp == "table1":
            print("\n" + "#" * 80)
            print("# Running Table 1: Main Comparison (100 problems, T=10^5)")
            print("#" * 80)
            from lsa_inference.experiments.exp_main import main as run_table1
            kwargs = {}
            if args.n_problems is not None:
                kwargs["n_problems"] = args.n_problems
            if args.n_traj is not None:
                kwargs["n_traj"] = args.n_traj
            if args.T is not None:
                kwargs["T"] = args.T
            if args.n_workers is not None:
                kwargs["n_workers"] = args.n_workers
            run_table1(**kwargs)
        elif exp == "table2":
            print("\n" + "#" * 80)
            print("# Running Table 2: Effect of Batch Number K (T=10^6)")
            print("#" * 80)
            from lsa_inference.experiments.exp_batch_k import main as run_table2
            kwargs = {}
            if args.T is not None:
                kwargs["T"] = args.T
            if args.n_traj is not None:
                kwargs["n_traj"] = args.n_traj
            run_table2(**kwargs)
        elif exp == "table3":
            print("\n" + "#" * 80)
            print("# Running Table 3: Effect of Trajectory Length T")
            print("#" * 80)
            from lsa_inference.experiments.exp_traj_len import main as run_table3
            kwargs = {}
            if args.n_traj is not None:
                kwargs["n_traj"] = args.n_traj
            run_table3(**kwargs)
        elif exp == "bootstrap":
            print("\n" + "#" * 80)
            print("# Running Bootstrap Comparison (T=10^6)")
            print("#" * 80)
            from lsa_inference.experiments.exp_bootstrap import main as run_bootstrap
            kwargs = {}
            if args.T is not None:
                kwargs["T"] = args.T
            if args.n_traj is not None:
                kwargs["n_traj"] = args.n_traj
            if args.n_workers is not None:
                kwargs["n_workers"] = args.n_workers
            run_bootstrap(**kwargs)


if __name__ == "__main__":
    main()
