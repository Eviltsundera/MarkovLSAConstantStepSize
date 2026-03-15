#!/usr/bin/env python3
"""Main entry point to run all or individual experiments.

Usage:
    python run_experiments.py              # run all experiments
    python run_experiments.py table1       # Table 1 only
    python run_experiments.py table2       # Table 2 only
    python run_experiments.py table3       # Table 3 only
    python run_experiments.py bootstrap    # Bootstrap comparison only
"""

import sys


def main():
    experiments = sys.argv[1:] if len(sys.argv) > 1 else [
        'table1', 'table2', 'table3', 'bootstrap'
    ]

    for exp in experiments:
        if exp == 'table1':
            print("\n" + "#" * 80)
            print("# Running Table 1: Main Comparison (100 problems, T=10^5)")
            print("#" * 80)
            from lsa_inference.experiments.exp_main import main as run_table1
            run_table1()
        elif exp == 'table2':
            print("\n" + "#" * 80)
            print("# Running Table 2: Effect of Batch Number K (T=10^6)")
            print("#" * 80)
            from lsa_inference.experiments.exp_batch_k import main as run_table2
            run_table2()
        elif exp == 'table3':
            print("\n" + "#" * 80)
            print("# Running Table 3: Effect of Trajectory Length T")
            print("#" * 80)
            from lsa_inference.experiments.exp_traj_len import main as run_table3
            run_table3()
        elif exp == 'bootstrap':
            print("\n" + "#" * 80)
            print("# Running Bootstrap Comparison (T=10^6)")
            print("#" * 80)
            from lsa_inference.experiments.exp_bootstrap import main as run_bootstrap
            run_bootstrap()
        else:
            print(f"Unknown experiment: {exp}")
            print("Available: table1, table2, table3, bootstrap")
            sys.exit(1)


if __name__ == '__main__':
    main()
