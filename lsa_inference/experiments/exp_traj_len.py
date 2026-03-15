"""Table 3: Effect of trajectory length T (single problem, 500 trajectories).

Methods: RR(0.2+0.02), constant alpha=0.2, constant alpha=0.02,
         diminishing 0.2/sqrt(k).
T values: 10^3, 10^4, 10^5, 10^6.
"""

import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from lsa_inference.markov_chain import generate_transition_matrix, simulate_chain
from lsa_inference.lsa_problem import generate_A, generate_b, compute_theta_star
from lsa_inference.lsa_runner import run_lsa_batched, run_lsa_diminishing
from lsa_inference.batch_inference import compute_covariance
from lsa_inference.rr_extrapolation import run_rr_extrapolation
from lsa_inference.utils import l2_error, ci_width, coverage


def main():
    n_states = 10
    d = 5
    n_traj = 500
    T_values = [1_000, 10_000, 100_000, 1_000_000]

    print(f"[Config] n_traj={n_traj}, n_states={n_states}, d={d}")
    print(f"[Config] T values: {[f'{t:,}' for t in T_values]}")

    rng = np.random.default_rng(456)
    P, pi = generate_transition_matrix(n_states, rng)
    A_list, A_bar = generate_A(n_states, d, pi, rng)
    b_list = generate_b(n_states, d, rng)
    theta_star = compute_theta_star(A_list, b_list, pi)

    evals = np.linalg.eigvals(A_bar)
    print(f"[Problem] ||θ*||={np.linalg.norm(theta_star):.4f}, "
          f"max_re(λ(Ā))={np.max(np.real(evals)):.4f}")

    methods = ['RR', 'alpha_0.2', 'alpha_0.02', 'dim_0.2']
    method_labels = {
        'RR': 'RR (0.2+0.02)',
        'alpha_0.2': 'α=0.2 (const)',
        'alpha_0.02': 'α=0.02 (const)',
        'dim_0.2': '0.2/√k (dim)',
    }

    results = []
    t_start_all = time.time()

    for ti, T in enumerate(T_values):
        K = max(int(T ** 0.3), 5)
        burn_in = min(1000, T // 10)

        print(f"\n{'='*60}")
        print(f"[T={T:,}] Starting ({ti+1}/{len(T_values)}), K={K}, burn_in={burn_in}")
        print(f"{'='*60}")
        t_T_start = time.time()
        cov_accum = {m: [] for m in methods}
        l2_accum = {m: [] for m in methods}

        log_interval = max(1, n_traj // 10)

        for traj_idx in tqdm(range(n_traj), desc=f"T={T:,}"):
            traj_rng = np.random.default_rng(rng.integers(0, 2**31))
            traj = simulate_chain(P, pi, T, traj_rng)

            # RR
            tt, St, lo, hi, se, n = run_rr_extrapolation(
                A_list, b_list, traj, [0.2, 0.02], K, burn_in)
            cov_accum['RR'].append(float(lo <= theta_star[0] <= hi))
            l2_accum['RR'].append(l2_error(tt, theta_star))

            # Constant alpha=0.2
            bm, n = run_lsa_batched(A_list, b_list, traj, 0.2, K, burn_in)
            tb, Sh = compute_covariance(bm, n)
            cov_accum['alpha_0.2'].append(
                coverage(theta_star, tb, Sh, K, n))
            l2_accum['alpha_0.2'].append(l2_error(tb, theta_star))

            # Constant alpha=0.02
            bm, n = run_lsa_batched(A_list, b_list, traj, 0.02, K, burn_in)
            tb, Sh = compute_covariance(bm, n)
            cov_accum['alpha_0.02'].append(
                coverage(theta_star, tb, Sh, K, n))
            l2_accum['alpha_0.02'].append(l2_error(tb, theta_star))

            # Diminishing 0.2/sqrt(k)
            bm, n_eff = run_lsa_diminishing(A_list, b_list, traj, 0.2, 0.5, K)
            tb, Sh = compute_covariance(bm, n_eff)
            cov_accum['dim_0.2'].append(
                coverage(theta_star, tb, Sh, K, n_eff))
            l2_accum['dim_0.2'].append(l2_error(tb, theta_star))

            # Periodic progress log
            if (traj_idx + 1) % log_interval == 0:
                done = traj_idx + 1
                t_elapsed = time.time() - t_T_start
                t_per_traj = t_elapsed / done
                t_remaining = t_per_traj * (n_traj - done)
                print(f"  [{done}/{n_traj}] Running coverage: "
                      f"RR={np.mean(cov_accum['RR'])*100:.1f}%, "
                      f"α0.2={np.mean(cov_accum['alpha_0.2'])*100:.1f}%, "
                      f"α0.02={np.mean(cov_accum['alpha_0.02'])*100:.1f}%, "
                      f"dim={np.mean(cov_accum['dim_0.2'])*100:.1f}% "
                      f"| {t_elapsed:.0f}s elapsed, ~{t_remaining:.0f}s left")

        t_T = time.time() - t_T_start
        print(f"\n[T={T:,}] Done in {t_T:.1f}s ({t_T/60:.1f}min)")
        for m in methods:
            vals = np.array(cov_accum[m])
            mean_cov = np.mean(vals) * 100
            se_cov = np.std(vals) / np.sqrt(n_traj) * 100
            mean_l2 = np.mean(l2_accum[m])
            print(f"  {method_labels[m]}: cov={mean_cov:.1f}% ± {se_cov:.1f}%, "
                  f"L2={mean_l2:.2e}")
            results.append({
                'T': T,
                'method': method_labels[m],
                'coverage_pct': mean_cov,
                'se_pct': se_cov,
                'l2_mean': mean_l2,
            })

    t_total = time.time() - t_start_all
    df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print(f"Table 3: Effect of Trajectory Length T")
    print(f"Total time: {t_total:.0f}s ({t_total/60:.1f}min)")
    print("=" * 60)
    pivot = df.pivot(index='T', columns='method', values='coverage_pct')
    print(pivot.to_string())

    df.to_csv('results_table3.csv', index=False)
    print("\nResults saved to results_table3.csv")


if __name__ == '__main__':
    main()
