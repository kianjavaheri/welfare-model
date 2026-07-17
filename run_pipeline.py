"""End-to-end pipeline: load -> clean -> DML estimate -> policy simulation.

Usage:
    python run_pipeline.py --dat cps_00001.dat --xml cps_00001.xml
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ubi_dml.dml_pipeline import fit_causal_forest_dml, fit_linear_dml
from ubi_dml.policy_simulation import UBI_ANNUAL_DOLLARS, simulate_ubi_effect
from ubi_dml.preprocessing import load_and_clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dat", default="cps_00001.dat")
    parser.add_argument("--xml", default="cps_00001.xml")
    parser.add_argument("--cv", type=int, default=5, help="Cross-fitting folds")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--cate-hist-out", default="cate_histogram.png")
    parser.add_argument("--skip-causal-forest", action="store_true",
                        help="Skip the heterogeneous-effects step (slower: builds a 1000-tree forest)")
    args = parser.parse_args()

    print(f"Loading {args.dat} using layout from {args.xml} ...")
    df = load_and_clean(args.dat, args.xml)
    print(f"Analytic sample: {len(df):,} person-year-pairs "
          f"(ages {df['AGE_1'].min()}-{df['AGE_1'].max()}, "
          f"years {sorted(df['YEAR_1'].unique())} -> {sorted(df['YEAR_2'].unique())})")

    print("\nFitting LinearDML (XGBoost nuisances, Neyman-orthogonal partialling-out) ...")
    linear_est = fit_linear_dml(df, cv=args.cv, random_state=args.random_state)
    sim = simulate_ubi_effect(linear_est, ubi_amount=UBI_ANNUAL_DOLLARS)

    print("\n=== Average causal effect (LinearDML) ===")
    print(f"beta (d wage_t+1 / d unearned_income_t): {sim.beta:.4f}")
    print(f"stderr: {sim.stderr:.4f}   95% CI: [{sim.ci_lower:.4f}, {sim.ci_upper:.4f}]")

    print(f"\n=== Policy simulation: ${UBI_ANNUAL_DOLLARS:,}/year UBI ===")
    print(f"Predicted change in next-year wage income: "
          f"${sim.predicted_wage_change:,.2f} "
          f"(95% CI: [${sim.predicted_wage_change_ci[0]:,.2f}, "
          f"${sim.predicted_wage_change_ci[1]:,.2f}])")

    if not args.skip_causal_forest:
        print("\nFitting CausalForestDML for personalized (heterogeneous) treatment effects ...")
        _, cate, _ = fit_causal_forest_dml(df, cv=args.cv, random_state=args.random_state)

        print(f"\nCATE summary across {len(cate):,} people:")
        print(f"  mean={np.mean(cate):.4f}  median={np.median(cate):.4f}  "
              f"std={np.std(cate):.4f}  [{np.min(cate):.4f}, {np.max(cate):.4f}]")

        plt.figure(figsize=(8, 5))
        plt.hist(cate, bins=50, color="#4C72B0", edgecolor="white")
        plt.axvline(sim.beta, color="black", linestyle="--", label=f"LinearDML ATE = {sim.beta:.3f}")
        plt.xlabel(r"Personalized treatment effect $\hat\theta(X_i)$ "
                   r"(d future wage / d unearned income)")
        plt.ylabel("Number of people")
        plt.title("CausalForestDML: distribution of personalized treatment effects")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.cate_hist_out, dpi=150)
        print(f"\nSaved CATE histogram to {args.cate_hist_out}")


if __name__ == "__main__":
    main()
