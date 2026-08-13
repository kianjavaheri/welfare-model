"""End-to-end pipeline: load -> response-weight -> clean -> DML estimate ->
policy report.

Usage:
    python run_pipeline.py \
        --baseline Data/ICPSR_37871/DS0001/37871-0001-Data.tsv \
        --followup Data/ICPSR_37871/DS0002/37871-0002-Data.tsv
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ubi_dml.dml_pipeline import fit_causal_forest_dml, fit_employment_dml, fit_linear_dml, fit_response_weights
from ubi_dml.policy_simulation import format_policy_report
from ubi_dml.preprocessing import (
    TREATMENT_COL,
    build_employment_sample,
    clean,
    load_raw,
    prepare_response_sample,
    summarize_employment_by_arm,
    summarize_response_by_arm,
    summarize_wage_by_arm,
    summarize_wage_response_by_arm,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="Data/ICPSR_37871/DS0001/37871-0001-Data.tsv",
                        help="Path to the DS0001 baseline file (.dta/.csv/.tsv/.tab)")
    parser.add_argument("--followup", default="Data/ICPSR_37871/DS0002/37871-0002-Data.tsv",
                        help="Path to the DS0002 Year-1 follow-up file (.dta/.csv/.tsv/.tab)")
    parser.add_argument("--cv", type=int, default=5, help="Cross-fitting folds")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--cate-hist-out", default="cate_histogram.png")
    parser.add_argument("--skip-causal-forest", action="store_true",
                        help="Skip the heterogeneous-effects step (slower: builds a 1000-tree forest)")
    args = parser.parse_args()

    print(f"Loading {args.baseline} + {args.followup} ...")
    raw = load_raw(args.baseline, args.followup)

    response_sample = prepare_response_sample(raw)
    print(f"Baseline-complete sample: {len(response_sample):,} mothers "
          f"(treatment arm: {int(response_sample[TREATMENT_COL].sum()):,} high-cash, "
          f"{int((1 - response_sample[TREATMENT_COL]).sum()):,} low-cash)")
    summarize_response_by_arm(response_sample)
    summarize_wage_response_by_arm(response_sample)

    print("\nFitting response-propensity models (cross-fitted XGBoost, corrects for "
          "differential item non-response on both the earnings question and the "
          "hours question the wage sample also depends on -- see "
          "preprocessing.py docstring) ...")
    weighted = fit_response_weights(response_sample, cv=args.cv, random_state=args.random_state)

    df = clean(weighted)
    print(f"\nWage-analytic sample: {len(df):,} mothers "
          f"(treatment arm: {int(df[TREATMENT_COL].sum()):,} high-cash, "
          f"{int((1 - df[TREATMENT_COL]).sum()):,} low-cash)")
    summarize_wage_by_arm(df)

    employment_df = build_employment_sample(weighted)
    print(f"\nEmployment-analytic sample: {len(employment_df):,} mothers "
          "(includes mothers with EMPLOYED=0, unlike the wage-analytic sample above)")
    summarize_employment_by_arm(employment_df)

    print("\nFitting employment LinearDML (extensive-margin check for post-treatment "
          "selection into the wage sample -- see preprocessing.py docstring) ...")
    _, employment_result = fit_employment_dml(employment_df, cv=args.cv, random_state=args.random_state)
    print(f"\n=== Effect of treatment on P(employed) ===")
    print(f"beta (probability points): {employment_result.beta:+.4f} "
          f"(stderr {employment_result.stderr:.4f}, "
          f"95% CI [{employment_result.ci_lower:+.4f}, {employment_result.ci_upper:+.4f}])")
    print("If this CI excludes 0, the wage-only ATE below should be read as "
          "conditional-on-employment, not as an unconditional effect on wages.")

    print("\nFitting LinearDML (XGBoost outcome model + XGBoost propensity model, "
          "Neyman-orthogonal partialling-out, discrete_treatment=True, response-weighted) ...")
    _, ate_result = fit_linear_dml(df, cv=args.cv, random_state=args.random_state)

    print("\n=== Average treatment effect (LinearDML) ===")
    print(format_policy_report(ate_result))

    if not args.skip_causal_forest:
        print("\nFitting CausalForestDML for personalized (heterogeneous) treatment effects ...")
        _, cate, cate_pct, forest_ate_result, _ = fit_causal_forest_dml(
            df, cv=args.cv, random_state=args.random_state
        )

        print("\n=== Average treatment effect (CausalForestDML) ===")
        print(format_policy_report(forest_ate_result))

        print(f"\nCATE summary across {len(cate):,} mothers (percentage wage impact):")
        print(f"  mean={np.mean(cate_pct):.2f}%  median={np.median(cate_pct):.2f}%  "
              f"std={np.std(cate_pct):.2f}%  [{np.min(cate_pct):.2f}%, {np.max(cate_pct):.2f}%]")

        plt.figure(figsize=(8, 5))
        plt.hist(cate_pct, bins=50, color="#4C72B0", edgecolor="white")
        plt.axvline(ate_result.pct_wage_impact, color="black", linestyle="--",
                    label=f"LinearDML ATE = {ate_result.pct_wage_impact:+.2f}%")
        plt.xlabel("Personalized treatment effect (% change in hourly wage)")
        plt.ylabel("Number of mothers")
        plt.title("CausalForestDML: distribution of personalized treatment effects")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.cate_hist_out, dpi=150)
        print(f"\nSaved CATE histogram to {args.cate_hist_out}")


if __name__ == "__main__":
    main()
