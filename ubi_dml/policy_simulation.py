"""Report the BFY cash-transfer effect in policy-relevant terms.

The prior CPS specification's treatment was a continuous dollar amount of
unearned income, so the fitted beta was a *slope* (d wage / d dollar) that
had to be multiplied by a hypothetical UBI grant size to become policy-
relevant. TREATA0 here is a binary RCT arm assignment, not a continuous
dollar shock -- beta already *is* the causal effect of "being assigned to
the high-cash arm" on log hourly wages, so there is no marginal-dollar slope
to rescale. The natural policy quantity is a level effect: the percentage
change in wages the high-cash arm causes relative to the low-cash arm,
exp(beta) - 1, which ATEResult.pct_wage_impact already carries.
"""
from __future__ import annotations

from ubi_dml.dml_pipeline import ATEResult


def format_policy_report(ate_result: ATEResult, policy_name: str = "BFY high-cash arm") -> str:
    return (
        f"Average treatment effect on ln(hourly wage): {ate_result.beta:.4f} "
        f"(stderr {ate_result.stderr:.4f}, "
        f"95% CI [{ate_result.ci_lower:.4f}, {ate_result.ci_upper:.4f}])\n"
        f"Percentage wage impact of the {policy_name}: "
        f"{ate_result.pct_wage_impact:+.2f}% "
        f"(95% CI [{ate_result.pct_wage_impact_ci[0]:+.2f}%, "
        f"{ate_result.pct_wage_impact_ci[1]:+.2f}%])"
    )
