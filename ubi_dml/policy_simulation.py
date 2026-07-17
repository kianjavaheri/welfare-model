"""Translate the fitted DML coefficient into a policy-relevant simulation.

beta is dY/dT: the change in year-(T+1) wage income per marginal dollar of
year-T unearned income. Since UBI proxy here is scaled in the same dollar
units as UNEARNED_INCOME_1, a $6,000/year UBI's predicted effect on labor
income is simply 6000 * beta (first-order / local approximation -- valid
under LinearDML's constant-effect specification; CausalForestDML's CATEs
give the distribution of that same local effect across the population
instead of a single number).
"""
from __future__ import annotations

from dataclasses import dataclass

from econml.dml import LinearDML

UBI_ANNUAL_DOLLARS = 6000


@dataclass
class PolicySimResult:
    beta: float
    stderr: float
    ci_lower: float
    ci_upper: float
    predicted_wage_change: float
    predicted_wage_change_ci: tuple[float, float]


def simulate_ubi_effect(est: LinearDML, ubi_amount: float = UBI_ANNUAL_DOLLARS) -> PolicySimResult:
    beta = float(est.effect(X=None)[0])
    inf = est.effect_inference(X=None)
    summary_df = inf.summary_frame()
    stderr = float(summary_df["stderr"].iloc[0])
    ci_lower = float(summary_df["ci_lower"].iloc[0])
    ci_upper = float(summary_df["ci_upper"].iloc[0])

    return PolicySimResult(
        beta=beta,
        stderr=stderr,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        predicted_wage_change=ubi_amount * beta,
        predicted_wage_change_ci=(ubi_amount * ci_lower, ubi_amount * ci_upper),
    )
