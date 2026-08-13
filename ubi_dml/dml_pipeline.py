"""Double Machine Learning estimation of the BFY cash-transfer -> log hourly
wage causal effect, per Chernozhukov et al. (2018) "Double/Debiased ML".

Why discrete_treatment=True and an XGBClassifier, not an XGBRegressor
------------------------------------------------------------------------
The prior CPS specification had a *continuous* treatment (dollars of
unearned income), so the nuisance model for E[T|X] was an XGBRegressor
predicting a dollar amount. TREATA0 here is a *binary* RCT arm assignment
(1 = high cash, 0 = low cash), so the correct nuisance model for E[T|X] is a
propensity score -- P(T=1|X) -- which needs a classifier's predict_proba,
not a regressor's predict. Passing discrete_treatment=True tells econml to
call model_t.predict_proba internally and to treat the recovered effect as
a discrete contrast (E[Y|T=1,X] - E[Y|T=0,X]) rather than a continuous
per-unit slope.

Identification note (why this pivot fixes the omitted-variable problem)
----------------------------------------------------------------------------
Under random assignment, T is independent of X and of every unobserved
confounder (wealth included) by design, so E[T|X] is constant at the
population randomization probability -- the propensity model here is only
sharpening precision (soaking up predictable variance in T before residual-
izing), not doing the identification work it would have to do in an
observational design. Likewise model_y/XGBRegressor for E[Y|X] only reduces
residual outcome variance; it does not need to capture "all confounding" the
way the CPS specification's outcome model implicitly had to. Cross-fitting
(cv-fold sample splitting between nuisance fitting and effect estimation)
still applies exactly as before, protecting beta from nuisance-overfitting
bias.

A second, separate pair of propensity models: response, not treatment
------------------------------------------------------------------------
fit_response_weights() below is not a DML nuisance model -- it runs before
and outside of the causal estimators, on preprocessing.prepare_response_sample()'s
output, to correct for differential item non-response. There are two
response events, not one, because the two analytic samples downstream need
different fields: build_employment_sample() only needs the Year-1 earnings
question answered (control: 8.8% missing vs. treatment: 4.1%, p=0.0045),
while clean()'s wage sample needs *both* the earnings question and the
Year-1 hours question (MHWEEKTOTALA1) answered, and the latter has its own,
separate non-response (~19.7% missing among earnings-responders) that an
earnings-only correction would silently miss -- see preprocessing.py's
"Differential item non-response" docstring note for the full derivation of
why one indicator isn't enough.

fit_response_weights() fits two cross-fitted XGBClassifiers -- one for
P(RESPONDED=1 | X, T) (earnings only), one for P(WAGE_RESPONDED=1 | X, T)
(earnings and hours both) -- each using out-of-fold predictions, exactly so
that no row's weight is estimated from a model that saw that row's own
response outcome (the same overfitting concern cross-fitting protects the
DML nuisance models from, applied here to a reweighting step instead of a
causal estimate), and turns each into its own inverse-probability-of-
response weight via preprocessing.attach_response_weights(). fit_linear_dml()
and fit_causal_forest_dml() use WAGE_RESPONSE_WEIGHT (matching clean()'s
sample); fit_employment_dml() uses RESPONSE_WEIGHT (matching
build_employment_sample()'s sample). Every effect estimated below is
response-weighted for the non-response event that actually determines its
sample, not just estimated on the (non-randomly incomplete) complete-case
subset.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML, LinearDML
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier, XGBRegressor

from ubi_dml.preprocessing import (
    CONFOUNDERS_BASELINE,
    EMPLOYED_COL,
    RESPONDED_COL,
    RESPONSE_WEIGHT_COL,
    TREATMENT_COL,
    WAGE_RESPONDED_COL,
    WAGE_RESPONSE_WEIGHT_COL,
    attach_response_weights,
)

NUMERIC_CONFOUNDERS = ["AGEA0", "HHCOMBINEDINCOMEA0", "HHNADULTA0", "HHNCHILDA0"]
CATEGORICAL_CONFOUNDERS = ["MRACEA0", "MEDLEVELA0", "MRELATESTATUSA0", "SITEA0"]

assert set(NUMERIC_CONFOUNDERS) | set(CATEGORICAL_CONFOUNDERS) == set(CONFOUNDERS_BASELINE)

# Floor on the response-propensity models' predicted P(responded=1|X,T),
# preventing a handful of near-zero predictions from producing extreme IPW
# weights. Overall earnings-response rate in this extract is ~91-96%;
# earnings-AND-hours response is lower (~75-83%) -- this is a defensive
# bound for both, not expected to bind for most rows.
MIN_RESPONSE_PROPENSITY = 0.05


def build_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode nominal baseline confounders (race, education level,
    relationship status, site); leave age/income/household-size numeric."""
    numeric = df[NUMERIC_CONFOUNDERS].astype(np.float64).reset_index(drop=True)
    categorical = pd.get_dummies(
        df[CATEGORICAL_CONFOUNDERS].astype(int).astype(str),
        prefix=CATEGORICAL_CONFOUNDERS,
        drop_first=True,
    ).reset_index(drop=True)
    # get_dummies returns bool columns; force a single homogeneous dtype
    # before this hits sklearn/xgboost/econml (mixed bool+float columns
    # silently coerce DataFrame.values to dtype=object, corrupting matmuls).
    return pd.concat([numeric, categorical], axis=1).astype(np.float64)


def build_response_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Design matrix for the response-propensity model: baseline confounders
    plus the treatment arm itself. T has to be a feature here (unlike in the
    causal DML nuisance models, where it's the target, not a feature) since
    response to the earnings question differs significantly by arm -- the
    model needs to be able to learn arm-specific response rates, not just
    covariate-driven ones."""
    X = build_design_matrix(df)
    T = df[[TREATMENT_COL]].astype(np.float64).reset_index(drop=True)
    return pd.concat([X, T], axis=1)


def _xgb_regressor(**overrides) -> XGBRegressor:
    params = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, verbosity=0,
                  random_state=0)
    params.update(overrides)
    return XGBRegressor(**params)


def _xgb_classifier(**overrides) -> XGBClassifier:
    params = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, verbosity=0,
                  random_state=0, objective="binary:logistic", eval_metric="logloss")
    params.update(overrides)
    return XGBClassifier(**params)


def _fit_propensity_and_attach_weights(
    df: pd.DataFrame,
    design: pd.DataFrame,
    target_col: str,
    weight_col: str,
    cv: int,
    random_state: int,
) -> pd.DataFrame:
    """Cross-fit P(target_col=1 | X, T) with an XGBClassifier (out-of-fold
    predictions, so no row's weight comes from a model that saw that row's
    own outcome), clip, and attach as weight_col via
    preprocessing.attach_response_weights(). Shared by fit_response_weights()
    for both the earnings-only and earnings-and-hours response events."""
    target = df[target_col].to_numpy(dtype=np.int64)

    propensities = cross_val_predict(
        _xgb_classifier(),
        design.to_numpy(),
        target,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        method="predict_proba",
    )[:, 1]
    propensities = np.clip(propensities, MIN_RESPONSE_PROPENSITY, 1.0)

    return attach_response_weights(df, propensities, responded_col=target_col, weight_col=weight_col)


def fit_response_weights(df: pd.DataFrame, cv: int = 5, random_state: int = 0) -> pd.DataFrame:
    """Estimate inverse-probability-of-response weights for both differential
    item non-response events this pipeline depends on -- see this module's
    and preprocessing.py's "Differential item non-response" docstrings. df
    must be preprocessing.prepare_response_sample()'s output (includes
    RESPONDED=0/WAGE_RESPONDED=0 rows -- each propensity model needs to see
    its own non-responders). Fits two independent cross-fitted XGBClassifiers
    on the same baseline-confounders-plus-arm design matrix -- one targeting
    RESPONDED (earnings question only, for build_employment_sample()), one
    targeting WAGE_RESPONDED (earnings and hours both, for clean()) -- and
    returns df with both RESPONSE_WEIGHT and WAGE_RESPONSE_WEIGHT attached."""
    design = build_response_design_matrix(df)
    df = _fit_propensity_and_attach_weights(
        df, design, RESPONDED_COL, RESPONSE_WEIGHT_COL, cv, random_state
    )
    df = _fit_propensity_and_attach_weights(
        df, design, WAGE_RESPONDED_COL, WAGE_RESPONSE_WEIGHT_COL, cv, random_state
    )
    return df


def pct_wage_impact(log_effect):
    """Convert a log-wage effect (or array of effects) to a percentage wage
    impact: exp(beta) - 1, as a percent. Valid for both the scalar ATE and
    an array of per-person CATEs since exp() is applied elementwise."""
    return (np.exp(log_effect) - 1.0) * 100.0


@dataclass
class ATEResult:
    beta: float                          # average effect on ln(hourly wage)
    stderr: float
    ci_lower: float
    ci_upper: float
    pct_wage_impact: float               # (exp(beta) - 1) * 100
    pct_wage_impact_ci: tuple[float, float]


def _build_ate_result(point: float, stderr: float, ci_lower: float, ci_upper: float) -> ATEResult:
    return ATEResult(
        beta=point,
        stderr=stderr,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        pct_wage_impact=pct_wage_impact(point),
        pct_wage_impact_ci=(pct_wage_impact(ci_lower), pct_wage_impact(ci_upper)),
    )


def _extract_effect_inference(inf) -> tuple[float, float, float]:
    """Extract (stderr, ci_lower, ci_upper) from an InferenceResults object
    (LinearDML.effect_inference)."""
    summary_df = inf.summary_frame()
    stderr = float(summary_df["stderr"].iloc[0])
    ci_lower = float(summary_df["ci_lower"].iloc[0])
    ci_upper = float(summary_df["ci_upper"].iloc[0])
    return stderr, ci_lower, ci_upper


def _summarize_effect_inference(point: float, inf) -> ATEResult:
    stderr, ci_lower, ci_upper = _extract_effect_inference(inf)
    return _build_ate_result(point, stderr, ci_lower, ci_upper)


def _summarize_ate_inference(inf) -> ATEResult:
    """Extract from a PopulationSummaryResults object (CausalForestDML.ate_inference)
    -- a different econml result type from effect_inference, with no
    summary_frame(): point/stderr/CI live on .mean_point/.stderr_mean/.conf_int_mean()."""
    point = float(inf.mean_point)
    stderr = float(inf.stderr_mean)
    ci_lower, ci_upper = inf.conf_int_mean()
    return _build_ate_result(point, stderr, float(ci_lower), float(ci_upper))


def fit_linear_dml(df: pd.DataFrame, cv: int = 5, random_state: int = 0) -> tuple[LinearDML, ATEResult]:
    """Fit LinearDML for a single average treatment effect: the average
    effect of the high-cash arm (vs. low-cash) on log hourly wages. X=None
    (no effect-modifier features) is what makes the fitted effect a single
    scalar rather than a per-person function -- W still carries every
    baseline confounder into both nuisance models for precision. df must
    come from preprocessing.clean() applied to a response-weighted sample
    (see preprocessing.py docstring) -- sample_weight=WAGE_RESPONSE_WEIGHT
    corrects for differential non-response on the earnings AND hours
    questions (clean()'s sample requires both)."""
    X = build_design_matrix(df)
    Y = df["LOG_HOURLY_WAGE"].to_numpy(dtype=np.float64)
    T = df[TREATMENT_COL].to_numpy(dtype=np.int64)
    weights = df[WAGE_RESPONSE_WEIGHT_COL].to_numpy(dtype=np.float64)

    est = LinearDML(
        model_y=_xgb_regressor(),
        model_t=_xgb_classifier(),
        discrete_treatment=True,
        cv=cv,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="econml.*")
        est.fit(Y, T, X=None, W=X.to_numpy(), sample_weight=weights)

    beta = float(est.effect(X=None)[0])
    inf = est.effect_inference(X=None)
    return est, _summarize_effect_inference(beta, inf)


def fit_causal_forest_dml(
    df: pd.DataFrame, cv: int = 5, random_state: int = 0, min_samples_leaf: int = 5
) -> tuple[CausalForestDML, np.ndarray, np.ndarray, ATEResult, pd.DataFrame]:
    """Fit CausalForestDML for *heterogeneous* (personalized) treatment
    effects. Returns the fitted estimator, the per-person CATE array (log
    points), the same array converted to percentage wage impact, the
    overall ATE (mean of the CATEs, with inference) as an ATEResult, and
    the X matrix used. df must come from preprocessing.clean() applied to a
    response-weighted sample -- sample_weight=WAGE_RESPONSE_WEIGHT corrects
    for differential non-response on the earnings AND hours questions, same
    as fit_linear_dml().

    min_samples_leaf default: the analytic BFY sample here is ~170-200
    mothers, not the 80K+ rows the prior CPS specification had. With
    min_samples_leaf=20 (fine for 80K rows), no split on a ~170-row honest-
    sample forest clears that bar, so every tree degenerates to a single
    root node and every person gets an identical, uninformative CATE
    (verified: std(cate)==0.0 at leaf=20 on this data). 5 is small enough to
    let the forest actually split, but at this sample size the resulting
    CATEs are still high-variance -- treat the *distribution* as suggestive
    of heterogeneity, not as reliable person-level point estimates."""
    X = build_design_matrix(df)
    Y = df["LOG_HOURLY_WAGE"].to_numpy(dtype=np.float64)
    T = df[TREATMENT_COL].to_numpy(dtype=np.int64)
    weights = df[WAGE_RESPONSE_WEIGHT_COL].to_numpy(dtype=np.float64)

    est = CausalForestDML(
        model_y=_xgb_regressor(),
        model_t=_xgb_classifier(),
        discrete_treatment=True,
        cv=cv,
        n_estimators=1000,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="econml.*")
        est.fit(Y, T, X=X.to_numpy(), W=None, sample_weight=weights)

    Xarr = X.to_numpy()
    cate = est.effect(Xarr)
    cate_pct = pct_wage_impact(cate)

    ate_inf = est.ate_inference(Xarr)
    ate_result = _summarize_ate_inference(ate_inf)

    return est, cate, cate_pct, ate_result, X


@dataclass
class EmploymentEffectResult:
    beta: float             # change in P(employed), as a fraction (0.05 = 5pp)
    stderr: float
    ci_lower: float
    ci_upper: float


def fit_employment_dml(
    df: pd.DataFrame, cv: int = 5, random_state: int = 0
) -> tuple[LinearDML, EmploymentEffectResult]:
    """Fit LinearDML for the effect of treatment on P(employed) -- the
    extensive-margin check for post-treatment selection into the wage
    sample (see preprocessing.py's "Extensive-margin (employment) selection"
    module docstring note). df must come from
    preprocessing.build_employment_sample() applied to a response-weighted
    sample: it needs mothers with EMPLOYED=0 included, which the wage-
    analytic sample excludes by construction, and sample_weight=
    RESPONSE_WEIGHT corrects for differential non-response on the earnings
    question the same way it does in fit_linear_dml(). Y is a 0/1 indicator
    rather than a log-wage, so this is a linear-probability-model DML --
    beta is a probability-point difference, not a log effect, and
    pct_wage_impact() does not apply to it."""
    X = build_design_matrix(df)
    Y = df[EMPLOYED_COL].to_numpy(dtype=np.float64)
    T = df[TREATMENT_COL].to_numpy(dtype=np.int64)
    weights = df[RESPONSE_WEIGHT_COL].to_numpy(dtype=np.float64)

    est = LinearDML(
        model_y=_xgb_regressor(),
        model_t=_xgb_classifier(),
        discrete_treatment=True,
        cv=cv,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="econml.*")
        est.fit(Y, T, X=None, W=X.to_numpy(), sample_weight=weights)

    beta = float(est.effect(X=None)[0])
    inf = est.effect_inference(X=None)
    stderr, ci_lower, ci_upper = _extract_effect_inference(inf)
    return est, EmploymentEffectResult(beta=beta, stderr=stderr, ci_lower=ci_lower, ci_upper=ci_upper)
