"""Double Machine Learning estimation of the unearned-income -> future-wage
causal effect, per Chernozhukov et al. (2018) "Double/Debiased ML".

Neyman orthogonality, briefly
------------------------------
A naive approach would regress INCWAGE_2 on UNEARNED_INCOME_1 and the raw
confounders X in one OLS/ML model. That target is *not* orthogonal to
nuisance estimation error: if the model of E[Y|X] or E[T|X] is even slightly
misspecified (near-guaranteed with a flexible learner like XGBoost, whose
regularization bias doesn't vanish fast enough), the bias leaks directly
into the coefficient on T.

DML instead partials out X from both Y and T *separately* (Robinson's
transformation): it fits ml_y: X -> Y and ml_t: X -> T with off-the-shelf
ML models, takes the residuals Y_res = Y - ml_y(X) and T_res = T - ml_t(X),
and estimates beta from a simple regression of Y_res on T_res. This
"partialling-out" moment condition has zero derivative with respect to
small errors in either nuisance model (Neyman orthogonality), so first-order
mistakes in ml_y / ml_t wash out and beta remains root-n consistent. Cross-
fitting (cv folds: nuisance models for each row are trained on the *other*
folds) removes the remaining overfitting bias from reusing the same data to
fit nuisances and the causal estimate. This is what protects the estimate
from omitted-variable-style bias driven by nonlinear confounding, as long as
X contains the relevant confounders.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML, LinearDML
from xgboost import XGBRegressor

NUMERIC_CONFOUNDERS = ["AGE_1", "EDUC_1", "NUMPREC_1", "NCHILD_1", "IND_1"]
CATEGORICAL_CONFOUNDERS = ["SEX_1", "RACE_1", "MARST_1", "METRO_1"]


def build_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode low-cardinality nominal confounders (SEX, RACE, MARST,
    METRO); leave AGE/EDUC/NUMPREC/NCHILD numeric. IND (4-digit industry
    code, ~270 categories) is fed to XGBoost as a raw numeric code rather
    than one-hot expanded -- tree splits on it still recover useful industry
    groupings without a several-hundred-column sparse expansion."""
    numeric = df[NUMERIC_CONFOUNDERS].astype(np.float64).reset_index(drop=True)
    categorical = pd.get_dummies(
        df[CATEGORICAL_CONFOUNDERS].astype(int).astype(str),
        prefix=CATEGORICAL_CONFOUNDERS,
        drop_first=True,
    ).reset_index(drop=True)
    # get_dummies returns bool columns; force everything to float64 so the
    # concatenated frame is a single homogeneous dtype before it hits
    # sklearn/xgboost/econml (mixed bool+float columns silently coerce
    # DataFrame.values to dtype=object, which corrupts downstream matmuls).
    return pd.concat([numeric, categorical], axis=1).astype(np.float64)


def _xgb(**overrides) -> XGBRegressor:
    params = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, verbosity=0,
                  random_state=0)
    params.update(overrides)
    return XGBRegressor(**params)


def fit_linear_dml(df: pd.DataFrame, cv: int = 5, random_state: int = 0) -> LinearDML:
    """Fit LinearDML for a single constant coefficient beta: the average
    dollar-for-dollar effect of year-T unearned income on year-(T+1) wage
    income. X=None (no effect-modifier features) is what makes the fitted
    treatment effect a single scalar rather than a per-person function --
    W carries every confounder into both nuisance models so they're still
    fully controlled for, they just don't drive treatment-effect
    heterogeneity in this specification."""
    X = build_design_matrix(df)
    Y = df["INCWAGE_2"].to_numpy(dtype=np.float64)
    T = df["UNEARNED_INCOME_1"].to_numpy(dtype=np.float64)
    weights = df["ASECWT_1"].to_numpy(dtype=np.float64)

    est = LinearDML(
        model_y=_xgb(),
        model_t=_xgb(),
        discrete_treatment=False,
        cv=cv,
        random_state=random_state,
    )
    # econml's weighted-least-squares final stage (StatsModelsLinearRegression)
    # emits benign numpy RuntimeWarnings ("divide by zero"/"overflow" in
    # matmul) on some cv folds when dollar-scale regressors are combined
    # with sample weights; np.linalg.lstsq(rcond=None) resolves them via SVD
    # regardless, and the returned point estimate/CI are finite and stable
    # (checked against unweighted and rescaled reruns). Silenced narrowly so
    # it doesn't bury real signal in this specific expected case.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="econml.*")
        est.fit(Y, T, X=None, W=X.to_numpy(), sample_weight=weights)
    return est


def fit_causal_forest_dml(df: pd.DataFrame, cv: int = 5, random_state: int = 0) -> tuple[CausalForestDML, np.ndarray, pd.DataFrame]:
    """Fit CausalForestDML for *heterogeneous* (personalized) treatment
    effects: beta as a function of X rather than one number. Returns the
    fitted estimator, the per-person CATE array, and the X matrix used."""
    X = build_design_matrix(df)
    Y = df["INCWAGE_2"].to_numpy(dtype=np.float64)
    T = df["UNEARNED_INCOME_1"].to_numpy(dtype=np.float64)
    weights = df["ASECWT_1"].to_numpy(dtype=np.float64)

    est = CausalForestDML(
        model_y=_xgb(),
        model_t=_xgb(),
        discrete_treatment=False,
        cv=cv,
        n_estimators=1000,
        min_samples_leaf=20,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="econml.*")
        est.fit(Y, T, X=X.to_numpy(), W=None, sample_weight=weights)
    cate = est.effect(X.to_numpy())
    return est, cate, X
