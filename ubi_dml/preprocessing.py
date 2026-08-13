"""Preprocessing for the Baby's First Years (ICPSR 37871) cash-transfer RCT.

Design pivot from the prior CPS specification
-----------------------------------------------
The previous version of this module cleaned an *observational* IPUMS CPS
extract, where the "treatment" (unearned/asset income) was something people
selected into based on unobserved wealth -- DML's Neyman-orthogonal
partialling-out controls for the confounders X you give it, but it cannot
control for wealth that was never measured, so beta was contaminated by
omitted-variable bias no matter how flexible the nuisance models were.

Baby's First Years is a randomized cash-transfer trial: mothers were
randomly assigned to a high-cash ($333/month) or low-cash ($20/month) group
at TREATA0. Random assignment means T is independent of X (and of every
unobserved confounder, including wealth) *by construction*, so the omitted-
variable problem that motivated this refactor is gone at the design stage --
DML/XGBoost nuisance models are still used here, but only to soak up
residual outcome variance and sharpen the confidence interval, not to buy
identification the way they had to in the CPS specification. See
dml_pipeline.py for how this changes the treatment nuisance model.

Two files, not one
-------------------
ICPSR 37871 ships baseline and each follow-up wave as *separate* datasets
(DS0001 = baseline/A0, DS0002 = Year-1 follow-up/A1, DS0004 = Year-3/A3,
...), joined by the respondent ID PUBLICSAMPLEID -- there is no single file
containing both TREATA0 and the Year-1 earnings/hours fields. load_raw()
takes the two file paths and inner-joins them.

Column names, corrected against the actual ICPSR 37871 codebook/user guide
(DS0001-Codebook-ICPSR.pdf, DS0002-Codebook-ICPSR.pdf, DS0002-User_guide.pdf)
-------------------------------------------------------------------------------
    TREATA0          treatment arm (1 = high cash, 0 = low cash), DS0001/baseline
    HHMOMEARNEDA1    mother's earned income, Year-1 follow-up, DS0002. This is
                     the *generated* variable (2019$-inflation-adjusted, proper
                     .d/.r/. missing coding, $2,500 floor applied below $5,000
                     per the user guide's "Generated Variables" income section)
                     -- NOT the raw TOTALEARNEDA1, which is a self-reported
                     amount in whatever period unit (year/month/week) the
                     respondent chose and needs TOTALEARNEDUNITA1 to interpret.
                     Per the DS0002 questionnaire (K2_Amt), the recall period
                     is the full previous calendar year ("[prevYear]"), so this
                     is a genuine annual figure -- WEEKS_PER_YEAR=52 below is
                     the correct order of magnitude, not an arbitrary guess.
    MHWEEKTOTALA1    mother's total hours worked per week, all jobs, Year-1
                     follow-up, DS0002 (there is no MOMHOURSRETURNA1 in this
                     extract -- that name doesn't exist in the codebook).
    AGEA0, MRACEA0, MEDLEVELA0, MRELATESTATUSA0, HHCOMBINEDINCOMEA0,
    HHNADULTA0, HHNCHILDA0, SITEA0   baseline (period-T) confounders, DS0001
    PUBLICSAMPLEID   respondent ID shared across every ICPSR 37871 dataset

Known limitation of the hourly-wage construction -- diagnosed, not a bug
----------------------------------------------------------------------------
HOURLY_WAGE = HHMOMEARNEDA1 / (MHWEEKTOTALA1 * 52) divides *last year's total*
earnings by *current* usual weekly hours. Those two are only on the same
basis for a mother who worked the same hours all year. Empirically, in this
extract, median implied wage falls monotonically as current hours rise
(~$18/hr at 1-10 hrs/week down to ~$5/hr at 40+ hrs/week) -- consistent with
many mothers having returned to work only partway through the prior year
(postpartum), so a full year of earnings undercounts what their *current*
hours would suggest, understating the rate for high-hours mothers far more
than it overstates it for a handful of low-hours ones. There is no
weeks-worked-during-the-earnings-period field in this extract to fix this
properly (MTOTALMONTHSWORKA1 is "months worked since the child's birth", not
months worked during [prevYear] -- a different, non-comparable window,
checked and rejected as a substitute). MIN_HOURLY_WAGE=7.25 already does the
heavy lifting here: of 368 mothers with positive earnings and hours, 192
(52%) fall below minimum wage on this ratio and get dropped. That is a real,
large sample-completeness cost, not a bug to trim away further -- report it.
WEEKLY_HOURS bounds below are a secondary safeguard against the (smaller,
still real) upward-bias tail at very low hours and against clear data-entry
outliers (max observed: 128 hours/week). Checked and cleared: the hours- and
wage-band filters themselves exclude employed mothers at nearly identical
rates by arm (~74% cut in both control and treatment), so -- unlike the
non-response issue below -- they are not a source of differential selection.

Differential item non-response -- corrected via IPW, not just documented
----------------------------------------------------------------------------
Mothers do not respond to the Year-1 earnings question at the same rate by
arm: on the baseline-complete-case sample, HHMOMEARNEDA1 is missing for
8.80% of low-cash mothers (52/591) vs. 4.09% of high-cash mothers (16/391)
-- a two-proportion z-test gives z=2.84, p=0.0045. This sits *upstream* of
the extensive-margin (employment) check below: that check can only compare
employment rates among mothers who answered the question, so it cannot
detect this on its own. If non-response correlates with true (unobserved)
wages differently across arms -- plausible, since financially unstable
mothers with informal/irregular income are the ones most likely to find
"how much did you earn last year" hard to answer, and the two arms differ by
design in financial stability -- this is a differential-nonresponse threat
to internal validity that DML's orthogonality cannot fix, because it's a
sample-construction problem upstream of estimation.

The fix implemented here is inverse-probability-of-response weighting -- but
there are *two* separate response events to correct for, not one, because
the wage sample needs two different fields and each is missing on its own
schedule. HHMOMEARNEDA1 (earnings) is missing for a different, and
differently-sized, set of mothers than MHWEEKTOTALA1 (hours): among
baseline-complete earnings-responders, MHWEEKTOTALA1 is additionally missing
for 19.7% (179/907) -- 14.7% of low-cash employed responders vs. 16.9% of
high-cash employed responders. That arm gap isn't significant on its own
(p~0.47 at this sample size), but the same logic that motivates weighting
for earnings non-response applies here too: a mother missing on hours is
just as absent from the wage-analytic sample as one missing on earnings,
and an earlier version of this module only modeled the former, silently
folding the latter into the (already large, separately-documented)
hours/wage-band trim as if it were pure measurement-bound trimming rather
than a second non-response channel.

RESPONDED (this module's original indicator) tracks earnings-question
response alone and is what build_employment_sample()/fit_employment_dml()
need, since the employment analysis only requires HHMOMEARNEDA1.
WAGE_RESPONDED tracks whether *both* HHMOMEARNEDA1 and MHWEEKTOTALA1 are
present -- what clean()/fit_linear_dml()/fit_causal_forest_dml() actually
need, since HOURLY_WAGE can't be constructed without both. Both are
computed in prepare_response_sample() below. dml_pipeline.fit_response_weights()
fits two separate cross-fitted P(RESPONDED=1|X,T) / P(WAGE_RESPONDED=1|X,T)
propensity models on the full baseline-complete sample (responders and
non-responders alike) and turns each into its own inverse-probability-of-
response weight (RESPONSE_WEIGHT, WAGE_RESPONSE_WEIGHT), so responders with
covariate/arm combinations that under-respond -- on either field -- count
for more, standing in for the similar non-responders the analytic sample is
missing. See dml_pipeline.fit_response_weights() for the propensity models
themselves (kept in dml_pipeline.py, not here, to avoid this module
depending on dml_pipeline's XGBoost/design-matrix code -- see that module's
docstring). clean() and build_employment_sample() below both expect to be
called on the *output* of fit_response_weights(), not on
prepare_response_sample()'s output directly; clean() carries
WAGE_RESPONSE_WEIGHT through (for fit_linear_dml()/fit_causal_forest_dml())
and build_employment_sample() carries RESPONSE_WEIGHT through (for
fit_employment_dml()), each passed as sample_weight.

Extensive-margin (employment) selection -- checked, not assumed
--------------------------------------------------------------------
Restricting the wage sample to employed mothers conditions on a
post-treatment outcome (employment), which can bias the wage ATE if
treatment shifts who is employed. Checked directly in this extract, among
mothers who responded to the earnings question: employment rate is 68.09%
in the low-cash arm vs. 68.00% in the high-cash arm -- statistically
indistinguishable. This check is necessary but, per the note above, not
sufficient on its own: it says nothing about the mothers who never answered
the question at all, which is exactly what the response-weighting
correction above is for (both the earnings-response and wage-response
variants -- see above). build_employment_sample() / summarize_employment_by_arm()
below make the employment-balance check reproducible and
dml_pipeline.fit_employment_dml() estimates it formally with (response-
weighted) DML rather than a raw mean-difference. Even combined, these checks
do not rule out finer-grained (e.g. compositional) selection within each
arm, so the wage ATE should still be reported as conditional-on-employment
-and-response, not as an unconditional effect on wages for the full
randomized population.

Missing-value handling
------------------------
DS0001/DS0002 columns come back from pd.read_csv as `object` dtype: blank
cells are literal empty strings (`''`), not NaN, and coexist with numeric
strings and ICPSR's negative sentinel codes for item non-response. A raw
`.astype(float)` throws on the blank strings, so every column that needs to
be numeric is first run through `pd.to_numeric(errors="coerce")` (blanks and
any other non-numeric text become NaN), then the sentinel codes below are
nulled out on top of that -- on *every* numeric field used downstream,
including HHMOMEARNEDA1/MHWEEKTOTALA1 (an earlier version of this module
only nulled sentinels on the baseline confounders; a -888/-999 "don't
know"/"refused" response to the earnings question would then have been
silently treated as "responded with negative earnings" rather than as
non-response, corrupting the very non-response indicator the IPW correction
above depends on). Verified against the DS0002 codebook -- every numeric
field's "(Range of) Missing Values" annotation reads exactly "-999, -888, ."
(Refused / Don't know / System missing); there is no positive-sentinel (e.g.
9999-style) convention anywhere in this extract, so MISSING_SENTINELS below
is the complete set, not a guess.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ID_COL = "PUBLICSAMPLEID"

TREATMENT_COL = "TREATA0"          # 1 = high-cash arm, 0 = low-cash arm
INCWAGE_COL = "HHMOMEARNEDA1"      # mother's generated earned income, Year-1 follow-up ($/yr)
HOURS_COL = "MHWEEKTOTALA1"        # mother's total hours/week, all jobs, Year-1 follow-up
EMPLOYED_COL = "EMPLOYED"          # 1 = positive Year-1 earnings reported, 0 = not
RESPONDED_COL = "RESPONDED"        # 1 = HHMOMEARNEDA1 answered (not missing/refused/don't know)
RESPONSE_WEIGHT_COL = "RESPONSE_WEIGHT"  # 1 / P(RESPONDED=1 | X, T), Hajek-normalized
WAGE_RESPONDED_COL = "WAGE_RESPONDED"    # 1 = BOTH HHMOMEARNEDA1 and MHWEEKTOTALA1 answered
WAGE_RESPONSE_WEIGHT_COL = "WAGE_RESPONSE_WEIGHT"  # 1 / P(WAGE_RESPONDED=1 | X, T), Hajek-normalized

# See "Column names" note above -- HHMOMEARNEDA1's recall period is the full
# prior calendar year, so 52 is the correct annualization factor, not a
# fixed to work around a missing field.
WEEKS_PER_YEAR = 52

CONFOUNDERS_BASELINE = [
    "AGEA0",                # mother's age
    "MRACEA0",              # mother's race/ethnicity
    "MEDLEVELA0",           # mother's education level
    "MRELATESTATUSA0",      # marital/cohabitation status
    "HHCOMBINEDINCOMEA0",   # baseline household income
    "HHNADULTA0",           # number of household adults
    "HHNCHILDA0",           # number of household children
    "SITEA0",               # study site (LA, MN, NE, NY)
]

# See "Missing-value handling" note above -- the complete set per the DS0002 codebook.
MISSING_SENTINELS = [-999, -888]

MIN_HOURLY_WAGE = 7.25
MAX_HOURLY_WAGE = 250.00

# See "Known limitation" note above: secondary safeguard, not the primary
# filter (the wage-rate band above does most of the exclusion). Floor of 10
# damps the low-hours upward-bias tail seen empirically (median implied wage
# ~$18/hr under 10 hrs/week vs. $6-9/hr elsewhere); ceiling of 80 rules out
# the physically-implausible reports (max observed: 128 hrs/week) while
# still allowing genuine multiple-job mothers through.
MIN_WEEKLY_HOURS = 10
MAX_WEEKLY_HOURS = 80


def _read_delimited(path: str) -> pd.DataFrame:
    """Load one ICPSR 37871 dataset file. Supports the formats ICPSR ships
    restricted-use files in: Stata (.dta) and delimited text (.csv/.tsv/.tab).
    low_memory=False since these files mix numeric and blank-string values
    within a column, which otherwise triggers pandas' chunked dtype-guessing
    and a DtypeWarning."""
    if path.endswith(".dta"):
        return pd.read_stata(path)
    if path.endswith((".csv", ".tab", ".tsv")):
        sep = "\t" if path.endswith((".tab", ".tsv")) else ","
        return pd.read_csv(path, sep=sep, low_memory=False)
    raise ValueError(f"Unsupported file extension for {path!r}; expected .dta, .csv, .tsv, or .tab")


def load_raw(baseline_path: str, followup_path: str) -> pd.DataFrame:
    """Load the DS0001 baseline file and the DS0002 Year-1 follow-up file and
    inner-join them on PUBLICSAMPLEID. Only the follow-up columns this
    pipeline needs are pulled in, so a follow-up file with hundreds of extra
    survey columns doesn't bloat the merge."""
    baseline = _read_delimited(baseline_path)
    followup = _read_delimited(followup_path)
    followup_cols = [ID_COL, INCWAGE_COL, HOURS_COL]
    return baseline.merge(followup[followup_cols], on=ID_COL, how="inner")


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert object-dtype columns (blank strings mixed with numeric
    strings) to float, turning blanks/non-numeric text into NaN rather than
    raising -- see "Missing-value handling" module docstring note."""
    df = df.copy()
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _null_out_missing_sentinels(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        df.loc[df[col].isin(MISSING_SENTINELS), col] = np.nan
    return df


def encode_treatment(df: pd.DataFrame) -> pd.DataFrame:
    """Encode TREATA0 as a clean 0/1 int8 indicator, failing loudly if the
    column contains anything other than the two expected arm codes."""
    df = df.copy()
    unique_vals = set(df[TREATMENT_COL].dropna().unique())
    if not unique_vals <= {0, 1}:
        raise ValueError(
            f"{TREATMENT_COL} must be binary 0/1 (1=high cash, 0=low cash); "
            f"found {sorted(unique_vals)}"
        )
    df[TREATMENT_COL] = df[TREATMENT_COL].astype(np.int8)
    return df


def prepare_response_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline-complete-case sample including EVERY mother regardless of
    whether she answered the Year-1 earnings and/or hours questions --
    RESPONDED=0 / WAGE_RESPONDED=0 rows included, unlike clean()'s or
    build_employment_sample()'s output. This is what
    dml_pipeline.fit_response_weights() needs to fit its two response-
    propensity models (P(RESPONDED|X,T) and P(WAGE_RESPONDED|X,T)) on, since
    each has to see both its responders and its non-responders. Coerces
    numeric types, nulls ICPSR sentinel codes on every numeric field (see
    "Missing-value handling" docstring note), drops incomplete baseline
    cases, and encodes treatment.

    RESPONDED tracks earnings-question (HHMOMEARNEDA1) response alone --
    what build_employment_sample() needs, since employment only requires
    earnings. WAGE_RESPONDED additionally requires MHWEEKTOTALA1 (hours) --
    what clean() needs, since HOURLY_WAGE can't be constructed without both
    fields. See "Differential item non-response" module docstring note for
    why these are two separate indicators rather than one."""
    numeric_cols = [TREATMENT_COL, INCWAGE_COL, HOURS_COL] + CONFOUNDERS_BASELINE
    df = _coerce_numeric(df, numeric_cols)
    df = _null_out_missing_sentinels(df, [INCWAGE_COL, HOURS_COL] + CONFOUNDERS_BASELINE)

    required = [TREATMENT_COL] + CONFOUNDERS_BASELINE
    df = df.dropna(subset=required)
    df = encode_treatment(df)

    df[RESPONDED_COL] = df[INCWAGE_COL].notna()
    df[WAGE_RESPONDED_COL] = df[RESPONDED_COL] & df[HOURS_COL].notna()
    return df.reset_index(drop=True)


def attach_response_weights(
    df: pd.DataFrame,
    propensities: np.ndarray,
    responded_col: str = RESPONDED_COL,
    weight_col: str = RESPONSE_WEIGHT_COL,
) -> pd.DataFrame:
    """Add weight_col = 1 / P(responded_col=1 | X, T) to df, Hajek-normalized
    (rescaled to mean 1 across responded_col==True rows) to reduce variance
    relative to raw Horvitz-Thompson weights -- a constant overall rescaling
    doesn't change what a weighted DML fit estimates, but it keeps the
    weights interpretable and keeps the *effective* sample size close to the
    responder count rather than being inflated or deflated by an arbitrary
    scale. `propensities` must align 1:1 with df's row order (e.g. from
    dml_pipeline.fit_response_weights()'s cross-fitted classifier).
    responded_col/weight_col default to the earnings-response pair
    (RESPONDED/RESPONSE_WEIGHT); pass WAGE_RESPONDED_COL/
    WAGE_RESPONSE_WEIGHT_COL to attach the wage-response weight instead --
    see "Differential item non-response" module docstring note."""
    df = df.copy()
    propensities = np.asarray(propensities, dtype=np.float64)
    raw_weight = 1.0 / propensities

    responded = df[responded_col].to_numpy(dtype=bool)
    raw_weight = raw_weight / raw_weight[responded].mean()

    df[weight_col] = raw_weight
    return df


def construct_hourly_wage(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to mothers employed in period T+1 (positive earnings, and
    hours within [MIN_WEEKLY_HOURS, MAX_WEEKLY_HOURS] -- see "Known
    limitation" module docstring note for why hours are bounded, not just
    required positive) and construct the hourly wage rate."""
    df = df.copy()
    employed = (
        (df[INCWAGE_COL] > 0)
        & df[HOURS_COL].between(MIN_WEEKLY_HOURS, MAX_WEEKLY_HOURS)
    )
    df = df[employed].copy()
    df["HOURLY_WAGE"] = df[INCWAGE_COL] / (df[HOURS_COL] * WEEKS_PER_YEAR)
    return df


def trim_hourly_wage(
    df: pd.DataFrame, lower: float = MIN_HOURLY_WAGE, upper: float = MAX_HOURLY_WAGE
) -> pd.DataFrame:
    """Drop implausible hourly rates outside [lower, upper]. In this extract
    this is the dominant filter (it removes the annual-earnings/current-
    hours mismatch cases described in the module docstring), not a minor
    outlier trim -- see summarize_wage_by_arm() to inspect what it's
    dropping before trusting the result."""
    return df[df["HOURLY_WAGE"].between(lower, upper)].copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full wage-analytic-sample construction. Expects df to already carry
    WAGE_RESPONDED/WAGE_RESPONSE_WEIGHT -- i.e. the output of
    dml_pipeline.fit_response_weights(prepare_response_sample(raw)), not raw
    survey data. Restricts to wage-responders (both earnings and hours
    answered, weighted, not just the raw complete cases -- see "Differential
    item non-response" docstring note), then to T+1 employed workers within
    plausible hours/wage bounds, and takes logs. WAGE_RESPONSE_WEIGHT is
    carried through into the output so fit_linear_dml()/
    fit_causal_forest_dml() can pass it as sample_weight."""
    df = df[df[WAGE_RESPONDED_COL]].copy()
    df = construct_hourly_wage(df)
    df = trim_hourly_wage(df)

    df["LOG_HOURLY_WAGE"] = np.log(df["HOURLY_WAGE"])

    keep_cols = ([TREATMENT_COL, "HOURLY_WAGE", "LOG_HOURLY_WAGE", WAGE_RESPONSE_WEIGHT_COL]
                 + CONFOUNDERS_BASELINE)
    return df[keep_cols].reset_index(drop=True)


def build_employment_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Build the extensive-margin analytic sample: every *responding*
    mother, not just the ones with positive wages -- this is the sample the
    post-treatment-selection check in dml_pipeline.fit_employment_dml()
    needs, since it must include the mothers who weren't employed
    (EMPLOYED=0) that the wage sample above deliberately excludes. Expects
    the same response-weighted input as clean() -- see that function's
    docstring."""
    df = df[df[RESPONDED_COL]].copy()
    df[EMPLOYED_COL] = (df[INCWAGE_COL] > 0).astype(np.int8)

    keep_cols = [TREATMENT_COL, EMPLOYED_COL, RESPONSE_WEIGHT_COL] + CONFOUNDERS_BASELINE
    return df[keep_cols].reset_index(drop=True)


def summarize_response_by_arm(df: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic: Year-1 earnings-question response rate by treatment arm
    on prepare_response_sample()'s output (before restricting to
    responders). This is the direct evidence for the differential-item-non-
    response finding that motivates the IPW correction -- see module
    docstring."""
    summary = df.groupby(TREATMENT_COL)[RESPONDED_COL].agg(["mean", "sum", "count"])
    print("\n=== Year-1 earnings-question RESPONDED rate by treatment arm ===")
    print(summary)
    return summary


def summarize_wage_response_by_arm(df: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic: WAGE_RESPONDED (both earnings AND hours answered) rate by
    treatment arm on prepare_response_sample()'s output. Distinct from
    summarize_response_by_arm() above -- that one only tracks the earnings
    question; this one tracks the additional attrition from the hours
    question (MHWEEKTOTALA1) that clean()'s wage sample also depends on but
    the earnings-only indicator misses. See "Differential item non-response"
    module docstring note."""
    summary = df.groupby(TREATMENT_COL)[WAGE_RESPONDED_COL].agg(["mean", "sum", "count"])
    print("\n=== Year-1 earnings-AND-hours WAGE_RESPONDED rate by treatment arm ===")
    print(summary)
    return summary


def summarize_wage_by_arm(df: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic: HOURLY_WAGE distribution by treatment arm on the final
    wage-analytic sample (post-trim), unweighted, plus the response-weighted
    mean for comparison. Run this before fitting DML -- if the two arms'
    distributions look implausible in ways beta alone wouldn't reveal (e.g.
    one arm's IQR sitting entirely at the wage floor), that's a sign the
    sample restriction itself, not the treatment effect, is doing the
    work."""
    summary = df.groupby(TREATMENT_COL)["HOURLY_WAGE"].describe()
    print("\n=== HOURLY_WAGE by treatment arm (wage-analytic sample, unweighted) ===")
    print(summary)

    weighted_mean = df.groupby(TREATMENT_COL).apply(
        lambda g: np.average(g["HOURLY_WAGE"], weights=g[WAGE_RESPONSE_WEIGHT_COL])
    )
    print("\nResponse-weighted mean HOURLY_WAGE by arm:")
    print(weighted_mean)
    return summary


def summarize_employment_by_arm(df: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic: employment rate by treatment arm on the full (not wage-
    restricted) employment-analytic sample from build_employment_sample().
    This is the direct evidence for or against extensive-margin selection
    into the wage sample -- see "Extensive-margin" module docstring note."""
    summary = df.groupby(TREATMENT_COL)[EMPLOYED_COL].agg(["mean", "sum", "count"])
    print("\n=== EMPLOYED rate by treatment arm (employment-analytic sample) ===")
    print(summary)
    return summary
