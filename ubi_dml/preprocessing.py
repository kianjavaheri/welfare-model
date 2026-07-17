"""Load and clean the linked IPUMS CPS ASEC extract.

cps_00001.dat is an IPUMS *CPS linked/matched* extract: each row already
represents one person (CPSIDP) observed in two adjacent ASEC years, with
every variable duplicated as `<VAR>_1` (year T, the baseline year) and
`<VAR>_2` (year T+1, the follow-up year). IPUMS performs the CPSIDP match
itself, so there is no long-format, stack-by-year merge to do for this
extract (see panel.py for a generic merge function that handles the
long-format case, for extracts that are *not* pre-linked).

Missing-value handling note
----------------------------
The task brief assumed a single generic missing code (999999). The actual
IPUMS convention -- confirmed against both the DDI codebook and the raw
data -- is "all 9s spanning the field's declared width", which differs by
variable: INCWAGE/INCRETIR are 8-digit fields (missing = 99999999), while
INCINT/INCDIVID/INCRENT/INCOTHER are 7-digit fields (missing = 9999999).
Hardcoding 999999 would silently fail to null out ~19% of INCWAGE_1 values
in this extract. We instead derive the missing code from each variable's
width in the DDI, so it's correct regardless of which extract is loaded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ubi_dml.ddi_parser import VarSpec, colspecs_for_fwf, decimals_map, names_for_fwf, parse_ddi

# Continuous income fields that use the IPUMS "all 9s of field width" missing
# convention. We only need the _1 (baseline year) values for the unearned
# income treatment and the _2 (follow-up year) value for the wage outcome,
# but we clean both suffixes for anyone reusing this module differently.
INCOME_VARS = [
    "INCWAGE_1", "INCWAGE_2",
    "INCINT_1", "INCINT_2",
    "INCDIVID_1", "INCDIVID_2",
    "INCRENT_1", "INCRENT_2",
    "INCRETIR_1", "INCRETIR_2",
    "INCOTHER_1", "INCOTHER_2",
]

# Categorical fields with an explicit NIU/missing/blank numeric code (per
# the DDI <catgry> labels), distinct from legitimate substantive categories
# such as EDUC=000 ("no schooling") or IND=0000 ("NIU, not employed") which
# we deliberately keep as real category values rather than nulling out.
CATEGORICAL_MISSING_CODES = {
    "SEX_1": [9], "SEX_2": [9],
    "MARST_1": [9], "MARST_2": [9],
    "RACE_1": [999], "RACE_2": [999],
    "METRO_1": [9], "METRO_2": [9],
    "EDUC_1": [999], "EDUC_2": [999],
}

TREATMENT_COMPONENTS_YEAR1 = ["INCINT_1", "INCDIVID_1", "INCRENT_1", "INCRETIR_1", "INCOTHER_1"]

CONFOUNDERS_YEAR1 = [
    "AGE_1", "SEX_1", "RACE_1", "MARST_1", "EDUC_1",
    "NUMPREC_1", "NCHILD_1", "METRO_1", "IND_1",
]

# Working-age restriction. CPS ASEC only asks income questions of people
# aged 15+; below that, INCWAGE/EDUC are NIU by survey design (confirmed in
# this extract: EDUC_1 == 001 "NIU or blank" and INCWAGE_1 == 99999999 both
# occur exactly 27,060 times -- the same underlying under-15 respondents).
# We use 18-64 specifically (not 15+) to keep the analytic population to
# working-age adults, so the labor-supply outcome isn't mechanically driven
# by full-time schooling (16-17) or Social Security retirement (65+), both
# of which operate through channels other than an unearned-income effect.
MIN_WORKING_AGE = 18
MAX_WORKING_AGE = 64


def load_raw(dat_path: str, xml_path: str) -> tuple[pd.DataFrame, list[VarSpec]]:
    """Read the fixed-width .dat file using column positions from the DDI."""
    specs = parse_ddi(xml_path)
    df = pd.read_fwf(
        dat_path,
        colspecs=colspecs_for_fwf(specs),
        names=names_for_fwf(specs),
        header=None,
        dtype="int64",
    )
    # Apply implied decimals (e.g. ASECWT_1 has dcml=4 -> divide by 10,000).
    for name, dcml in decimals_map(specs).items():
        df[name] = df[name] / (10 ** dcml)
    return df, specs


def _missing_code_for_width(width: int) -> int:
    return int("9" * width)


def clean(df: pd.DataFrame, specs: list[VarSpec]) -> pd.DataFrame:
    """Apply IPUMS missing-code recoding, sample restriction, and treatment
    construction. Returns one row per (CPSIDP, year-pair) analytic unit."""
    df = df.copy()
    width_by_name = {s.name: s.width for s in specs}

    # --- 1. Null out IPUMS missing/NIU codes -----------------------------
    # Columns start as int64; cast to float64 first so NaN can be assigned.
    for var in INCOME_VARS:
        if var not in df.columns:
            continue
        missing_code = _missing_code_for_width(width_by_name[var])
        df[var] = df[var].astype(float)
        df.loc[df[var] == missing_code, var] = np.nan

    for var, codes in CATEGORICAL_MISSING_CODES.items():
        if var not in df.columns:
            continue
        df[var] = df[var].astype(float)
        df.loc[df[var].isin(codes), var] = np.nan

    # --- 2. Restrict to the working-age analytic sample (baseline year) --
    df = df[df["AGE_1"].between(MIN_WORKING_AGE, MAX_WORKING_AGE)]

    # --- 3. Construct the treatment: total unearned income in year T -----
    # Missing components are treated as $0 contribution to the sum (a
    # person can validly have no interest income while still reporting
    # rent income, etc.) rather than nulling the whole aggregate.
    df[TREATMENT_COMPONENTS_YEAR1] = df[TREATMENT_COMPONENTS_YEAR1].fillna(0)
    df["UNEARNED_INCOME_1"] = df[TREATMENT_COMPONENTS_YEAR1].sum(axis=1)

    # --- 4. Drop residual missingness on the outcome and confounders -----
    # (complete-case / listwise deletion; after the age restriction this is
    # a small share of rows and represents genuine item nonresponse rather
    # than a structural survey-universe exclusion.)
    required = ["INCWAGE_2", "ASECWT_1"] + CONFOUNDERS_YEAR1
    df = df.dropna(subset=required)

    # INCWAGE_2 == NaN after step 1 already isolated true missingness;
    # genuine zero earners keep their reported 0, so no fillna(0) here.
    df["INCWAGE_2"] = df["INCWAGE_2"].astype(float)

    keep_cols = [
        "CPSIDP", "YEAR_1", "YEAR_2", "ASECWT_1",
        "INCWAGE_2", "UNEARNED_INCOME_1",
    ] + CONFOUNDERS_YEAR1
    return df[keep_cols].reset_index(drop=True)


def load_and_clean(dat_path: str, xml_path: str) -> pd.DataFrame:
    df, specs = load_raw(dat_path, xml_path)
    return clean(df, specs)
