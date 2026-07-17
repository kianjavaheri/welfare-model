"""Generic long-format -> T/T+1 longitudinal merge on CPSIDP.

cps_00001.dat does NOT need this function: it was pulled from IPUMS CPS's
own longitudinal-linking extract builder, which already ships one row per
matched person with every variable duplicated as `<VAR>_1` (year T) /
`<VAR>_2` (year T+1) -- see preprocessing.py. This module exists for the
more general case the task brief describes: a *long*-format CPS ASEC
extract (one row per person-year, stacked across survey years) that has not
been pre-linked, which is what you get from a standard (non-longitudinal)
IPUMS CPS extract spanning multiple ASEC samples.

CPSIDP uniquely identifies a person within the CPS's 4-8-4 rotation pattern,
but a naive merge on CPSIDP alone can still stitch together two different
people if a household's line number was reassigned between waves. IPUMS's
own guidance is to sanity-check candidate matches on fields that shouldn't
change between adjacent ASEC waves (sex; age advancing by ~1 year). We apply
that check here rather than trusting CPSIDP in isolation.
"""
from __future__ import annotations

import pandas as pd


def build_longitudinal_panel(
    df_long: pd.DataFrame,
    id_col: str = "CPSIDP",
    year_col: str = "YEAR",
    year_gap: int = 1,
    validate_sex_col: str | None = "SEX",
    validate_age_col: str | None = "AGE",
    max_age_drift: int = 2,
) -> pd.DataFrame:
    """Merge a long-format (one row per person-year) panel into T -> T+1
    pairs on `id_col`, suffixing columns `_1` (year T) and `_2` (year T+1).

    Only consecutive-year pairs (YEAR_2 == YEAR_1 + year_gap) are kept, so
    there is no way for a "future" row to leak into a "past" row's features
    (each person contributes at most one T -> T+1 pair per starting year).
    """
    left = df_long.rename(columns={c: f"{c}_1" for c in df_long.columns if c != id_col})
    right = df_long.rename(columns={c: f"{c}_2" for c in df_long.columns if c != id_col})

    merged = left.merge(right, on=id_col, how="inner")
    merged = merged[merged[f"{year_col}_2"] == merged[f"{year_col}_1"] + year_gap]

    if validate_sex_col:
        merged = merged[merged[f"{validate_sex_col}_1"] == merged[f"{validate_sex_col}_2"]]

    if validate_age_col:
        age_delta = merged[f"{validate_age_col}_2"] - merged[f"{validate_age_col}_1"]
        merged = merged[age_delta.between(0, max_age_drift)]

    return merged.reset_index(drop=True)
