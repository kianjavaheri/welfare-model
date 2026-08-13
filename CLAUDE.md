# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo does

Estimates the causal effect of Baby's First Years (BFY, ICPSR 37871)'s unconditional cash
transfer on maternal log hourly wages via Double Machine Learning (`LinearDML` for the ATE,
`CausalForestDML` for heterogeneous per-mother effects).

This project originally estimated the effect of unearned/asset income on future wages using
*observational* IPUMS CPS ASEC data (`ddi_parser.py`, `panel.py`). That design was pivoted away
from because its positive coefficient reflected omitted-variable bias from unobserved wealth —
CPS collects no net worth data, so DML's orthogonality could partial out the confounders it was
given but not the wealth it was never given. `ddi_parser.py` and `panel.py` are unused leftovers
from that era; nothing in the current pipeline imports them.

## Commands

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install libomp   # xgboost needs the OpenMP runtime on macOS

# Run the full pipeline
python run_pipeline.py \
    --baseline Data/ICPSR_37871/DS0001/37871-0001-Data.tsv \
    --followup Data/ICPSR_37871/DS0002/37871-0002-Data.tsv

# Useful flags: --cv (cross-fitting folds, default 5), --random-state,
# --cate-hist-out (histogram path), --skip-causal-forest (skip the slower forest fit)
```

There is no test suite, linter, or build step in this repo — `run_pipeline.py` end-to-end against
real data is the only current way to verify a change. A full run (without `--skip-causal-forest`)
writes `cate_histogram.png` to the repo root; it's a regenerable output artifact, not something to
commit or treat as source.

This repo is not a git repository (no `.git`). `Data/ICPSR_37871/` holds restricted-use ICPSR
microdata (BFY) and is **not currently gitignored** — `.gitignore` only excludes the old
`cps_*.dat/.xml/.csv` extracts from the CPS-era pipeline. Add a `Data/` entry before ever
`git init`-ing this repo, since ICPSR's terms of use prohibit redistributing this data.

## Architecture

### Pipeline order matters — don't call `clean()`/`build_employment_sample()` on raw data

The pipeline is a strict sequence, not independent utility functions:

```
load_raw(baseline, followup)              # ubi_dml/preprocessing.py — inner-joins DS0001 + DS0002 on PUBLICSAMPLEID
  -> prepare_response_sample(df)           # baseline-complete-case sample, INCLUDING non-responders to the earnings and/or hours questions
  -> fit_response_weights(df, cv, seed)    # ubi_dml/dml_pipeline.py — two cross-fitted XGBClassifiers, P(RESPONDED|X,T) and P(WAGE_RESPONDED|X,T) -> IPW weights
  -> clean(weighted)                       # wage-analytic sample: employed + plausible hours/wage bounds + log
  -> build_employment_sample(weighted)     # extensive-margin sample: every earnings-responder, EMPLOYED=0 included
```

`clean()` and `build_employment_sample()` both *expect* their input to already carry the response
columns from `fit_response_weights()` — `clean()` filters on `WAGE_RESPONDED`/carries
`WAGE_RESPONSE_WEIGHT` (needs both earnings and hours), `build_employment_sample()` filters on
`RESPONDED`/carries `RESPONSE_WEIGHT` (needs earnings only) — rather than doing their own
missingness handling. Calling them directly on `load_raw()`'s output will raise a `KeyError`. See
`run_pipeline.py` for the canonical call order.

### Two-file data model

ICPSR 37871 ships each wave as a separate dataset file (DS0001 = baseline/`A0`, DS0002 = Year-1
follow-up/`A1`, DS0004 = Year-3/`A3`, ...) joined on `PUBLICSAMPLEID`. There is no single file
containing both `TREATA0` and the Year-1 earnings/hours fields — `load_raw()` requires both paths.

### Column semantics (hard-won from the actual ICPSR codebook/user guide, not guessable from names)

- `TREATA0` — binary RCT arm (1 = high-cash \$333/mo, 0 = low-cash \$20/mo), DS0001.
- `HHMOMEARNEDA1` — mother's *generated* (inflation-adjusted, `$2,500` floor below `$5,000`,
  proper missing coding) earned income for the **entire prior calendar year**, DS0002. Not the
  raw `TOTALEARNEDA1`, which is in whatever period unit the respondent chose.
- `MHWEEKTOTALA1` — mother's usual hours/week across all jobs **at the time of the Year-1
  interview**, DS0002. There is no `MOMHOURSRETURNA1` in this extract.
- `HOURLY_WAGE = HHMOMEARNEDA1 / (MHWEEKTOTALA1 * 52)` — see "Known limitations" below for why
  this ratio is noisier than it looks.
- `CONFOUNDERS_BASELINE` (in `preprocessing.py`): `AGEA0, MRACEA0, MEDLEVELA0, MRELATESTATUSA0,
  HHCOMBINEDINCOMEA0, HHNADULTA0, HHNCHILDA0, SITEA0` — all DS0001/baseline, never post-treatment.
  `dml_pipeline.py` splits this same list into `NUMERIC_CONFOUNDERS`/`CATEGORICAL_CONFOUNDERS` (for
  passthrough vs. one-hot encoding in `build_design_matrix()`), and asserts at import time that the
  two partitions union back to exactly `CONFOUNDERS_BASELINE` — adding/removing a confounder means
  updating both files, or the assert fails on import.
- ICPSR sentinel codes for missing data are `-999`/`-888` (Refused/Don't know) — a fixed,
  verified-against-codebook set (`MISSING_SENTINELS`), not a generic missing-value guess. Applied
  to *every* numeric field including the outcome fields, not just the confounders.

### Module responsibilities

- **`ubi_dml/preprocessing.py`** — loading, missing-code/sentinel cleaning, sample construction
  (response/wage/employment samples), and print-based diagnostics (`summarize_*_by_arm`). Doesn't
  import `dml_pipeline` (would be circular); the response-*propensity model itself* lives in
  `dml_pipeline.py` even though the *sample construction* around it lives here.
- **`ubi_dml/dml_pipeline.py`** — all XGBoost/econml modeling: `build_design_matrix()` (one-hot
  encodes nominal confounders), the two response-propensity models (`fit_response_weights`, one
  for earnings response and one for earnings-and-hours response), and the three DML fits
  (`fit_linear_dml`, `fit_causal_forest_dml`, `fit_employment_dml`). `T` is a
  binary RCT arm, so every DML estimator uses `discrete_treatment=True` with an `XGBClassifier`
  propensity model for `E[T|X]` (not an `XGBRegressor` — that was only correct for the old
  continuous-dollar CPS treatment).
- **`ubi_dml/policy_simulation.py`** — formats an `ATEResult` into a percentage-wage-impact
  report (`(exp(beta) - 1) * 100`). Thin on purpose: since `T` is a binary arm rather than a
  continuous dollar amount, there's no dollar-shock rescaling to do (unlike the old CPS
  specification's policy simulation).
- **`run_pipeline.py`** — CLI orchestrating the full sequence above and printing every diagnostic.

### Known limitations baked into the design (don't "simplify" these away)

1. **Earnings/hours measurement mismatch**: `HHMOMEARNEDA1` is a full prior-year recall;
   `MHWEEKTOTALA1` is point-in-time. For mothers who returned to work partway through the recall
   year (common in a postpartum sample), this understates the true hourly rate. There's no
   weeks-worked-during-the-recall-window field to fix this directly (`MTOTALMONTHSWORKA1` is
   months worked *since the child's birth* — a different, non-comparable window; already checked
   and rejected as a substitute). `MIN_HOURLY_WAGE`/`MAX_HOURLY_WAGE` (`$7.25`–`$250`) and
   `MIN_WEEKLY_HOURS`/`MAX_WEEKLY_HOURS` (`10`–`80`) bounds absorb the resulting noise rather than
   eliminate it — the wage floor alone drops roughly half of otherwise-employed respondents. This
   is documented, checked-for-differential-arm-selection (and cleared), and intentional.
2. **Differential item non-response**: mothers answer the earnings question at significantly
   different rates by arm (~9.8% missing low-cash vs. ~4.3% high-cash, p=0.0045) — a threat to
   internal validity that sits upstream of, and isn't visible to, the employment-balance check
   alone. This isn't the only response event the pipeline has to correct for: constructing
   `HOURLY_WAGE` also needs the hours question (`MHWEEKTOTALA1`), which has its own, separate
   non-response (~19.7% missing among earnings-responders) that an earnings-only correction can't
   see or fix. `fit_response_weights()` fits two independent cross-fitted `XGBClassifier`s —
   `P(RESPONDED|X,T)` for earnings and `P(WAGE_RESPONDED|X,T)` for earnings-and-hours — and
   `attach_response_weights()` (Hájek-normalized) turns each into its own weight
   (`RESPONSE_WEIGHT`, `WAGE_RESPONSE_WEIGHT`); `build_employment_sample()` uses the former,
   `clean()` uses the latter. Corrected via inverse-probability-of-response weighting on both
   fields, not by ignoring either or dropping the finding.
3. **Extensive-margin (employment) selection**: restricting the wage sample to employed mothers
   conditions on a post-treatment outcome. Checked via `fit_employment_dml()` against the
   unrestricted (`build_employment_sample()`) sample — currently balanced across arms, but the
   wage ATE should still be reported as conditional-on-employment-and-response, not as an
   unconditional effect.
4. **Small-sample causal forest**: `CausalForestDML`'s `min_samples_leaf` defaults to `5`, not
   econml's typical `20` — at this sample size (~150-200 mothers), `min_samples_leaf=20` produces
   a degenerate forest where every tree fails to split and every person gets an identical CATE.

Full narrative/derivation for all of the above (including the ICPSR codebook citations) lives in
the module docstrings in `preprocessing.py` and `dml_pipeline.py`, and in `README.md`.
