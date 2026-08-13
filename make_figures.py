"""Generate report-ready figures for the BFY cash-transfer wage analysis.

Runs the full pipeline (identical to run_pipeline.py's sequence) and writes
six PNGs to figures/, one per figure referenced in the write-up. Re-run
whenever the underlying data or pipeline code changes so the figures stay in
sync with the reported numbers.

Usage:
    python make_figures.py \
        --baseline Data/ICPSR_37871/DS0001/37871-0001-Data.tsv \
        --followup Data/ICPSR_37871/DS0002/37871-0002-Data.tsv
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from ubi_dml import dml_pipeline as dp
from ubi_dml import preprocessing as pp

# ---------------------------------------------------------------------------
# Style. Arm identity (low-cash / high-cash) is color-consistent across every
# figure that compares arms; figures that compare estimators or show a single
# distribution use a neutral single hue instead, so color never means two
# different things in the same report. Palette is the dataviz-skill default:
# categorical slots 1 (blue) and 2 (orange), an adjacent pair validated for
# colorblind-safety and light-surface contrast.
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"       # low-cash arm ($20/mo)
ORANGE = "#eb6834"     # high-cash arm ($333/mo)
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

# Ordinal blue ramp (light -> dark) for the single-series funnel, where color
# tracks stage order rather than a second category.
FUNNEL_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
})


def _clean_ax(ax, hide_left=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["left"].set_visible(not hide_left)
    if not hide_left:
        ax.spines["left"].set_color(BASELINE)
    ax.tick_params(length=0)


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Figure 1 -- sample construction funnel
# ---------------------------------------------------------------------------
def fig_sample_funnel(stages: list[tuple[str, int]]):
    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    colors = FUNNEL_RAMP[: len(stages)]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    y = np.arange(len(stages))[::-1]
    ax.barh(y, values, color=colors, height=0.62, zorder=3)

    for yi, v in zip(y, values):
        pct = 100 * v / values[0]
        ax.text(v + max(values) * 0.025, yi, f"{v:,}  ({pct:.0f}%)",
                va="center", ha="left", fontsize=10, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlim(0, max(values) * 1.22)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("Mothers (n); % of the randomized total in parentheses", color=MUTED, fontsize=10)
    ax.grid(axis="x", zorder=0)
    _clean_ax(ax, hide_left=True)
    ax.set_title("Figure 1. Sample construction: from randomization to\nthe wage-analytic sample",
                 loc="left", fontsize=12.5, color=INK, pad=14)
    _save(fig, "fig1_sample_funnel.png")


# ---------------------------------------------------------------------------
# Figure 2 -- item response rates by arm
# ---------------------------------------------------------------------------
def fig_response_rates(earnings_low, earnings_high, wage_low, wage_high):
    groups = ["Answered earnings\nquestion", "Answered earnings\n& hours questions"]
    low_vals = [earnings_low, wage_low]
    high_vals = [earnings_high, wage_high]

    x = np.arange(len(groups))
    w = 0.32
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    bars_low = ax.bar(x - w / 2, low_vals, width=w, color=BLUE, zorder=3, label="Low-cash arm ($20/mo)")
    bars_high = ax.bar(x + w / 2, high_vals, width=w, color=ORANGE, zorder=3, label="High-cash arm ($333/mo)")

    for bars in (bars_low, bars_high):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 1.5, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=10, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10.5)
    ax.set_ylabel("Response rate (%)", color=MUTED, fontsize=10)
    ax.set_ylim(0, 112)
    ax.grid(axis="y", zorder=0)
    _clean_ax(ax)
    ax.legend(frameon=False, loc="lower left", fontsize=9.5)
    ax.set_title("Figure 2. Item response rates by treatment arm", loc="left", fontsize=12.5, color=INK, pad=14)
    _save(fig, "fig2_response_rates.png")


# ---------------------------------------------------------------------------
# Figure 3 -- hourly wage distribution by arm
# ---------------------------------------------------------------------------
def fig_wage_distribution(low, high, mean_low, mean_high):
    combined = np.concatenate([low.to_numpy(), high.to_numpy()])
    display_cap = min(combined.max(), np.percentile(combined, 96) * 1.25)
    show_low = low[low <= display_cap]
    show_high = high[high <= display_cap]
    n_excluded = (len(low) - len(show_low)) + (len(high) - len(show_high))

    bins = np.linspace(pp.MIN_HOURLY_WAGE, display_cap, 20)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.hist(show_low, bins=bins, color=BLUE, alpha=0.55, zorder=3, label="Low-cash arm ($20/mo)")
    ax.hist(show_high, bins=bins, color=ORANGE, alpha=0.55, zorder=3, label="High-cash arm ($333/mo)")
    ax.axvline(mean_low, color=BLUE, lw=2, linestyle="--", zorder=4)
    ax.axvline(mean_high, color=ORANGE, lw=2, linestyle="--", zorder=4)

    ax.set_xlabel("Hourly wage ($)", color=MUTED, fontsize=10)
    ax.set_ylabel("Number of mothers", color=MUTED, fontsize=10)
    ax.grid(axis="y", zorder=0)
    _clean_ax(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    if n_excluded:
        ax.text(0.99, 0.70, f"{n_excluded} additional high earner(s)\nabove ${display_cap:.0f}/hr not shown",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                color=MUTED, style="italic")
    ax.set_title("Figure 3. Hourly wage distribution by treatment arm\n(dashed lines: response-weighted mean)",
                 loc="left", fontsize=12.5, color=INK, pad=14)
    _save(fig, "fig3_wage_distribution.png")


# ---------------------------------------------------------------------------
# Figure 4 -- extensive-margin check: employment rate by arm
# ---------------------------------------------------------------------------
def fig_employment(emp_low, emp_high, beta, ci_lo, ci_hi):
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    x = [0, 1]
    vals = [emp_low, emp_high]
    ax.bar(x, vals, width=0.5, color=[BLUE, ORANGE], zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 1.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=11, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(["Low-cash arm\n($20/mo)", "High-cash arm\n($333/mo)"], fontsize=10.5)
    ax.set_ylabel("Employed at Year 1 (%)", color=MUTED, fontsize=10)
    y_top = max(vals) * 1.42
    ax.set_ylim(0, y_top)
    ax.grid(axis="y", zorder=0)
    _clean_ax(ax)

    bracket_y = max(vals) * 1.20
    ax.plot([0, 0, 1, 1], [vals[0] + 3, bracket_y, bracket_y, vals[1] + 3], color=MUTED, lw=1, zorder=2)
    ax.text(0.5, bracket_y + y_top * 0.02,
            f"DML effect: {beta * 100:+.1f} pp\n(95% CI [{ci_lo * 100:+.1f}, {ci_hi * 100:+.1f}])",
            ha="center", va="bottom", fontsize=9, color=INK_SECONDARY)

    ax.set_title("Figure 4. Extensive-margin check: employment rate\nby treatment arm",
                 loc="left", fontsize=12.5, color=INK, pad=14)
    _save(fig, "fig4_employment_rate.png")


# ---------------------------------------------------------------------------
# Figure 5 -- wage-impact estimates, LinearDML vs. CausalForestDML
# ---------------------------------------------------------------------------
def fig_ate_comparison(linear_est, linear_lo, linear_hi, forest_est, forest_lo, forest_hi):
    rows = ["CausalForestDML\n(mean of per-mother effects)", "LinearDML\n(single average effect)"]
    ests = [forest_est, linear_est]
    los = [forest_lo, linear_lo]
    his = [forest_hi, linear_hi]

    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    y = np.arange(len(rows))
    for yi, est, lo, hi in zip(y, ests, los, his):
        ax.plot([lo, hi], [yi, yi], color=INK_SECONDARY, lw=2, zorder=2, solid_capstyle="round")
        ax.plot(est, yi, "o", color=BLUE, markersize=10, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=2)

    ax.axvline(0, color=BASELINE, lw=1, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(rows, fontsize=10.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    xr = max(abs(min(los)), abs(max(his))) * 1.25
    ax.set_xlim(-xr, xr)

    # Anchor value labels in axes-fraction x-space (outside the plotted data
    # range entirely) rather than at a data-coordinate offset, so a wide CI
    # whisker can never run underneath the text -- see dataviz skill note on
    # measuring before placing a label near a mark's own extent.
    label_transform = ax.get_yaxis_transform()
    for yi, est, lo, hi in zip(y, ests, los, his):
        ax.text(1.03, yi, f"{est:+.1f}%  [{lo:+.1f}, {hi:+.1f}]",
                transform=label_transform, va="center", ha="left",
                fontsize=10, color=INK, clip_on=False)

    fig.subplots_adjust(left=0.27, right=0.66, top=0.85, bottom=0.16)
    ax.set_xlabel("Wage impact of the high-cash arm (%)", color=MUTED, fontsize=10)
    ax.grid(axis="x", zorder=0)
    _clean_ax(ax, hide_left=True)
    ax.set_title("Figure 5. Estimated wage impact of the high-cash arm, by estimator",
                 loc="left", fontsize=12.5, color=INK, pad=14)
    fig.savefig(os.path.join(FIG_DIR, "fig5_ate_comparison.png"), dpi=200)
    plt.close(fig)
    print(f"wrote {os.path.join(FIG_DIR, 'fig5_ate_comparison.png')}")


# ---------------------------------------------------------------------------
# Figure 6 -- CATE distribution
# ---------------------------------------------------------------------------
def fig_cate_distribution(cate_pct, ate_pct):
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    bins = np.linspace(cate_pct.min(), cate_pct.max(), 26)
    counts, _, _ = ax.hist(cate_pct, bins=bins, color=BLUE, alpha=0.85, zorder=3,
                            edgecolor=SURFACE, linewidth=0.6)
    ax.axvline(ate_pct, color=ORANGE, lw=2, linestyle="--", zorder=4)
    ax.text(ate_pct, counts.max() * 1.04, f"LinearDML ATE = {ate_pct:+.1f}%",
            ha="center", va="bottom", fontsize=9.5, color=ORANGE)

    ax.set_xlabel("Personalized wage impact (%)", color=MUTED, fontsize=10)
    ax.set_ylabel("Number of mothers", color=MUTED, fontsize=10)
    ax.set_ylim(0, counts.max() * 1.22)
    ax.grid(axis="y", zorder=0)
    _clean_ax(ax)
    ax.set_title("Figure 6. CausalForestDML: distribution of per-mother wage-impact estimates (CATEs)",
                 loc="left", fontsize=12.5, color=INK, pad=14)
    _save(fig, "fig6_cate_distribution.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="Data/ICPSR_37871/DS0001/37871-0001-Data.tsv")
    parser.add_argument("--followup", default="Data/ICPSR_37871/DS0002/37871-0002-Data.tsv")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)

    raw = pp.load_raw(args.baseline, args.followup)
    response_sample = pp.prepare_response_sample(raw)
    weighted = dp.fit_response_weights(response_sample, cv=args.cv, random_state=args.random_state)

    wage_responded = weighted[weighted[pp.WAGE_RESPONDED_COL]]
    post_hours_band = pp.construct_hourly_wage(wage_responded)
    wage_df = pp.clean(weighted)
    employment_df = pp.build_employment_sample(weighted)

    # The randomized total (1,000 = 600 low-cash + 400 high-cash) is a fixed,
    # documented quantity for this study, not derivable from response_sample
    # (which already restricts to baseline-covariate-complete rows).
    stages = [
        ("Randomized", 1000),
        ("Baseline-covariate-complete", len(response_sample)),
        ("Answered earnings question", int(response_sample[pp.RESPONDED_COL].sum())),
        ("Answered earnings & hours questions", int(response_sample[pp.WAGE_RESPONDED_COL].sum())),
        ("Employed, hours in [10, 80]/week", len(post_hours_band)),
        ("Final wage-analytic sample", len(wage_df)),
    ]
    fig_sample_funnel(stages)

    resp_by_arm = response_sample.groupby(pp.TREATMENT_COL)[pp.RESPONDED_COL].mean() * 100
    wage_resp_by_arm = response_sample.groupby(pp.TREATMENT_COL)[pp.WAGE_RESPONDED_COL].mean() * 100
    fig_response_rates(resp_by_arm[0], resp_by_arm[1], wage_resp_by_arm[0], wage_resp_by_arm[1])

    low_wage = wage_df.loc[wage_df[pp.TREATMENT_COL] == 0, "HOURLY_WAGE"]
    high_wage = wage_df.loc[wage_df[pp.TREATMENT_COL] == 1, "HOURLY_WAGE"]
    low_w = wage_df.loc[wage_df[pp.TREATMENT_COL] == 0, pp.WAGE_RESPONSE_WEIGHT_COL]
    high_w = wage_df.loc[wage_df[pp.TREATMENT_COL] == 1, pp.WAGE_RESPONSE_WEIGHT_COL]
    mean_low = float(np.average(low_wage, weights=low_w))
    mean_high = float(np.average(high_wage, weights=high_w))
    fig_wage_distribution(low_wage, high_wage, mean_low, mean_high)

    emp_by_arm = employment_df.groupby(pp.TREATMENT_COL)[pp.EMPLOYED_COL].mean() * 100
    _, employment_result = dp.fit_employment_dml(employment_df, cv=args.cv, random_state=args.random_state)
    fig_employment(emp_by_arm[0], emp_by_arm[1],
                   employment_result.beta, employment_result.ci_lower, employment_result.ci_upper)

    _, ate_result = dp.fit_linear_dml(wage_df, cv=args.cv, random_state=args.random_state)
    _, cate, cate_pct, forest_ate_result, _ = dp.fit_causal_forest_dml(
        wage_df, cv=args.cv, random_state=args.random_state
    )
    fig_ate_comparison(
        ate_result.pct_wage_impact, ate_result.pct_wage_impact_ci[0], ate_result.pct_wage_impact_ci[1],
        forest_ate_result.pct_wage_impact, forest_ate_result.pct_wage_impact_ci[0], forest_ate_result.pct_wage_impact_ci[1],
    )
    fig_cate_distribution(cate_pct, ate_result.pct_wage_impact)

    print("\nAll figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
