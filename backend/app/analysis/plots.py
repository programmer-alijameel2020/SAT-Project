"""
Plot generation for the results/report system. All functions return a
matplotlib Figure; the caller (report_generator.py) saves each as both
PDF (for the file directory / paper figures) and PNG (for web display).

Backend is forced to 'Agg' (headless, no display) since this runs
server-side -- must happen before pyplot is imported anywhere.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from .metrics import (
    wilson_ci, paired_odds_ratio_ci, mcnemar_exact_pvalue,
    group_recovery, paired_discordant_counts,
)

MODE_COLORS = {"raw": "#94a3b8", "homography": "#2f6b62", "affine": "#c9a227"}
FIGSIZE_WIDE = (9, 5)
FIGSIZE_SQUARE = (7, 6)
FIGSIZE_TALL = (7, 9)


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)


def plot_overall_recovery(gap_rows: list) -> plt.Figure:
    """Headline figure: pooled recovery rate per mode (raw/affine/
    homography, whichever are actually present), with Wilson 95% CI
    error bars. The exact McNemar p-value annotation is specifically
    raw-vs-homography (the primary ablation this project is built
    around) regardless of how many other modes are shown as bars."""
    by_mode = group_recovery(gap_rows, lambda r: r["mode"])
    modes = sorted(by_mode.keys())

    fig, ax = plt.subplots(figsize=(6, 5.5))
    rates, los, his, ns = [], [], [], []
    for m in modes:
        rec, tot = by_mode[m]
        rate = rec / tot if tot else 0
        lo, hi = wilson_ci(rec, tot)
        rates.append(rate); los.append(rate - lo); his.append(hi - rate); ns.append(tot)

    ax.bar(modes, rates, color=[MODE_COLORS.get(m, "#888") for m in modes],
           width=0.5, yerr=[los, his], capsize=6,
           error_kw=dict(ecolor="#333", linewidth=1.3))
    for i, (m, rate, n) in enumerate(zip(modes, rates, ns)):
        ax.text(i, rate + his[i] + 0.015, f"{rate:.1%}\n(n={n})", ha="center", fontsize=10)

    counts = paired_discordant_counts(gap_rows)
    if ("all",) in counts:
        _, _, n01, n10 = counts[("all",)]
        p = mcnemar_exact_pvalue(n01, n10)
        if p is not None:
            sig = "*" * (3 if p < 0.001 else 2 if p < 0.01 else 1 if p < 0.05 else 0)
            ax.set_title(f"Pooled Genuine-Occlusion Recovery Rate\n"
                         f"McNemar exact p={p:.4g} {sig}  (n={n01+n10} discordant pairs)",
                         fontsize=11)
    ax.set_ylabel("Recovery rate (Wilson 95% CI)")
    ax.set_ylim(0, max(rates) * 1.4 if rates else 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    _style_ax(ax)
    fig.tight_layout()
    return fig


def plot_recovery_by_bucket(gap_rows: list) -> plt.Figure:
    """Grouped bar: recovery rate by occlusion-duration bucket x mode
    (all modes present, not hardcoded to two)."""
    buckets = ["short_1_5", "medium_6_15", "long_16_plus"]
    bucket_labels = ["Short (1-5f)", "Medium (6-15f)", "Long (16f+)"]
    by_group = group_recovery(gap_rows, lambda r: (r["mode"], r["duration_bucket"]))
    modes = sorted(set(r["mode"] for r in gap_rows))

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    x = np.arange(len(buckets))
    width = 0.8 / max(1, len(modes))
    for i, m in enumerate(modes):
        rates, errs_lo, errs_hi = [], [], []
        for b in buckets:
            rec, tot = by_group.get((m, b), (0, 0))
            rate = rec / tot if tot else 0
            lo, hi = wilson_ci(rec, tot)
            rates.append(rate); errs_lo.append(rate - lo); errs_hi.append(hi - rate)
        offset = (i - (len(modes) - 1) / 2) * width
        ax.bar(x + offset, rates, width, label=m, color=MODE_COLORS.get(m, "#888"),
               yerr=[errs_lo, errs_hi], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_ylabel("Recovery rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("Recovery Rate by Occlusion Duration")
    ax.legend(frameon=False)
    _style_ax(ax)
    fig.tight_layout()
    return fig


def plot_recovery_by_sequence(gap_rows: list) -> plt.Figure:
    """Horizontal grouped bar of recovery rate per sequence x mode (all
    modes present), sorted by sample size so the most-informative
    sequences are easiest to read."""
    by_group = group_recovery(gap_rows, lambda r: (r["sequence"], r["mode"]))
    modes = sorted(set(r["mode"] for r in gap_rows))
    sequences = sorted(set(r["sequence"] for r in gap_rows),
                        key=lambda s: -sum(t for (seq, _), (rec, t) in by_group.items() if seq == s))

    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(sequences) + 1)))
    y = np.arange(len(sequences))
    height = 0.8 / max(1, len(modes))
    for i, m in enumerate(modes):
        rates = []
        for seq in sequences:
            rec, tot = by_group.get((seq, m), (0, 0))
            rates.append(rec / tot if tot else 0)
        offset = (i - (len(modes) - 1) / 2) * height
        ax.barh(y + offset, rates, height, label=m, color=MODE_COLORS.get(m, "#888"))
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s}  (n={sum(t for (sq,_),(_,t) in by_group.items() if sq==s)//len(modes)})"
                         for s in sequences])
    ax.set_xlabel("Recovery rate")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("Recovery Rate by Sequence")
    ax.legend(frameon=False)
    ax.invert_yaxis()
    _style_ax(ax)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_forest_odds_ratio(gap_rows: list) -> plt.Figure:
    """Forest plot: per-sequence odds ratio (homography vs raw) with 95%
    CI on a log axis, plus the pooled estimate -- the standard way to
    show whether an effect is consistent across independent groups
    (here: sequences) rather than driven by one large one."""
    counts = paired_discordant_counts(gap_rows, group_key_fn=lambda r: (r["sequence"],))
    pooled = paired_discordant_counts(gap_rows)

    rows = []
    for (seq,), (both, neither, n01, n10) in counts.items():
        nd = n01 + n10
        if nd == 0:
            continue
        orv, lo, hi = paired_odds_ratio_ci(n01, n10)
        rows.append((seq, orv, lo, hi, nd))
    rows.sort(key=lambda r: r[0])

    if ("all",) in pooled:
        _, _, n01, n10 = pooled[("all",)]
        p_or, p_lo, p_hi = paired_odds_ratio_ci(n01, n10)
        rows.append(("POOLED", p_or, p_lo, p_hi, n01 + n10))

    fig, ax = plt.subplots(figsize=(7, max(3, 0.55 * len(rows) + 1.5)))
    y = np.arange(len(rows))
    for i, (seq, orv, lo, hi, nd) in enumerate(rows):
        color = "#2f6b62" if seq == "POOLED" else "#5b8db8"
        marker = "D" if seq == "POOLED" else "o"
        ax.plot([lo, hi], [i, i], color=color, linewidth=1.5, zorder=1)
        ax.scatter([orv], [i], color=color, marker=marker, s=70, zorder=2)
    ax.axvline(1.0, color="#c0392b", linestyle="--", linewidth=1, label="OR = 1 (no effect)")
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[0]} (n={r[4]})" for r in rows])
    ax.set_xlabel("Odds ratio (homography vs raw), log scale")
    ax.set_title("Forest Plot: Effect Consistency Across Sequences")
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower right")
    _style_ax(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_discordant_breakdown(gap_rows: list) -> plt.Figure:
    """Grouped bar per sequence: how many gaps each mode uniquely
    recovered (discordant pairs only) -- directly visualizes where each
    mode's wins are concentrated."""
    counts = paired_discordant_counts(gap_rows, group_key_fn=lambda r: (r["sequence"],))
    sequences = sorted(seq for (seq,) in counts.keys())

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    x = np.arange(len(sequences))
    width = 0.35
    n01_vals = [counts[(s,)][2] for s in sequences]
    n10_vals = [counts[(s,)][3] for s in sequences]
    ax.bar(x - width / 2, n01_vals, width, label="homography won (raw failed)",
           color=MODE_COLORS["homography"])
    ax.bar(x + width / 2, n10_vals, width, label="raw won (homography failed)",
           color=MODE_COLORS["raw"])
    ax.set_xticks(x)
    ax.set_xticklabels(sequences, rotation=30, ha="right")
    ax.set_ylabel("Discordant gap count")
    ax.set_title("Where Each Mode Uniquely Wins, by Sequence")
    ax.legend(frameon=False)
    _style_ax(ax)
    fig.tight_layout()
    return fig


def plot_recovery_vs_duration(gap_rows: list) -> plt.Figure:
    """Line plot: recovery rate as a function of exact occlusion
    duration (binned per-frame where sample size allows), showing where
    along the duration axis the two modes diverge, rather than just the
    coarse 3-bucket summary."""
    durations = sorted(set(r["duration"] for r in gap_rows))
    by_group = group_recovery(gap_rows, lambda r: (r["mode"], r["duration"]))

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for m in sorted(set(r["mode"] for r in gap_rows)):
        xs, ys, ns = [], [], []
        for d in durations:
            rec, tot = by_group.get((m, d), (0, 0))
            if tot == 0:
                continue
            xs.append(d); ys.append(rec / tot); ns.append(tot)
        sizes = [max(15, min(120, n * 4)) for n in ns]
        ax.plot(xs, ys, color=MODE_COLORS.get(m, "#888"), linewidth=1.2, alpha=0.6, zorder=1)
        ax.scatter(xs, ys, s=sizes, color=MODE_COLORS.get(m, "#888"), label=m,
                   alpha=0.85, zorder=2, edgecolors="white", linewidth=0.5)
    ax.set_xlabel("Occlusion duration (frames)")
    ax.set_ylabel("Recovery rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("Recovery Rate vs. Exact Occlusion Duration\n(point size = sample count at that duration)")
    ax.legend(frameon=False)
    _style_ax(ax)
    fig.tight_layout()
    return fig


def plot_latency_distribution(run_rows: list) -> plt.Figure:
    """Box plot of per-run mean total latency, by mode -- the efficiency
    side of the evaluation, alongside the accuracy-focused plots above."""
    by_mode = {}
    for m in ["raw", "homography", "affine"]:
        vals = [r["mean_total_latency_ms"] for r in run_rows
                if r["mode"] == m and r["mean_total_latency_ms"] is not None]
        if vals:
            by_mode[m] = vals

    fig, ax = plt.subplots(figsize=(6, 5.5))
    if by_mode:
        modes = list(by_mode.keys())
        data = [by_mode[m] for m in modes]
        bp = ax.boxplot(data, labels=modes, patch_artist=True, widths=0.5,
                         medianprops=dict(color="#222", linewidth=1.5))
        for patch, m in zip(bp["boxes"], modes):
            patch.set_facecolor(MODE_COLORS.get(m, "#888"))
            patch.set_alpha(0.7)
    ax.set_ylabel("Mean per-frame total latency (ms)")
    ax.set_title("Latency Distribution Across Runs")
    _style_ax(ax)
    fig.tight_layout()
    return fig


def plot_motion_injection_validation(run_rows: list) -> plt.Figure:
    """Scatter: configured vs. measured synthetic platform-motion
    magnitude, with a y=x reference line -- validates that the injection
    mechanism actually delivers what was requested (this caught a real
    bug earlier in the project when the two diverged badly)."""
    xs = [r["platform_motion_configured_px"] for r in run_rows
          if r["platform_motion_configured_px"] is not None and r["platform_motion_mean_px"] is not None]
    ys = [r["platform_motion_mean_px"] for r in run_rows
          if r["platform_motion_configured_px"] is not None and r["platform_motion_mean_px"] is not None]

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    if xs:
        ax.scatter(xs, ys, alpha=0.5, color="#2f6b62", edgecolors="white", linewidth=0.4)
        lim = max(max(xs, default=1), max(ys, default=1)) * 1.15
        ax.plot([0, lim], [0, lim], color="#c0392b", linestyle="--", linewidth=1, label="y = x (perfect match)")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Configured drift (px/frame)")
    ax.set_ylabel("Measured drift, mean (px/frame)")
    ax.set_title("Synthetic Motion Injection: Configured vs. Measured")
    ax.legend(frameon=False)
    _style_ax(ax)
    fig.tight_layout()
    return fig


def plot_homography_fallback_rates(run_rows: list) -> plt.Figure:
    """Histogram of per-run homography-estimation fallback rate -- how
    often the primary ORB+RANSAC estimate failed its validity gate and
    fell back to affine/identity. Low values across the board indicate
    the platform-motion estimator itself is reliable."""
    vals = [r["homography_fallback_rate"] for r in run_rows if r["homography_fallback_rate"] is not None]
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    if vals:
        ax.hist(vals, bins=min(20, max(5, len(set(vals)))), color="#5b8db8", edgecolor="white")
    ax.set_xlabel("Homography fallback rate (fraction of frames)")
    ax.set_ylabel("Number of runs")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("Homography Estimation Reliability Across All Runs")
    _style_ax(ax)
    fig.tight_layout()
    return fig


def plot_recovery_by_category(gap_rows: list) -> plt.Figure:
    """Grouped bar: recovery rate by target category (car/plane/ship/
    train/SDM-Car) x mode -- the category-level effect (e.g. a mechanism
    that helps small erratic targets more than large steady ones)."""
    by_group = group_recovery(gap_rows, lambda r: (r["category"], r["mode"]))
    categories = sorted(set(r["category"] for r in gap_rows))
    modes = sorted(set(r["mode"] for r in gap_rows))

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    x = np.arange(len(categories))
    width = 0.8 / max(1, len(modes))
    for i, m in enumerate(modes):
        rates, errs_lo, errs_hi = [], [], []
        for c in categories:
            rec, tot = by_group.get((c, m), (0, 0))
            rate = rec / tot if tot else 0
            lo, hi = wilson_ci(rec, tot)
            rates.append(rate); errs_lo.append(rate - lo); errs_hi.append(hi - rate)
        offset = (i - (len(modes) - 1) / 2) * width
        ax.bar(x + offset, rates, width, label=m, color=MODE_COLORS.get(m, "#888"),
               yerr=[errs_lo, errs_hi], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Recovery rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("Recovery Rate by Target Category")
    ax.legend(frameon=False)
    _style_ax(ax)
    fig.tight_layout()
    return fig


def plot_sequence_bucket_heatmap(gap_rows: list) -> plt.Figure:
    """Heatmap: rows = sequences, columns = duration buckets, value =
    (homography recovery rate - raw recovery rate). Diverging colormap
    centered at 0 -- green means homography helps in that cell, red
    means raw does better -- lets many sequences x buckets be scanned
    at once rather than reading dozens of small bar charts."""
    buckets = ["short_1_5", "medium_6_15", "long_16_plus"]
    bucket_labels = ["Short", "Medium", "Long"]
    by_group = group_recovery(gap_rows, lambda r: (r["sequence"], r["duration_bucket"], r["mode"]))
    sequences = sorted(set(r["sequence"] for r in gap_rows))

    grid = np.full((len(sequences), len(buckets)), np.nan)
    for i, seq in enumerate(sequences):
        for j, b in enumerate(buckets):
            h_rec, h_tot = by_group.get((seq, b, "homography"), (0, 0))
            r_rec, r_tot = by_group.get((seq, b, "raw"), (0, 0))
            if h_tot and r_tot:
                grid[i, j] = (h_rec / h_tot) - (r_rec / r_tot)

    fig, ax = plt.subplots(figsize=(6, max(3, 0.5 * len(sequences) + 1.5)))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels(bucket_labels)
    ax.set_yticks(range(len(sequences)))
    ax.set_yticklabels(sequences)
    for i in range(len(sequences)):
        for j in range(len(buckets)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i,j]:+.0%}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Recovery rate: homography - raw")
    ax.set_title("Recovery Advantage by Sequence x Occlusion Duration")
    fig.tight_layout()
    return fig
