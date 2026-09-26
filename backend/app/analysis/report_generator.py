"""
Orchestrates a full report: loads every matching experiment from disk,
computes every table, generates every figure (saved as both PDF and
PNG), and writes a summary.json the web Results page and the standalone
CLI script both read from -- this is the single place "run everything"
happens, so the web UI and the terminal tool can never drift apart.
"""
from __future__ import annotations

import csv
import datetime
import json
import os
import uuid

from . import plots as P
from .metrics import (
    load_experiment_records, group_recovery, paired_discordant_counts,
    wilson_ci, paired_odds_ratio_ci, mcnemar_exact_pvalue, mcnemar_log10_pvalue,
    bucket_of, build_paired_records,
)

FIGURES = [
    ("overall_recovery", "Pooled Recovery Rate (headline)", P.plot_overall_recovery, "gap"),
    ("recovery_by_bucket", "Recovery Rate by Occlusion Duration", P.plot_recovery_by_bucket, "gap"),
    ("recovery_by_sequence", "Recovery Rate by Sequence", P.plot_recovery_by_sequence, "gap"),
    ("forest_odds_ratio", "Forest Plot: Effect Consistency", P.plot_forest_odds_ratio, "gap"),
    ("discordant_breakdown", "Discordant Wins by Sequence", P.plot_discordant_breakdown, "gap"),
    ("recovery_vs_duration", "Recovery Rate vs. Exact Duration", P.plot_recovery_vs_duration, "gap"),
    ("recovery_by_category", "Recovery Rate by Target Category", P.plot_recovery_by_category, "gap"),
    ("sequence_bucket_heatmap", "Advantage Heatmap (Sequence x Duration)", P.plot_sequence_bucket_heatmap, "gap"),
    ("latency_distribution", "Latency Distribution", P.plot_latency_distribution, "run"),
    ("motion_injection_validation", "Motion Injection: Configured vs Measured", P.plot_motion_injection_validation, "run"),
    ("homography_fallback_rates", "Homography Estimation Reliability", P.plot_homography_fallback_rates, "run"),
]


def _write_table(rows: list, path_no_ext: str, fieldnames: list):
    """Writes the same table as both .csv (for spreadsheets/papers) and
    .json (for the web Results page to render directly as HTML)."""
    with open(path_no_ext + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(path_no_ext + ".json", "w") as f:
        json.dump(rows, f, indent=2, default=str)


def _build_tables(gap_rows: list, run_rows: list, tables_dir: str) -> list:
    manifest = []

    # 1. Summary by mode
    by_mode = group_recovery(gap_rows, lambda r: r["mode"])
    rows = []
    for m, (rec, tot) in sorted(by_mode.items()):
        lo, hi = wilson_ci(rec, tot)
        rows.append(dict(mode=m, recovered=rec, total=tot,
                          recovery_rate=round(rec / tot, 4) if tot else None,
                          ci_low=round(lo, 4), ci_high=round(hi, 4)))
    _write_table(rows, os.path.join(tables_dir, "summary_by_mode"),
                 ["mode", "recovered", "total", "recovery_rate", "ci_low", "ci_high"])
    manifest.append(dict(name="summary_by_mode", title="Summary by Mode"))

    # 2. Summary by mode x duration bucket
    by_bucket = group_recovery(gap_rows, lambda r: (r["mode"], r["duration_bucket"]))
    rows = []
    for (m, b), (rec, tot) in sorted(by_bucket.items()):
        lo, hi = wilson_ci(rec, tot)
        rows.append(dict(mode=m, duration_bucket=b, recovered=rec, total=tot,
                          recovery_rate=round(rec / tot, 4) if tot else None,
                          ci_low=round(lo, 4), ci_high=round(hi, 4)))
    _write_table(rows, os.path.join(tables_dir, "summary_by_mode_bucket"),
                 ["mode", "duration_bucket", "recovered", "total", "recovery_rate", "ci_low", "ci_high"])
    manifest.append(dict(name="summary_by_mode_bucket", title="Summary by Mode x Duration Bucket"))

    # 3. Summary by sequence x mode, with per-sequence McNemar + odds ratio
    by_seq_mode = group_recovery(gap_rows, lambda r: (r["sequence"], r["mode"]))
    discordant_by_seq = paired_discordant_counts(gap_rows, group_key_fn=lambda r: (r["sequence"],))
    sequences = sorted(set(r["sequence"] for r in gap_rows))
    rows = []
    for seq in sequences:
        raw_rec, raw_tot = by_seq_mode.get((seq, "raw"), (0, 0))
        hom_rec, hom_tot = by_seq_mode.get((seq, "homography"), (0, 0))
        both, neither, n01, n10 = discordant_by_seq.get((seq,), (0, 0, 0, 0))
        p = mcnemar_exact_pvalue(n01, n10)
        log10p = mcnemar_log10_pvalue(n01, n10)
        orv, or_lo, or_hi = paired_odds_ratio_ci(n01, n10) if (n01 + n10) else (None, None, None)
        rows.append(dict(
            sequence=seq, raw_recovered=raw_rec, raw_total=raw_tot,
            raw_rate=round(raw_rec / raw_tot, 4) if raw_tot else None,
            homography_recovered=hom_rec, homography_total=hom_tot,
            homography_rate=round(hom_rec / hom_tot, 4) if hom_tot else None,
            n_discordant=n01 + n10, favor_homography=n01, favor_raw=n10,
            # Full precision, NOT rounded to a fixed decimal count -- a
            # p-value this small (routinely far below 1e-6 for
            # well-powered sequences) rounds to a hard 0.0 otherwise,
            # destroying the actual magnitude. log10_p is the
            # numerically robust, directly citable form ("p < 10^-88").
            mcnemar_p=p, mcnemar_log10_p=round(log10p, 3) if log10p is not None else None,
            odds_ratio=round(orv, 4) if orv else None,
            or_ci_low=round(or_lo, 4) if or_lo else None,
            or_ci_high=round(or_hi, 4) if or_hi else None,
        ))
    _write_table(rows, os.path.join(tables_dir, "summary_by_sequence"),
                 ["sequence", "raw_recovered", "raw_total", "raw_rate",
                  "homography_recovered", "homography_total", "homography_rate",
                  "n_discordant", "favor_homography", "favor_raw", "mcnemar_p", "mcnemar_log10_p",
                  "odds_ratio", "or_ci_low", "or_ci_high"])
    manifest.append(dict(name="summary_by_sequence", title="Summary by Sequence (with per-sequence McNemar)"))

    # 4. Summary by category x mode
    by_cat_mode = group_recovery(gap_rows, lambda r: (r["category"], r["mode"]))
    categories = sorted(set(r["category"] for r in gap_rows))
    rows = []
    for cat in categories:
        raw_rec, raw_tot = by_cat_mode.get((cat, "raw"), (0, 0))
        hom_rec, hom_tot = by_cat_mode.get((cat, "homography"), (0, 0))
        rows.append(dict(
            category=cat, raw_recovered=raw_rec, raw_total=raw_tot,
            raw_rate=round(raw_rec / raw_tot, 4) if raw_tot else None,
            homography_recovered=hom_rec, homography_total=hom_tot,
            homography_rate=round(hom_rec / hom_tot, 4) if hom_tot else None,
        ))
    _write_table(rows, os.path.join(tables_dir, "summary_by_category"),
                 ["category", "raw_recovered", "raw_total", "raw_rate",
                  "homography_recovered", "homography_total", "homography_rate"])
    manifest.append(dict(name="summary_by_category", title="Summary by Target Category"))

    # 5. Latency / efficiency summary by mode
    rows = []
    for m in sorted(set(r["mode"] for r in run_rows if r["mode"])):
        vals = [r["mean_total_latency_ms"] for r in run_rows if r["mode"] == m and r["mean_total_latency_ms"] is not None]
        geo_vals = [r["mean_geometry_latency_ms"] for r in run_rows if r["mode"] == m and r["mean_geometry_latency_ms"] is not None]
        p95_vals = [r["p95_total_latency_ms"] for r in run_rows if r["mode"] == m and r["p95_total_latency_ms"] is not None]
        rows.append(dict(
            mode=m, n_runs=len(vals),
            mean_total_latency_ms=round(sum(vals) / len(vals), 3) if vals else None,
            mean_geometry_latency_ms=round(sum(geo_vals) / len(geo_vals), 3) if geo_vals else None,
            mean_p95_latency_ms=round(sum(p95_vals) / len(p95_vals), 3) if p95_vals else None,
        ))
    _write_table(rows, os.path.join(tables_dir, "latency_summary"),
                 ["mode", "n_runs", "mean_total_latency_ms", "mean_geometry_latency_ms", "mean_p95_latency_ms"])
    manifest.append(dict(name="latency_summary", title="Latency / Efficiency Summary"))

    # 6. Raw paired gap-level records -- full reviewer-reproducibility
    # export: every genuine gap, both conditions' outcomes on one row,
    # everything needed to independently recompute every statistic in
    # this report from scratch.
    paired_records = build_paired_records(gap_rows)
    _write_table(paired_records, os.path.join(tables_dir, "paired_gap_records"),
                 ["dataset_source", "category", "sequence", "seed", "gt_id", "start", "end",
                  "duration", "duration_bucket", "raw_recovered", "homography_recovered",
                  "raw_run_uid", "homography_run_uid", "concordance"])
    manifest.append(dict(name="paired_gap_records",
                          title="Raw Paired Gap-Level Records (full reproducibility export)"))

    # 7. Platform-motion injection: configured vs. measured, per sequence
    by_seq_motion: dict = {}
    for r in run_rows:
        if r["platform_motion_mean_px"] is None:
            continue
        by_seq_motion.setdefault(r["sequence"], []).append(
            (r["platform_motion_mean_px"], r["platform_motion_configured_px"]))
    rows = []
    for seq in sorted(by_seq_motion.keys()):
        measured = [v[0] for v in by_seq_motion[seq]]
        configured = [v[1] for v in by_seq_motion[seq] if v[1] is not None]
        rows.append(dict(
            sequence=seq, n_runs=len(measured),
            configured_px_per_frame=configured[0] if configured else None,
            measured_mean_px_per_frame=round(sum(measured) / len(measured), 4),
            measured_min_px_per_frame=round(min(measured), 4),
            measured_max_px_per_frame=round(max(measured), 4),
        ))
    _write_table(rows, os.path.join(tables_dir, "platform_motion_by_sequence"),
                 ["sequence", "n_runs", "configured_px_per_frame", "measured_mean_px_per_frame",
                  "measured_min_px_per_frame", "measured_max_px_per_frame"])
    manifest.append(dict(name="platform_motion_by_sequence",
                          title="Synthetic Motion Injection: Configured vs. Measured, by Sequence"))

    # 8. Internal (tracker bookkeeping) vs. GT-anchored occlusion counts,
    # per sequence -- direct, per-sequence evidence for the inflation
    # finding that originally motivated building the GT-anchored
    # evaluator at all (previously only shown as a single car_001
    # anecdote: 851 internal events vs. 14 genuine gaps).
    by_seq_counts: dict = {}
    for r in run_rows:
        if r["n_occlusion_events_internal"] is None or r["n_genuine_gt_gaps"] is None:
            continue
        by_seq_counts.setdefault(r["sequence"], []).append(
            (r["n_occlusion_events_internal"], r["n_genuine_gt_gaps"]))
    rows = []
    for seq in sorted(by_seq_counts.keys()):
        internal_vals = [v[0] for v in by_seq_counts[seq]]
        true_vals = set(v[1] for v in by_seq_counts[seq])
        true_n = true_vals.pop() if len(true_vals) == 1 else max(true_vals)
        mean_internal = sum(internal_vals) / len(internal_vals)
        rows.append(dict(
            sequence=seq, n_genuine_gt_gaps=true_n,
            mean_internal_occlusion_events=round(mean_internal, 1),
            min_internal_occlusion_events=min(internal_vals),
            max_internal_occlusion_events=max(internal_vals),
            inflation_ratio_mean=round(mean_internal / true_n, 2) if true_n else None,
        ))
    _write_table(rows, os.path.join(tables_dir, "internal_vs_gt_occlusion_counts"),
                 ["sequence", "n_genuine_gt_gaps", "mean_internal_occlusion_events",
                  "min_internal_occlusion_events", "max_internal_occlusion_events", "inflation_ratio_mean"])
    manifest.append(dict(name="internal_vs_gt_occlusion_counts",
                          title="Internal Tracker Bookkeeping vs. GT-Anchored Occlusion Counts (inflation evidence)"))

    return manifest


def generate_report(experiments_root: str, name_prefix: str, reports_root: str) -> dict:
    gap_rows, run_rows = load_experiment_records(experiments_root, name_prefix)

    if not gap_rows:
        return dict(
            error=(f"No experiment data found under '{experiments_root}' matching "
                   f"name prefix '{name_prefix}'. Run some experiments first."),
        )

    report_id = f"{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    report_dir = os.path.join(reports_root, report_id)
    figures_dir = os.path.join(report_dir, "figures")
    tables_dir = os.path.join(report_dir, "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # ---- figures ----------------------------------------------------------
    figure_manifest = []
    for key, title, fn, data_kind in FIGURES:
        data = gap_rows if data_kind == "gap" else run_rows
        if not data:
            continue
        try:
            fig = fn(data)
        except Exception as e:
            figure_manifest.append(dict(name=key, title=title, error=str(e)))
            continue
        pdf_path = os.path.join(figures_dir, f"{key}.pdf")
        png_path = os.path.join(figures_dir, f"{key}.png")
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        figure_manifest.append(dict(name=key, title=title))

    # ---- tables -------------------------------------------------------------
    table_manifest = _build_tables(gap_rows, run_rows, tables_dir)

    # ---- headline summary -----------------------------------------------------
    by_mode = group_recovery(gap_rows, lambda r: r["mode"])
    pooled = paired_discordant_counts(gap_rows)
    _, _, n01, n10 = pooled.get(("all",), (0, 0, 0, 0))
    p_value = mcnemar_exact_pvalue(n01, n10)
    log10_p_value = mcnemar_log10_pvalue(n01, n10)
    or_val, or_lo, or_hi = paired_odds_ratio_ci(n01, n10) if (n01 + n10) else (None, None, None)

    raw_rec, raw_tot = by_mode.get("raw", (0, 0))
    hom_rec, hom_tot = by_mode.get("homography", (0, 0))

    summary = dict(
        report_id=report_id,
        generated_at=datetime.datetime.utcnow().isoformat(),
        name_prefix=name_prefix,
        experiments_root=experiments_root,
        n_runs=len(run_rows),
        n_gaps_total=len(gap_rows),
        n_sequences=len(set(r["sequence"] for r in gap_rows)),
        n_categories=len(set(r["category"] for r in gap_rows)),
        raw_recovered=raw_rec, raw_total=raw_tot,
        raw_recovery_rate=round(raw_rec / raw_tot, 4) if raw_tot else None,
        homography_recovered=hom_rec, homography_total=hom_tot,
        homography_recovery_rate=round(hom_rec / hom_tot, 4) if hom_tot else None,
        n_discordant_pairs=n01 + n10,
        favor_homography=n01, favor_raw=n10,
        # Full precision -- do NOT round this to a fixed decimal count.
        # At this sample size the true p-value is routinely far below
        # 1e-6 (e.g. the campaign that prompted this fix: p ~ 1.4e-88),
        # and round(p, 6) silently collapses any such value to a hard
        # 0.0, which is not a number a paper can report or a reviewer
        # can verify. mcnemar_log10_p is the robust, directly citable
        # form for exactly this situation ("p < 10^-88").
        mcnemar_p_value=p_value,
        mcnemar_log10_p_value=round(log10_p_value, 3) if log10_p_value is not None else None,
        odds_ratio=round(or_val, 4) if or_val else None,
        odds_ratio_ci_low=round(or_lo, 4) if or_lo else None,
        odds_ratio_ci_high=round(or_hi, 4) if or_hi else None,
        figures=figure_manifest,
        tables=table_manifest,
    )

    with open(os.path.join(report_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary
