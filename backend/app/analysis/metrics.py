"""
Statistical core for the results/report generator.

Everything here operates on GAP-LEVEL rows (one row per genuine ground-
truth occlusion gap, per run) and RUN-LEVEL rows (one row per experiment),
built by loading every experiment folder's config.json + results.json
from disk. This is the single source of truth both the API-driven web
Results page and the standalone generate_report.py CLI script read from,
so the two can never silently disagree.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

WILSON_Z = 1.959963985  # z for 95% two-sided CI


def wilson_ci(successes: int, total: int, z: float = WILSON_Z) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion -- used instead of
    the naive normal-approximation interval because it stays well-behaved
    (never goes below 0 or above 1) even for small samples or proportions
    near 0/1, both of which are common here (recovery rates are often
    under 30%, and some sequences have very few genuine gaps)."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    margin = (z * math.sqrt((p * (1 - p) + z ** 2 / (4 * total)) / total)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def paired_odds_ratio_ci(n01: int, n10: int, z: float = WILSON_Z) -> Tuple[float, float, float]:
    """
    Odds ratio (and its 95% CI) for paired binary outcomes, from the
    discordant-pair counts only (n01 = condition-B-only successes,
    n10 = condition-A-only successes) -- the standard approach for
    McNemar-style paired data, since concordant pairs (both succeeded or
    both failed) carry no information about which condition is better.
    Uses a Haldane-Anscombe continuity correction (+0.5 to both counts)
    when either count is zero, since log(0) is undefined and a raw zero
    count doesn't mean the true rate is exactly zero -- just that it
    wasn't observed in this sample.
    """
    a, b = n01, n10
    if a == 0 or b == 0:
        a, b = a + 0.5, b + 0.5
    or_est = a / b
    log_or = math.log(or_est)
    se = math.sqrt(1 / a + 1 / b)
    return (or_est, math.exp(log_or - z * se), math.exp(log_or + z * se))


def mcnemar_exact_pvalue(n01: int, n10: int) -> Optional[float]:
    """Two-sided exact McNemar test p-value via the binomial distribution
    -- avoids adding scipy as a hard dependency of this module by
    implementing the exact binomial tail sum directly (matches
    scipy.stats.binomtest(n01, n01+n10, 0.5) exactly). Uses a log-sum-exp
    accumulation rather than naively summing exp() terms directly: with
    thousands of highly skewed discordant pairs, the true p-value can be
    astronomically small (e.g. 1e-80 range), and naive summation risks
    every individual term underflowing to a true 0.0 before they're ever
    added -- log-sum-exp stays numerically valid down to values Python
    floats can represent at all (~1e-308)."""
    n = n01 + n10
    if n == 0:
        return None
    log_p = mcnemar_log10_pvalue(n01, n10)
    if log_p is None:
        return None
    log_p_nat = log_p * math.log(10)
    return math.exp(log_p_nat) if log_p_nat > -700 else 0.0  # true underflow only past ~1e-304


def mcnemar_log10_pvalue(n01: int, n10: int) -> Optional[float]:
    """log10 of the two-sided exact McNemar p-value -- the numerically
    robust and directly citable form for very small p-values (e.g.
    reporting "p < 10^-85" in a paper is both more informative and more
    honest than a value that rounds to 0.000000). Computed via
    log-sum-exp so it stays accurate however small the true p-value is,
    unlike computing p first and taking its log afterward."""
    n = n01 + n10
    if n == 0:
        return None
    k = min(n01, n10)

    def log_binom(n_, k_):
        return math.lgamma(n_ + 1) - math.lgamma(k_ + 1) - math.lgamma(n_ - k_ + 1)

    log_p_half = -n * math.log(2)
    log_terms = [log_binom(n, i) + log_p_half for i in range(0, k + 1)]
    upper_start = n - k
    log_terms += [log_binom(n, i) + log_p_half for i in range(upper_start, n + 1) if i > k]

    # log-sum-exp: log(sum(exp(x_i))) = m + log(sum(exp(x_i - m))), m = max(x_i)
    m = max(log_terms)
    total = sum(math.exp(t - m) for t in log_terms)
    log_p_nat = m + math.log(total)
    return min(0.0, log_p_nat / math.log(10))


def bucket_of(duration: int) -> str:
    if duration <= 5:
        return "short_1_5"
    if duration <= 15:
        return "medium_6_15"
    return "long_16_plus"


def category_of(dataset_source: str, sequence_name: Optional[str]) -> str:
    """VISO sequence names are '<category>_<seq_id>' (e.g. 'car_001') --
    the category is the natural grouping axis. SDM-Car sequence names
    ('1-2', '10-2', ...) carry no comparable category, so the dataset
    source itself is used as the category for those rows."""
    if dataset_source == "VISO" and sequence_name and "_" in sequence_name:
        return sequence_name.split("_")[0]
    return dataset_source or "unknown"


def load_experiment_records(
    experiments_root: str, name_prefix: str = ""
) -> Tuple[List[dict], List[dict]]:
    """
    Scans every experiment folder under experiments_root, keeps the ones
    whose config name starts with name_prefix (empty prefix = all), and
    returns (gap_rows, run_rows):

      gap_rows: one row per genuine GT occlusion gap per run --
        dataset_source, category, sequence, seed, mode, run_uid, gt_id,
        start, end, duration, duration_bucket, recovered

      run_rows: one row per experiment (run-level diagnostics) --
        dataset_source, category, sequence, seed, mode, run_uid,
        n_genuine_gt_gaps, n_recovered, recovery_rate,
        mean_total_latency_ms, mean_geometry_latency_ms,
        p95_total_latency_ms, homography_fallback_rate,
        platform_motion_mean_px, platform_motion_configured_px,
        n_frames, cancelled
    """
    gap_rows: List[dict] = []
    run_rows: List[dict] = []

    if not os.path.isdir(experiments_root):
        return gap_rows, run_rows

    for entry in sorted(os.listdir(experiments_root)):
        if entry.startswith("_"):
            continue  # skip the _reports output directory itself
        folder = os.path.join(experiments_root, entry)
        config_path = os.path.join(folder, "config.json")
        results_path = os.path.join(folder, "results.json")
        if not (os.path.isfile(config_path) and os.path.isfile(results_path)):
            continue
        try:
            with open(config_path) as f:
                config = json.load(f)
            with open(results_path) as f:
                results = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        name = config.get("name", "")
        if name_prefix and not name.startswith(name_prefix):
            continue

        dataset_source = config.get("dataset_source", "unknown")
        sequence = config.get("sequence_name")
        seed = config.get("synth_motion_seed")
        mode = config.get("trajectory_mode")
        category = category_of(dataset_source, sequence)

        run_rows.append(dict(
            dataset_source=dataset_source, category=category, sequence=sequence,
            seed=seed, mode=mode, run_uid=entry,
            n_genuine_gt_gaps=(results.get("gt_anchored_evaluation") or {}).get("n_genuine_gt_gaps"),
            n_recovered=(results.get("gt_anchored_evaluation") or {}).get("n_recovered"),
            recovery_rate=(results.get("gt_anchored_evaluation") or {}).get("recovery_rate"),
            mean_total_latency_ms=results.get("mean_total_latency_ms"),
            mean_geometry_latency_ms=results.get("mean_geometry_latency_ms"),
            p95_total_latency_ms=results.get("p95_total_latency_ms"),
            homography_fallback_rate=results.get("homography_fallback_rate"),
            platform_motion_mean_px=(results.get("platform_motion_px") or {}).get("mean"),
            platform_motion_configured_px=config.get("synth_motion_px_per_frame"),
            n_frames=results.get("n_frames"),
            cancelled=results.get("cancelled"),
            # The tracker's own internal occlusion bookkeeping -- kept
            # specifically to substantiate the "internal count is
            # inflated relative to genuine GT gaps" finding with real
            # per-sequence numbers, not just the single car_001 anecdote
            # that originally surfaced it.
            n_occlusion_events_internal=results.get("n_occlusion_events"),
        ))

        gt_eval = results.get("gt_anchored_evaluation")
        if not gt_eval:
            continue
        for gap in gt_eval.get("gap_details", []):
            gap_rows.append(dict(
                dataset_source=dataset_source, category=category, sequence=sequence,
                seed=seed, mode=mode, run_uid=entry,
                gt_id=gap["gt_id"], start=gap["start"], end=gap["end"],
                duration=gap["duration"], duration_bucket=bucket_of(gap["duration"]),
                recovered=gap["recovered"],
            ))

    return gap_rows, run_rows


def group_recovery(gap_rows: List[dict], key_fn) -> Dict[tuple, Tuple[int, int]]:
    """Groups gap_rows by key_fn(row) -> key, returns {key: (recovered, total)}."""
    out: Dict[tuple, List[int]] = defaultdict(lambda: [0, 0])
    for row in gap_rows:
        k = key_fn(row)
        out[k][1] += 1
        if row["recovered"]:
            out[k][0] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


def paired_discordant_counts(gap_rows: List[dict], group_key_fn=None) -> Dict[tuple, Tuple[int, int, int, int]]:
    """
    Pairs raw vs. homography outcomes for the SAME underlying gap
    (identified by sequence, seed, gt_id, start, end -- independent of
    mode), grouped by group_key_fn(row) if given (else one global group).
    Returns {group_key: (n_both, n_neither, n01, n10)} where n01 = raw
    failed & homography won, n10 = raw won & homography failed.
    """
    paired: Dict[tuple, Dict[str, bool]] = defaultdict(dict)
    group_of_pair: Dict[tuple, tuple] = {}
    for row in gap_rows:
        pair_key = (row["dataset_source"], row["sequence"], row["seed"],
                    row["gt_id"], row["start"], row["end"])
        paired[pair_key][row["mode"]] = row["recovered"]
        if group_key_fn is not None:
            group_of_pair[pair_key] = group_key_fn(row)
        else:
            group_of_pair[pair_key] = ("all",)

    counts: Dict[tuple, List[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for pair_key, outcomes in paired.items():
        if "raw" not in outcomes or "homography" not in outcomes:
            continue
        g = group_of_pair[pair_key]
        r, h = outcomes["raw"], outcomes["homography"]
        if r and h:
            counts[g][0] += 1
        elif not r and not h:
            counts[g][1] += 1
        elif not r and h:
            counts[g][2] += 1
        else:
            counts[g][3] += 1
    return {k: tuple(v) for k, v in counts.items()}


def build_paired_records(gap_rows: List[dict]) -> List[dict]:
    """
    One row per genuine occlusion gap, with BOTH conditions' outcomes on
    the same row (wide format) -- the exact reviewer-reproducibility
    artifact for the paired analysis: everything needed to independently
    recompute Wilson CIs, the exact McNemar p-value, and paired odds
    ratios from scratch, without needing any of this project's own code.
    Rows where only one mode's outcome exists (an incomplete pair) are
    still included with the missing side as None, rather than silently
    dropped, so the export is a complete audit trail of every genuine
    gap the campaign touched.
    """
    paired: Dict[tuple, dict] = {}
    for row in gap_rows:
        pair_key = (row["dataset_source"], row["sequence"], row["seed"],
                    row["gt_id"], row["start"], row["end"])
        if pair_key not in paired:
            paired[pair_key] = dict(
                dataset_source=row["dataset_source"], category=row["category"],
                sequence=row["sequence"], seed=row["seed"], gt_id=row["gt_id"],
                start=row["start"], end=row["end"], duration=row["duration"],
                duration_bucket=row["duration_bucket"],
                raw_recovered=None, homography_recovered=None,
                raw_run_uid=None, homography_run_uid=None,
            )
        paired[pair_key][f"{row['mode']}_recovered"] = row["recovered"]
        paired[pair_key][f"{row['mode']}_run_uid"] = row["run_uid"]

    out = []
    for rec in paired.values():
        rr, hr = rec["raw_recovered"], rec["homography_recovered"]
        if rr is None or hr is None:
            concordance = "incomplete_pair"
        elif rr == hr:
            concordance = "concordant"
        else:
            concordance = "discordant_favor_homography" if hr and not rr else "discordant_favor_raw"
        rec["concordance"] = concordance
        out.append(rec)
    out.sort(key=lambda r: (r["dataset_source"], r["sequence"], r["seed"], r["gt_id"], r["start"]))
    return out
