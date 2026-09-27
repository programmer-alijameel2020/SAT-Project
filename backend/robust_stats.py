"""
Cluster-robust statistics for the occlusion-recovery study (any number of modes).

Modes are detected from columns named <mode>_recovered. Each comparison in
`wanted` (see main) runs when both of its modes are present.

Why this exists
---------------
paired_gap_records.csv holds 16,290 rows, but these are 3,258 unique
ground-truth gaps x 5 injection seeds, nested in 37 sequences. Exact McNemar
on all rows treats them as independent and overstates significance. This
script reports three levels of analysis, from most to least conservative:

  L1  sequence level   : Wilcoxon signed-rank on per-sequence rate differences
                         + sequence-cluster bootstrap CI of the pooled difference
  L2  unique-gap level : seeds averaged per gap, exact sign test
  L3  gap x seed level : exact McNemar + paired odds ratio (descriptive only)

Usage:  python robust_stats.py paired_gap_records.csv [--boot 5000] [--out robust_stats.csv]
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats

GAP_KEY = ["sequence", "gt_id", "start", "end"]


def load(path):
    d = pd.read_csv(path)
    for c in [c for c in d.columns if c.endswith("_recovered")]:
        d[c] = d[c].astype(str).str.strip().str.lower().eq("true")
    return d


def modes(d):
    return [c[: -len("_recovered")] for c in d.columns if c.endswith("_recovered")]


def _pair(d, a, b):
    """Return copy of d with columns x (mode a) and y (mode b) as booleans."""
    x = d.copy()
    x["_a"] = x[f"{a}_recovered"]
    x["_b"] = x[f"{b}_recovered"]
    return x


def sequence_level(d, n_boot, rng):
    sq = d.groupby("sequence")[["_a", "_b"]].mean()
    diff = sq["_b"] - sq["_a"]
    nz = diff[diff != 0]
    w = stats.wilcoxon(nz) if len(nz) >= 2 else None
    seqs = d["sequence"].unique()
    groups = {s: d[d["sequence"] == s] for s in seqs}
    boot = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(seqs, len(seqs), replace=True)
        x = pd.concat([groups[s] for s in pick])
        boot[i] = x["_b"].mean() - x["_a"].mean()
    return {
        "n_sequences": len(seqs),
        "rate_a": float(d["_a"].mean()), "rate_b": float(d["_b"].mean()),
        "seq_pos": int((diff > 0).sum()), "seq_neg": int((diff < 0).sum()),
        "seq_zero": int((diff == 0).sum()),
        "wilcoxon_p": float(w.pvalue) if w else np.nan,
        "pooled_diff": float(d["_b"].mean() - d["_a"].mean()),
        "boot_ci_low": float(np.percentile(boot, 2.5)),
        "boot_ci_high": float(np.percentile(boot, 97.5)),
    }


def gap_level(d):
    g = d.groupby(GAP_KEY)[["_a", "_b"]].mean()
    diff = g["_b"] - g["_a"]
    pos, neg = int((diff > 0).sum()), int((diff < 0).sum())
    p = stats.binomtest(pos, pos + neg).pvalue if pos + neg else np.nan
    return {"n_unique_gaps": len(g), "gap_pos": pos, "gap_neg": neg,
            "gap_tie": int((diff == 0).sum()), "sign_test_p": float(p)}


def row_level(d):
    b = int((d["_b"] & ~d["_a"]).sum())
    c = int((~d["_b"] & d["_a"]).sum())
    p = stats.binomtest(b, b + c).pvalue if b + c else np.nan
    orr = b / c if c else np.inf
    se = np.sqrt(1 / b + 1 / c) if b and c else np.nan
    return {"n_rows": len(d), "favor_b": b, "favor_a": c, "mcnemar_p": float(p),
            "odds_ratio": float(orr),
            "or_ci_low": float(np.exp(np.log(orr) - 1.96 * se)) if b and c else np.nan,
            "or_ci_high": float(np.exp(np.log(orr) + 1.96 * se)) if b and c else np.nan}


def run(d, label, a, b, n_boot, rng):
    x = _pair(d, a, b)
    out = {"comparison": f"{b} vs {a}", "subset": label}
    out.update(sequence_level(x, n_boot, rng))
    out.update(gap_level(x))
    out.update(row_level(x))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="robust_stats.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    d = load(a.csv)

    subsets = [("ALL", d)]
    subsets += [(f"source={s}", x) for s, x in d.groupby("dataset_source")]
    subsets += [(f"category={c}", x) for c, x in d.groupby("category")]
    subsets += [(f"bucket={k}", x) for k, x in d.groupby("duration_bucket")]
    subsets.append(("ALL_excl_ship_045", d[d["sequence"] != "ship_045"]))

    ms = set(modes(d))
    # (baseline a, variant b): each isolates one factor of the design.
    wanted = [
        ("raw", "homography"),      # headline: compensation (+refit) vs raw refit
        ("raw", "affine"),
        ("raw_kf", "raw"),          # windowed refit, no compensation
        ("raw_kf", "gmc"),          # compensation, no refit (state warping)
        ("gmc", "affine"),          # KEY: refit on compensated history vs state warping alone
        ("raw_kf", "affine"),       # both components together
        ("affine", "homography"),   # transform model order
        ("homography", "oracle"),   # estimation error vs perfect transform
    ]
    pairs = [(a_, b_) for a_, b_ in wanted if a_ in ms and b_ in ms]
    if not pairs:
        raise SystemExit(f"No comparable mode pairs among {sorted(ms)}")

    rows = []
    for (ma, mb) in pairs:
        for label, x in subsets:
            if x["sequence"].nunique() < 2:
                continue
            rows.append(run(x, label, ma, mb, a.boot, rng))
    res = pd.DataFrame(rows)
    res.to_csv(a.out, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(res.round(4))


if __name__ == "__main__":
    main()