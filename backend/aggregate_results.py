"""
Aggregate a batch experiment campaign's results into a single paired
comparison and run McNemar's exact test for statistical significance --
the correct test for this design, since each genuine occlusion gap is
evaluated under BOTH trajectory-mode conditions (raw and homography)
with the SAME underlying synthetic-motion realization (same seed),
making this a paired binary-outcome comparison, not an
independent-samples one.

Usage:
    python aggregate_results.py --experiments-root "D:\\...\\Sat-project\\experiments" --prefix batch_
"""
import argparse
import json
import os
from collections import defaultdict

from scipy.stats import binomtest


def load_batch_results(experiments_root, prefix):
    """Returns list of (config, results) for every experiment folder
    whose config name starts with `prefix`."""
    out = []
    for entry in os.listdir(experiments_root):
        folder = os.path.join(experiments_root, entry)
        config_path = os.path.join(folder, "config.json")
        results_path = os.path.join(folder, "results.json")
        if not (os.path.isfile(config_path) and os.path.isfile(results_path)):
            continue
        with open(config_path) as f:
            config = json.load(f)
        if not config.get("name", "").startswith(prefix):
            continue
        with open(results_path) as f:
            results = json.load(f)
        out.append((config, results))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments-root", required=True)
    ap.add_argument("--prefix", default="batch_")
    args = ap.parse_args()

    runs = load_batch_results(args.experiments_root, args.prefix)
    print(f"Loaded {len(runs)} batch experiment result(s) matching prefix '{args.prefix}'\n")
    if not runs:
        print("Nothing to aggregate -- check --experiments-root and --prefix.")
        return

    # key: (sequence_name, seed, gt_id, start, end) -> {mode: recovered}
    paired = defaultdict(dict)
    per_mode_totals = defaultdict(lambda: [0, 0])                         # mode -> [recovered, total]
    per_mode_bucket = defaultdict(lambda: defaultdict(lambda: [0, 0]))    # mode -> bucket -> [rec, total]

    n_runs_with_eval = 0
    per_sequence = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # sequence -> mode -> [rec, total]
    for config, results in runs:
        mode = config.get("trajectory_mode")
        seed = config.get("synth_motion_seed")
        seq = config.get("sequence_name")
        gt_eval = results.get("gt_anchored_evaluation")
        if not gt_eval:
            continue
        n_runs_with_eval += 1
        for gap in gt_eval.get("gap_details", []):
            key = (seq, seed, gap["gt_id"], gap["start"], gap["end"])
            paired[key][mode] = gap["recovered"]

            per_mode_totals[mode][1] += 1
            if gap["recovered"]:
                per_mode_totals[mode][0] += 1

            dur = gap["duration"]
            bucket = "short_1_5" if dur <= 5 else "medium_6_15" if dur <= 15 else "long_16_plus"
            per_mode_bucket[mode][bucket][1] += 1
            if gap["recovered"]:
                per_mode_bucket[mode][bucket][0] += 1

            per_sequence[seq][mode][1] += 1
            if gap["recovered"]:
                per_sequence[seq][mode][0] += 1

    print(f"({n_runs_with_eval}/{len(runs)} runs had a gt_anchored_evaluation block)\n")

    print("=== Per-mode pooled recovery (across all sequences/seeds) ===")
    for mode, (rec, tot) in per_mode_totals.items():
        rate = f"{rec/tot:.1%}" if tot else "n/a"
        print(f"  {mode:12s}: {rec}/{tot}  ({rate})")

    print("\n=== Per-mode recovery by duration bucket ===")
    for mode, buckets in per_mode_bucket.items():
        print(f"  {mode}:")
        for b in ("short_1_5", "medium_6_15", "long_16_plus"):
            rec, tot = buckets[b]
            rate = f"{rec/tot:.1%}" if tot else "n/a"
            print(f"    {b:15s}: {rec}/{tot}  ({rate})")

    print("\n=== Per-sequence recovery (checking the pooled result isn't dominated by one sequence) ===")
    for seq in sorted(per_sequence.keys()):
        modes = per_sequence[seq]
        raw_rec, raw_tot = modes.get("raw", [0, 0])
        hom_rec, hom_tot = modes.get("homography", [0, 0])
        share = f"{raw_tot / sum(t for _, t in per_mode_totals.values()) * 100:.0f}%" if raw_tot else "0%"
        raw_rate = f"{raw_rec/raw_tot:.1%}" if raw_tot else "n/a"
        hom_rate = f"{hom_rec/hom_tot:.1%}" if hom_tot else "n/a"
        direction = ("homography higher" if hom_tot and raw_tot and hom_rec/hom_tot > raw_rec/raw_tot
                     else "raw higher" if hom_tot and raw_tot and hom_rec/hom_tot < raw_rec/raw_tot
                     else "tied/n-a")
        print(f"  {seq:12s} (~{share} of pooled pairs): raw {raw_rec}/{raw_tot} ({raw_rate})  "
              f"vs  homography {hom_rec}/{hom_tot} ({hom_rate})  -> {direction}")

    # McNemar's exact test on paired (raw, homography) outcomes
    n_01 = 0   # raw failed, homography succeeded
    n_10 = 0   # raw succeeded, homography failed
    n_both = 0
    n_neither = 0
    n_incomplete = 0
    for key, outcomes in paired.items():
        if "raw" not in outcomes or "homography" not in outcomes:
            n_incomplete += 1
            continue
        r, h = outcomes["raw"], outcomes["homography"]
        if r and h:
            n_both += 1
        elif not r and not h:
            n_neither += 1
        elif not r and h:
            n_01 += 1
        else:
            n_10 += 1

    print("\n=== Paired comparison (McNemar) ===")
    print(f"  Both recovered:                {n_both}")
    print(f"  Neither recovered:             {n_neither}")
    print(f"  raw failed, homography won:    {n_01}")
    print(f"  raw won, homography failed:    {n_10}")
    print(f"  (incomplete pairs skipped:     {n_incomplete})")

    n_discordant = n_01 + n_10
    if n_discordant == 0:
        print("\n  No discordant pairs -- McNemar's test needs at least one to be informative.")
    else:
        result = binomtest(n_01, n_discordant, p=0.5, alternative="two-sided")
        print(f"\n  McNemar's exact test (two-sided exact binomial on {n_discordant} discordant pairs):")
        print(f"    p-value = {result.pvalue:.4f}")
        if result.pvalue < 0.05:
            print(f"    -> statistically significant at alpha=0.05")
        else:
            print(f"    -> NOT statistically significant at alpha=0.05 "
                  f"(need more sequences/seeds for more discordant pairs)")

    # Per-sequence McNemar -- the real check of whether the pooled result
    # reflects a genuine, generalizing effect or is dominated by whichever
    # sequence happens to contribute the most gaps.
    per_seq_discordant = defaultdict(lambda: [0, 0])   # seq -> [n_01, n_10]
    for key, outcomes in paired.items():
        if "raw" not in outcomes or "homography" not in outcomes:
            continue
        seq = key[0]
        r, h = outcomes["raw"], outcomes["homography"]
        if not r and h:
            per_seq_discordant[seq][0] += 1
        elif r and not h:
            per_seq_discordant[seq][1] += 1

    print("\n=== Per-sequence McNemar (does the effect hold independently in each sequence?) ===")
    for seq in sorted(per_seq_discordant.keys()):
        n01, n10 = per_seq_discordant[seq]
        nd = n01 + n10
        if nd == 0:
            print(f"  {seq:12s}: no discordant pairs -- not informative")
            continue
        res = binomtest(n01, nd, p=0.5, alternative="two-sided")
        sig = "significant" if res.pvalue < 0.05 else "not significant"
        favor = "homography" if n01 > n10 else "raw" if n10 > n01 else "tied"
        print(f"  {seq:12s}: {nd} discordant ({n01} favor homography, {n10} favor raw), "
              f"p={res.pvalue:.4f} ({sig}, direction favors {favor})")


if __name__ == "__main__":
    main()
