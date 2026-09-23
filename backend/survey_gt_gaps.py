"""
Survey genuine ground-truth occlusion gaps across every sequence in a
dataset (VISO or SDM-Car), to help choose which sequences are worth
including in a multi-sequence experimental campaign.

Uses the SAME loader classes (VISOLoader / SDMCarLoader) the live system
itself uses for annotation parsing, rather than re-implementing format
parsing here -- this is what keeps the survey's gap counts trustworthy:
a fix to the real parser (e.g. the VISO corner-vs-xywh format fix found
on ship_045) automatically applies here too, instead of needing a second
copy of the parsing logic kept in sync by hand.

Usage:
    python survey_gt_gaps.py --dataset VISO "D:\\...\\VISO\\mot\\mot"
    python survey_gt_gaps.py --dataset SDM-Car "D:\\...\\SDM-Car\\train"
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.datasets.viso import VISOLoader
from app.datasets.sdmcar import SDMCarLoader
from analyze_gt_occlusions import find_gaps, bucket

LOADERS = {"VISO": VISOLoader, "SDM-Car": SDMCarLoader}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Dataset root directory (VISO's mot/mot, or SDM-Car's train/test/validation)")
    ap.add_argument("--dataset", choices=list(LOADERS.keys()), default="VISO")
    ap.add_argument("--out", default=None, help="CSV output path (default: gt_gap_survey_<dataset>.csv)")
    args = ap.parse_args()
    out_path = args.out or f"gt_gap_survey_{args.dataset.replace('-', '')}.csv"

    loader_cls = LOADERS[args.dataset]
    loader = loader_cls(args.root)
    sequences = loader.list_sequences()
    print(f"Found {len(sequences)} sequence(s) for {args.dataset} under {args.root}\n")

    if not sequences:
        print("No sequences found -- check the root path.")
        return

    rows = []
    for seq in sequences:
        annotations = loader.load_annotations(seq)
        tracks = (annotations or {}).get("tracks", {})
        n_tracks = len(tracks)
        all_gaps = []
        max_frame = 0
        for tid, entries in tracks.items():
            frames = sorted(set(e["frame"] for e in entries))
            if frames:
                max_frame = max(max_frame, frames[-1])
            all_gaps.extend(find_gaps(frames))

        buckets = defaultdict(int)
        for gap in all_gaps:
            buckets[bucket(gap[2])] += 1

        rows.append(dict(
            dataset=args.dataset, sequence=seq.name, n_frames_reported=seq.n_frames,
            n_frames_gt=max_frame, n_tracks=n_tracks, n_gaps=len(all_gaps),
            short=buckets["short_1_5"], medium=buckets["medium_6_15"], long=buckets["long_16_plus"],
        ))

    rows.sort(key=lambda r: -r["n_gaps"])

    print(f"{'sequence':18s} {'frames':>7s} {'tracks':>7s} {'gaps':>6s} {'short':>6s} {'med':>5s} {'long':>5s}")
    for r in rows:
        print(f"{r['sequence']:18s} {r['n_frames_gt']:7d} {r['n_tracks']:7d} {r['n_gaps']:6d} "
              f"{r['short']:6d} {r['medium']:5d} {r['long']:5d}")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nFull survey written to {out_path}")


if __name__ == "__main__":
    main()
