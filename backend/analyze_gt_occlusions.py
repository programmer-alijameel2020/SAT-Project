"""
Ground-truth occlusion gap analyzer.

Independent of anything our pipeline does: VISO's gt.txt only writes a
row for a track ID on frames where it's actually visible/annotated. A
gap in a track's frame sequence (frames present: 40-55, then nothing,
then resumes at 62) is therefore a GENUINE occlusion, recorded by the
dataset's own annotators -- not something inferred from our noisy
placeholder detector losing a blob.

This answers one specific, useful question before anything else: how
much real occlusion does this sequence actually contain? If genuine
occlusion is rare or mostly short here, that alone would explain why
raw vs. homography trajectory representations looked similar in the
tracker runs -- there may not be enough real long-occlusion cases in
this particular sequence to distinguish the two mechanisms, regardless
of whether the mechanism itself is sound.

Usage:
    python analyze_gt_occlusions.py "D:\\...\\VISO\\mot\\mot\\car\\001\\gt\\gt.txt"
"""
import argparse
from collections import defaultdict

from app.datasets.viso import parse_viso_gt_file


def load_gt(path):
    """frame,id,x,y,w,h,... -- delegates to the shared, format-tolerant
    VISO parser (app/datasets/viso.py), which auto-detects both confirmed
    delimiter variants (comma vs. whitespace) and box encodings (x,y,w,h
    vs. x1,y1,x2,y2 corners) across VISO's own releases. Only frame
    numbers are needed here -- gap-finding is a presence/absence question,
    independent of box format."""
    tracks_with_boxes = parse_viso_gt_file(path)
    tracks = defaultdict(list)
    for tid, entries in tracks_with_boxes.items():
        tracks[tid] = [e["frame"] for e in entries]
    return tracks


def find_gaps(frames):
    """Given a sorted list of frame numbers a track was visible in,
    return (gap_start, gap_end, duration) for every missing-frame span
    strictly between its first and last appearance."""
    frames = sorted(set(frames))
    gaps = []
    for a, b in zip(frames, frames[1:]):
        if b - a > 1:
            gaps.append((a + 1, b - 1, b - a - 1))
    return gaps


def bucket(duration):
    if duration <= 5:
        return "short_1_5"
    if duration <= 15:
        return "medium_6_15"
    return "long_16_plus"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt_path", help="Path to a VISO gt.txt file")
    args = ap.parse_args()

    tracks = load_gt(args.gt_path)
    print(f"Loaded {len(tracks)} ground-truth track(s) from {args.gt_path}\n")

    all_gaps = []
    per_track_span = []
    for tid, frames in tracks.items():
        frames_sorted = sorted(set(frames))
        span = frames_sorted[-1] - frames_sorted[0] + 1
        n_visible = len(frames_sorted)
        per_track_span.append((tid, frames_sorted[0], frames_sorted[-1], span, n_visible))
        gaps = find_gaps(frames_sorted)
        for (start, end, dur) in gaps:
            all_gaps.append(dict(track_id=tid, start=start, end=end, duration=dur))

    print(f"Total genuine occlusion gaps found: {len(all_gaps)}\n")

    buckets = {"short_1_5": [], "medium_6_15": [], "long_16_plus": []}
    for g in all_gaps:
        buckets[bucket(g["duration"])].append(g)

    print("Ground-truth occlusion-duration breakdown:")
    for name, items in buckets.items():
        durations = [g["duration"] for g in items]
        avg = sum(durations) / len(durations) if durations else 0
        print(f"  {name:15s}: {len(items):4d} gaps"
              + (f"  (avg duration {avg:.1f} frames)" if items else ""))

    if all_gaps:
        print("\nLongest 10 genuine occlusion gaps:")
        for g in sorted(all_gaps, key=lambda x: -x["duration"])[:10]:
            print(f"  track {g['track_id']:4d}: frames {g['start']}-{g['end']}"
                  f"  (duration {g['duration']} frames)")
    else:
        print("\nNo gaps found -- every track was visible on every frame between "
              "its first and last appearance. This sequence may have very little "
              "genuine (dataset-annotated) occlusion.")

    print(f"\nPer-track visibility spans (first 10 of {len(per_track_span)}):")
    for tid, first, last, span, n_vis in sorted(per_track_span)[:10]:
        missing = span - n_vis
        print(f"  id {tid:4d}: frames {first}-{last} (span {span}), "
              f"visible {n_vis} frames, missing {missing} frames")


if __name__ == "__main__":
    main()
