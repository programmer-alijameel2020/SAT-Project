"""
Ground-truth-anchored occlusion-recovery evaluator.

Fixes a confound found empirically: the tracker's own internal occlusion
bookkeeping (tracker.summary()'s n_occlusion_events) counts ANY
unmatched-then-rematched track as a recovery, including brief association
noise unrelated to genuine occlusion -- confirmed directly, with a
GT-detector run reporting 260-330 "events" against a ground truth
containing only 14 real gaps for VISO car_001.

This evaluator instead:
  1. Finds genuine occlusion gaps directly from the ground-truth
     annotations themselves (a gap in a GT track's own frame sequence --
     see analyze_gt_occlusions.py for the standalone version of this
     logic), independent of anything the tracker does.
  2. Matches the tracker's own reported track IDs to GT IDs per frame via
     IoU + Hungarian assignment, using only tracks the tracker itself
     reported as matched (visible) that frame.
  3. For each genuine GT gap, checks whether the tracker's assigned ID
     for that GT track is the SAME immediately before and immediately
     after the gap. That is Recovery@K exactly as originally specified
     (spec Section 13), evaluated against real occlusion rather than
     internal tracker bookkeeping.

Only meaningful when detector_mode="ground_truth" -- with the placeholder
detector, detections aren't frame-aligned to GT identities, so this
evaluator isn't wired in for that mode.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Optional
from collections import defaultdict

from ..tracking.association import iou


def find_gt_gaps(gt_tracks: Dict[int, list]) -> List[dict]:
    """Same logic as analyze_gt_occlusions.py's find_gaps(), applied to
    every GT track. Returns a flat list of genuine occlusion gaps, each
    a dict(gt_id, start, end, duration)."""
    gaps = []
    for gt_id, entries in gt_tracks.items():
        frames = sorted(set(e["frame"] for e in entries))
        for a, b in zip(frames, frames[1:]):
            if b - a > 1:
                gaps.append(dict(gt_id=gt_id, start=a + 1, end=b - 1, duration=b - a - 1))
    return gaps


class GTAnchoredEvaluator:
    def __init__(self, gt_tracks: Dict[int, list], iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        # NOTE: gt_tracks is used ONLY to determine which frames each GT
        # id is genuinely visible in (i.e. to find the real occlusion
        # gaps) -- that's a presence/absence question, unaffected by
        # synthetic-motion warping. It is NOT used as a source of box
        # POSITIONS for matching: with motion injection enabled, the
        # tracker only ever sees WARPED boxes, so comparing against
        # unwarped ground-truth positions here would silently fail IoU
        # matching for every frame once cumulative drift exceeds a few
        # pixels -- confirmed directly (100% null tracker_id_before/after
        # across all 14 gaps, including trivial 1-frame gaps, on a run
        # with real injected motion). Callers must pass each frame's
        # ACTUAL observed GT box positions (post-warp, if applicable) to
        # accumulate_frame() instead.
        self.gaps = find_gt_gaps(gt_tracks)

        self.assignments: Dict[int, Dict[int, int]] = {}   # frame -> {gt_id: tracker_id}

        # General ID-switch count across CONTINUOUSLY visible GT tracks
        # (i.e. NOT spanning a genuine gap) -- a separate robustness stat
        # from occlusion-specific recovery, reported alongside it.
        self.id_switches = 0
        self._prev_assignment: Dict[int, int] = {}
        self._prev_frame: Optional[int] = None

    def accumulate_frame(self, frame_idx: int, tracker_boxes: List[dict],
                          gt_boxes_this_frame: Dict[int, list]):
        """
        tracker_boxes: list of {"id": tracker_id, "box": [x,y,w,h]} for
        tracks the tracker itself reported as matched (visible) this frame.
        gt_boxes_this_frame: {gt_id: [x,y,w,h]} -- the ACTUAL ground-truth
        box positions as observed this frame (i.e. already warped by any
        synthetic motion injection, exactly as the tracker itself saw them).
        """
        assignment: Dict[int, int] = {}

        if gt_boxes_this_frame and tracker_boxes:
            gt_ids = list(gt_boxes_this_frame.keys())
            cost = np.ones((len(gt_ids), len(tracker_boxes)))
            for i, gid in enumerate(gt_ids):
                for j, tb in enumerate(tracker_boxes):
                    iou_val = iou(np.array(gt_boxes_this_frame[gid]), np.array(tb["box"]))
                    cost[i, j] = 1.0 - iou_val
            row_idx, col_idx = linear_sum_assignment(cost)
            for r, c in zip(row_idx, col_idx):
                if cost[r, c] <= (1.0 - self.iou_threshold):
                    assignment[gt_ids[r]] = tracker_boxes[c]["id"]

        self.assignments[frame_idx] = assignment

        if self._prev_frame is not None and frame_idx - self._prev_frame == 1:
            for gid, tid in assignment.items():
                prev_tid = self._prev_assignment.get(gid)
                if prev_tid is not None and prev_tid != tid:
                    self.id_switches += 1

        self._prev_assignment = assignment
        self._prev_frame = frame_idx

    def finalize(self) -> dict:
        recovered_gaps, lost_gaps = [], []
        buckets = {"short_1_5": [0, 0], "medium_6_15": [0, 0], "long_16_plus": [0, 0]}
        gap_details = []

        for gap in self.gaps:
            tid_before = self.assignments.get(gap["start"] - 1, {}).get(gap["gt_id"])
            tid_after = self.assignments.get(gap["end"] + 1, {}).get(gap["gt_id"])
            is_recovered = (tid_before is not None and tid_before == tid_after)

            key = ("short_1_5" if gap["duration"] <= 5 else
                   "medium_6_15" if gap["duration"] <= 15 else "long_16_plus")
            buckets[key][1] += 1
            if is_recovered:
                buckets[key][0] += 1
                recovered_gaps.append(gap)
            else:
                lost_gaps.append(gap)

            # Diagnostic detail per gap -- distinguishes "no tracker
            # assignment found at the boundary frame at all" (None) from
            # "an assignment existed but the identity changed" (two
            # different real IDs), since those imply different root
            # causes (missing association entirely vs. genuine ID switch).
            gap_details.append(dict(
                gt_id=gap["gt_id"], start=gap["start"], end=gap["end"],
                duration=gap["duration"], tracker_id_before=tid_before,
                tracker_id_after=tid_after, recovered=is_recovered,
            ))

        n_total = len(self.gaps)
        return dict(
            n_genuine_gt_gaps=n_total,
            n_recovered=len(recovered_gaps),
            n_lost=len(lost_gaps),
            recovery_rate=(len(recovered_gaps) / n_total) if n_total else None,
            recovery_by_duration_bucket={
                k: dict(recovered=v[0], total=v[1], rate=(v[0] / v[1] if v[1] else None))
                for k, v in buckets.items()
            },
            general_id_switches=self.id_switches,
            gap_details=gap_details,
            note=("Anchored directly to genuine ground-truth occlusion gaps "
                  "(a gap in the GT track's own annotated frame sequence), "
                  "NOT the tracker's internal occlusion bookkeeping. This is "
                  "the trustworthy number for the raw-vs-homography ablation "
                  "-- compare this block between runs, not n_occlusion_events."),
        )
