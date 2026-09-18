"""
Ground-truth-as-detector.

Bypasses the placeholder frame-differencing detector entirely by feeding
the dataset's own annotated bounding boxes to the tracker as if they were
detections, with confidence 1.0. This exists specifically to ISOLATE the
tracker/association/geometry-compensation mechanism from detector
quality when evaluating occlusion recovery -- with the placeholder
detector, the overwhelming majority of "occlusion events" turned out to
be detector dropout rather than genuine structural occlusion (see
analyze_gt_occlusions.py, which found only 14 genuine occlusion gaps in
a car_001 sequence whose placeholder-detector-driven runs reported
850+ "occlusion events").

When this detector is used, an occlusion event in the tracker's own
metrics corresponds directly to a genuine gap in the ground truth --
exactly the phenomenon the research is about -- rather than a mixture of
that and detector noise.

This is NOT a substitute for a real neural detector long-term (Phase 3 of
the roadmap) -- it deliberately "cheats" by using the answer key, so
absolute accuracy numbers from a ground-truth-detector run are not
representative of real-world deployed performance. Its value is narrow
and specific: a clean ablation of the tracking/geometry mechanism itself,
decoupled from detection quality, while a real detector is still pending.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List

from .simple_detector import Detection


class GroundTruthDetector:
    def __init__(self, tracks: Dict[int, list]):
        """
        tracks: the value of annotations["tracks"] from a dataset loader's
        load_annotations(), i.e. {track_id: [{"frame": int,
        "box": [x, y, w, h]}, ...]}.
        """
        self._by_frame: Dict[int, List[dict]] = {}
        for tid, entries in tracks.items():
            for e in entries:
                self._by_frame.setdefault(e["frame"], []).append(
                    dict(track_id=tid, box=e["box"])
                )
        self.n_frames_with_annotations = len(self._by_frame)

    def detect_for_frame(self, frame_idx: int) -> List[Detection]:
        entries = self._by_frame.get(frame_idx, [])
        dets = []
        for e in entries:
            x, y, w, h = e["box"]
            cx, cy = x + w / 2.0, y + h / 2.0
            dets.append(Detection(
                box=np.array([x, y, w, h], dtype=np.float64),
                center=np.array([cx, cy], dtype=np.float64),
                confidence=1.0,
                source_id=e["track_id"],
            ))
        return dets
