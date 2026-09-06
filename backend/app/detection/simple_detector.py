"""
Placeholder host detector for the prototype.

The research plan is explicit that the detector, appearance model, and
association solver are CONTROLLED components, not the claimed novelty
(Section 4: "The detector ... are not the claimed novelty; they are
controlled components used to measure the contribution of the proposed
representation.").

For the first prototype we use a lightweight frame-differencing + contour
blob detector so the full pipeline (detection -> geometry -> prediction ->
association -> metrics) runs end-to-end today, on both VISO and SDM-Car
frames, without requiring a trained CNN detector. This module is a
straightforward drop-in point: swap `SimpleMotionDetector.detect()` for a
YOLOv8 / CenterNet-style detector in Phase 3 of the roadmap without
touching anything downstream.
"""
from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List


@dataclass
class Detection:
    box: np.ndarray       # [x, y, w, h] top-left + size
    center: np.ndarray    # [cx, cy]
    confidence: float
    appearance: np.ndarray = None  # optional descriptor (mean color patch)


class SimpleMotionDetector:
    """Background-subtraction + accumulative frame-differencing detector,
    tuned for small, dim, dense moving-object scenes typical of satellite
    video (roughly following the accumulative multi-frame-differencing
    idea used as the VISO benchmark's own baseline)."""

    def __init__(
        self,
        min_area: int = 4,
        max_area: int = 2500,
        diff_thresh: int = 18,
        history_frames: int = 3,
        max_detections_per_frame: int = 250,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.diff_thresh = diff_thresh
        self.history_frames = history_frames
        self.max_detections_per_frame = max_detections_per_frame
        self._buffer: List[np.ndarray] = []

    def reset(self):
        self._buffer = []

    def detect(self, frame_gray: np.ndarray) -> List[Detection]:
        self._buffer.append(frame_gray.astype(np.float32))
        if len(self._buffer) > self.history_frames:
            self._buffer.pop(0)
        if len(self._buffer) < 2:
            return []

        acc = np.zeros_like(frame_gray, dtype=np.float32)
        for i in range(1, len(self._buffer)):
            acc += np.abs(self._buffer[i] - self._buffer[i - 1])
        acc /= max(1, len(self._buffer) - 1)

        _, mask = cv2.threshold(acc.astype(np.uint8), self.diff_thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w / 2.0, y + h / 2.0
            conf = float(min(1.0, area / (self.max_area * 0.1)))
            detections.append(Detection(
                box=np.array([x, y, w, h], dtype=np.float64),
                center=np.array([cx, cy], dtype=np.float64),
                confidence=max(conf, 0.15),
            ))

        # Hard safety valve: regardless of how min_area/max_area/diff_thresh
        # are tuned, association cost grows with n_tracks * n_detections.
        # On real (noisy) satellite frames a poorly-tuned threshold can
        # produce thousands of spurious blobs and make the run appear
        # to hang rather than crash. Keep only the most confident
        # detections up to a hard cap so a bad threshold degrades
        # tracking QUALITY, not the ability to finish a run.
        if len(detections) > self.max_detections_per_frame:
            detections.sort(key=lambda d: d.confidence, reverse=True)
            detections = detections[:self.max_detections_per_frame]

        return detections
