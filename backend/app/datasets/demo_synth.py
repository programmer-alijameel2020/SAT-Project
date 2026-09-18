"""
Synthetic demo sequence generator.

Not one of the two research datasets -- this exists purely so the whole
system (frontend -> backend -> geometry -> tracker -> DB -> results
browser) can be exercised end-to-end without waiting on real dataset
downloads, and to give a fast, reproducible regression check for the
pipeline itself.

It simulates exactly the phenomenon the research targets: a moving
"platform" (simulated by panning/zooming the sampled background) plus a
handful of small targets moving independently and passing behind a
static occluder for a few frames.
"""
from __future__ import annotations

import os
import numpy as np
import cv2
from typing import List, Optional

from .base import BaseDatasetLoader, SequenceInfo


class DemoSyntheticLoader(BaseDatasetLoader):
    source_name = "demo"

    def __init__(self, root_dir: str = None, width=480, height=320,
                 n_frames=120, n_targets=6, seed=42):
        super().__init__(root_dir or "demo://synthetic")
        self.W, self.H = width, height
        self.n_frames = n_frames
        self.n_targets = n_targets
        self.seed = seed

    def list_sequences(self) -> List[SequenceInfo]:
        return [SequenceInfo(
            name="synthetic_demo_01", path="demo://synthetic_demo_01",
            n_frames=self.n_frames, has_annotations=True, source=self.source_name,
        )]

    def _make_background(self, rng):
        bg = (rng.integers(60, 90, size=(self.H * 3, self.W * 3, 3))).astype(np.uint8)
        # a few "roads"/structures for texture that ORB can key on
        for _ in range(40):
            x1, y1 = rng.integers(0, self.W * 3, 2)
            x2, y2 = x1 + rng.integers(-80, 80), y1 + rng.integers(-80, 80)
            cv2.line(bg, (x1, y1), (x2, y2), tuple(int(v) for v in rng.integers(90, 140, 3)), 2)
        for _ in range(60):
            x, y = rng.integers(0, self.W * 3, 2)
            cv2.circle(bg, (x, y), int(rng.integers(2, 6)),
                       tuple(int(v) for v in rng.integers(100, 160, 3)), -1)
        return bg

    def _generate(self):
        rng = np.random.default_rng(self.seed)
        bg = self._make_background(rng)
        # platform trajectory: smooth pan + slight zoom drift (simulates orbital motion)
        cx, cy = self.W * 1.0, self.H * 1.0
        vcx, vcy = rng.uniform(0.6, 1.4), rng.uniform(-0.5, 0.5)
        occluder = dict(x=int(self.W * 0.55), y=int(self.H * 0.35),
                         w=int(self.W * 0.12), h=int(self.H * 0.35))

        targets = []
        for i in range(self.n_targets):
            targets.append(dict(
                id=i + 1,
                pos=np.array([rng.uniform(0.1, 0.9) * self.W, rng.uniform(0.1, 0.9) * self.H]),
                vel=rng.uniform(-1.5, 1.5, size=2),
                size=int(rng.integers(4, 9)),
            ))

        frames, gt = [], {}
        for t in range(self.n_frames):
            cx += vcx
            cy += vcy
            x0, y0 = int(cx), int(cy)
            crop = bg[y0:y0 + self.H, x0:x0 + self.W].copy()
            if crop.shape[0] != self.H or crop.shape[1] != self.W:
                crop = cv2.resize(crop, (self.W, self.H))

            for tr in targets:
                tr["pos"] += tr["vel"]
                if tr["pos"][0] < 0 or tr["pos"][0] > self.W:
                    tr["vel"][0] *= -1
                if tr["pos"][1] < 0 or tr["pos"][1] > self.H:
                    tr["vel"][1] *= -1
                px, py = tr["pos"]
                s = tr["size"]
                behind_occluder = (
                    occluder["x"] <= px <= occluder["x"] + occluder["w"]
                    and occluder["y"] <= py <= occluder["y"] + occluder["h"]
                )
                gt.setdefault(tr["id"], []).append(dict(
                    frame=t, box=[float(px - s / 2), float(py - s / 2), float(s), float(s)],
                    occluded=bool(behind_occluder),
                ))
                if not behind_occluder:
                    cv2.rectangle(crop, (int(px - s / 2), int(py - s / 2)),
                                  (int(px + s / 2), int(py + s / 2)), (30, 220, 30), -1)

            cv2.rectangle(crop, (occluder["x"], occluder["y"]),
                          (occluder["x"] + occluder["w"], occluder["y"] + occluder["h"]),
                          (70, 70, 70), -1)
            frames.append(crop)
        return frames, gt

    def frame_iterator(self, sequence: SequenceInfo, max_frames: Optional[int] = None):
        frames, _ = self._generate()
        n = min(len(frames), max_frames) if max_frames else len(frames)
        for i in range(n):
            yield i, frames[i]

    def load_annotations(self, sequence: SequenceInfo) -> Optional[dict]:
        _, gt = self._generate()
        return {"tracks": gt}
