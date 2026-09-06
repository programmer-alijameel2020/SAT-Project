"""
Every experiment run gets its own folder under EXPERIMENTS_ROOT, named by
its experiment_uid, so the frontend can browse past runs as a plain file
system (spec requirement #4).

Layout created for each run:

  experiments/<uid>/
    config.json          full parameter set as submitted from the frontend
    results.json          tracker.summary() + occlusion event list
    homography_log.json  per-frame homography quality diagnostics
    frames/               sampled overlay frames (jpg) for visual review
    figures/               generated plots (recovery-by-duration bar chart, etc.)
    events.jsonl           full raw pipeline event stream (one JSON object/line)
"""
from __future__ import annotations

import os
import json
import cv2

EXPERIMENTS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "..", "experiments"
)
EXPERIMENTS_ROOT = os.path.abspath(EXPERIMENTS_ROOT)


class ExperimentStore:
    def __init__(self, experiment_uid: str):
        self.uid = experiment_uid
        self.dir = os.path.join(EXPERIMENTS_ROOT, experiment_uid)
        self.frames_dir = os.path.join(self.dir, "frames")
        self.figures_dir = os.path.join(self.dir, "figures")
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        self._events_path = os.path.join(self.dir, "events.jsonl")

    def save_config(self, config: dict):
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2, default=str)

    def save_results(self, results: dict):
        with open(os.path.join(self.dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

    def save_homography_log(self, log: list):
        with open(os.path.join(self.dir, "homography_log.json"), "w") as f:
            json.dump(log, f, indent=2, default=str)

    def append_event(self, event: dict):
        with open(self._events_path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def save_overlay_frame(self, frame_idx: int, frame_bgr, tracks_snapshot: list):
        img = frame_bgr.copy()
        for tr in tracks_snapshot:
            if tr["box"] is None:
                continue
            x, y, w, h = [int(v) for v in tr["box"]]
            color = (0, 220, 0) if tr["visible"] else (0, 165, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            label = f"#{tr['id']}" + ("" if tr["visible"] else f" occ:{tr['occlusion_age']}")
            cv2.putText(img, label, (x, max(0, y - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, color, 1, cv2.LINE_AA)
        out_path = os.path.join(self.frames_dir, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(out_path, img)
        return out_path

    def list_files(self) -> dict:
        def walk(d):
            out = []
            for root, _, files in os.walk(d):
                for f in sorted(files):
                    full = os.path.join(root, f)
                    out.append(dict(
                        name=f, rel_path=os.path.relpath(full, self.dir),
                        size_bytes=os.path.getsize(full),
                    ))
            return out
        return dict(root=self.dir, files=walk(self.dir))
