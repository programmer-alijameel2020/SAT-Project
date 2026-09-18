"""
VISO dataset loader.

CONFIRMED layout (via inspect_dataset.py against a real extraction):

    VISO/mot/mot/<category>/<seq_id>/img/000001.jpg ...
    VISO/mot/mot/<category>/<seq_id>/gt/gt.txt (+ gt_centroid.txt)

where <category> in {car, plane, ship, train} (VISO's four moving-object
classes -- "train" here means the vehicle, not a data split) and <seq_id>
is a zero-padded sequence number. Point `dataset_root` at
".../VISO/mot/mot" to scan only real MOT sequences and skip the
Detection_coco_format / Detection_voc_format / SOT siblings, which are not
MOT-track data and would otherwise get picked up as false "sequences" if
you pointed dataset_root at the VISO root itself.

This loader still scans generically (os.walk for any folder with >=5
images) rather than hard-coding the exact depth, since the VISO releases
have varied in exact nesting across mirrors -- but sequence *naming* is
now derived from the parent folders so the dropdown shows "car_001"
instead of a raw path dump.
"""
from __future__ import annotations

import os
import glob
import cv2
import numpy as np
from typing import List, Optional

from .base import BaseDatasetLoader, SequenceInfo

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv")


class VISOLoader(BaseDatasetLoader):
    source_name = "VISO"
    gt_frame_index_base = 1   # gt.txt frame numbering starts at 1, not 0

    def list_sequences(self) -> List[SequenceInfo]:
        sequences = []
        if not os.path.isdir(self.root_dir):
            return sequences

        # Case A: sub-folders that are themselves image sequences
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            imgs = sorted([f for f in filenames if f.lower().endswith(IMG_EXTS)])
            if len(imgs) >= 5:
                gt_path = self._find_gt(dirpath)
                seq_name = self._pretty_name(dirpath)
                sequences.append(SequenceInfo(
                    name=seq_name, path=dirpath, n_frames=len(imgs),
                    has_annotations=gt_path is not None, source=self.source_name,
                ))

        # Case B: raw video files directly under root
        for f in glob.glob(os.path.join(self.root_dir, "**", "*"), recursive=True):
            if f.lower().endswith(VIDEO_EXTS):
                cap = cv2.VideoCapture(f)
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if n > 0:
                    sequences.append(SequenceInfo(
                        name=os.path.splitext(os.path.basename(f))[0], path=f,
                        n_frames=n, has_annotations=False, source=self.source_name,
                    ))
        return sequences

    def _pretty_name(self, dirpath: str) -> str:
        """Derive a readable sequence name from the folder path, e.g.
        '.../VISO/mot/mot/car/001/img' -> 'car_001'. Strips a trailing
        'img'/'images' component (the images live one level below the
        sequence's own natural name), then joins the parent + this
        folder's names."""
        rel = os.path.relpath(dirpath, self.root_dir)
        parts = [p for p in rel.split(os.sep) if p not in (".", "")]
        if parts and parts[-1].lower() in ("img", "images", "imgs"):
            parts = parts[:-1]
        if not parts:
            return os.path.basename(dirpath) or "sequence"
        # keep at most the last two path components for readability
        # (category + sequence id), e.g. "car_001" rather than the full path
        return "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]

    def _find_gt(self, dirpath: str) -> Optional[str]:
        for cand in ("gt.txt", "gt/gt.txt", "../gt/gt.txt", "annotations.txt"):
            p = os.path.join(dirpath, cand)
            if os.path.isfile(p):
                return p
        return None

    def frame_iterator(self, sequence: SequenceInfo, max_frames: Optional[int] = None):
        if os.path.isdir(sequence.path):
            imgs = sorted([f for f in os.listdir(sequence.path) if f.lower().endswith(IMG_EXTS)])
            if max_frames:
                imgs = imgs[:max_frames]
            for i, fn in enumerate(imgs):
                frame = cv2.imread(os.path.join(sequence.path, fn))
                if frame is not None:
                    yield i, frame
        else:
            cap = cv2.VideoCapture(sequence.path)
            i = 0
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames and i >= max_frames):
                    break
                yield i, frame
                i += 1
            cap.release()

    def load_annotations(self, sequence: SequenceInfo) -> Optional[dict]:
        gt_path = self._find_gt(sequence.path) if os.path.isdir(sequence.path) else None
        if gt_path is None:
            return None
        tracks = {}
        with open(gt_path, "r") as fh:
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                frame, tid, x, y, w, h = map(float, parts[:6])
                tracks.setdefault(int(tid), []).append(
                    dict(frame=int(frame), box=[x, y, w, h])
                )
        return {"tracks": tracks} if tracks else None
