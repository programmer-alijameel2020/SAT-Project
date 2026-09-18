"""
SDM-Car dataset loader.

CORRECTED based on the actual downloaded layout (confirmed via
inspect_dataset.py against a real extraction): SDM-Car ships as paired
video + CSV files directly in each split folder, NOT per-frame image
folders with label files as originally assumed:

    train/1-1.avi        train/1-1-gt.csv
    train/10-1.avi        train/10-1-gt.csv
    ...

Each sequence is one .avi file; its ground truth is the CSV with the same
basename plus "-gt.csv" suffix.

NOTE ON CSV COLUMN FORMAT: not yet verified against a real file's header.
The parser below tries, in order: (1) a header row with recognizable
column names, (2) a fully-numeric first row treated as headerless with
the common frame,id,x,y,w,h ordering. If a file matches neither pattern,
its rows are kept raw under "unparsed_rows" rather than silently
mis-parsed -- annotations aren't used by the running pipeline yet (only
frame iteration is), so this fails safe rather than blocking anything.
Paste the first few lines of one real *-gt.csv and this can be locked
down exactly.
"""
from __future__ import annotations

import os
import csv
import cv2
from typing import List, Optional

from .base import BaseDatasetLoader, SequenceInfo
from .viso import VIDEO_EXTS

# Column name variants we'll recognize if a header row is present.
_COL_ALIASES = {
    "frame": {"frame", "frame_id", "frameid", "frame_idx", "fr"},
    "id": {"id", "track_id", "trackid", "object_id", "obj_id"},
    "x": {"x", "bb_left", "left", "x1"},
    "y": {"y", "bb_top", "top", "y1"},
    "w": {"w", "width", "bbox_w", "bb_width"},
    "h": {"h", "height", "bbox_h", "bb_height"},
}


class SDMCarLoader(BaseDatasetLoader):
    source_name = "SDM-Car"

    def list_sequences(self) -> List[SequenceInfo]:
        sequences = []
        if not os.path.isdir(self.root_dir):
            return sequences

        for dirpath, _, filenames in os.walk(self.root_dir):
            for fn in filenames:
                if not fn.lower().endswith(VIDEO_EXTS):
                    continue
                video_path = os.path.join(dirpath, fn)
                cap = cv2.VideoCapture(video_path)
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if n_frames <= 0:
                    continue
                seq_id = os.path.splitext(fn)[0]              # e.g. "1-1"
                gt_csv = self._gt_csv_path(dirpath, seq_id)
                sequences.append(SequenceInfo(
                    name=seq_id, path=video_path, n_frames=n_frames,
                    has_annotations=gt_csv is not None, source=self.source_name,
                ))
        return sequences

    def _gt_csv_path(self, dirpath: str, seq_id: str) -> Optional[str]:
        cand = os.path.join(dirpath, f"{seq_id}-gt.csv")
        return cand if os.path.isfile(cand) else None

    def frame_iterator(self, sequence: SequenceInfo, max_frames: Optional[int] = None):
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
        seq_id = os.path.splitext(os.path.basename(sequence.path))[0]
        gt_csv = self._gt_csv_path(os.path.dirname(sequence.path), seq_id)
        if gt_csv is None:
            return None

        with open(gt_csv, newline="") as fh:
            reader = list(csv.reader(fh))
        if not reader:
            return None

        header_row = [c.strip().lower() for c in reader[0]]
        col_idx = {}
        for canon, aliases in _COL_ALIASES.items():
            for i, cell in enumerate(header_row):
                if cell in aliases:
                    col_idx[canon] = i
                    break

        has_header = len(col_idx) >= 4   # recognized enough columns
        data_rows = reader[1:] if has_header else reader

        if not has_header:
            # fall back to the common headerless MOT-style ordering,
            # only if the first row is fully numeric (else this file's
            # format is genuinely unknown -- return it raw)
            def is_numeric_row(row):
                try:
                    [float(c) for c in row]
                    return True
                except ValueError:
                    return False

            if not data_rows or not is_numeric_row(data_rows[0]):
                return {"tracks": {}, "unparsed_rows": reader[:20],
                        "note": "Unrecognized CSV format -- verify column layout."}
            col_idx = {"frame": 0, "id": 1, "x": 2, "y": 3, "w": 4, "h": 5}

        tracks: dict = {}
        for row in data_rows:
            if len(row) <= max(col_idx.values()):
                continue
            try:
                frame = int(float(row[col_idx["frame"]]))
                tid = int(float(row[col_idx["id"]])) if "id" in col_idx else -1
                x = float(row[col_idx["x"]])
                y = float(row[col_idx["y"]])
                w = float(row[col_idx["w"]])
                h = float(row[col_idx["h"]])
            except (ValueError, KeyError):
                continue
            tracks.setdefault(tid, []).append(dict(frame=frame, box=[x, y, w, h]))

        return {"tracks": tracks} if tracks else None
