"""Common interface all dataset loaders implement, so the pipeline
orchestrator can treat VISO, SDM-Car, and the synthetic demo dataset
identically (spec requirement: "must be able to handle both two datasets
according to the user selection")."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class SequenceInfo:
    name: str
    path: str
    n_frames: int
    has_annotations: bool
    source: str          # "VISO" | "SDM-Car" | "demo"


class BaseDatasetLoader:
    """Interface: list available sequences, iterate frames as BGR arrays."""

    source_name = "base"

    # Frame numbering convention used by this dataset's ground-truth
    # annotation files, relative to frame_iterator()'s 0-indexed output.
    # VISO's gt.txt starts at frame=1 (matching its 000001.jpg filenames)
    # while frame_iterator's enumerate() yields the first frame as 0 --
    # without correcting for this, GroundTruthDetector.detect_for_frame()
    # silently reads each frame's ground truth from one frame ahead of
    # the actual image. Confirmed via SDM-Car's gt.csv, which starts at
    # frame=0 and needs no correction -- this is dataset-specific, not
    # universal, hence a per-loader class attribute rather than a global
    # constant.
    gt_frame_index_base = 0

    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def list_sequences(self) -> List[SequenceInfo]:
        raise NotImplementedError

    def frame_iterator(self, sequence: SequenceInfo, max_frames: Optional[int] = None):
        """Yields (frame_idx, frame_bgr) tuples."""
        raise NotImplementedError

    def load_annotations(self, sequence: SequenceInfo) -> Optional[dict]:
        """Returns ground-truth tracks if available, else None."""
        return None
