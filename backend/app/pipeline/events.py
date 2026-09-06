"""Canonical stage names for the live pipeline event stream. The frontend
dashboard keys its animation state off these exact strings."""

STAGE_INIT = "initializing"
STAGE_DATASET_SCAN = "dataset_scanning"
STAGE_SEQUENCE_SELECTED = "sequence_selected"
STAGE_VIDEO_TO_FRAMES = "video_to_frames"
STAGE_FRAME_LOADED = "frame_loaded"
STAGE_DETECTION = "detection"
STAGE_HOMOGRAPHY = "homography_estimation"
STAGE_DECOMPOSITION = "motion_decomposition"
STAGE_PREDICTION = "residual_prediction"
STAGE_ASSOCIATION = "association"
STAGE_FRAME_COMPLETE = "frame_complete"
STAGE_METRICS = "computing_metrics"
STAGE_SAVING = "saving_results"
STAGE_COMPLETED = "completed"
STAGE_FAILED = "failed"
STAGE_CANCELLED = "cancelled"


def make_event(stage: str, seq: int, **payload) -> dict:
    return dict(stage=stage, seq=seq, **payload)
