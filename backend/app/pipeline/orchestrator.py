"""
The experiment orchestrator. This is what turns a parameter set from the
frontend into: dataset scan -> sequence selection -> video-to-frames ->
per-frame (homography -> decomposition -> prediction -> association) ->
metrics -> saved results, with a live event emitted at every step (spec
requirement #1) and every artifact written into the experiment's own
output folder (spec requirement #4).
"""
from __future__ import annotations

import asyncio
import datetime
import traceback
from typing import Callable, Awaitable, Dict

from ..datasets.viso import VISOLoader
from ..datasets.sdmcar import SDMCarLoader
from ..datasets.demo_synth import DemoSyntheticLoader
from ..detection.simple_detector import SimpleMotionDetector
from ..tracking.tracker import MotionGroundedTracker
from ..storage.experiment_store import ExperimentStore
from ..db.database import SessionLocal
from ..db.models import Experiment
from ..utils.device import detect_device
from . import events as EV

EmitFn = Callable[[dict], Awaitable[None]]

LOADERS = {
    "VISO": VISOLoader,
    "SDM-Car": SDMCarLoader,
    "demo": DemoSyntheticLoader,
}

# One cancellation flag per running experiment. A run checks its own flag
# once per frame and stops cleanly (marks status="cancelled", writes
# whatever results have accumulated so far) rather than being killed
# mid-write. Cleared automatically when the run ends either way.
_cancel_flags: Dict[str, asyncio.Event] = {}


def request_cancel(experiment_uid: str) -> bool:
    """Called by the API route. Returns True if a running experiment was
    found and signalled, False if there was nothing to cancel."""
    flag = _cancel_flags.get(experiment_uid)
    if flag is None:
        return False
    flag.set()
    return True


async def run_experiment(experiment_uid: str, params: dict, emit: EmitFn):
    seq_no = 0

    async def send(stage: str, **payload):
        nonlocal seq_no
        ev = EV.make_event(stage, seq_no, **payload)
        seq_no += 1
        await emit(ev)
        store.append_event(ev)
        await asyncio.sleep(0)  # yield so the socket actually flushes

    store = ExperimentStore(experiment_uid)
    store.save_config(params)
    db = SessionLocal()
    exp_row = db.query(Experiment).filter_by(experiment_uid=experiment_uid).first()
    cancel_flag = asyncio.Event()
    _cancel_flags[experiment_uid] = cancel_flag

    try:
        exp_row.status = "running"
        exp_row.started_at = datetime.datetime.utcnow()
        db.commit()

        await send(EV.STAGE_INIT, message="Experiment initialized", params=params)

        device_info = detect_device()
        await send(EV.STAGE_INIT,
                   message=f"Compute device: {device_info.device}"
                            + (f" ({device_info.gpu_name})" if device_info.gpu_name else ""),
                   device=device_info.to_dict())

        # ---- dataset scan --------------------------------------------------
        dataset_source = params["dataset_source"]
        loader_cls = LOADERS[dataset_source]
        root_dir = params.get("dataset_root", "")
        loader = loader_cls(root_dir) if dataset_source != "demo" else loader_cls()

        await send(EV.STAGE_DATASET_SCAN, message=f"Scanning {dataset_source} at '{root_dir}'")
        sequences = loader.list_sequences()
        await send(EV.STAGE_DATASET_SCAN, message=f"Found {len(sequences)} sequence(s)",
                   n_sequences=len(sequences),
                   sequences=[dict(name=s.name, n_frames=s.n_frames,
                                    has_annotations=s.has_annotations) for s in sequences[:50]])

        if not sequences:
            raise RuntimeError(
                f"No sequences found under '{root_dir}' for dataset '{dataset_source}'. "
                f"Check the dataset_root path."
            )

        seq_name = params.get("sequence_name")
        sequence = next((s for s in sequences if s.name == seq_name), sequences[0])
        if seq_name and sequence.name != seq_name:
            await send(EV.STAGE_SEQUENCE_SELECTED,
                       message=(f"Requested sequence '{seq_name}' not found in this scan -- "
                                f"falling back to '{sequence.name}'. Re-run 'Scan for sequences' "
                                f"on the setup screen after switching dataset source to avoid this."),
                       sequence_name=sequence.name, n_frames=sequence.n_frames,
                       has_annotations=sequence.has_annotations, fallback_used=True)
        else:
            await send(EV.STAGE_SEQUENCE_SELECTED, sequence_name=sequence.name,
                       n_frames=sequence.n_frames, has_annotations=sequence.has_annotations,
                       fallback_used=False)

        # ---- video -> frame array -------------------------------------------
        max_frames = params.get("max_frames") or sequence.n_frames
        await send(EV.STAGE_VIDEO_TO_FRAMES,
                   message=f"Decoding sequence into frame array (max {max_frames} frames)")

        # ---- build tracker ---------------------------------------------------
        detector = SimpleMotionDetector(
            min_area=params.get("det_min_area", 4),
            max_area=params.get("det_max_area", 2500),
            diff_thresh=params.get("det_diff_thresh", 18),
            max_detections_per_frame=params.get("det_max_detections", 250),
        )
        tracker = MotionGroundedTracker(
            trajectory_mode=params.get("trajectory_mode", "homography"),
            window_length=params.get("window_length", 8),
            kmax_occlusion=params.get("kmax_occlusion", 20),
            min_hits_to_confirm=params.get("min_hits_to_confirm", 3),
            predictor_model=params.get("predictor_model", "constant_velocity"),
            homography_kwargs=dict(
                min_inliers=params.get("min_inliers", 15),
                min_inlier_ratio=params.get("min_inlier_ratio", 0.35),
                max_reproj_error=params.get("max_reproj_error", 3.0),
            ),
        )

        save_every = max(1, params.get("save_frame_every", 5))
        step_delay = params.get("step_delay_ms", 0) / 1000.0

        prev_frame = None
        n_processed = 0
        cancelled = False
        for frame_idx, frame_bgr in loader.frame_iterator(sequence, max_frames=max_frames):
            if cancel_flag.is_set():
                cancelled = True
                break

            await send(EV.STAGE_FRAME_LOADED, frame=frame_idx,
                       message=f"Frame {frame_idx} loaded", shape=list(frame_bgr.shape))

            import cv2
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            dets = detector.detect(gray)
            await send(EV.STAGE_DETECTION, frame=frame_idx, n_detections=len(dets))

            result = tracker.step(frame_bgr, prev_frame, dets)

            await send(EV.STAGE_HOMOGRAPHY, frame=frame_idx,
                       mode=result["homography_mode"], quality=result["homography_quality"])
            await send(EV.STAGE_DECOMPOSITION, frame=frame_idx,
                       message="Target histories reprojected to compensated coordinates")
            await send(EV.STAGE_PREDICTION, frame=frame_idx,
                       message="Residual predictor + Kalman forecast updated")
            await send(EV.STAGE_ASSOCIATION, frame=frame_idx,
                       n_matched=result["n_matched"], n_new=result["n_new"],
                       n_occluded=result["n_occluded"], n_dead=result["n_dead"],
                       tracks=result["tracks_snapshot"])

            if frame_idx % save_every == 0:
                store.save_overlay_frame(frame_idx, frame_bgr, result["tracks_snapshot"])

            # NOTE: 'frame' must not also be passed via **result, since
            # result already contains a 'frame' key -- passing both would
            # raise "got multiple values for keyword argument 'frame'".
            await send(EV.STAGE_FRAME_COMPLETE, **{
                k: v for k, v in result.items() if k != "tracks_snapshot"
            })

            prev_frame = frame_bgr
            n_processed += 1
            if step_delay > 0:
                await asyncio.sleep(step_delay)

        # ---- metrics -----------------------------------------------------------
        if cancelled:
            await send(EV.STAGE_METRICS,
                       message=f"Run cancelled after {n_processed} frame(s) -- "
                                f"computing metrics over what was processed so far")
        else:
            await send(EV.STAGE_METRICS, message="Computing occlusion-recovery and efficiency metrics")
        summary = tracker.summary()
        summary["n_frames_processed"] = n_processed
        summary["cancelled"] = cancelled
        summary["compute_device"] = device_info.to_dict()

        await send(EV.STAGE_SAVING, message="Writing results to experiment folder")
        store.save_results(summary)
        store.save_homography_log(tracker.homography_log)

        exp_row.status = "cancelled" if cancelled else "completed"
        exp_row.results_json = summary
        exp_row.finished_at = datetime.datetime.utcnow()
        db.commit()

        await send(EV.STAGE_CANCELLED if cancelled else EV.STAGE_COMPLETED,
                   summary=summary, output_dir=store.dir)

    except Exception as e:
        tb = traceback.format_exc()
        if exp_row is not None:
            exp_row.status = "failed"
            exp_row.error_message = f"{e}\n{tb}"
            exp_row.finished_at = datetime.datetime.utcnow()
            db.commit()
        await send(EV.STAGE_FAILED, error=str(e), traceback=tb)
    finally:
        _cancel_flags.pop(experiment_uid, None)
        db.close()
