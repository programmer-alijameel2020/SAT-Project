"""
The experiment orchestrator. This is what turns a parameter set from the
frontend into: dataset scan -> sequence selection -> video-to-frames ->
per-frame (homography -> decomposition -> prediction -> association) ->
metrics -> saved results, with a live event emitted at every step (spec
requirement #1) and every artifact written into the experiment's own
output folder (spec requirement #4).

Additions for the final campaign:
  * provenance (git commit, dirty flag, library versions) is written into
    config.json of every run, so every reported number is traceable to one
    code version;
  * oracle mode receives the TRUE platform transform of the injected drift,
    G_{t-1->t} = W_t W_{t-1}^{-1}, where W_t is the injector's cumulative
    warp H_{0->t}; it requires inject_synthetic_motion=True;
  * the OpenCV RNG is seeded identically for every run and mode.
"""
from __future__ import annotations

import asyncio
import datetime
import platform
import subprocess
import traceback
from pathlib import Path

import cv2
import numpy as np
from typing import Callable, Awaitable, Dict

from ..datasets.viso import VISOLoader
from ..datasets.sdmcar import SDMCarLoader
from ..datasets.demo_synth import DemoSyntheticLoader
from ..detection.simple_detector import SimpleMotionDetector
from ..detection.gt_detector import GroundTruthDetector
from ..evaluation.gt_evaluator import GTAnchoredEvaluator
from ..datasets.motion_injection import generate_translation_walk, warp_frame, warp_box
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


def _provenance() -> dict:
    """Code version + environment for config.json. Never raises."""
    here = Path(__file__).resolve().parent

    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=here, capture_output=True,
                                  text=True, timeout=10).stdout.strip() or None
        except Exception:
            return None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return dict(
        git_commit=commit,
        git_dirty=(bool(status) if commit else None),
        python=platform.python_version(),
        opencv=cv2.__version__,
        numpy=np.__version__,
        machine=platform.processor() or platform.machine(),
        recorded_at_utc=datetime.datetime.utcnow().isoformat(timespec="seconds"),
    )


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
    params = dict(params)
    params["provenance"] = _provenance()
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

        # ---- synthetic platform-motion injection (optional) ------------------
        # See app/datasets/motion_injection.py docstring for why this exists:
        # confirmed empirically that VISO's released frames carry ~0 platform
        # motion, so without this, raw-vs-homography has nothing to compensate
        # for regardless of whether the mechanism works.
        detector_mode = params.get("detector_mode", "placeholder")
        inject_motion = params.get("inject_synthetic_motion", False)
        synth_H_list = None
        if inject_motion:
            synth_H_list = generate_translation_walk(
                n_frames=max_frames,
                px_per_frame=params.get("synth_motion_px_per_frame", 2.0),
                seed=params.get("synth_motion_seed", 42),
            )
            await send(EV.STAGE_INIT,
                       message=(f"Synthetic platform motion injection ENABLED: "
                                f"~{params.get('synth_motion_px_per_frame', 2.0)} px/frame "
                                f"random-walk drift applied to observed frames"
                                + (" and ground-truth boxes" if detector_mode == "ground_truth" else "")
                                + ". Absolute accuracy numbers from this run reflect a "
                                  "synthetic stress test, not real-world platform motion."))

        # ---- build detector ---------------------------------------------------
        gt_detector = None
        gt_evaluator = None
        detector = None

        if detector_mode == "ground_truth":
            annotations = loader.load_annotations(sequence)
            if not annotations or not annotations.get("tracks"):
                raise RuntimeError(
                    f"detector_mode='ground_truth' was requested but sequence "
                    f"'{sequence.name}' has no parsable annotations. Ground-truth "
                    f"detection mode requires a sequence with a gt.txt (VISO) or "
                    f"gt.csv (SDM-Car) file."
                )
            gt_detector = GroundTruthDetector(annotations["tracks"])
            gt_evaluator = GTAnchoredEvaluator(annotations["tracks"])
            await send(EV.STAGE_INIT,
                       message=(f"Using ground-truth boxes as detections "
                                f"({gt_detector.n_frames_with_annotations} annotated frames) "
                                f"-- this isolates the tracker/geometry mechanism from "
                                f"detector quality. Occlusion events now correspond "
                                f"directly to genuine gaps in the ground truth, not "
                                f"detector dropout."))
        else:
            detector = SimpleMotionDetector(
                min_area=params.get("det_min_area", 4),
                max_area=params.get("det_max_area", 2500),
                diff_thresh=params.get("det_diff_thresh", 18),
                max_detections_per_frame=params.get("det_max_detections", 250),
            )

        # ---- build tracker ---------------------------------------------------
        # min_hits_to_confirm exists to protect against the PLACEHOLDER
        # detector's spurious one-frame blobs -- a track that goes unmatched
        # even once before reaching min_hits is killed immediately (see
        # tracker.py step() section 4) rather than waiting through the full
        # occlusion window, since an unconfirmed track is more likely noise
        # than a real target. That assumption is WRONG when using the
        # ground-truth detector: every detection there is verified real by
        # definition, so the same rule instead kills genuinely valid,
        # continuously-visible tracks on any transient association hiccup,
        # spawning a fresh track ID next frame -- counted as a spurious
        # "occlusion event" even though nothing was actually occluded. This
        # was confirmed directly: a GT-detector run reported 260 occlusion
        # events on a 260-frame sequence whose ground truth contains only
        # 14 genuine gaps. Forcing min_hits_to_confirm=1 in ground-truth
        # mode makes every GT-derived track confirmed on its first
        # observation, removing this failure mode entirely.
        effective_min_hits = 1 if detector_mode == "ground_truth" else params.get("min_hits_to_confirm", 3)
        if detector_mode == "ground_truth" and params.get("min_hits_to_confirm", 3) != 1:
            await send(EV.STAGE_INIT,
                       message=(f"min_hits_to_confirm overridden to 1 for ground-truth mode "
                                f"(requested {params.get('min_hits_to_confirm', 3)}) -- the "
                                f"tentative-track pruning it controls exists to protect against "
                                f"noisy detector blobs, which don't exist when using verified "
                                f"ground-truth boxes."))

        tracker = MotionGroundedTracker(
            opencv_seed=0,
            trajectory_mode=params.get("trajectory_mode", "homography"),
            window_length=params.get("window_length", 8),
            kmax_occlusion=params.get("kmax_occlusion", 20),
            min_hits_to_confirm=effective_min_hits,
            predictor_model=params.get("predictor_model", "constant_velocity"),
            homography_kwargs=dict(
                min_inliers=params.get("min_inliers", 15),
                min_inlier_ratio=params.get("min_inlier_ratio", 0.35),
                max_reproj_error=params.get("max_reproj_error", 3.0),
            ),
        )

        oracle_mode = tracker.spec.estimator == "oracle"
        if oracle_mode and synth_H_list is None:
            raise RuntimeError(
                "trajectory_mode='oracle' requires inject_synthetic_motion=True: the oracle "
                "transform is the injected drift itself, which is only known when injected.")
        prev_H_synth = None

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

            # Apply synthetic platform-motion drift, if enabled, BEFORE
            # anything downstream sees this frame. This is the key
            # architectural point: injection happens upstream of the
            # tracker's own homography estimator, so that estimator (and
            # everything after it) is completely unmodified -- it just
            # estimates whatever motion is actually present in the frames
            # it's handed, synthetic or real, exactly as it would for a
            # genuinely unstabilized satellite video.
            if synth_H_list is not None and frame_idx < len(synth_H_list):
                H_synth = synth_H_list[frame_idx]
                frame_bgr = warp_frame(frame_bgr, H_synth)
            else:
                H_synth = None

            import cv2
            if gt_detector is not None:
                gt_frame_idx = frame_idx + getattr(loader, "gt_frame_index_base", 0)
                dets = gt_detector.detect_for_frame(gt_frame_idx)
                if H_synth is not None:
                    for d in dets:
                        new_box = warp_box(d.box.tolist(), H_synth)
                        d.box = np.array(new_box, dtype=np.float64)
                        d.center = np.array([new_box[0] + new_box[2] / 2.0,
                                             new_box[1] + new_box[3] / 2.0], dtype=np.float64)
            else:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                dets = detector.detect(gray)
            await send(EV.STAGE_DETECTION, frame=frame_idx, n_detections=len(dets))

            if oracle_mode and prev_frame is not None:
                if H_synth is None or prev_H_synth is None:
                    raise RuntimeError(f"oracle mode: no injected transform for frame {frame_idx}")
                # True platform transform between consecutive OBSERVED frames.
                tracker.set_oracle_transform(H_synth @ np.linalg.inv(prev_H_synth))

            result = tracker.step(frame_bgr, prev_frame, dets)
            prev_H_synth = H_synth

            if gt_evaluator is not None:
                matched_this_frame = [
                    dict(id=t["id"], box=t["box"])
                    for t in result["tracks_snapshot"] if t["visible"] and t["box"] is not None
                ]
                # dets here are the GT-derived detections for this exact
                # frame, already warped by synthetic motion injection if
                # enabled -- the same coordinate system the tracker itself
                # operated in, which is what the evaluator needs (see
                # gt_evaluator.py docstring for why the raw unwarped
                # annotations can't be used directly for this).
                gt_boxes_this_frame = {
                    d.source_id: d.box.tolist() for d in dets if d.source_id is not None
                }
                gt_evaluator.accumulate_frame(gt_frame_idx, matched_this_frame, gt_boxes_this_frame)

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
        if gt_evaluator is not None:
            summary["gt_anchored_evaluation"] = gt_evaluator.finalize()

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