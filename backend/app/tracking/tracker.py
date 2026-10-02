"""
The unified proposed system (spec Section 11 / research-plan "Final system
statement"): a tracking-by-detection loop in which

  1. PlatformMotionEstimator estimates G_{t->t+1} from video content,
  2. each track's history is expressed in platform-compensated coordinates,
  3. a fixed residual predictor + Kalman filter forecasts short-horizon
     target motion during occlusion,
  4. the forecast is reprojected to image coordinates for association,
  5. Hungarian assignment with Mahalanobis/IoU gating matches detections,
  6. occlusion age / confidence governs reactivation and track death.

`trajectory_mode` selects one row of representation.MODE_SPECS
(raw_kf | raw | gmc | affine | homography | oracle). A mode only decides
(a) which transform fills the chain, (b) whether the Kalman state is
transported through G_{t-1->t} every frame, and (c) whether the Kalman
velocity is replaced by the robust windowed refit. Detector, association,
gates, noise parameters and track management are identical across modes.

Fixes relative to the version used for the first campaign:
  1. affine mode builds its chain from PRIMARY 6-DOF affine estimates
     (previously it silently reused the homography chain);
  2. TransformChain indexing (decomposition.py) no longer lags one frame;
  3. the Kalman state is transported into each new frame (recursive form of
     Eq. 7.4, spec Sec. 7.3) instead of predicting in stale coordinates and
     reprojecting with a single step, so platform motion accumulated during
     an occlusion is carried by the forecast;
  4. no silent fallback to raw history on representation errors;
  5. fallback / platform-motion statistics are measured against the
     primary model; the full transform is logged per frame;
  6. OpenCV RNG seeded at construction for run-to-run determinism.
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from ..geometry.homography import PlatformMotionEstimator, HomographyResult, transform_point
from ..geometry.decomposition import TransformChain
from ..geometry.representation import represent_history, Mode, spec
from ..geometry.residual_predictor import robust_fit, confidence_weights
from ..geometry.kalman import CompensatedKalmanFilter
from ..detection.simple_detector import Detection
from .association import associate


@dataclass
class Track:
    track_id: int
    kf: CompensatedKalmanFilter
    history: List[tuple] = field(default_factory=list)   # (frame_idx, xy, conf)
    box: np.ndarray = None
    visible: bool = True
    occlusion_age: int = 0
    total_age: int = 0
    hits: int = 1                      # number of frames actually matched to a detection
    confirmed: bool = False            # True once hits >= min_hits_to_confirm
    id_switches: int = 0
    reappeared: bool = False
    occlusion_event_start: Optional[int] = None
    max_occlusion_seen: int = 0


class MotionGroundedTracker:
    def __init__(
        self,
        trajectory_mode: Mode = "homography",
        window_length: int = 8,
        kmax_occlusion: int = 20,
        min_hits_to_confirm: int = 2,
        max_age_before_death: int = None,
        predictor_model: str = "constant_velocity",
        homography_kwargs: dict = None,
        opencv_seed: int = 0,
    ):
        self.spec = spec(trajectory_mode)          # raises on unknown modes
        self.mode = trajectory_mode
        cv2.setRNGSeed(int(opencv_seed))
        self.L = window_length
        self.kmax = kmax_occlusion
        self.min_hits = min_hits_to_confirm
        # A track that hasn't been seen for longer than the occlusion
        # horizon (K_max) is, by definition, no longer a survivable
        # occlusion -- it should die. max_age_before_death defaults to
        # kmax_occlusion itself rather than an independent hardcoded
        # value, so the "Max occlusion horizon" the user sets in the UI
        # actually governs track death, not just the prediction window.
        # Pass max_age_before_death explicitly only to allow a track to
        # outlive the occlusion horizon for some other reason (rare).
        self.max_age = max_age_before_death if max_age_before_death is not None else kmax_occlusion
        self.predictor_model = predictor_model

        # Transform source follows the mode table. Non-compensating modes keep
        # the homography estimator only to log platform motion (never consumed).
        # The oracle mode takes G_{t-1->t} from set_oracle_transform() instead.
        self.estimator = None
        if self.spec.estimator != "oracle":
            kwargs = dict(homography_kwargs or {})
            kwargs.pop("model", None)
            self.estimator = PlatformMotionEstimator(**kwargs, model=self.spec.estimator)
        self._oracle_G: Optional[np.ndarray] = None
        self.chain = TransformChain()
        self.chain.model = self.spec.estimator

        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
        self.frame_idx = -1

        # diagnostics accumulated over the run
        self.homography_log: List[dict] = []
        self.occlusion_events: List[dict] = []
        self.geometry_time_ms: List[float] = []
        self.total_time_ms: List[float] = []

    # ------------------------------------------------------------------ #
    def set_oracle_transform(self, G_prev_to_t: np.ndarray) -> None:
        """Oracle mode only: supply the TRUE platform transform G_{t-1->t}
        (from the motion injector) before calling step() for frame t >= 1."""
        self._oracle_G = np.asarray(G_prev_to_t, dtype=np.float64)

    def step(self, frame_bgr: np.ndarray, prev_frame_bgr: Optional[np.ndarray],
              detections: List[Detection]) -> dict:
        import time
        t0 = time.perf_counter()
        self.frame_idx += 1
        t = self.frame_idx

        # 1) platform-motion estimation ---------------------------------
        tg0 = time.perf_counter()
        if prev_frame_bgr is None:
            hres = HomographyResult(np.eye(3), "identity", 0, 0, 0.0, 0.0, 1.0, True,
                                    model=self.spec.estimator)
        elif self.spec.estimator == "oracle":
            if self._oracle_G is None:
                raise RuntimeError(f"oracle mode: no transform supplied for frame {t}")
            hres = HomographyResult(self._oracle_G, "oracle", 0, 0, 1.0, 0.0, 1.0, True,
                                    model="oracle")
        else:
            boxes = [tr.box.tolist() for tr in self.tracks.values() if tr.box is not None]
            hres: HomographyResult = self.estimator.estimate(prev_frame_bgr, frame_bgr, boxes)
        self._oracle_G = None
        self.chain.push(hres.H)
        if len(self.chain) != t + 1:
            raise RuntimeError(f"TransformChain out of sync: {len(self.chain)} frames at t={t}")

        # Translation magnitude of the platform transform, measured at the
        # image center. Fit quality says how CONFIDENTLY the transform was
        # estimated, not how MUCH the platform moved.
        img_h, img_w = frame_bgr.shape[:2]
        center = np.array([img_w / 2.0, img_h / 2.0])
        displaced = transform_point(hres.H, center)
        translation_px = float(np.linalg.norm(displaced - center))

        self.homography_log.append(dict(
            frame=t, model=hres.model, mode=hres.mode, gate_passed=bool(hres.gate_passed),
            n_matches=hres.n_matches, n_inliers=hres.n_inliers,
            inlier_ratio=round(hres.inlier_ratio, 4), reproj_error=round(hres.reproj_error, 4),
            quality=round(hres.quality_score, 4), translation_px=round(translation_px, 3),
            H=[round(float(v), 8) for v in np.asarray(hres.H).ravel()],
        ))
        self.geometry_time_ms.append((time.perf_counter() - tg0) * 1000)

        # 2) predict every existing track forward ------------------------
        # Compensating modes first transport each state into frame-t
        # coordinates (position through G, velocity/covariance through its
        # Jacobian), then predict. The prediction is already in frame-t
        # image coordinates, so no reprojection is needed.
        G_step = self.chain.G(t - 1, t) if (self.spec.compensate and t >= 1) else None
        predicted = []
        for tid, tr in self.tracks.items():
            if G_step is not None:
                tr.kf.transport(G_step)
            q_pred, P_pred = tr.kf.predict()
            p_img = q_pred
            S = tr.kf.Hm @ P_pred @ tr.kf.Hm.T + tr.kf.R
            predicted.append((tid, p_img, tr.box if tr.box is not None
                               else np.array([p_img[0]-3, p_img[1]-3, 6, 6]), S, tr.occlusion_age))

        # 3) associate -----------------------------------------------------
        matches, unmatched_tracks, unmatched_dets = associate(predicted, detections)

        matched_ids = set()
        for (pi, di) in matches:
            tid = predicted[pi][0]
            tr = self.tracks[tid]
            det = detections[di]
            tr.kf.update(det.center)
            tr.box = det.box
            tr.visible = True
            if tr.occlusion_age > 0:
                tr.reappeared = True
                self.occlusion_events.append(dict(
                    track_id=tid, start=tr.occlusion_event_start, end=t,
                    duration=tr.occlusion_age, recovered=True,
                ))
            tr.occlusion_age = 0
            tr.total_age += 1
            tr.hits += 1
            if tr.hits >= self.min_hits:
                tr.confirmed = True
            tr.history.append((t, det.center.copy(), det.confidence))
            if len(tr.history) > self.L:
                tr.history.pop(0)
            matched_ids.add(tid)

        # 4) unmatched tracks -> occluded, coast on prediction ------------
        dead = []
        for pi in unmatched_tracks:
            tid = predicted[pi][0]
            tr = self.tracks[tid]
            tr.visible = False
            if tr.occlusion_age == 0:
                tr.occlusion_event_start = t
            tr.occlusion_age += 1
            tr.total_age += 1
            tr.max_occlusion_seen = max(tr.max_occlusion_seen, tr.occlusion_age)

            if not tr.confirmed:
                # Tentative track that went unmatched even once: almost
                # certainly a spurious detector blob. Kill it immediately.
                dead.append(tid)
            elif tr.occlusion_age > self.max_age:
                dead.append(tid)
                if tr.occlusion_event_start is not None:
                    self.occlusion_events.append(dict(
                        track_id=tid, start=tr.occlusion_event_start, end=t,
                        duration=tr.occlusion_age, recovered=False,
                    ))
        for tid in dead:
            del self.tracks[tid]

        # 5) unmatched detections -> spawn new tracks ----------------------
        new_ids = []
        for di in unmatched_dets:
            det = detections[di]
            kf = CompensatedKalmanFilter()
            kf.initialize(det.center)
            tid = self._next_id
            self._next_id += 1
            tr = Track(track_id=tid, kf=kf, box=det.box, visible=True,
                       confirmed=(self.min_hits <= 1))
            tr.history.append((t, det.center.copy(), det.confidence))
            self.tracks[tid] = tr
            new_ids.append(tid)

        # 6) windowed robust refit of the velocity (refit modes only) -------
        for tid, tr in (self.tracks.items() if self.spec.refit else []):
            if len(tr.history) < 2:
                continue
            hist = [(k, p) for (k, p, c) in tr.history]
            confs = np.array([c for (_, _, c) in tr.history])
            # No silent fallback: a failure here would otherwise quietly turn
            # a compensated run into a raw run and corrupt the ablation.
            q_hist = represent_history(hist, t, self.mode, self.chain)
            taus = np.array([k - t for k, _ in hist], dtype=np.float64)
            w = confidence_weights(
                det_conf=confs,
                assoc_dist=np.zeros(len(confs)),
                geom_quality=np.full(len(confs), hres.quality_score if hres.quality_score > 0 else 1.0),
                ages=np.array([t - k for k, _ in hist], dtype=np.float64),
            )
            fit = robust_fit(q_hist, taus, w, model=self.predictor_model)
            v_est = fit.v
            tr.kf.x[2:] = v_est

        self.total_time_ms.append((time.perf_counter() - t0) * 1000)

        return dict(
            frame=t,
            n_detections=len(detections),
            n_tracks_active=sum(1 for tr in self.tracks.values() if tr.confirmed),
            n_tracks_tentative=sum(1 for tr in self.tracks.values() if not tr.confirmed),
            n_matched=len(matches),
            n_new=len(new_ids),
            n_occluded=sum(1 for tr in self.tracks.values() if tr.confirmed and not tr.visible),
            n_dead=len(dead),
            homography_mode=hres.mode,
            homography_quality=round(hres.quality_score, 4),
            tracks_snapshot=[
                dict(id=tid, box=tr.box.tolist() if tr.box is not None else None,
                     visible=tr.visible, occlusion_age=tr.occlusion_age,
                     confirmed=tr.confirmed)
                for tid, tr in self.tracks.items()
            ],
        )

    # ------------------------------------------------------------------ #
    def summary(self) -> dict:
        recovered = [e for e in self.occlusion_events if e["recovered"]]
        lost = [e for e in self.occlusion_events if not e["recovered"]]
        n_events = len(self.occlusion_events)
        recovery_rate = len(recovered) / n_events if n_events else None
        mean_geom_ms = float(np.mean(self.geometry_time_ms)) if self.geometry_time_ms else 0.0
        mean_total_ms = float(np.mean(self.total_time_ms)) if self.total_time_ms else 0.0
        primary = self.spec.estimator
        # Fallback = any frame whose transform did not come from the primary
        # model (for homography runs this is identical to the old definition).
        fallback_rate = (
            sum(1 for h in self.homography_log if h["mode"] != primary)
            / max(1, len(self.homography_log))
        )
        translations = [h["translation_px"] for h in self.homography_log if h["mode"] == primary]
        platform_motion_px = dict(
            mean=round(float(np.mean(translations)), 3) if translations else None,
            median=round(float(np.median(translations)), 3) if translations else None,
            max=round(float(np.max(translations)), 3) if translations else None,
            p95=round(float(np.percentile(translations, 95)), 3) if translations else None,
            n_frames_measured=len(translations),
        )
        return dict(
            trajectory_mode=self.mode,
            estimator_model=primary,
            n_frames=self.frame_idx + 1,
            platform_motion_px=platform_motion_px,
            n_occlusion_events=n_events,
            n_recovered=len(recovered),
            n_lost=len(lost),
            recovery_rate_overall=recovery_rate,
            mean_geometry_latency_ms=round(mean_geom_ms, 3),
            mean_total_latency_ms=round(mean_total_ms, 3),
            p95_total_latency_ms=round(float(np.percentile(self.total_time_ms, 95)), 3)
            if self.total_time_ms else 0.0,
            homography_fallback_rate=round(fallback_rate, 4),
            recovery_by_duration_bucket=self._recovery_by_bucket(),
        )

    def _recovery_by_bucket(self) -> dict:
        buckets = {"short_1_5": [0, 0], "medium_6_15": [0, 0], "long_16_plus": [0, 0]}
        for e in self.occlusion_events:
            d = e["duration"]
            key = "short_1_5" if d <= 5 else ("medium_6_15" if d <= 15 else "long_16_plus")
            buckets[key][1] += 1
            if e["recovered"]:
                buckets[key][0] += 1
        return {
            k: dict(recovered=v[0], total=v[1], rate=(v[0] / v[1] if v[1] else None))
            for k, v in buckets.items()
        }