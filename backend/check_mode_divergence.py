"""
Robust platform-motion estimation (homography or affine primary model).

Implements spec Section 3 (Estimating platform motion):
 - background feature correspondences (ORB, target-region rejection)
 - robust homography estimation (RANSAC + optional Huber refinement)
 - inlier set / quality statistics (r_H, eps_H, q_H)
 - validity gate + affine/identity fallback (Eq. 3.10 - 3.11)

Primary model is selected at construction time via `model`:
  model="homography" : RANSAC homography -> Huber refine -> gate;
                       on gate failure: 4-DOF similarity fallback
                       (cv2.estimateAffinePartial2D), then identity.
                       THIS PATH IS UNCHANGED from the version used for the
                       reported raw/homography campaign.
  model="affine"     : RANSAC full 6-DOF affine (cv2.estimateAffine2D) ->
                       Huber refine (6 params) -> SAME gate (m0, r0, eps0);
                       on gate failure: identity. Never falls back to, or
                       reuses, a homography estimate.
"""
from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HomographyResult:
    """Result of estimating the platform transform between two frames."""
    H: np.ndarray                 # 3x3 operational transform G_{t->t+1}
    mode: str                     # "homography" | "affine" | "identity"
    n_matches: int
    n_inliers: int
    inlier_ratio: float
    reproj_error: float
    quality_score: float          # q_H in [0,1], sigmoid-based (Eq 3.9)
    gate_passed: bool
    model: str = "homography"     # primary model this estimator was configured with


class PlatformMotionEstimator:
    """
    Estimates the dominant background (platform-induced) transform between
    consecutive frames from video content alone -- no ephemeris/attitude
    metadata required (Assumption 5.2 / 5.3 in the research plan).
    """

    def __init__(
        self,
        min_inliers: int = 15,
        min_inlier_ratio: float = 0.35,
        max_reproj_error: float = 3.0,
        ransac_reproj_threshold: float = 3.0,
        max_features: int = 800,
        huber_delta: float = 2.0,
        quality_alpha_r: float = 6.0,
        quality_alpha_eps: float = 0.6,
        model: str = "homography",
    ):
        if model not in ("homography", "affine"):
            raise ValueError(f"PlatformMotionEstimator: unknown model {model!r} "
                             "(expected 'homography' or 'affine')")
        self.model = model
        self.m0 = min_inliers
        self.r0 = min_inlier_ratio
        self.eps0 = max_reproj_error
        self.ransac_thresh = ransac_reproj_threshold
        self.orb = cv2.ORB_create(nfeatures=max_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.huber_delta = huber_delta
        self.alpha_r = quality_alpha_r
        self.alpha_eps = quality_alpha_eps

    # ------------------------------------------------------------------ #
    def _reject_target_regions(self, kps, mask_boxes):
        """Drop keypoints that fall inside detection / predicted-target boxes
        (Section 3.1: 'Features inside detection masks ... should be removed')."""
        if not mask_boxes:
            return list(range(len(kps)))
        keep = []
        for i, kp in enumerate(kps):
            x, y = kp.pt
            inside = False
            for (bx, by, bw, bh) in mask_boxes:
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    inside = True
                    break
            if not inside:
                keep.append(i)
        return keep

    def _huber_refine(self, H0: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Nonlinear refinement of the RANSAC estimate on inliers using a
        Huber loss on symmetric transfer error (Eq. 3.2-3.5)."""
        from scipy.optimize import least_squares

        def residuals(h):
            H = np.append(h, 1.0).reshape(3, 3)
            src_h = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
            return (src_h - dst).ravel()

        h0 = (H0 / H0[2, 2]).ravel()[:8]
        try:
            res = least_squares(
                residuals, h0, loss="huber", f_scale=self.huber_delta, max_nfev=200
            )
            H_ref = np.append(res.x, 1.0).reshape(3, 3)
            return H_ref
        except Exception:
            return H0

    # ------------------------------------------------------------------ #
    def estimate(
        self,
        frame_t: np.ndarray,
        frame_t1: np.ndarray,
        target_boxes_t: Optional[list] = None,
    ) -> HomographyResult:
        """Estimate G_{t->t+1} with the configured primary model.
        homography: falls back to similarity, then identity (Eq. 3.10-3.11).
        affine:     falls back to identity only."""
        gray_t = cv2.cvtColor(frame_t, cv2.COLOR_BGR2GRAY) if frame_t.ndim == 3 else frame_t
        gray_t1 = cv2.cvtColor(frame_t1, cv2.COLOR_BGR2GRAY) if frame_t1.ndim == 3 else frame_t1

        kp1, des1 = self.orb.detectAndCompute(gray_t, None)
        kp2, des2 = self.orb.detectAndCompute(gray_t1, None)

        if des1 is None or des2 is None or len(kp1) < self.m0 or len(kp2) < self.m0:
            return self._tag(HomographyResult(np.eye(3), "identity", 0, 0, 0.0, 0.0, 0.0, False))

        keep1 = set(self._reject_target_regions(kp1, target_boxes_t))
        matches = self.bf.knnMatch(des1, des2, k=2)
        good = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.queryIdx not in keep1:
                continue
            if m.distance < 0.75 * n.distance:
                good.append(m)

        n_matches = len(good)
        if n_matches < self.m0:
            return self._tag(HomographyResult(np.eye(3), "identity", n_matches, 0, 0.0, 0.0, 0.0, False))

        src = np.float32([kp1[m.queryIdx].pt for m in good])
        dst = np.float32([kp2[m.trainIdx].pt for m in good])

        if self.model == "affine":
            return self._tag(self._estimate_affine_primary(src, dst, n_matches))
        return self._tag(self._estimate_homography_primary(src, dst, n_matches))

    # ------------------------------------------------------------------ #
    def _tag(self, res: HomographyResult) -> HomographyResult:
        res.model = self.model
        return res

    def _estimate_homography_primary(self, src, dst, n_matches) -> HomographyResult:
        """Original homography path -- logic unchanged."""

        H0, inlier_mask = cv2.findHomography(
            src, dst, cv2.RANSAC, self.ransac_thresh, maxIters=2000, confidence=0.995
        )
        if H0 is None:
            return self._affine_fallback(src, dst, n_matches)

        inlier_mask = inlier_mask.ravel().astype(bool)
        n_inliers = int(inlier_mask.sum())
        if n_inliers < self.m0:
            return self._affine_fallback(src, dst, n_matches)

        H_refined = self._huber_refine(H0, src[inlier_mask], dst[inlier_mask])
        H_refined = H_refined / H_refined[2, 2]

        # recompute reprojection stats with the refined H
        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H_refined).reshape(-1, 2)
        err = np.linalg.norm(proj - dst, axis=1)
        inlier_mask2 = err <= self.ransac_thresh
        n_inliers2 = int(inlier_mask2.sum())
        inlier_ratio = n_inliers2 / max(1, n_matches)
        reproj_error = float(np.median(err[inlier_mask2])) if n_inliers2 > 0 else float(np.median(err))

        quality = 1.0 / (1.0 + np.exp(-(self.alpha_r * (inlier_ratio - self.r0)
                                         - self.alpha_eps * (reproj_error - self.eps0))))

        gate = (n_inliers2 >= self.m0) and (inlier_ratio >= self.r0) and (reproj_error <= self.eps0)

        if gate:
            return HomographyResult(H_refined, "homography", n_matches, n_inliers2,
                                     inlier_ratio, reproj_error, float(quality), True)
        return self._affine_fallback(src, dst, n_matches)

    # ------------------------------------------------------------------ #
    def _huber_refine_affine(self, A0: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Huber-loss refinement of a 2x3 affine on inliers (6 parameters),
        mirroring _huber_refine for the homography path."""
        from scipy.optimize import least_squares

        def residuals(a):
            A = a.reshape(2, 3)
            return (src @ A[:, :2].T + A[:, 2] - dst).ravel()

        try:
            res = least_squares(residuals, A0.ravel().astype(np.float64), loss="huber",
                                f_scale=self.huber_delta, max_nfev=200)
            return res.x.reshape(2, 3)
        except Exception:
            return A0

    def _estimate_affine_primary(self, src, dst, n_matches) -> HomographyResult:
        """Full 6-DOF affine as the PRIMARY model, gated by the same
        (m0, r0, eps0) criteria as the homography path. Identity on failure."""
        A0, inlier_mask = cv2.estimateAffine2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_thresh,
            maxIters=2000, confidence=0.995,
        )
        if A0 is None or inlier_mask is None:
            return HomographyResult(np.eye(3), "identity", n_matches, 0, 0.0, 0.0, 0.0, False)
        inlier_mask = inlier_mask.ravel().astype(bool)
        if int(inlier_mask.sum()) < self.m0:
            return HomographyResult(np.eye(3), "identity", n_matches, int(inlier_mask.sum()),
                                     0.0, 0.0, 0.0, False)

        src64, dst64 = src.astype(np.float64), dst.astype(np.float64)
        A = self._huber_refine_affine(A0, src64[inlier_mask], dst64[inlier_mask])

        proj = src64 @ A[:, :2].T + A[:, 2]
        err = np.linalg.norm(proj - dst64, axis=1)
        inlier_mask2 = err <= self.ransac_thresh
        n_inliers2 = int(inlier_mask2.sum())
        inlier_ratio = n_inliers2 / max(1, n_matches)
        reproj_error = float(np.median(err[inlier_mask2])) if n_inliers2 > 0 else float(np.median(err))

        quality = 1.0 / (1.0 + np.exp(-(self.alpha_r * (inlier_ratio - self.r0)
                                         - self.alpha_eps * (reproj_error - self.eps0))))
        gate = (n_inliers2 >= self.m0) and (inlier_ratio >= self.r0) and (reproj_error <= self.eps0)

        if not gate:
            return HomographyResult(np.eye(3), "identity", n_matches, n_inliers2,
                                     inlier_ratio, reproj_error, 0.0, False)
        G = np.eye(3)
        G[:2, :] = A
        return HomographyResult(G, "affine", n_matches, n_inliers2,
                                 inlier_ratio, reproj_error, float(quality), True)

    def _affine_fallback(self, src, dst, n_matches) -> HomographyResult:
        """Fallback for the HOMOGRAPHY path only (Eq. 3.11). Note: this is a
        4-DOF similarity (estimateAffinePartial2D), not a full affine."""
        A, inlier_mask = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_thresh
        )
        if A is None:
            return HomographyResult(np.eye(3), "identity", n_matches, 0, 0.0, 0.0, 0.0, False)
        H = np.eye(3)
        H[:2, :] = A
        n_inliers = int(inlier_mask.ravel().astype(bool).sum()) if inlier_mask is not None else 0
        ratio = n_inliers / max(1, n_matches)
        gate = n_inliers >= max(6, self.m0 // 2)
        return HomographyResult(H, "affine" if gate else "identity", n_matches, n_inliers,
                                 ratio, 0.0, 0.5 if gate else 0.0, gate)


def compose(H_list: list[np.ndarray]) -> np.ndarray:
    """Cumulative transform H_{k->t} = H_{t-1->t} ... H_{k->k+1}  (Eq. 2.5)."""
    H = np.eye(3)
    for Hi in H_list:
        H = Hi @ H
    return H / H[2, 2]


def dehomogenize(p: np.ndarray) -> np.ndarray:
    """pi(.) operator, Eq. 2.2."""
    p = np.asarray(p, dtype=np.float64)
    if p.shape[-1] == 3:
        return p[..., :2] / p[..., 2:3]
    return p


def to_homogeneous(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    ones = np.ones(p.shape[:-1] + (1,))
    return np.concatenate([p, ones], axis=-1)


def transform_point(H: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply H to a 2D point (or Nx2 array of points), returning pi(H p~)."""
    p = np.asarray(p, dtype=np.float64)
    single = p.ndim == 1
    pts = p.reshape(-1, 2)
    ph = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    out = (H @ ph.T).T
    out = out[:, :2] / out[:, 2:3]
    return out[0] if single else out