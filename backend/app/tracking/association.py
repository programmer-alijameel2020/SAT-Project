"""
Association and reactivation (spec Section 10 / research-plan Section 11).

Composite cost combining Mahalanobis gating, IoU, and optional appearance,
solved by Hungarian assignment (scipy.optimize.linear_sum_assignment).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """box = [x, y, w, h]."""
    ax1, ay1, ax2, ay2 = box_a[0], box_a[1], box_a[0] + box_a[2], box_a[1] + box_a[3]
    bx1, by1, bx2, by2 = box_b[0], box_b[1], box_b[0] + box_b[2], box_b[1] + box_b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-9 else 0.0


def mahalanobis(innovation: np.ndarray, S: np.ndarray) -> float:
    try:
        Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        Sinv = np.linalg.pinv(S)
    return float(innovation.T @ Sinv @ innovation)


def associate(
    predicted_centers: list,          # list of (track_id, xy, box, S_cov, occlusion_age)
    detections: list,                 # list of Detection
    gate_chi2: float = chi2.ppf(0.95, df=2),
    lambda_g: float = 1.0,
    lambda_iou: float = 3.0,
    lambda_app: float = 0.0,
    max_cost: float = 1e6,
    kmax_reactivation: int = 30,
):
    """
    Returns (matches, unmatched_tracks, unmatched_dets) where matches is a
    list of (track_idx, det_idx) pairs into the input lists.
    """
    n_t, n_d = len(predicted_centers), len(detections)
    if n_t == 0 or n_d == 0:
        return [], list(range(n_t)), list(range(n_d))

    cost = np.full((n_t, n_d), max_cost)
    gated = np.zeros((n_t, n_d), dtype=bool)

    for i, (tid, xy, box, S, age) in enumerate(predicted_centers):
        gate = gate_chi2 * (1.0 + 0.05 * min(age, kmax_reactivation))  # widen with occlusion age
        for j, det in enumerate(detections):
            innov = det.center - xy
            d_mah = mahalanobis(innov, S)
            if d_mah > gate:
                continue
            d_iou = 1.0 - iou(box, det.box)
            c = lambda_g * d_mah + lambda_iou * d_iou
            cost[i, j] = c
            gated[i, j] = True

    row_idx, col_idx = linear_sum_assignment(cost)
    matches, matched_t, matched_d = [], set(), set()
    for r, c in zip(row_idx, col_idx):
        if gated[r, c]:
            matches.append((r, c))
            matched_t.add(r)
            matched_d.add(c)

    unmatched_tracks = [i for i in range(n_t) if i not in matched_t]
    unmatched_dets = [j for j in range(n_d) if j not in matched_d]
    return matches, unmatched_tracks, unmatched_dets
