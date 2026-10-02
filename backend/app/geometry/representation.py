"""
Trajectory representation operators R (spec Section 12: Baseline and
ablation operators). This is the single switch point that makes the
raw-vs-homography ablation causal and clean: the SAME downstream
predictor (residual_predictor.robust_fit + kalman.CompensatedKalmanFilter)
consumes whichever representation is selected here.

R_raw:  identity -- fit directly on raw pixel history
R_affine: history reprojected through the affine fallback transform only
R_hom:  history reprojected through the full homography chain (Eq. 12.1-12.3)
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Literal

from .decomposition import TransformChain, compensated_history

Mode = Literal["raw", "affine", "homography"]


def represent_history(
    history: List[Tuple[int, np.ndarray]],
    t_ref: int,
    mode: Mode,
    chain: TransformChain = None,
) -> np.ndarray:
    """
    history: list of (frame_idx, xy) visible observations for one track.
    Returns an (N,2) array in the representation requested by `mode`.
    """
    if mode == "raw":
        return np.array([p for _, p in history], dtype=np.float64)

    if chain is None:
        raise ValueError(f"mode={mode} requires a TransformChain")

    # affine and homography both use the same compensated_history() call --
    # the distinction lives in *how* the chain's per-step transforms were
    # estimated upstream (PlatformMotionEstimator returns mode='affine' when
    # the homography gate fails, so the chain already encodes the fallback).
    return compensated_history(chain, history, t_ref)


def reproject_forecast(
    q_forecast: np.ndarray,
    t_ref: int,
    t_target: int,
    mode: Mode,
    chain: TransformChain = None,
) -> np.ndarray:
    """
    Maps a forecast made in the reference-frame coordinate system back to
    image coordinates at t_target (Eq. 7.4 / Phom in Eq. 12.5).
    For mode='raw' this is the identity map (no compensation was applied).
    """
    if mode == "raw" or chain is None:
        return q_forecast
    from .homography import transform_point
    G = chain.G(t_ref, t_target)
    return transform_point(G, q_forecast)
