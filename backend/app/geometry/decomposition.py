"""
Target-motion decomposition: platform-compensated coordinates.

Implements spec Section 4 / research-plan Section 7:
  q_k(t) = pi( G_{k->t} [p_k; 1] )
which maps a target's raw pixel history into a common "compensated"
coordinate system, removing the dominant platform-induced component so a
downstream predictor operates on residual (target-attributable) motion.
"""
from __future__ import annotations

import numpy as np
from typing import List

from .homography import dehomogenize, transform_point


class TransformChain:
    """
    Maintains the per-frame operational transforms G_{t->t+1} and provides
    cumulative transforms G_{k->t} in either direction (Eq. 2.5 / 2.7),
    without needing to recompute a full chain product on every query
    (keeps a running cache of cumulative-from-frame-0 transforms).
    """

    def __init__(self):
        self._step_transforms: List[np.ndarray] = []   # G_{t->t+1}, index t
        self._cum_from_0: List[np.ndarray] = [np.eye(3)]  # G_{0->t}, index t

    def push(self, G_t_to_t1: np.ndarray):
        """Register the transform from the newest frame to the next one."""
        self._step_transforms.append(G_t_to_t1)
        cum = G_t_to_t1 @ self._cum_from_0[-1]
        self._cum_from_0.append(cum / cum[2, 2])

    def __len__(self):
        return len(self._cum_from_0)

    def G(self, k: int, t: int) -> np.ndarray:
        """Cumulative transform mapping frame-k coordinates to frame-t's
        reference system: G_{k->t} = G_{0->t} G_{0->k}^{-1}."""
        if k == t:
            return np.eye(3)
        Gk = self._cum_from_0[k]
        Gt = self._cum_from_0[t]
        G = Gt @ np.linalg.inv(Gk)
        return G / G[2, 2]


def compensated_position(chain: TransformChain, p_k: np.ndarray, k: int, t: int) -> np.ndarray:
    """q_k(t) = pi(G_{k->t} [p_k;1])  (Eq. 4.1)."""
    G = chain.G(k, t)
    return transform_point(G, p_k)


def compensated_history(chain: TransformChain, history: List[tuple], t_ref: int) -> np.ndarray:
    """
    history: list of (frame_index, p_xy) for a single track's visible
    observations. Returns an (N,2) array of q_k(t_ref) values -- the
    trajectory expressed in the frame-t_ref reference system (Eq. 4.1).
    """
    out = np.zeros((len(history), 2), dtype=np.float64)
    for i, (k, p) in enumerate(history):
        out[i] = compensated_position(chain, np.asarray(p), k, t_ref)
    return out


def residual_of_stationary_check(chain: TransformChain, p_k: np.ndarray, k: int) -> np.ndarray:
    """
    epsilon_{k+1} = p_{k+1} - pi(G_{k->k+1} [p_k;1])  (Eq. 4.6)
    Useful for diagnostics: near-zero for background, large for real targets.
    """
    G = chain.G(k, k + 1)
    return transform_point(G, p_k)
