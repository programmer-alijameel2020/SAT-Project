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
    Maintains per-frame operational transforms and cumulative transforms
    G_{k->t} in either direction (Eq. 2.5 / 2.7), with a running cache of
    cumulative-from-frame-0 transforms.

    Indexing contract (FIXED):
      push() is called exactly once per frame, in frame order.
      The call made while processing frame t registers G_{t-1->t};
      the call for frame 0 registers the identity (no previous frame).
      After that, _cum_from_0[t] == G_{0->t} and G(k, t) == G_{k->t}.

    The previous implementation pre-seeded the cache with an identity and
    then also appended the frame-0 identity, so _cum_from_0[t] held
    G_{0->t-1} and G(k, t) silently returned G_{k-1->t-1}: every
    compensation and reprojection used a transform lagged by one frame.
    """

    def __init__(self):
        self._step_transforms: List[np.ndarray] = []   # index t: G_{t-1->t} (index 0: I)
        self._cum_from_0: List[np.ndarray] = []        # index t: G_{0->t}
        self.model = None                              # set by the tracker

    def push(self, G_prev_to_t: np.ndarray):
        """Register G_{t-1->t} for the frame currently being processed."""
        G_prev_to_t = np.asarray(G_prev_to_t, dtype=np.float64)
        self._step_transforms.append(G_prev_to_t)
        if not self._cum_from_0:
            self._cum_from_0.append(np.eye(3))          # frame 0: G_{0->0}
            return
        cum = G_prev_to_t @ self._cum_from_0[-1]
        self._cum_from_0.append(cum / cum[2, 2])

    def __len__(self):
        """Number of frames registered (frame indices 0 .. len-1)."""
        return len(self._cum_from_0)

    def G(self, k: int, t: int) -> np.ndarray:
        """Cumulative transform mapping frame-k coordinates into frame t:
        G_{k->t} = G_{0->t} G_{0->k}^{-1}. Valid for k <= t and k > t."""
        n = len(self._cum_from_0)
        if not (0 <= k < n and 0 <= t < n):
            raise IndexError(f"TransformChain.G({k}, {t}) outside registered frames 0..{n - 1}")
        if k == t:
            return np.eye(3)
        G = self._cum_from_0[t] @ np.linalg.inv(self._cum_from_0[k])
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