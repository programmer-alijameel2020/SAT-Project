"""
Trajectory representation operators and the mode table for the ablation
(spec Section 12). Every mode runs the SAME tracker, detector, association
and Kalman filter; a mode only sets three switches:

  estimator   which transform fills the TransformChain
              (None -> chain unused; raw modes still log platform motion)
  compensate  whether the Kalman state is transported through G_{t-1->t}
              each frame and the history is expressed in frame-t coordinates
  refit       whether the Kalman velocity is overwritten each frame by the
              robust windowed fit on the (compensated or raw) history

  mode        estimator    compensate  refit   role
  ----------  -----------  ----------  ------  --------------------------------
  raw_kf      -            no          no      plain Kalman (SORT-like) baseline
  raw         -            no          yes     raw-pixel windowed refit
  gmc         affine       yes         no      state-warping CMC (BoT-SORT-style)
  affine      affine       yes         yes     proposed, affine transform
  homography  homography   yes         yes     proposed, projective transform
  oracle      injected GT  yes         yes     proposed, perfect transform (upper bound)

The 2x2 block {raw_kf, raw, gmc, affine} is a full factorial over
(compensation) x (windowed refit); homography and oracle vary only the
transform source with everything else fixed.
"""
from __future__ import annotations

from typing import List, Tuple, Literal, Dict, NamedTuple

import numpy as np

from .decomposition import TransformChain, compensated_history

Mode = Literal["raw_kf", "raw", "gmc", "affine", "homography", "oracle"]


class ModeSpec(NamedTuple):
    estimator: str      # "homography" | "affine" | "oracle"  (transform that fills the chain)
    compensate: bool
    refit: bool


MODE_SPECS: Dict[str, ModeSpec] = {
    "raw_kf":     ModeSpec("homography", False, False),
    "raw":        ModeSpec("homography", False, True),
    "gmc":        ModeSpec("affine",     True,  False),
    "affine":     ModeSpec("affine",     True,  True),
    "homography": ModeSpec("homography", True,  True),
    "oracle":     ModeSpec("oracle",     True,  True),
}
VALID_MODES = tuple(MODE_SPECS)


def spec(mode: str) -> ModeSpec:
    if mode not in MODE_SPECS:
        raise ValueError(f"Unknown trajectory mode {mode!r}; expected one of {VALID_MODES}")
    return MODE_SPECS[mode]


def _check_chain(mode: str, chain: TransformChain) -> None:
    s = spec(mode)
    if not s.compensate:
        return
    if chain is None:
        raise ValueError(f"mode={mode} requires a TransformChain")
    if getattr(chain, "model", None) != s.estimator:
        raise RuntimeError(
            f"mode={mode} requires a chain built by {s.estimator!r}, "
            f"got {getattr(chain, 'model', None)!r}")


def represent_history(
    history: List[Tuple[int, np.ndarray]],
    t_ref: int,
    mode: Mode,
    chain: TransformChain = None,
) -> np.ndarray:
    """(N,2) history in raw pixels (non-compensating modes) or in the
    frame-t_ref compensated system q_k(t_ref) (Eq. 4.1)."""
    _check_chain(mode, chain)
    if not spec(mode).compensate:
        return np.array([p for _, p in history], dtype=np.float64)
    return compensated_history(chain, history, t_ref)


def reproject_forecast(
    q_forecast: np.ndarray,
    t_ref: int,
    t_target: int,
    mode: Mode,
    chain: TransformChain = None,
) -> np.ndarray:
    """Map a forecast made in frame-t_ref coordinates into frame t_target
    (Eq. 7.4). Identity for non-compensating modes. The tracker itself now
    transports the Kalman state every frame and no longer needs this; it is
    kept for analysis scripts (e.g. multi-step forecast error)."""
    _check_chain(mode, chain)
    if not spec(mode).compensate:
        return q_forecast
    from .homography import transform_point
    return transform_point(chain.G(t_ref, t_target), q_forecast)