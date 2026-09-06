"""
Residual trajectory model: weighted least-squares constant-velocity (and
optional quadratic) fit in compensated coordinates.

Implements spec Section 5 (Residual trajectory model) and Section 6
(Confidence weighting).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ResidualFit:
    a: np.ndarray          # position intercept (2,)
    v: np.ndarray          # velocity (2,)
    accel: Optional[np.ndarray] = None   # only for quadratic model
    model: str = "constant_velocity"

    def predict(self, tau: float) -> np.ndarray:
        if self.model == "quadratic" and self.accel is not None:
            return self.a + self.v * tau + 0.5 * self.accel * tau ** 2
        return self.a + self.v * tau


def confidence_weights(
    det_conf: np.ndarray,
    assoc_dist: np.ndarray,
    geom_quality: np.ndarray,
    ages: np.ndarray,
    sigma_a: float = 15.0,
    tau_w: float = 8.0,
    conf_min: float = 0.05,
) -> np.ndarray:
    """
    omega_k = omega_det * omega_assoc * omega_geom * omega_age   (Eq. 6.1-6.5)
    All inputs are arrays over the observation window.
    """
    w_det = np.clip(det_conf, conf_min, 1.0)
    w_assoc = np.exp(-(assoc_dist ** 2) / (2 * sigma_a ** 2))
    w_geom = np.clip(geom_quality, 0.0, 1.0)
    w_age = np.exp(-ages / tau_w)
    w = w_det * w_assoc * w_geom * w_age
    m = w.max()
    return w / m if m > 1e-9 else np.ones_like(w) / len(w)


def fit_constant_velocity(
    q_hist: np.ndarray,      # (N, 2) compensated positions
    taus: np.ndarray,        # (N,) relative frame offsets tau_k = k - t
    weights: Optional[np.ndarray] = None,
) -> ResidualFit:
    """
    Weighted least-squares solve of Eq. 5.2 / 5.4 via a QR-backed lstsq
    (never an explicit matrix inverse, per Section 5.1 implementation note).
    """
    n = q_hist.shape[0]
    if weights is None:
        weights = np.ones(n)
    sw = np.sqrt(weights)
    D = np.stack([np.ones(n), taus], axis=1)          # (N,2) design matrix
    Dw = D * sw[:, None]
    theta = np.zeros((2, 2))
    for dim in range(2):
        yw = q_hist[:, dim] * sw
        sol, *_ = np.linalg.lstsq(Dw, yw, rcond=None)
        theta[:, dim] = sol
    a = theta[0]
    v = theta[1]
    return ResidualFit(a=a, v=v, model="constant_velocity")


def fit_quadratic(
    q_hist: np.ndarray,
    taus: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> ResidualFit:
    """Ablation model, Eq. 5.5-5.6."""
    n = q_hist.shape[0]
    if n < 3:
        return fit_constant_velocity(q_hist, taus, weights)
    if weights is None:
        weights = np.ones(n)
    sw = np.sqrt(weights)
    D = np.stack([np.ones(n), taus, 0.5 * taus ** 2], axis=1)
    Dw = D * sw[:, None]
    theta = np.zeros((3, 2))
    for dim in range(2):
        yw = q_hist[:, dim] * sw
        sol, *_ = np.linalg.lstsq(Dw, yw, rcond=None)
        theta[:, dim] = sol
    return ResidualFit(a=theta[0], v=theta[1], accel=theta[2], model="quadratic")


def robust_fit(
    q_hist: np.ndarray,
    taus: np.ndarray,
    weights: Optional[np.ndarray] = None,
    model: str = "constant_velocity",
    huber_delta: float = 5.0,
    n_iter: int = 3,
) -> ResidualFit:
    """
    Iteratively-reweighted least squares wrapper (Eq. 5.7): downweights
    observations whose residual to the current fit exceeds huber_delta.
    """
    n = q_hist.shape[0]
    w = np.ones(n) if weights is None else weights.copy()
    fit_fn = fit_quadratic if model == "quadratic" else fit_constant_velocity
    fit = fit_fn(q_hist, taus, w)
    for _ in range(n_iter):
        pred = np.stack([fit.predict(t) for t in taus])
        resid = np.linalg.norm(q_hist - pred, axis=1)
        huber_w = np.where(resid <= huber_delta, 1.0, huber_delta / np.maximum(resid, 1e-6))
        w = w * huber_w
        w = w / max(w.max(), 1e-9)
        fit = fit_fn(q_hist, taus, w)
    return fit
