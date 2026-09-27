"""
Kalman realization of the residual predictor in compensated coordinates.

Implements spec Section 8: state x = [u, v, u_dot, v_dot]^T, constant-
velocity transition F, white-acceleration process noise Q, position-only
measurement H. During occlusion the measurement update is skipped and
only the predict step runs (Eq. 8.10), which is exactly the mechanism
that lets the filter "coast" through an occlusion using the compensated
(platform-motion-free) coordinate frame instead of raw pixels.

FIX (state transport): the filter state lives in the coordinates of the
most recent frame. For compensated modes, transport(G) must be called once
per frame with G_{t-1->t} BEFORE predict(): it maps the position through
the projective transform, the velocity through the transform's Jacobian at
that position, and the covariance through the corresponding block-Jacobian.
This is the recursive prediction form of the formulation (spec Sec. 7.3):
over an occlusion of k frames the accumulated platform motion is carried by
k successive transports, instead of only the last frame's transform as in
the previous implementation (which predicted in stale coordinates and
reprojected with a single step).
"""
from __future__ import annotations

import numpy as np


class CompensatedKalmanFilter:
    def __init__(self, q_accel: float = 1.0, r_meas: float = 4.0, dt: float = 1.0):
        self.dt = dt
        self.qa = q_accel
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        self.Q = q_accel * np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2],
        ], dtype=np.float64)
        self.Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.R = r_meas * np.eye(2)

        self.x = np.zeros(4)
        self.P = np.eye(4) * 10.0

    def initialize(self, pos_xy: np.ndarray, vel_xy: np.ndarray = None):
        self.x = np.array([pos_xy[0], pos_xy[1],
                            0.0 if vel_xy is None else vel_xy[0],
                            0.0 if vel_xy is None else vel_xy[1]])
        self.P = np.eye(4) * 10.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy(), self.P.copy()

    @staticmethod
    def projective_jacobian(G: np.ndarray, p: np.ndarray) -> np.ndarray:
        """d pi(G [p;1]) / dp  (2x2). For an affine G this is its linear part."""
        y = G @ np.array([p[0], p[1], 1.0])
        w = y[2]
        u = y[:2] / w
        return (G[:2, :2] - np.outer(u, G[2, :2])) / w

    def transport(self, G: np.ndarray):
        """Map state and covariance from frame t-1 coordinates into frame t."""
        G = np.asarray(G, dtype=np.float64)
        p = self.x[:2].copy()
        J = self.projective_jacobian(G, p)
        y = G @ np.array([p[0], p[1], 1.0])
        self.x[:2] = y[:2] / y[2]
        self.x[2:] = J @ self.x[2:]
        T = np.zeros((4, 4))
        T[:2, :2] = J
        T[2:, 2:] = J
        self.P = T @ self.P @ T.T

    def update(self, z_xy: np.ndarray):
        y = z_xy - self.Hm @ self.x
        S = self.Hm @ self.P @ self.Hm.T + self.R
        K = self.P @ self.Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.Hm) @ self.P

    def occlusion_confidence(self, occlusion_age: int, tau_l: float = 8.0,
                              sigma_s: float = 20.0, geom_quality: float = 1.0) -> float:
        """eta_{t+tau} = exp(-tr(S)/2 sigma_S^2) * exp(-tau/tau_l) * q_H  (Eq. 8.11)."""
        S = self.Hm @ self.P @ self.Hm.T + self.R
        term1 = np.exp(-np.trace(S) / (2 * sigma_s ** 2))
        term2 = np.exp(-occlusion_age / tau_l)
        return float(term1 * term2 * geom_quality)