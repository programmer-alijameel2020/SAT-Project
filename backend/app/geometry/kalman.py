"""
Kalman realization of the residual predictor in compensated coordinates.

Implements spec Section 8: state x = [u, v, u_dot, v_dot]^T, constant-
velocity transition F, white-acceleration process noise Q, position-only
measurement H. During occlusion the measurement update is skipped and
only the predict step runs (Eq. 8.10), which is exactly the mechanism
that lets the filter "coast" through an occlusion using the compensated
(platform-motion-free) coordinate frame instead of raw pixels.
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
