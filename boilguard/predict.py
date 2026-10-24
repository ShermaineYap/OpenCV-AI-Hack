"""Physics-informed time-to-overflow prediction.

Foam on heated milk / starch water grows *exponentially* at onset (each bubble
film traps gas that generates more bubbles) and then saturates or collapses as
drainage catches up (logistic growth, see :mod:`boilguard.sim`). In log space
this is simply a growth rate k(t) that drifts:

    y = log(foam_mm + h0),   dy/dt = k,   dk/dt = kd,   kd ~ random walk

We track x = [y, k, kd] with an OpenCV ``cv2.KalmanFilter`` whose transition
matrix is rebuilt for every (variable) time step, and whose measurement noise
comes from the vision confidence (low confidence / steam -> trust less).

Overflow happens when foam reaches the calm freeboard D0. Instead of a single
ETA we draw Monte-Carlo samples from the Kalman posterior and solve the
kinematic quadratic for each, giving a calibrated distribution:

    P(overflow within T seconds), ETA median, ETA 10th percentile.

A plateau (kd < 0, peak below the rim) is predicted as "no overflow", which is
what keeps false interventions low on simmering pots.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Forecast:
    t: float
    foam_mm: float
    rate: float            # k, 1/s (foam e-folding rate)
    accel: float           # kd
    eta_p50: float         # s, inf if no overflow within horizon
    eta_p10: float
    p_10s: float
    p_20s: float
    p_lag: float           # P(overflow within actuation lag + 3 s margin)
    rise_mm_s: float
    fill: float = 0.0      # foam / calm freeboard (1.0 = at the rim)

    def to_dict(self):
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, float):
                out[k] = None if not math.isfinite(v) else round(v, 4)
            else:
                out[k] = v
        return out


class FoamKalman:
    H0 = 6.0          # mm, log offset (keeps log finite at zero foam; tuned on dev set)
    HORIZON = 60.0

    def __init__(self, D0_mm: float, q: float = 0.001, n_samples: int = 400, seed: int = 0):
        self.D0 = float(D0_mm)
        self.q = q
        kf = cv2.KalmanFilter(3, 1, 0, cv2.CV_64F)
        kf.measurementMatrix = np.array([[1.0, 0.0, 0.0]])
        kf.statePost = np.array([[math.log(self.H0)], [0.0], [0.0]])
        kf.errorCovPost = np.diag([0.3, 0.01, 0.002]).astype(np.float64)
        self.kf = kf
        self.t = None
        self.rng = np.random.default_rng(seed)
        self.n = n_samples
        self.Y_over = math.log(self.D0 + self.H0)
        self.last = None

    def _F_Q(self, dt):
        F = np.array([[1, dt, 0.5 * dt * dt], [0, 1, dt], [0, 0, 1]], np.float64)
        q = self.q
        Q = q * np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                          [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                          [dt ** 3 / 6, dt ** 2 / 2, dt]], np.float64)
        return F, Q + np.diag([1e-5, 1e-5, 1e-5])

    def update(self, t: float, foam_mm: float, confidence: float, fog: float = 0.0,
               lag_s: float = 4.0) -> Forecast:
        kf = self.kf
        dt = 0.1 if self.t is None else max(1e-3, t - self.t)
        self.t = t
        F, Q = self._F_Q(dt)
        kf.transitionMatrix = F
        kf.processNoiseCov = Q
        kf.predict()
        foam = max(0.0, foam_mm)
        sigma_mm = 1.6 / max(confidence, 0.12) + 4.0 * fog
        r = (sigma_mm / (foam + self.H0)) ** 2
        z = math.log(foam + self.H0)
        innov = z - float(kf.statePre[0, 0])
        s2 = float(kf.errorCovPre[0, 0]) + r
        if innov * innov > 16 * s2:            # 4-sigma gate: inflate instead of reject
            r *= innov * innov / (16 * s2) * 4
        kf.measurementNoiseCov = np.array([[r]])
        if confidence >= 0.05:
            kf.correct(np.array([[z]]))
        else:
            kf.statePost = kf.statePre.copy()
            kf.errorCovPost = kf.errorCovPre.copy()
        x = kf.statePost
        x[1, 0] = float(np.clip(x[1, 0], -1.5, 1.5))
        x[2, 0] = float(np.clip(x[2, 0], -1.0, 1.0))
        kf.statePost = x
        self.last = self.forecast(t, lag_s)
        return self.last

    def forecast(self, t: float, lag_s: float = 4.0) -> Forecast:
        x = self.kf.statePost.ravel()
        P = self.kf.errorCovPost
        try:
            L = np.linalg.cholesky(P + 1e-9 * np.eye(3))
        except np.linalg.LinAlgError:
            L = np.diag(np.sqrt(np.abs(np.diag(P))))
        S = x[None, :] + self.rng.standard_normal((self.n, 3)) @ L.T
        # physics prior: foam can grow at most exponentially once heating is steady,
        # so growth-rate acceleration is not extrapolated upward (only saturation, kd<0)
        kd = np.minimum(S[:, 2], 0.01)
        tau = self._crossing(S[:, 0], S[:, 1], kd)
        foam = math.exp(x[0]) - self.H0
        rise = x[1] * (foam + self.H0)
        p = lambda T: float(np.mean(tau <= T))
        srt = np.sort(tau)
        p50 = float(srt[int(0.5 * self.n)])
        p10 = float(srt[int(0.1 * self.n)])
        return Forecast(t=t, foam_mm=foam, rate=float(x[1]), accel=float(x[2]),
                        eta_p50=p50, eta_p10=p10, p_10s=p(10.0), p_20s=p(20.0),
                        p_lag=p(lag_s + 3.0), rise_mm_s=float(rise),
                        fill=float(max(0.0, foam) / max(self.D0, 1e-3)))

    def _crossing(self, y, k, kd):
        """Smallest tau in (0, HORIZON] with y + k tau + kd tau^2 / 2 = Y_over."""
        c = y - self.Y_over
        a = 0.5 * kd
        tau = np.full_like(y, np.inf)
        already = c >= 0
        tau[already] = 0.0
        lin = (np.abs(a) < 1e-6) & ~already & (k > 0)
        tau[lin] = -c[lin] / k[lin]
        quad = (np.abs(a) >= 1e-6) & ~already
        disc = k * k - 4 * a * c
        okq = quad & (disc >= 0)
        sq = np.sqrt(np.where(okq, disc, 0.0))
        r1 = np.where(okq, (-k - sq) / np.where(quad, 2 * a, 1), np.inf)
        r2 = np.where(okq, (-k + sq) / np.where(quad, 2 * a, 1), np.inf)
        r1 = np.where(r1 > 0, r1, np.inf)
        r2 = np.where(r2 > 0, r2, np.inf)
        tau = np.where(okq, np.minimum(r1, r2), tau)
        tau[tau > self.HORIZON] = np.inf
        return tau


class LinearBaseline:
    """Linear extrapolation of the freeboard over the last `window` seconds."""

    def __init__(self, window=3.0):
        self.buf = deque()
        self.window = window

    def update(self, t, freeboard_mm):
        self.buf.append((t, freeboard_mm))
        while self.buf and t - self.buf[0][0] > self.window:
            self.buf.popleft()
        if len(self.buf) < 4:
            return math.inf
        ts, fs = np.array(self.buf).T
        slope = np.polyfit(ts - ts[-1], fs, 1)[0]
        if slope >= -0.05:
            return math.inf
        return float(max(0.0, fs[-1]) / -slope)


class ThresholdBaseline:
    """What a simple 'boil-over sensor' does: alarm when foam is near the rim."""

    def __init__(self, D0_mm, frac=0.35, min_mm=12.0):
        self.level = max(min_mm, frac * D0_mm)

    def alarm(self, freeboard_mm):
        return freeboard_mm <= self.level
