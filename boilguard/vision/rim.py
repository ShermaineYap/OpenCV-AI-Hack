"""Pot-rim localisation and elliptical polar unwrapping (OpenCV 5).

A range-hood camera sees the circular rim of a pot as an ellipse (camera tilt).
We find it once at start-up from a median of the first frames:

1. Median-of-frames background (removes bubbles, noise, passing hands).
2. CLAHE + Gaussian blur + Canny edges, closed with a small dilation.
3. ``cv2.findContours`` -> ``cv2.fitEllipse`` on every long contour.
4. Each candidate is scored by *edge support* (fraction of 360 sampled ellipse
   points that land on an edge), size plausibility and roundness.
5. The winner is refined radially: we unwrap a band around it with
   ``cv2.remap`` and locate the inner edge of the bright rim lip from the
   angle-averaged radial profile. That inner edge is where liquid spills over.

The resulting :class:`RimModel` builds remap tables that turn any frame into a
polar image (rows = angle, cols = normalised radius rho, rho=1 at the inner rim),
where a rising liquid surface is a vertical front moving right.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RimModel:
    cx: float
    cy: float
    a: float          # semi-axis along `angle` (px), at the inner rim
    b: float          # other semi-axis (px)
    angle: float      # degrees, cv2.fitEllipse convention
    score: float = 0.0

    def as_cv_ellipse(self, scale: float = 1.0):
        return ((self.cx, self.cy), (2 * self.a * scale, 2 * self.b * scale), self.angle)

    def point(self, rho: np.ndarray, theta: np.ndarray):
        ca, sa = math.cos(math.radians(self.angle)), math.sin(math.radians(self.angle))
        x = self.a * rho * np.cos(theta)
        y = self.b * rho * np.sin(theta)
        return self.cx + x * ca - y * sa, self.cy + x * sa + y * ca

    def polar_maps(self, rho_min=0.55, rho_max=1.12, n_r=240, n_theta=360):
        rho = np.linspace(rho_min, rho_max, n_r, dtype=np.float32)
        th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False, dtype=np.float32)
        R, T = np.meshgrid(rho, th)
        X, Y = self.point(R, T)
        return X.astype(np.float32), Y.astype(np.float32), rho

    def mask(self, shape, scale=1.0):
        m = np.zeros(shape[:2], np.uint8)
        cv2.ellipse(m, self.as_cv_ellipse(scale), 255, -1, cv2.LINE_AA)
        return m


def _edge_support(edges: np.ndarray, ell, n=360) -> float:
    (cx, cy), (w, h), ang = ell
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    x, y = w / 2 * np.cos(th), h / 2 * np.sin(th)
    X = np.round(cx + x * ca - y * sa).astype(int)
    Y = np.round(cy + x * sa + y * ca).astype(int)
    H, W = edges.shape
    ok = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
    if ok.mean() < 0.9:
        return 0.0
    return float((edges[Y[ok], X[ok]] > 0).sum() / n)


def detect_rim(frames: list[np.ndarray], min_frac=0.25, max_frac=0.98) -> RimModel:
    """Detect the pot rim from a list of BGR frames (calm, start of recording)."""
    stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames[:15]])
    bg = np.median(stack, axis=0).astype(np.uint8)
    H, W = bg.shape
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = cv2.GaussianBlur(clahe.apply(bg), (5, 5), 1.2)
    edges = cv2.Canny(g, 40, 110)
    edges_d = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges_d, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    short = min(H, W)
    cands = []
    for c in contours:
        if len(c) < 80:
            continue
        ell = cv2.fitEllipse(c)
        (cx, cy), (w, h), ang = ell
        major, minor = max(w, h), min(w, h)
        if not (min_frac * short <= major <= max_frac * short * 1.15) or minor / max(major, 1) < 0.6:
            continue
        if not (0 <= cx < W and 0 <= cy < H):
            continue
        sup = _edge_support(edges_d, ell)
        if sup >= 0.3:
            cands.append((sup * (0.6 + 0.4 * minor / major), major, ell))
    best, best_score = None, 0.0
    if cands:
        top = max(s for s, _, _ in cands)
        # Concentric rings (liquid edge, lip, pot body) all fit well; the rim is the
        # largest well-supported one, refined inward to the lip's inner edge below.
        strong = [c for c in cands if c[0] >= 0.45 * top]
        best_score, _, best = max(strong, key=lambda c: c[1])
    if best is None or best_score < 0.25:
        circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.5, minDist=short,
                                   param1=110, param2=40,
                                   minRadius=int(min_frac * short / 2), maxRadius=int(max_frac * short / 2))
        if circles is None:
            raise RuntimeError("Pot rim not found: make sure the whole pot rim is in view")
        x, y, r = circles[0, 0]
        best, best_score = ((float(x), float(y)), (2 * float(r), 2 * float(r)), 0.0), 0.3
    (cx, cy), (w, h), ang = best
    model = RimModel(cx, cy, w / 2, h / 2, ang, best_score)
    return refine_rim(model, bg)


def _polar(model: RimModel, gray: np.ndarray, lo, hi, n_r, n_t=360):
    X, Y, rho = model.polar_maps(lo, hi, n_r, n_t)
    pol = cv2.remap(gray, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    return cv2.GaussianBlur(pol, (0, 0), sigmaX=1.2, sigmaY=1.5), rho


def refine_rim(model: RimModel, gray_bg: np.ndarray, lip_max=0.07) -> RimModel:
    """Snap the ellipse to the inner edge of the rim lip (the spill line).

    Physical cue: hood lights sit next to the camera, so the up-facing rim lip is
    a thin *bright ridge* in the angle-averaged radial profile, darker on both
    sides (vertical inner wall below it, stove top outside). A bright liquid such
    as milk is a wide plateau, not a ridge, so it is not confused with the lip.

    1. Elliptical polar unwrap (cv2.remap) of rho in [0.75, 1.15].
    2. Ridge with the highest two-sided prominence -> lip.
    3. Spill line = strongest outward-rising step inside the ridge.
    4. Per-angle sub-pixel edge points near that radius -> cv2.fitEllipse with
       one round of residual trimming, giving an accurate rim ellipse.
    """
    pol, rho = _polar(model, gray_bg, 0.75, 1.15, 400)
    drho = rho[1] - rho[0]
    P = pol.mean(axis=0)
    w = max(3, int(0.08 / drho))
    best_i, best_prom = None, 0.0
    for i in range(2, len(P) - 2):
        if P[i] < P[i - 2:i + 3].max():
            continue
        left = P[max(0, i - w):i].min() if i > 0 else P[i]
        right = P[i + 1:i + 1 + w].min() if i + 1 < len(P) else P[i]
        prom = P[i] - max(left, right)
        if prom > best_prom:
            best_i, best_prom = i, prom
    if best_i is None or best_prom < 6:
        return model
    j0 = max(1, best_i - int(lip_max / drho))
    gpos = np.gradient(P)
    k = j0 + int(np.argmax(gpos[j0:best_i + 1]))
    s0 = float(rho[k])
    # per-angle edge points around the spill line
    pol2, rho2 = _polar(model, gray_bg, s0 - 0.02, s0 + 0.02, 80)
    # signed: going outward the lip is brighter than the wall below it, while
    # the lip -> stove edge falls; angles where this fails are trimmed below.
    gm = np.maximum(cv2.Sobel(pol2, cv2.CV_32F, 1, 0, ksize=3), 0)
    idx = gm.argmax(axis=1)
    strength = gm[np.arange(len(idx)), idx]
    th = np.linspace(0, 2 * np.pi, pol2.shape[0], endpoint=False)
    keep = (strength > max(1.0, np.percentile(strength, 30))) & (idx > 1) & (idx < pol2.shape[1] - 2)
    r_sub = idx.astype(np.float64)
    ii = np.where(keep)[0]
    y0, y1, y2 = gm[ii, idx[ii] - 1], gm[ii, idx[ii]], gm[ii, idx[ii] + 1]
    den = y0 - 2 * y1 + y2
    r_sub[ii] += np.where(np.abs(den) > 1e-6, 0.5 * (y0 - y2) / np.where(np.abs(den) > 1e-6, den, 1), 0)
    rr = np.interp(r_sub, np.arange(len(rho2)), rho2)
    X, Y = model.point(rr[keep], th[keep])
    pts = np.stack([X, Y], 1).astype(np.float32)
    if len(pts) < 20:
        return RimModel(model.cx, model.cy, model.a * s0, model.b * s0, model.angle, model.score)
    ell = cv2.fitEllipse(pts)
    for _ in range(2):
        cand = RimModel(ell[0][0], ell[0][1], ell[1][0] / 2, ell[1][1] / 2, ell[2])
        # residual in normalised radius
        dx, dy = pts[:, 0] - cand.cx, pts[:, 1] - cand.cy
        ca, sa = math.cos(math.radians(cand.angle)), math.sin(math.radians(cand.angle))
        u, v = dx * ca + dy * sa, -dx * sa + dy * ca
        res = np.abs(np.sqrt((u / cand.a) ** 2 + (v / cand.b) ** 2) - 1)
        good = res < max(0.006, 2.5 * np.median(res))
        if good.sum() < 20:
            break
        pts = pts[good]
        ell = cv2.fitEllipse(pts)
    (cx, cy), (w2, h2), ang = ell
    return RimModel(cx, cy, w2 / 2, h2 / 2, ang, model.score)
