"""Freeboard (rim-to-surface distance) measurement from a single frame.

Key idea: the *inner wall of the pot is static background*. In the elliptical
polar unwrap (rows = angle, cols = normalised radius rho, rho = 1 at the spill
line) the visible wall is the band between the liquid edge and the rim. As
foam rises it covers the wall from the inside out, so per angle we only need to
find where "differs from the calm reference" turns into "matches the reference".

Per frame:
1. ``cv2.remap`` into polar space (precomputed maps, ~1 ms).
2. Photometric normalisation against the rim lip (always visible): robust
   gain/offset fit compensates steam haze, exposure drift and flicker. The gain
   doubles as a steam/fog index.
3. Colour distance to the calm reference, smoothed with ``cv2.GaussianBlur``.
4. Per-angle 1-D change-point search (cumulative sums, fully vectorised) for the
   foam front between the calm liquid edge and the rim.
5. Rows whose rim lip is itself covered (hand, spoon) are rejected; the front
   radius is the median of valid rows, its MAD gives a confidence.
6. Pinhole geometry turns radius into depth below the rim:
   r_surface / r_rim = H / (H + d)  =>  d = H * (1/rho - 1).

Farneback optical flow on the liquid region gives an agitation index that flags
boiling onset before any foam has formed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

from .rim import RimModel, detect_rim


@dataclass
class Measurement:
    t: float
    freeboard_mm: float
    foam_mm: float
    rho: float
    confidence: float
    valid_frac: float
    lip_covered: float
    fog: float
    sharpness: float
    agitation: float
    overflow: bool
    ms: float = 0.0

    def to_dict(self):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


class SurfaceMeter:
    RHO_MIN, RHO_MAX, N_R, N_T = 0.55, 1.08, 265, 360

    def __init__(self, cam_height_mm: float, rim: RimModel | None = None):
        self.H = float(cam_height_mm)
        self.rim = rim
        self.ready = False
        self._prev_small = None

    # ----------------------------------------------------------- calibration
    def calibrate(self, frames: list[np.ndarray]):
        if self.rim is None:
            self.rim = detect_rim(frames)
        X, Y, rho = self.rim.polar_maps(self.RHO_MIN, self.RHO_MAX, self.N_R, self.N_T)
        self.map_x, self.map_y, self.rho = X, Y, rho
        self.i_rim = int(np.searchsorted(rho, 0.995))            # spill line
        self.lip = slice(int(np.searchsorted(rho, 1.008)), int(np.searchsorted(rho, 1.028)))
        # static, always-visible band (bright lip + dark stove just outside) for the haze fit:
        # its strong contrast gives a well-conditioned per-angle regression
        self.haze_band = slice(int(np.searchsorted(rho, 1.005)), int(np.searchsorted(rho, 1.075)))
        self._setup_registration(frames)
        pols = np.stack([self._polar(f, self._register(f)) for f in frames[:15]])
        self.ref = np.median(pols, axis=0)
        self.ref_lip_std = float(self.ref[:, self.lip].std()) + 1e-3
        self.ref_lip_rows = np.median(self.ref[:, self.lip], axis=1)
        gray = cv2.cvtColor(self.ref.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.ref_sharp = float(cv2.Laplacian(gray, cv2.CV_32F).var()) + 1e-3
        # calm liquid edge per angle: outermost strong colour step inside the rim
        g = cv2.GaussianBlur(self.ref, (0, 0), sigmaX=1.2, sigmaY=2.5)
        gx = np.linalg.norm(np.diff(g, axis=1), axis=2)
        lo = int(np.searchsorted(rho, 0.62))
        hi = self.i_rim - 2
        band = gx[:, lo:hi]
        thr = 0.5 * band.max(axis=1, keepdims=True)
        strong = band >= thr
        j = band.shape[1] - 1 - np.argmax(strong[:, ::-1], axis=1)      # outermost strong step
        # climb to the local gradient maximum, then sub-sample parabola refinement
        rr = np.arange(self.N_T)
        for _ in range(4):
            nxt = np.minimum(j + 1, band.shape[1] - 1)
            j = np.where(band[rr, nxt] > band[rr, j], nxt, j)
        jm, jp = np.maximum(j - 1, 0), np.minimum(j + 1, band.shape[1] - 1)
        y0, y1, y2 = band[rr, jm], band[rr, j], band[rr, jp]
        den = y0 - 2 * y1 + y2
        off = np.where(np.abs(den) > 1e-6, 0.5 * (y0 - y2) / np.where(np.abs(den) > 1e-6, den, 1), 0.0)
        pos = lo + j + 0.5 + np.clip(off, -0.5, 0.5)      # diff index i sits between i and i+1
        pos = cv2.medianBlur(pos.astype(np.float32).reshape(-1, 1), 5).ravel()
        self.i_s0 = np.clip(np.round(pos).astype(int), lo, self.i_rim - 1)
        self.rho_s0 = float(np.median(np.interp(pos, np.arange(len(rho)), rho)))
        self.D0 = self.depth(self.rho_s0)
        # noise level of the colour distance on calm frames
        ds = [self._dist(p) for p in pols[-5:]]
        wall_mask = np.zeros_like(ds[0], bool)
        for r in range(self.N_T):
            wall_mask[r, self.i_s0[r] + 2:self.i_rim] = True
        noise = np.concatenate([d[wall_mask] for d in ds]) if wall_mask.any() else np.array([5.0])
        self.tau = float(np.clip(np.percentile(noise, 95) * 2.0 + 6.0, 10.0, 45.0))
        self._prev_small = None
        self.ready = True
        return self

    def _setup_registration(self, frames):
        """Rim-anchored phase correlation: a window around the rim (lip + stove,
        interior masked out because foam changes it) is registered against the
        calm median with cv2.phaseCorrelate to cancel hood vibration/camera shake."""
        r = self.rim
        R = max(r.a, r.b)
        x0, y0 = int(max(0, r.cx - 1.35 * R)), int(max(0, r.cy - 1.35 * R))
        h, w = frames[0].shape[:2]
        x1, y1 = int(min(w, r.cx + 1.35 * R)), int(min(h, r.cy + 1.35 * R))
        self._roi = (x0, y0, x1, y1)
        m_out = np.zeros((h, w), np.float32)
        cv2.ellipse(m_out, r.as_cv_ellipse(1.3), 1.0, -1)
        cv2.ellipse(m_out, r.as_cv_ellipse(0.96), 0.0, -1)
        m_out = cv2.GaussianBlur(m_out, (0, 0), 3)
        self._reg_mask = m_out[y0:y1, x0:x1]
        self._reg_win = cv2.createHanningWindow(self._reg_mask.shape[::-1], cv2.CV_32F)
        g = np.median(np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames[:15]]), 0)
        self._reg_ref = self._reg_patch(g.astype(np.float32))
        self.shift = (0.0, 0.0)

    def _reg_patch(self, gray):
        x0, y0, x1, y1 = self._roi
        p = gray[y0:y1, x0:x1].astype(np.float32)
        p = p - float((p * self._reg_mask).sum() / (self._reg_mask.sum() + 1e-6))
        return p * self._reg_mask

    def _register(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        (dx, dy), resp = cv2.phaseCorrelate(self._reg_ref, self._reg_patch(g), self._reg_win)
        # Hood vibration is a few pixels. Large or weak peaks mean something else moved
        # (a hand or spoon crossing the rim): keep the previous estimate instead.
        px, py = self.shift
        if resp < 0.1 or abs(dx) > 6 or abs(dy) > 6:
            dx, dy = 0.0, 0.0            # unreliable (steam, occluder): assume mean position
        self.shift = (float(dx), float(dy))
        return self.shift

    def depth(self, rho: float) -> float:
        return max(0.0, self.H * (1.0 / max(rho, 1e-3) - 1.0))

    # --------------------------------------------------------------- helpers
    def _polar(self, frame, shift=(0.0, 0.0)):
        return cv2.remap(frame, self.map_x + np.float32(shift[0]), self.map_y + np.float32(shift[1]),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)

    def _normalise(self, pol):
        """Undo steam haze with the atmospheric scattering model used in dehazing:

            I = J * t + A * (1 - t)       (t = transmission, A = airlight)

        For every angle we know J on the rim lip and the stove just outside it (calm
        reference) and observe I, so a
        per-angle least-squares line I = t * J + b (b = A(1 - t)) gives t and b. Steam is
        smooth in space, so t and b are median-filtered and smoothed around the rim; the
        median also rejects sectors hidden by a hand or spoon. Inverting the model
        restores the wall/foam appearance; 1 - t is the steam (fog) index.
        """
        cur = pol[:, self.haze_band].reshape(self.N_T, -1)
        ref = self.ref[:, self.haze_band].reshape(self.N_T, -1)
        xm, ym = ref.mean(1, keepdims=True), cur.mean(1, keepdims=True)
        vx = ((ref - xm) ** 2).mean(1)
        cov = ((ref - xm) * (cur - ym)).mean(1)
        t = np.where(vx > 150.0, cov / np.maximum(vx, 1e-6), np.nan)
        t_med = np.nanmedian(t) if np.isfinite(t).any() else 1.0
        t = np.where(np.isfinite(t), t, t_med)
        t = np.clip(t, 0.3, 1.2).astype(np.float32)
        b = (ym.ravel() - t * xm.ravel()).astype(np.float32)

        def smooth(v):
            v3 = np.concatenate([v[-30:], v, v[:30]]).reshape(-1, 1).astype(np.float32)
            v3 = cv2.medianBlur(cv2.medianBlur(v3, 5), 5)
            v3 = cv2.GaussianBlur(v3, (1, 0), sigmaX=0.1, sigmaY=6)
            return v3.ravel()[30:-30]

        t, b = smooth(t), smooth(b)
        out = (pol - b[:, None, None]) / t[:, None, None]
        return out, float(np.median(t))

    @staticmethod
    def _cdist(a, b):
        d = cv2.subtract(a, b)
        sq = cv2.transform(cv2.multiply(d, d), np.ones((1, 3), np.float32))
        return cv2.sqrt(sq).reshape(a.shape[:2])

    def _dist(self, pol_n, ref=None):
        # tolerant to +-1 sample radial misalignment (residual jitter, rim-fit error)
        ref = self.ref if ref is None else ref
        pol_n = np.ascontiguousarray(pol_n, np.float32)
        ref = np.ascontiguousarray(ref, np.float32)
        d0 = self._cdist(pol_n, ref)
        d1 = self._cdist(np.ascontiguousarray(pol_n[:, 1:]), np.ascontiguousarray(ref[:, :-1]))
        d2 = self._cdist(np.ascontiguousarray(pol_n[:, :-1]), np.ascontiguousarray(ref[:, 1:]))
        d = d0.copy()
        d[:, 1:] = np.minimum(d[:, 1:], d1)
        d[:, :-1] = np.minimum(d[:, :-1], d2)
        return cv2.GaussianBlur(d, (0, 0), sigmaX=1.5, sigmaY=2.0)

    # --------------------------------------------------------------- measure
    def measure(self, frame: np.ndarray, t: float) -> Measurement:
        t0 = cv2.getTickCount()
        pol = self._polar(frame, self._register(frame))
        pol_n, gain = self._normalise(pol)
        c0 = int(max(0, self.i_s0.min() - 6))
        c1 = int(min(pol.shape[1], self.lip.stop + 4))
        D = np.zeros(pol.shape[:2], np.float32)
        D[:, c0:c1] = self._dist(pol_n[:, c0:c1], self.ref[:, c0:c1])
        tau = self.tau / max(gain, 0.35)          # haze amplifies normalised noise
        # change point per row on [i_s0 - 2, i_rim): covered (D>tau) then wall (D<tau)
        lo = int(max(0, self.i_s0.min() - 3))
        seg = D[:, lo:self.i_rim]
        a = np.maximum(0.0, tau - seg)            # cost if called "covered"
        b = np.maximum(0.0, seg - tau)            # cost if called "wall"
        n = seg.shape[1]
        rows = np.arange(self.N_T)[:, None]
        cols = np.arange(n + 1)[None, :]
        ca = np.concatenate([np.zeros((self.N_T, 1)), np.cumsum(a, 1)], 1)
        cb = np.concatenate([np.zeros((self.N_T, 1)), np.cumsum(b, 1)], 1)
        start = (self.i_s0 - lo)[:, None]
        # covered from row-specific calm edge up to k; wall from k to rim
        cost = (ca - np.take_along_axis(ca, np.clip(start, 0, n), 1)) * (cols >= start) \
            + (cb[:, -1:] - cb)
        cost = np.where(cols >= start - 2, cost, np.inf)
        k = np.argmin(cost, axis=1)
        front = np.clip(lo + k, 0, len(self.rho) - 1)
        rho_rows = self.rho[np.minimum(front, len(self.rho) - 1)]
        # lip coverage: hand/spoon (or spilled foam) over the rim
        lip_cov = (D[:, self.lip] > tau).mean(axis=1)
        covered_rows = lip_cov > 0.5
        at_rim = front >= self.i_rim - 1
        valid = ~covered_rows | at_rim
        lip_covered = float(covered_rows.mean())
        # occlusion vs overflow: overflow covers the lip AND the front is at the rim
        overflow = bool((covered_rows & at_rim).mean() > 0.3)
        if valid.sum() < 20:
            valid = np.ones_like(valid)
        rv = rho_rows[valid]
        rho_s = float(np.median(rv))
        mad = float(np.median(np.abs(rv - rho_s))) + 1e-4
        valid_frac = float(valid.mean())
        conf = float(np.clip(valid_frac * np.exp(-mad / 0.012) * min(1.0, gain / 0.5), 0.0, 1.0))
        if not overflow:
            conf *= float(1.0 - min(1.0, 1.5 * lip_covered))   # something is over the rim
        fb = 0.0 if overflow else self.depth(min(rho_s, 1.0))
        # sharpness of the lip band (focus / condensation on lens)
        gray = cv2.cvtColor(np.clip(pol_n, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F).var() / self.ref_sharp)
        # agitation: Farneback optical flow on the liquid region (downscaled)
        liq = gray[:, :max(8, int(np.median(front)) - 6)]
        small = cv2.resize(liq, (max(8, liq.shape[1] // 2), self.N_T // 2), interpolation=cv2.INTER_AREA)
        agit = 0.0
        if self._prev_small is not None and self._prev_small.shape == small.shape:
            flow = cv2.calcOpticalFlowFarneback(self._prev_small, small, None, 0.5, 2, 9, 2, 5, 1.1, 0)
            agit = float(np.median(np.linalg.norm(flow, axis=2)) * 2.0)
        self._prev_small = small
        ms = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000
        self.last_front = front
        self.last_valid = valid
        return Measurement(t=t, freeboard_mm=fb, foam_mm=max(0.0, self.D0 - fb), rho=rho_s,
                           confidence=conf, valid_frac=valid_frac, lip_covered=lip_covered,
                           fog=float(np.clip(1.0 - gain, 0, 1)), sharpness=sharp, agitation=agit,
                           overflow=overflow, ms=ms)

    # ------------------------------------------------------------ visualise
    def front_points(self):
        """Image coordinates of the detected foam front (valid rows only)."""
        th = np.linspace(0, 2 * np.pi, self.N_T, endpoint=False)
        rr = self.rho[self.last_front]
        X, Y = self.rim.point(rr, th)
        return np.stack([X, Y], 1)[self.last_valid]
