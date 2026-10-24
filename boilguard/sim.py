"""Physically-grounded boil-over simulator (the BoilGuard synthetic benchmark).

Renders what a downward-looking range-hood camera sees when a pot of milk, soup or
pasta water is heated, and exposes exact ground truth (foam height, freeboard,
overflow time). It is also a closed-loop environment: heat, hood-fan and human
actions change the physics, so policies can be compared on prevented overflows.

Physics (all lengths in mm, time in s)
--------------------------------------
* Foam height h above the calm surface follows logistic growth driven by the
  *effective* heater power Pe, minus drainage when power is reduced:

      dh/dt = Pe * g0 * h * (1 - h / hmax)  -  lam * (1 - Pe) * h

  Exponential onset with saturation is the standard first-order description of
  protein/starch-stabilised foams (gas generation vs. film drainage).
* The heater has thermal inertia:  dPe/dt = (P - Pe) / tau_heat
  (induction ~1 s, ceramic/coil ~4-8 s). This lag is why early prediction matters.
* Overflow happens when h reaches the calm freeboard D0 (surface reaches the rim).
* Camera geometry is a pinhole looking down from height H above the rim, so a
  surface d mm below the rim of a pot of radius R appears with radius
  r = R_px * H / (H + d). The vision pipeline inverts exactly this relation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

LIQUIDS = {
    # name: (liquid BGR, foam BGR)
    "milk": ((214, 222, 226), (246, 249, 250)),
    "pasta_water": ((176, 186, 184), (232, 236, 236)),
    "tomato_soup": ((40, 78, 178), (120, 160, 228)),
    "coconut_curry": ((92, 168, 214), (196, 226, 240)),
}


@dataclass
class SimParams:
    seed: int = 0
    width: int = 640
    height: int = 480
    fps: float = 15.0
    duration: float = 75.0
    liquid: str = "milk"
    pot_radius_mm: float = 95.0
    cam_height_mm: float = 420.0       # camera height above the rim
    freeboard_mm: float = 38.0         # D0: calm surface depth below rim
    t_boil: float = 12.0               # foaming onset
    g0: float = 0.28                   # foam growth rate at full power (1/s)
    hmax_mm: float = 80.0              # foam saturation height
    lam: float = 0.45                  # drainage rate when power reduced (1/s)
    tau_heat: float = 4.0              # heater thermal lag (s)
    heater: str = "ceramic"
    steam_max: float = 0.35            # 0..1 steam opacity
    tilt_deg: float = 8.0
    tilt_dir_deg: float = 30.0
    rim_px: float = 160.0
    center_jitter: tuple = (0.0, 0.0)
    occlusions: int = 1                # spoon/hand passes
    gain: float = 1.0
    noise: float = 3.0
    shake_px: float = 0.8
    jpeg_q: int = 80
    light_dir_deg: float = 210.0

    @property
    def will_overflow_uncontrolled(self) -> bool:
        return self.hmax_mm > self.freeboard_mm

    def to_dict(self):
        return asdict(self)


def random_params(seed: int, overflow: bool | None = None, hard: bool = False,
                  width: int = 640, height: int = 480) -> SimParams:
    """Sample a scenario. overflow=None -> 75% overflow-prone, 25% safe simmer."""
    rng = np.random.default_rng(seed)
    if overflow is None:
        overflow = rng.random() < 0.75
    D0 = float(rng.uniform(22, 55))
    hmax = D0 * (rng.uniform(1.25, 3.0) if overflow else rng.uniform(0.35, 0.85))
    heater = str(rng.choice(["induction", "ceramic", "coil"], p=[0.35, 0.4, 0.25]))
    tau = {"induction": rng.uniform(0.8, 1.8), "ceramic": rng.uniform(3.5, 6.0),
           "coil": rng.uniform(5.0, 8.0)}[heater]
    steam_hi = 0.75 if hard else 0.5
    scale = min(width, height) / 480.0
    return SimParams(
        seed=seed, width=width, height=height,
        liquid=str(rng.choice(list(LIQUIDS), p=[0.45, 0.25, 0.15, 0.15])),
        pot_radius_mm=float(rng.uniform(75, 120)),
        cam_height_mm=float(rng.uniform(320, 560)),
        freeboard_mm=D0,
        t_boil=float(rng.uniform(8, 22)),
        g0=float(rng.uniform(0.18, 0.42)),
        hmax_mm=float(hmax),
        lam=float(rng.uniform(0.3, 0.6)),
        tau_heat=float(tau), heater=heater,
        steam_max=float(rng.uniform(0.05, steam_hi)),
        tilt_deg=float(rng.uniform(0, 14)),
        tilt_dir_deg=float(rng.uniform(0, 360)),
        rim_px=float(rng.uniform(140, 180) * scale),
        center_jitter=(float(rng.uniform(-40, 40) * scale), float(rng.uniform(-25, 25) * scale)),
        occlusions=int(rng.choice([0, 1, 2], p=[0.4, 0.4, 0.2]) + (1 if hard else 0)),
        gain=float(rng.uniform(0.75, 1.15)),
        noise=float(rng.uniform(1.5, 6.0 if hard else 4.5)),
        shake_px=float(rng.uniform(0.0, 2.0 if hard else 1.2)),
        jpeg_q=int(rng.integers(55, 92)),
        light_dir_deg=float(rng.uniform(0, 360)),
        duration=75.0,
    )


@dataclass
class SimState:
    t: float = 0.0
    h: float = 0.0          # foam height (mm)
    P: float = 1.0          # commanded heater power
    Pe: float = 1.0         # effective heater power
    steam: float = 0.0
    fan: bool = False
    spill: float = 0.0
    overflow_time: float | None = None
    events: list = field(default_factory=list)


class BoilSim:
    """Closed-loop boil-over environment with a renderer."""

    def __init__(self, p: SimParams):
        self.p = p
        self.rng = np.random.default_rng(p.seed + 1000)
        self.s = SimState()
        self.dt = 1.0 / p.fps
        self.frame_idx = 0
        W, H = p.width, p.height
        self.c = np.array([W / 2 + p.center_jitter[0], H / 2 + p.center_jitter[1]], np.float64)
        # Ellipse from tilt: minor axis compressed by cos(tilt) along tilt direction.
        self.ax_major = p.rim_px
        self.ax_minor = p.rim_px * math.cos(math.radians(p.tilt_deg))
        self.ang = p.tilt_dir_deg % 180.0
        self._build_static()
        self._bubbles = self._init_bubbles(450)
        self._occl = self._init_occlusions()
        self._steam_tex = self._noise_texture(int(H * 1.6), int(W * 1.6), 9).astype(np.float32)

    # ------------------------------------------------------------------ physics
    def surface_radius_px(self, depth_mm: float) -> float:
        p = self.p
        return p.rim_px * p.cam_height_mm / (p.cam_height_mm + max(depth_mm, 0.0))

    @property
    def freeboard_mm(self) -> float:
        return max(self.p.freeboard_mm - self.s.h, 0.0)

    def set_power(self, P: float, source: str = "agent"):
        if abs(P - self.s.P) > 1e-6:
            self.s.events.append({"t": round(self.s.t, 2), "event": "power", "P": P, "by": source})
        self.s.P = float(np.clip(P, 0.0, 1.0))

    def set_fan(self, on: bool, source: str = "agent"):
        if on != self.s.fan:
            self.s.events.append({"t": round(self.s.t, 2), "event": "fan", "on": on, "by": source})
        self.s.fan = on

    def _physics(self, dt: float):
        p, s = self.p, self.s
        s.Pe += (s.P - s.Pe) * (1 - math.exp(-dt / p.tau_heat))
        if s.t >= p.t_boil:
            if s.h <= 0 and s.Pe > 0.6:
                s.h = 0.4
            if s.h > 0:
                grow = s.Pe * p.g0 * s.h * (1 - s.h / p.hmax_mm)
                drain = p.lam * (1 - s.Pe) * s.h
                s.h = max(0.0, s.h + (grow - drain) * dt)
        if s.h >= p.freeboard_mm:
            if s.overflow_time is None:
                s.overflow_time = s.t
                s.events.append({"t": round(s.t, 2), "event": "OVERFLOW"})
            excess = s.Pe * p.g0 * s.h * max(0.0, 1 - s.h / p.hmax_mm)
            s.spill += excess * dt
            s.h = min(s.h, p.freeboard_mm * 1.02)
        else:
            s.spill = max(0.0, s.spill - 0.5 * dt)
        heat = 1.0 / (1 + math.exp(-(s.t - p.t_boil + 4) / 2.5)) * min(1.0, s.Pe * 1.3 + 0.2)
        target = p.steam_max * heat * (0.22 if s.fan else 1.0)
        s.steam += (target - s.steam) * (1 - math.exp(-dt / (2.0 if s.fan else 4.0)))

    def step(self, render: bool = True) -> np.ndarray | None:
        """Advance one frame and return the rendered BGR frame (None if render=False)."""
        self._physics(self.dt)
        frame = self.render() if render else None
        self.s.t += self.dt
        self.frame_idx += 1
        return frame

    def truth(self) -> dict:
        s = self.s
        return {"t": round(s.t, 4), "foam_mm": s.h, "freeboard_mm": self.freeboard_mm,
                "P": s.P, "Pe": s.Pe, "steam": s.steam, "fan": s.fan,
                "overflowed": s.overflow_time is not None}

    def natural_overflow_time(self) -> float | None:
        """Overflow time with the heater left at full power (no intervention)."""
        twin = BoilSim.__new__(BoilSim)
        twin.p, twin.s, twin.dt = self.p, SimState(), self.dt
        steps = int(self.p.duration / self.dt)
        for _ in range(steps):
            twin._physics(twin.dt)
            if twin.s.overflow_time is not None:
                return twin.s.overflow_time
            twin.s.t += twin.dt
        return None

    # ---------------------------------------------------------------- rendering
    def _noise_texture(self, h, w, octaves):
        tex = np.zeros((h, w), np.float32)
        amp, tot = 1.0, 0.0
        for o in range(octaves):
            sz = 2 ** (o + 2)
            small = self.rng.random((sz, int(sz * w / h) + 1)).astype(np.float32)
            tex += amp * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
            tot += amp
            amp *= 0.55
        tex /= tot
        tex = (tex - tex.min()) / (tex.max() - tex.min() + 1e-6)
        return tex

    def _ellipse_pts(self, scale: float, center=None, n=180):
        c = self.c if center is None else center
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        a, b = self.ax_major * scale, self.ax_minor * scale
        ca, sa = math.cos(math.radians(self.ang)), math.sin(math.radians(self.ang))
        x = a * np.cos(th); y = b * np.sin(th)
        X = c[0] + x * ca - y * sa
        Y = c[1] + x * sa + y * ca
        return np.stack([X, Y], 1)

    def _ellipse_mask(self, scale, center=None, shape=None):
        H, W = shape or (self.p.height, self.p.width)
        m = np.zeros((H, W), np.uint8)
        pts = np.round(self._ellipse_pts(scale, center) * 8).astype(np.int32)
        cv2.fillPoly(m, [pts], 255, cv2.LINE_AA, shift=3)
        return m

    def _build_static(self):
        p = self.p
        H, W = p.height, p.width
        rng = self.rng
        # stove top: dark glass/steel with sheen and a few crumbs
        base = self._noise_texture(H, W, 5)
        yy, xx = np.mgrid[0:H, 0:W]
        sheen = np.exp(-(((xx - W * 0.2) / (W * 0.6)) ** 2 + ((yy - H * 0.1) / (H * 0.5)) ** 2))
        stove = 28 + 22 * base + 30 * sheen
        bg = np.dstack([stove * 1.05, stove, stove * 0.95]).clip(0, 255).astype(np.uint8)
        for _ in range(25):
            cx, cy = int(rng.integers(0, W)), int(rng.integers(0, H))
            cv2.circle(bg, (cx, cy), int(rng.integers(1, 3)), (60, 70, 80), -1, cv2.LINE_AA)
        # pot handle
        ang = math.radians(self.ang + 90 + rng.uniform(-30, 30))
        hx = self.c + np.array([math.cos(ang), math.sin(ang)]) * self.ax_major * 1.02
        hx2 = self.c + np.array([math.cos(ang), math.sin(ang)]) * self.ax_major * 1.75
        cv2.line(bg, tuple(np.int32(hx)), tuple(np.int32(hx2)), (35, 35, 38), int(p.rim_px * 0.16), cv2.LINE_AA)
        cv2.line(bg, tuple(np.int32(hx)), tuple(np.int32(hx2)), (70, 70, 74), int(p.rim_px * 0.05), cv2.LINE_AA)
        # pot outer body: visible as a thin crescent on the side facing the camera tilt
        tilt_off = self.c + np.array([math.cos(math.radians(self.ang + 90)),
                                      math.sin(math.radians(self.ang + 90))]) * p.rim_px * math.sin(math.radians(p.tilt_deg)) * 0.25
        outer = self._ellipse_mask(1.035, tilt_off)
        bg[outer > 0] = (bg[outer > 0] * 0.3 + np.array([80, 80, 86]) * 0.7).astype(np.uint8)
        # rim lip: bright steel ring with angular highlight
        lip = cv2.subtract(self._ellipse_mask(1.035), self._ellipse_mask(1.0))
        theta = np.arctan2(yy - self.c[1], xx - self.c[0])
        light = math.radians(p.light_dir_deg)
        lipval = 175 + 60 * np.cos(theta - light)
        lipimg = np.dstack([lipval, lipval, lipval * 1.02]).clip(0, 255).astype(np.uint8)
        bg = np.where(lip[..., None] > 0, lipimg, bg)
        self.bg = bg
        # inner wall texture (drawn under the liquid): brushed steel, darker away from light
        brushed = self._noise_texture(H, W, 6)
        wall = 105 + 55 * np.cos(theta - light - math.pi) + 18 * (brushed - 0.5)
        r_norm = np.hypot(xx - self.c[0], yy - self.c[1]) / p.rim_px
        wall = wall * (0.75 + 0.25 * np.clip(r_norm, 0, 1))
        self.wall = np.dstack([wall, wall * 1.0, wall * 1.03]).clip(0, 255).astype(np.uint8)
        self.inner_mask = self._ellipse_mask(1.0)
        self._theta, self._light = theta, light
        shade = (1.0 + 0.08 * np.cos(theta - light)) * 235.0 / 1.08
        self._shade3 = np.dstack([shade] * 3).clip(0, 255).astype(np.uint8)
        self._dark_wall = (self.wall * 0.75).astype(np.uint8)
        self._k5 = np.ones((3, 3), np.uint8)
        self._noise_bank = [self.rng.normal(0, p.noise, (H, W, 3)).astype(np.int16) for _ in range(6)]
        self._white = np.full((H, W, 3), 228, np.uint8)

    def _init_bubbles(self, n):
        r = np.sqrt(self.rng.random(n)) * 0.97
        a = self.rng.random(n) * 2 * np.pi
        size = self.rng.uniform(0.012, 0.05, n)
        phase = self.rng.random(n)
        return {"r": r, "a": a, "size": size, "phase": phase, "speed": self.rng.uniform(0.3, 1.5, n)}

    def _init_occlusions(self):
        occ = []
        for _ in range(self.p.occlusions):
            t0 = float(self.rng.uniform(self.p.t_boil - 5, self.p.t_boil + 25))
            occ.append({"t0": t0, "dur": float(self.rng.uniform(0.8, 2.2)),
                        "ang": float(self.rng.uniform(0, 2 * np.pi)),
                        "kind": str(self.rng.choice(["spoon", "hand"]))})
        return occ

    def render(self) -> np.ndarray:
        p, s = self.p, self.s
        img = self.bg.copy()
        cv2.copyTo(self.wall, self.inner_mask, img)
        depth = self.freeboard_mm
        scale = self.surface_radius_px(depth) / p.rim_px
        # perspective: deeper surface shifts slightly along tilt direction
        shift = depth / p.cam_height_mm * math.sin(math.radians(p.tilt_deg)) * p.rim_px * 0.6
        ang = math.radians(self.ang + 90)
        sc = self.c + np.array([math.cos(ang), math.sin(ang)]) * shift
        smask = self._ellipse_mask(scale, sc)
        liq, foam = LIQUIDS[p.liquid]
        foam_frac = float(np.clip(s.h / max(p.freeboard_mm * 0.35, 1e-3), 0, 1))
        col = np.array(liq, np.float32) * (1 - foam_frac) + np.array(foam, np.float32) * foam_frac
        surf = cv2.multiply(self._shade3, (float(col[0]), float(col[1]), float(col[2]), 0.0), scale=1.0 / 235.0)
        # soft contact shadow ring where the surface meets the wall
        ring = cv2.subtract(cv2.dilate(smask, self._k5), smask)
        cv2.copyTo(self._dark_wall, ring, img)
        cv2.copyTo(surf, smask, img)
        self._draw_bubbles(img, sc, scale, col, foam_frac)
        if s.spill > 0.05 or s.overflow_time is not None:
            self._draw_spill(img, foam)
        self._draw_occlusions(img)
        img = self._apply_steam(img)
        img = self._apply_camera(img)
        return img

    def _draw_bubbles(self, img, sc, scale, col, foam_frac):
        p, s, b = self.p, self.s, self._bubbles
        if s.t < p.t_boil - 3:
            return
        pre = np.clip((s.t - (p.t_boil - 3)) / 3, 0, 1)
        n_active = int(len(b["r"]) * min(1.0, 0.04 * pre + 0.96 * foam_frac ** 0.7))
        if n_active <= 0:
            return
        b["phase"] = (b["phase"] + self.dt * b["speed"] * (0.5 + s.Pe)) % 1.0
        b["a"] += self.dt * 0.02 * (self.rng.random(len(b["a"])) - 0.5)
        R = p.rim_px * scale
        ca, sa = math.cos(math.radians(self.ang)), math.sin(math.radians(self.ang))
        k = self.ax_minor / self.ax_major
        col_dark = tuple(int(v) for v in np.clip(col * 0.78, 0, 255))
        col_hi = tuple(int(v) for v in np.clip(col * 1.08 + 12, 0, 255))
        for i in range(n_active):
            life = math.sin(math.pi * b["phase"][i])
            rad = max(1, int(R * b["size"][i] * (0.4 + 0.8 * life) * (0.5 + foam_frac)))
            x = b["r"][i] * R * math.cos(b["a"][i]); y = b["r"][i] * R * math.sin(b["a"][i]) * k
            if b["r"][i] * R + rad > R * 0.995:
                continue
            X = int(sc[0] + x * ca - y * sa); Y = int(sc[1] + x * sa + y * ca)
            cv2.circle(img, (X, Y), rad, col_dark, 1, cv2.LINE_AA)
            if rad > 2:
                cv2.circle(img, (X - rad // 3, Y - rad // 3), max(1, rad // 4), col_hi, -1, cv2.LINE_AA)

    def _draw_spill(self, img, foam):
        p, s = self.p, self.s
        amt = min(1.0, s.spill / 8.0 + 0.15)
        rng = np.random.default_rng(p.seed + 7)
        n = int(6 + 30 * amt)
        for i in range(n):
            a = rng.uniform(0, 2 * np.pi)
            rr = 1.0 + rng.uniform(0.0, 0.12 + 0.35 * amt)
            ca, sa = math.cos(math.radians(self.ang)), math.sin(math.radians(self.ang))
            x = self.ax_major * rr * math.cos(a); y = self.ax_minor * rr * math.sin(a)
            X = int(self.c[0] + x * ca - y * sa); Y = int(self.c[1] + x * sa + y * ca)
            cv2.circle(img, (X, Y), int(p.rim_px * rng.uniform(0.04, 0.12) * (0.4 + amt)), foam, -1, cv2.LINE_AA)

    def _draw_occlusions(self, img):
        p, t = self.p, self.s.t
        for o in self._occl:
            u = (t - o["t0"]) / o["dur"]
            if 0 <= u <= 1:
                a = o["ang"]
                d = np.array([math.cos(a), math.sin(a)])
                start = self.c - d * p.rim_px * 2.2 + np.array([-d[1], d[0]]) * p.rim_px * (u * 1.6 - 0.8)
                end = start + d * p.rim_px * 1.7
                if o["kind"] == "spoon":
                    cv2.line(img, tuple(np.int32(start)), tuple(np.int32(end)), (60, 60, 64), max(3, int(p.rim_px * 0.06)), cv2.LINE_AA)
                    cv2.ellipse(img, tuple(np.int32(end)), (int(p.rim_px * 0.18), int(p.rim_px * 0.11)),
                                math.degrees(a), 0, 360, (90, 90, 96), -1, cv2.LINE_AA)
                else:
                    cv2.line(img, tuple(np.int32(start)), tuple(np.int32(end)), (120, 150, 200), max(6, int(p.rim_px * 0.28)), cv2.LINE_AA)

    def _apply_steam(self, img):
        s = self.s.steam
        if s < 0.01:
            return img
        H, W = img.shape[:2]
        th, tw = self._steam_tex.shape
        oy = int((self.s.t * 25) % (th - H))
        ox = int((th - H) / 2 + 20 * math.sin(self.s.t * 0.4)) % max(1, tw - W)
        a = np.ascontiguousarray(self._steam_tex[oy:oy + H, ox:ox + W])
        a = np.clip(a * (0.95 * s) + 0.35 * s, 0, 0.92).astype(np.float32)
        return cv2.blendLinear(img, self._white, 1.0 - a, a)

    def _apply_camera(self, img):
        p = self.p
        H, W = img.shape[:2]
        flick = 1.0 + 0.02 * math.sin(self.s.t * 2 * math.pi * 0.7)
        out = cv2.convertScaleAbs(img, alpha=p.gain * flick)
        if p.shake_px > 0:
            dx, dy = self.rng.normal(0, p.shake_px, 2)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            out = cv2.warpAffine(out, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        noise = self._noise_bank[self.frame_idx % len(self._noise_bank)]
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, p.jpeg_q])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # ------------------------------------------------------------ calibration
    def true_rim(self):
        """Ground-truth rim ellipse in cv2.fitEllipse format."""
        return ((float(self.c[0]), float(self.c[1])),
                (2 * self.ax_major, 2 * self.ax_minor), float(self.ang))
