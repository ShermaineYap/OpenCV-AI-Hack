"""Annotated video rendering and H.264 writing."""
from __future__ import annotations

import math
import subprocess

import cv2
import numpy as np

STATE_COL = {"MONITOR": (160, 160, 160), "BOILING": (0, 200, 255), "ALERT": (0, 140, 255),
             "VERIFY": (255, 160, 0), "RESOLVED": (90, 200, 90), "OVERFLOW": (40, 40, 230)}


def risk_color(p):
    p = float(np.clip(p, 0, 1))
    return (int(80 * (1 - p)), int(200 * (1 - p) + 40 * p), int(80 + 175 * p))


def draw(frame, meter, meas, fc, agent, truth=None, last_actions=None, privacy=False):
    img = frame.copy()
    H, W = img.shape[:2]
    if privacy and meter.rim is not None:
        mask = meter.rim.mask(img.shape, 1.45)
        blur = cv2.GaussianBlur(img, (0, 0), 12)
        img = np.where(mask[..., None] > 0, img, blur)
    if meter.ready:
        cv2.ellipse(img, meter.rim.as_cv_ellipse(), (255, 220, 0), 1, cv2.LINE_AA)
        if meas is not None and hasattr(meter, "last_front"):
            pts = meter.front_points().astype(np.int32)
            col = risk_color(fc.p_lag if fc else 0)
            for p in pts[::3]:
                cv2.circle(img, tuple(p), 1, col, -1, cv2.LINE_AA)
    # side panel
    pw = 230
    panel = np.full((H, pw, 3), 24, np.uint8)
    y = 26

    def put(txt, col=(235, 235, 235), s=0.5, th=1):
        nonlocal y
        cv2.putText(panel, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, s, col, th, cv2.LINE_AA)
        y += int(26 * s / 0.5)

    put("BoilGuard", (255, 220, 0), 0.75, 2)
    st = agent.s.state if agent else "-"
    cv2.rectangle(panel, (10, y - 16), (pw - 10, y + 8), STATE_COL.get(st, (120, 120, 120)), -1)
    cv2.putText(panel, st, (16, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
    y += 30
    if meas is not None:
        put(f"freeboard {meas.freeboard_mm:5.1f} mm")
        put(f"confidence {meas.confidence:4.2f}  fog {meas.fog:4.2f}")
    if fc is not None:
        eta = "none" if not math.isfinite(fc.eta_p50) else f"{fc.eta_p50:4.1f} s"
        put(f"ETA overflow  {eta}", risk_color(fc.p_lag), 0.55, 2)
        put(f"P(<20s) {fc.p_20s:4.2f}  P(<lag) {fc.p_lag:4.2f}")
        put(f"growth k {fc.rate:+.2f}/s")
    if agent:
        put(f"heat {agent.s.heat:.2f}  fan {'ON' if agent.s.fan_on else 'off'}")
        put(f"analysis {agent.sampling_hz:.0f} Hz  mode {agent.cfg.mode}", (180, 180, 180), 0.42)
    # freeboard gauge
    if meas is not None and meter.ready and meter.D0 > 0:
        gx, gy0, gy1 = pw - 34, y + 6, H - 120
        cv2.rectangle(panel, (gx, gy0), (gx + 18, gy1), (70, 70, 70), 1)
        frac = np.clip(meas.foam_mm / meter.D0, 0, 1)
        yy = int(gy1 - frac * (gy1 - gy0))
        cv2.rectangle(panel, (gx + 2, yy), (gx + 16, gy1 - 1), risk_color(frac), -1)
        cv2.line(panel, (gx - 6, gy0), (gx + 24, gy0), (60, 60, 255), 2)
        cv2.putText(panel, "rim", (gx - 40, gy0 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 255), 1, cv2.LINE_AA)
        if truth is not None:
            ty = int(gy1 - np.clip(truth["foam_mm"] / max(meter.D0, 1), 0, 1) * (gy1 - gy0))
            cv2.line(panel, (gx - 4, ty), (gx + 22, ty), (255, 255, 255), 1)
    # action ticker
    y = H - 104
    for a in (last_actions or [])[-4:]:
        cv2.putText(panel, a[:30], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 230, 255), 1, cv2.LINE_AA)
        y += 20
    if truth is not None:
        cv2.putText(img, f"SIM t={truth['t']:.1f}s  true freeboard {truth['freeboard_mm']:.1f} mm  P={truth['P']:.2f}",
                    (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.hstack([img, panel])


class VideoWriter:
    """H.264 MP4 via the ffmpeg binary bundled with imageio-ffmpeg (browser friendly);
    falls back to OpenCV's mp4v writer."""

    def __init__(self, path, fps, size):
        self.path, self.fps, self.size = path, fps, size
        self.proc = None
        self.cv = None
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            w, h = size
            self.proc = subprocess.Popen(
                [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                 "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", path],
                stdin=subprocess.PIPE)
        except Exception:
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame):
        if self.proc:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.cv.write(frame)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        elif self.cv:
            self.cv.release()
