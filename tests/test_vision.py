import numpy as np
import pytest

from boilguard.sim import BoilSim, random_params
from boilguard.vision.rim import detect_rim
from boilguard.vision.surface import SurfaceMeter


def _calm(seed, n=15, **kw):
    sim = BoilSim(random_params(seed, **kw))
    return sim, [sim.step() for _ in range(n)]


@pytest.mark.parametrize("seed", [1, 2, 7, 11])
def test_rim_detection_accuracy(seed):
    sim, frames = _calm(seed)
    rim = detect_rim(frames)
    (cx, cy), (w, h), _ = sim.true_rim()
    assert np.hypot(rim.cx - cx, rim.cy - cy) < 3.0
    assert abs((rim.a + rim.b) / 2 - (w + h) / 4) < 3.0


def test_freeboard_tracks_rising_foam():
    sim, frames = _calm(2, overflow=True)
    m = SurfaceMeter(sim.p.cam_height_mm).calibrate(frames)
    assert abs(m.D0 - sim.p.freeboard_mm) < 6.0
    errs = []
    for i in range(int(40 * sim.p.fps)):
        f = sim.step()
        if i % 5 or sim.s.overflow_time is not None:
            continue
        errs.append(abs(m.measure(f, sim.s.t).freeboard_mm - sim.freeboard_mm))
    assert np.median(errs) < 5.0


def test_hand_over_rim_is_not_an_overflow():
    # seed 3000 (hard) has two hands crossing the rim at t~12.7-14.6 s before boiling
    sim = BoilSim(random_params(3000, overflow=False, hard=True))
    frames = [sim.step() for _ in range(15)]
    m = SurfaceMeter(sim.p.cam_height_mm).calibrate(frames)
    worst = 0.0
    while sim.s.t < 15.0:
        f = sim.step()
        if sim.s.t > 12.5:
            me = m.measure(f, sim.s.t)
            assert not me.overflow
            worst = max(worst, me.foam_mm)
    assert worst < 8.0
