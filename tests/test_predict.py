import math
import numpy as np

from boilguard.predict import FoamKalman


def _feed(kf, fn, t_end, dt=0.125, noise=0.8, seed=0):
    rng = np.random.default_rng(seed)
    t, fc = 0.0, None
    while t < t_end:
        t += dt
        fc = kf.update(t, max(0.0, fn(t) + rng.normal(0, noise)), 0.8)
    return t, fc


def test_exponential_growth_eta():
    D0 = 40.0
    foam = lambda t: 0.0 if t < 5 else 1.0 * math.exp(0.3 * (t - 5))   # reaches 40 mm at t~17.3
    kf = FoamKalman(D0)
    t, fc = _feed(kf, foam, 14.0)
    true_eta = 5 + math.log(40.0) / 0.3 - t
    assert fc.p_20s > 0.7
    assert abs(fc.eta_p50 - true_eta) < 2.5


def test_plateau_predicts_no_overflow():
    D0 = 40.0
    foam = lambda t: 0.0 if t < 5 else 15.0 * (1 - math.exp(-0.4 * (t - 5)))
    kf = FoamKalman(D0)
    _, fc = _feed(kf, foam, 30.0)
    # the decision quantity (overflow within heat lag + margin) must be low,
    # and the median forecast says "never"
    assert fc.p_lag < 0.1
    assert not math.isfinite(fc.eta_p50)


def test_calm_pot_low_risk():
    kf = FoamKalman(35.0)
    _, fc = _feed(kf, lambda t: 0.0, 20.0, noise=1.5)
    assert fc.p_lag < 0.05
    assert fc.p_20s < 0.35         # 20 s look-ahead is deliberately cautious; alerts are also fill-gated
