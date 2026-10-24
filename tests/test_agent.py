from boilguard.agent.policy import AgentConfig, BoilGuardAgent
from boilguard.agent.tools import Actuators, HumanChannel
from boilguard.predict import Forecast
from boilguard.vision.surface import Measurement


class Recorder(Actuators):
    def __init__(self):
        self.calls = []

    def set_heat(self, level, reason):
        self.calls.append(("set_heat", level)); return {"ok": True}

    def hood_fan(self, on, reason):
        self.calls.append(("hood_fan", on)); return {"ok": True}

    def notify(self, severity, message, evidence=None):
        self.calls.append(("notify", severity)); return {"ok": True}

    def alarm(self, on):
        self.calls.append(("alarm", on)); return {"ok": True}


def M(fb=20.0, conf=0.9, fog=0.0, overflow=False):
    return Measurement(t=0, freeboard_mm=fb, foam_mm=30 - fb, rho=0.95, confidence=conf, valid_frac=1.0,
                       lip_covered=0.0, fog=fog, sharpness=1.0, agitation=0.0, overflow=overflow)


def F(p_lag, p20=None, eta=5.0, fill=0.7, rate=0.3, rise=2.0):
    return Forecast(t=0, foam_mm=10, rate=rate, accel=0, eta_p50=eta, eta_p10=eta * 0.6, p_10s=p_lag,
                    p_20s=p20 if p20 is not None else p_lag, p_lag=p_lag, rise_mm_s=rise, fill=fill)


def test_visual_forecast_triggers_heat_reduction_and_verification():
    rec = Recorder()
    ag = BoilGuardAgent(AgentConfig(mode="auto_veto", heater="ceramic"), rec, HumanChannel())
    ag.step(1.0, M(), F(0.1, p20=0.1, eta=40))
    assert not any(c[0] == "set_heat" for c in rec.calls)
    ag.step(2.0, M(), F(0.9, eta=4.0))                    # urgent: ETA p10 < lag + veto
    assert ("set_heat", 0.35) in rec.calls
    assert ag.s.state == "VERIFY"
    # still rising after the verify deadline -> escalate to OFF + alarm
    ag.step(2.0 + 6 * 1.5 + 3.1, M(), F(0.95, eta=2.0))
    assert ("set_heat", 0.0) in rec.calls and ("alarm", True) in rec.calls


def test_notify_only_never_actuates():
    rec = Recorder()
    ag = BoilGuardAgent(AgentConfig(mode="notify_only"), rec, HumanChannel())
    for i in range(10):
        ag.step(float(i), M(fog=0.6, conf=0.2), F(0.95, eta=3.0))
    assert all(c[0] == "notify" for c in rec.calls)


def test_steam_triggers_hood_fan_active_perception():
    rec = Recorder()
    ag = BoilGuardAgent(AgentConfig(), rec, HumanChannel())
    ag.step(0.0, M(fog=0.5, conf=0.6), F(0.0, eta=float("inf"), fill=0.0))
    ag.step(1.2, M(fog=0.5, conf=0.6), F(0.0, eta=float("inf"), fill=0.0))
    assert ("hood_fan", True) in rec.calls


def test_overflow_on_rim_cuts_heat():
    rec = Recorder()
    ag = BoilGuardAgent(AgentConfig(), rec, HumanChannel())
    for i in range(3):
        ag.step(float(i), M(fb=0.0, overflow=True), F(1.0, eta=0.0))
    assert ("set_heat", 0.0) in rec.calls and ag.s.state == "OVERFLOW"
