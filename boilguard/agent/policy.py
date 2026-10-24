"""BoilGuard agent: perception -> decision -> action -> verification loop.

The vision forecast changes what the agent does next:

* boiling detected (agitation / foam)   -> raise camera analysis rate (active perception)
* steam haze / low confidence           -> switch on the hood fan to clear the view, and
                                           act earlier while visibility is poor
* P(overflow within 20 s) high          -> alert the cook with visual evidence
* P(overflow within heater lag) high    -> reduce heat (mode-dependent human control)
* after acting                          -> verify from video that foam growth reversed;
                                           if not, escalate to burner OFF + alarm
* overflow seen on the rim              -> burner OFF, alarm, report

Autonomy modes (human control):
* ``notify_only`` - never actuates; only alerts.
* ``ask_first``   - asks for approval and acts only if approved.
* ``auto_veto``   - announces the action with a short veto window; acts at once if the
                    predicted overflow is sooner than lag + veto. (default)
Restoring heat is always left to the person.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .llm import Narrator
from .tools import Actuators, HumanChannel

LOW_HEAT = 0.35

HEATER_LAG = {"induction": 2.0, "ceramic": 6.0, "coil": 8.0, "gas": 1.0}


@dataclass
class AgentConfig:
    mode: str = "auto_veto"
    heater: str = "ceramic"
    veto_s: float = 3.0
    # operating point chosen by offline replay on a separate dev set (eval/tune.py)
    heads_up_p20: float = 0.5          # quiet "foam rising, watching" notification
    alert_p20: float = 0.7             # boil-over warning (tuned on dev for few false warnings)
    alert_fill: float = 0.25
    act_p: float = 0.7
    act_fill: float = 0.4
    act_p_low_vis: float = 0.5
    fog_on: float = 0.3
    conf_low: float = 0.3
    idle_hz: float = 2.0
    boil_hz: float = 8.0
    alert_hz: float = 10.0

    @property
    def lag_s(self):
        return HEATER_LAG.get(self.heater, 6.0)


@dataclass
class AgentState:
    state: str = "MONITOR"
    alerted: bool = False
    heads_up: bool = False
    fan_on: bool = False
    heat: float = 1.0
    low_vis_since: float | None = None
    low_vis_mode: bool = False
    veto_deadline: float | None = None
    approval_id: str | None = None
    approval_deadline: float | None = None
    verify_deadline: float | None = None
    human_until: float = -1.0
    overflow_frames: int = 0
    last_notify: float = -1e9
    escalated: bool = False
    stable_since: float | None = None
    act_streak: int = 0
    events: list = field(default_factory=list)


class BoilGuardAgent:
    def __init__(self, cfg: AgentConfig, act: Actuators, human: HumanChannel,
                 narrator: Narrator | None = None):
        self.cfg, self.act, self.human = cfg, act, human
        self.nar = narrator or Narrator(use_bedrock=False)
        self.s = AgentState()
        self.trace: list[dict] = []
        self.sampling_hz = cfg.idle_hz

    # ----------------------------------------------------------------- utils
    def _tool(self, name, **kw):
        fn = {"set_heat": self.act.set_heat, "hood_fan": self.act.hood_fan,
              "notify": self.act.notify, "alarm": self.act.alarm}[name]
        res = fn(**kw)
        return {"tool": name, "args": {k: v for k, v in kw.items() if k != "evidence"}, "result": res}

    def _notify(self, t, severity, kind, ev):
        event = {"kind": kind, **ev}
        text, src = self.nar.message(event)
        if severity != "info":             # info = silent heads-up, no call to action
            self.human.alert(t, text)
        self.s.last_notify = t
        return self._tool("notify", severity=severity, message=text, evidence=ev), {"message": text, "by": src}

    def _log(self, t, decision, tools, why, m, f, extra=None):
        rec = {"t": round(t, 2), "state": self.s.state, "decision": decision, "why": why,
               "perception": {"freeboard_mm": round(m.freeboard_mm, 1), "confidence": round(m.confidence, 2),
                              "fog": round(m.fog, 2), "agitation": round(m.agitation, 3),
                              "lip_covered": round(m.lip_covered, 2)},
               "forecast": f.to_dict() if f else None, "tools": tools}
        if extra:
            rec.update(extra)
        self.trace.append(rec)

    # ------------------------------------------------------------------ step
    def step(self, t, m, f, human_events=None):
        """One perception-decision-action cycle. Returns the next sampling rate."""
        cfg, s = self.cfg, self.s
        lag = cfg.lag_s
        ev = {"freeboard_mm": round(m.freeboard_mm, 1),
              "eta_s": None if not math.isfinite(f.eta_p50) else round(f.eta_p50, 1),
              "p_overflow_20s": round(f.p_20s, 2), "confidence": round(m.confidence, 2)}

        # --- human input first: the person is always in charge
        for e in human_events or []:
            if e["type"] == "took_over":
                s.human_until = t + 20.0
                s.heat = min(s.heat, 0.3)
                self._log(t, "human_took_over", [], "cook acknowledged and turned the heat down", m, f)
            elif e["type"] in ("approve", "deny") and e.get("id") == s.approval_id:
                s.approval_id = None
                if e["type"] == "approve":
                    tools = [self._tool("set_heat", level=LOW_HEAT, reason="approved by cook")]
                    s.heat = LOW_HEAT
                    s.state, s.verify_deadline = "VERIFY", t + lag * 1.5 + 3
                    self._log(t, "reduce_heat (approved)", tools, "cook approved", m, f)
                else:
                    self._log(t, "approval_denied", [], "cook declined; monitoring only", m, f)
            elif e["type"] == "veto" and s.veto_deadline:
                s.veto_deadline = None
                s.human_until = t + 20.0
                self._log(t, "vetoed", [], "cook cancelled the action", m, f)

        # --- overflow seen on the rim
        s.overflow_frames = s.overflow_frames + 1 if m.overflow else 0
        if s.overflow_frames >= 3 and s.state != "OVERFLOW":
            tools = []
            if cfg.mode != "notify_only":
                tools.append(self._tool("set_heat", level=0.0, reason="overflow detected"))
                tools.append(self._tool("alarm", on=True))
                s.heat = 0.0
            n, meta = self._notify(t, "critical", "overflow", ev)
            tools.append(n)
            s.state = "OVERFLOW"
            self._log(t, "overflow_response", tools, "foam covering the rim lip", m, f, {"msg": meta})
            return self._rate(cfg.alert_hz)
        if s.state == "OVERFLOW":
            return self._rate(cfg.boil_hz)

        # --- active perception: steam/visibility
        low_vis = m.fog >= cfg.fog_on or m.confidence < cfg.conf_low
        if low_vis:
            s.low_vis_since = s.low_vis_since if s.low_vis_since is not None else t
            if not s.fan_on and t - s.low_vis_since >= 1.0 and cfg.mode != "notify_only":
                tools = [self._tool("hood_fan", on=True, reason="steam is hiding the pot rim")]
                s.fan_on = True
                self._log(t, "hood_fan_on", tools, f"low visibility (fog {m.fog:.2f}, conf {m.confidence:.2f})", m, f)
            if t - s.low_vis_since >= 6.0 and not s.low_vis_mode:
                s.low_vis_mode = True
                n, meta = self._notify(t, "info", "low_visibility", ev)
                self._log(t, "conservative_mode", [n], "visibility still poor after fan", m, f, {"msg": meta})
        else:
            s.low_vis_since = None
            if s.low_vis_mode and m.confidence > cfg.conf_low + 0.15:
                s.low_vis_mode = False
                self._log(t, "visibility_restored", [], "confidence recovered", m, f)
        act_p = cfg.act_p_low_vis if s.low_vis_mode else cfg.act_p

        # --- boiling onset -> sample faster
        if s.state == "MONITOR" and (f.foam_mm > 2.5 and f.rate > 0.05 or m.agitation > 0.6):
            s.state = "BOILING"
            self._log(t, "boiling_detected", [], "foam/agitation onset; analysis rate up", m, f,
                      {"sampling_hz": cfg.boil_hz})

        # --- pending veto window
        if s.veto_deadline is not None:
            if t >= s.veto_deadline or f.eta_p10 <= lag:
                s.veto_deadline = None
                tools = [self._tool("set_heat", level=LOW_HEAT, reason="predicted boil-over")]
                s.heat = LOW_HEAT
                n, meta = self._notify(t, "warning", "intervene", ev)
                tools.append(n)
                s.state, s.verify_deadline = "VERIFY", t + lag * 1.5 + 3
                self._log(t, "reduce_heat", tools, "veto window elapsed", m, f, {"msg": meta})
            return self._rate(cfg.alert_hz)

        # --- pending approval (ask_first)
        if s.approval_id is not None:
            if t >= s.approval_deadline:
                s.approval_id = None
                tools = [self._tool("alarm", on=True)]
                self._log(t, "approval_timeout_alarm", tools, "no answer; mode forbids acting alone", m, f)
            return self._rate(cfg.alert_hz)

        # --- verification after acting
        if s.state == "VERIFY":
            if f.rate < 0 or f.rise_mm_s <= 0.1:
                s.stable_since = s.stable_since or t
                if t - s.stable_since >= 2.0:
                    n, meta = self._notify(t, "info", "resolved", ev)
                    s.state = "RESOLVED"
                    self._log(t, "verified_receding", [n], "foam growth reversed after heat reduction", m, f,
                              {"msg": meta})
            else:
                s.stable_since = None
                if t >= s.verify_deadline and f.p_lag >= act_p and not s.escalated and cfg.mode != "notify_only":
                    tools = [self._tool("set_heat", level=0.0, reason="still rising after reduce"),
                             self._tool("alarm", on=True)]
                    s.heat, s.escalated = 0.0, True
                    n, meta = self._notify(t, "critical", "escalate", ev)
                    tools.append(n)
                    self._log(t, "escalate_heat_off", tools, "verification failed: foam still rising", m, f,
                              {"msg": meta})
            return self._rate(cfg.alert_hz)

        if s.state == "RESOLVED":
            if (f.p_lag >= act_p and f.rise_mm_s > 0.3 and f.fill > 0.5
                    and s.heat > 0 and not s.escalated and cfg.mode != "notify_only"):
                tools = [self._tool("set_heat", level=0.0, reason="rising again on low heat")]
                s.heat, s.escalated = 0.0, True
                s.state, s.verify_deadline = "VERIFY", t + lag * 1.5 + 3
                self._log(t, "heat_off_rerise", tools, "foam rising again", m, f)
            return self._rate(cfg.boil_hz)

        human_active = t < s.human_until

        # --- heads-up (silent) then warning
        if not s.heads_up and not s.alerted and f.p_20s >= cfg.heads_up_p20 and f.fill >= cfg.alert_fill:
            s.heads_up = True
            n, meta = self._notify(t, "info", "heads_up", ev)
            self._log(t, "heads_up", [n], f"foam rising; P(overflow<20s)={f.p_20s:.2f}", m, f, {"msg": meta})
        if not s.alerted and f.p_20s >= cfg.alert_p20 and f.fill >= cfg.alert_fill:
            s.alerted = True
            s.state = "ALERT"
            n, meta = self._notify(t, "warning", "alert", ev)
            self._log(t, "alert_cook", [n], f"P(overflow<20s)={f.p_20s:.2f}", m, f, {"msg": meta})

        # --- intervene
        # low-confidence evidence must persist for two analyses before the agent acts
        s.act_streak = s.act_streak + 1 if (f.p_lag >= act_p and f.fill >= cfg.act_fill) else 0
        need = 1 if m.confidence >= 0.4 or f.eta_p10 <= lag else 2
        if s.act_streak >= need and s.heat > LOW_HEAT:
            if human_active and f.p_lag < 0.9:
                return self._rate(cfg.alert_hz)       # cook is handling it
            if cfg.mode == "notify_only":
                if t - s.last_notify > 5.0:
                    n, meta = self._notify(t, "critical", "alert", ev)
                    self._log(t, "urgent_alert", [n], "notify-only mode", m, f, {"msg": meta})
            elif cfg.mode == "ask_first":
                s.approval_id = self.human.request_approval(t, "reduce_heat", lag)
                s.approval_deadline = t + max(lag, 4.0)
                n, meta = self._notify(t, "warning", "alert", ev)
                self._log(t, "request_approval", [n], f"P(overflow<lag)={f.p_lag:.2f}", m, f, {"msg": meta})
            else:
                if f.eta_p10 <= lag + cfg.veto_s:
                    tools = [self._tool("set_heat", level=LOW_HEAT, reason="predicted boil-over (urgent)")]
                    s.heat = LOW_HEAT
                    n, meta = self._notify(t, "warning", "intervene", ev)
                    tools.append(n)
                    s.state, s.verify_deadline = "VERIFY", t + lag * 1.5 + 3
                    self._log(t, "reduce_heat", tools,
                              f"P(overflow<lag)={f.p_lag:.2f}; ETA p10 {f.eta_p10:.1f}s < lag+veto", m, f, {"msg": meta})
                else:
                    s.veto_deadline = t + cfg.veto_s
                    n, meta = self._notify(t, "warning", "veto_window", {**ev, "veto_s": cfg.veto_s})
                    self._log(t, "announce_reduce_heat", [n], f"P(overflow<lag)={f.p_lag:.2f}", m, f, {"msg": meta})
            return self._rate(cfg.alert_hz)

        if s.state in ("BOILING", "ALERT"):
            return self._rate(cfg.alert_hz if s.alerted else cfg.boil_hz)
        return self._rate(cfg.idle_hz)

    def _rate(self, hz):
        self.sampling_hz = hz
        return hz
