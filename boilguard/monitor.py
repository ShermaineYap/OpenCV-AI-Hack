"""Session runner: camera/video/simulator -> vision -> Kalman -> agent -> actions.

The same loop runs on a laptop, in an AWS Lambda container (web demo / batch
analysis) and inside the evaluation harness.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .agent.llm import Narrator
from .agent.policy import AgentConfig, BoilGuardAgent, LOW_HEAT
from .agent.tools import (Actuators, LogActuators, NoHuman, SimActuators, SimHuman, MultiActuators,
                          sample_human)
from .overlay import VideoWriter, draw
from .predict import FoamKalman, LinearBaseline, ThresholdBaseline
from .sim import BoilSim, SimParams
from .vision.surface import SurfaceMeter

CALIB_FRAMES = 15


@dataclass
class RunOptions:
    policy: str = "boilguard"          # boilguard | notify_only | ask_first | threshold | linear | none
    human: str = "sample"              # sample | present | absent
    video_path: str | None = None
    privacy: bool = False
    use_bedrock: bool = False
    aws_mirror: bool = False
    cam_height_error: float = 0.0      # fractional error of the user-entered camera height
    seed: int = 0
    fixed_hz: float | None = None      # force a constant analysis rate (offline replay studies)


def _summary_common(timeline, trace, overflow_t, natural_t, extra):
    acts = [e for e in trace if e.get("tools")]
    first_alert = next((e["t"] for e in trace if e["decision"] in ("alert_cook", "urgent_alert", "request_approval",
                                                                     "announce_reduce_heat", "reduce_heat")), None)
    first_act = next((e["t"] for e in trace if e["decision"].startswith("reduce_heat")), None)
    ms = [r["ms"] for r in timeline if r.get("ms")]
    return {"overflowed": overflow_t is not None, "overflow_t": overflow_t, "natural_overflow_t": natural_t,
            "first_alert_t": first_alert, "first_action_t": first_act,
            "alert_lead_s": (natural_t - first_alert) if (natural_t and first_alert) else None,
            "n_tool_calls": sum(len(e["tools"]) for e in acts), "n_decisions": len(trace),
            "vision_ms_mean": float(np.mean(ms)) if ms else None,
            "analysed_frames": len(timeline), **extra}


def run_sim(params: SimParams, opts: RunOptions, agent_cfg: AgentConfig | None = None) -> dict:
    """Closed-loop run on the simulator. Returns summary, timeline, trace."""
    rng = np.random.default_rng(opts.seed + 99)
    sim = BoilSim(params)
    natural_t = sim.natural_overflow_time()
    render_all = opts.video_path is not None
    if opts.human == "sample":
        human = sample_human(rng, sim)
    else:
        human = SimHuman(sim, opts.human == "present", float(np.clip(rng.lognormal(np.log(6.0), 0.5), 2, 25)))
    if opts.policy == "none":
        writer = VideoWriter(opts.video_path, params.fps, (params.width + 230, params.height)) if render_all else None
        while sim.s.t < params.duration:
            frame = sim.step(render=render_all)
            if writer is not None:
                panel = np.full((params.height, 230, 3), 24, np.uint8)
                cv2.putText(panel, "No BoilGuard", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
                tr = sim.truth()
                cv2.putText(panel, f"t = {tr['t']:.1f} s", (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
                cv2.putText(panel, f"true freeboard {tr['freeboard_mm']:.1f} mm", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (235, 235, 235), 1, cv2.LINE_AA)
                if tr["overflowed"]:
                    cv2.putText(panel, "BOILED OVER", (12, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (60, 60, 240), 2, cv2.LINE_AA)
                writer.write(np.hstack([frame, panel]))
        if writer:
            writer.close()
        return {"summary": _summary_common([], [], sim.s.overflow_time, natural_t,
                                           {"policy": "none", "human_present": human.present}),
                "timeline": [], "trace": []}

    cfg = agent_cfg or AgentConfig(heater=params.heater)
    if opts.policy in ("notify_only", "ask_first"):
        cfg.mode = opts.policy
    act: Actuators = SimActuators(sim)
    if opts.aws_mirror:
        from .agent.tools import AwsActuators
        act = MultiActuators(act, AwsActuators(session_id=f"sim{opts.seed}"))
    agent = BoilGuardAgent(cfg, act, human, Narrator(use_bedrock=opts.use_bedrock))
    cam_h = params.cam_height_mm * (1 + opts.cam_height_error)
    frames = [sim.step() for _ in range(CALIB_FRAMES)]
    meter = SurfaceMeter(cam_h).calibrate(frames)
    (tcx, tcy), (tw, th), _ = sim.true_rim()
    rim_err = {"center_px": float(np.hypot(meter.rim.cx - tcx, meter.rim.cy - tcy)),
               "radius_px": float((meter.rim.a + meter.rim.b - (tw + th) / 2) / 2)}
    if meter.D0 < 8.0:
        agent.trace.append({"t": round(sim.s.t, 2), "state": "CALIBRATING", "decision": "request_rim_confirmation",
                            "why": f"calm freeboard {meter.D0:.1f} mm implausible", "tools": [], "forecast": None})
    kf = FoamKalman(meter.D0, seed=opts.seed)
    lin = LinearBaseline()
    thr = ThresholdBaseline(meter.D0)
    baseline_fired = False
    writer = None
    if render_all:
        writer = VideoWriter(opts.video_path, params.fps, (params.width + 230, params.height))
    timeline, ticker, pending_h = [], [], []
    next_t = sim.s.t
    meas = fc = None
    while sim.s.t < params.duration:
        hz = opts.fixed_hz or (agent.sampling_hz if opts.policy not in ("threshold", "linear") else 8.0)
        due = sim.s.t >= next_t - 1e-9
        frame = sim.step(render=due or render_all)
        t = sim.s.t
        truth = sim.truth()
        pending_h.extend(human.poll(t))
        if due:
            hevents, pending_h = pending_h, []
            next_t = t + 1.0 / hz
            meas = meter.measure(frame, t)
            fc = kf.update(t, meas.foam_mm, meas.confidence, meas.fog, lag_s=cfg.lag_s)
            if opts.policy == "threshold":
                if not baseline_fired and thr.alarm(meas.freeboard_mm):
                    baseline_fired = True
                    human.alert(t, "threshold alarm")
                    agent.trace.append({"t": round(t, 2), "state": "BASELINE", "decision": "alert_cook",
                                        "why": "freeboard below threshold", "tools": [{"tool": "notify"}]})
            elif opts.policy == "linear":
                eta = lin.update(t, meas.freeboard_mm)
                if not baseline_fired and eta <= cfg.lag_s + 2.0:
                    baseline_fired = True
                    sim.set_power(LOW_HEAT, "agent")
                    human.alert(t, "linear ETA alarm")
                    agent.trace.append({"t": round(t, 2), "state": "BASELINE", "decision": "reduce_heat",
                                        "why": f"linear ETA {eta:.1f}s", "tools": [{"tool": "set_heat"}]})
            else:
                n_before = len(agent.trace)
                agent.step(t, meas, fc, hevents)
                for e in agent.trace[n_before:]:
                    ticker.append(f"{e['t']:5.1f}s {e['decision']}")
            truth_eta = (natural_t - t) if natural_t else None
            timeline.append({"t": round(t, 3), "true_freeboard": round(truth["freeboard_mm"], 2),
                             "true_foam": round(truth["foam_mm"], 2), "freeboard": round(meas.freeboard_mm, 2),
                             "foam": round(meas.foam_mm, 2), "conf": round(meas.confidence, 3),
                             "fog": round(meas.fog, 3), "eta_p50": fc.eta_p50 if math.isfinite(fc.eta_p50) else None,
                             "eta_p10": fc.eta_p10 if math.isfinite(fc.eta_p10) else None,
                             "p20": round(fc.p_20s, 3), "plag": round(fc.p_lag, 3), "rate": round(fc.rate, 4),
                             "P": round(truth["P"], 2), "steam": round(truth["steam"], 3),
                             "state": agent.s.state, "hz": agent.sampling_hz, "ms": round(meas.ms, 2),
                             "true_eta_natural": round(truth_eta, 2) if truth_eta is not None else None,
                             "overflowed": truth["overflowed"]})
        if writer is not None and frame is not None:
            writer.write(draw(frame, meter, meas, fc, agent, truth, ticker, privacy=opts.privacy))
    if writer:
        writer.close()
    summ = _summary_common(timeline, agent.trace, sim.s.overflow_time, natural_t, {
        "policy": opts.policy, "mode": cfg.mode, "human_present": human.present,
        "human_reaction_s": round(human.reaction, 2), "human_took_over": human.took_over,
        "heater": params.heater, "lag_s": cfg.lag_s, "D0_true": round(params.freeboard_mm, 2),
        "D0_est": round(meter.D0, 2), "liquid": params.liquid, "rim_err": rim_err,
        "hmax_ratio": round(params.hmax_mm / params.freeboard_mm, 3), "will_overflow": params.will_overflow_uncontrolled,
        "intervened": any(e["decision"].startswith(("reduce_heat", "escalate", "heat_off")) for e in agent.trace),
        "sim_events": sim.s.events, "bedrock_calls": agent.nar.log})
    return {"summary": summ, "timeline": timeline, "trace": agent.trace, "params": params.to_dict()}


def run_video(path: str, cam_height_mm: float, heater="ceramic", mode="auto_veto", out_video=None,
              max_seconds=120.0, privacy=True, use_bedrock=False) -> dict:
    """Open-loop analysis of a recorded clip: decisions are logged as dry-run actions."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []

    def read():
        ok, f = cap.read()
        if not ok:
            return None
        h, w = f.shape[:2]
        if w > 640:
            f = cv2.resize(f, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)
        return f

    for _ in range(CALIB_FRAMES):
        f = read()
        if f is None:
            break
        frames.append(f)
    if len(frames) < 5:
        raise ValueError("video too short")
    meter = SurfaceMeter(cam_height_mm).calibrate(frames)
    cfg = AgentConfig(mode=mode, heater=heater)
    agent = BoilGuardAgent(cfg, LogActuators(), NoHuman(), Narrator(use_bedrock=use_bedrock))
    kf = FoamKalman(meter.D0)
    H, W = frames[0].shape[:2]
    writer = VideoWriter(out_video, fps, (W + 230, H)) if out_video else None
    timeline, ticker = [], []
    idx = len(frames)
    t = idx / fps
    next_t = t
    meas = fc = None
    while t < max_seconds:
        f = read()
        if f is None:
            break
        idx += 1
        t = idx / fps
        if t >= next_t:
            next_t = t + 1.0 / agent.sampling_hz
            meas = meter.measure(f, t)
            fc = kf.update(t, meas.foam_mm, meas.confidence, meas.fog, lag_s=cfg.lag_s)
            n0 = len(agent.trace)
            agent.step(t, meas, fc, [])
            for e in agent.trace[n0:]:
                ticker.append(f"{e['t']:5.1f}s {e['decision']}")
            timeline.append({"t": round(t, 3), "freeboard": round(meas.freeboard_mm, 2), "foam": round(meas.foam_mm, 2),
                             "conf": round(meas.confidence, 3), "fog": round(meas.fog, 3),
                             "eta_p50": fc.eta_p50 if math.isfinite(fc.eta_p50) else None,
                             "eta_p10": fc.eta_p10 if math.isfinite(fc.eta_p10) else None,
                             "p20": round(fc.p_20s, 3), "plag": round(fc.p_lag, 3), "rate": round(fc.rate, 4),
                             "state": agent.s.state, "hz": agent.sampling_hz, "ms": round(meas.ms, 2)})
        if writer is not None and meas is not None:
            writer.write(draw(f, meter, meas, fc, agent, None, ticker, privacy=privacy))
    cap.release()
    if writer:
        writer.close()
    over_t = next((r["t"] for r in timeline if r["freeboard"] <= 0.5), None)
    summ = _summary_common(timeline, agent.trace, over_t, None, {
        "policy": "boilguard (dry-run)", "mode": mode, "D0_est": round(meter.D0, 2), "fps": fps,
        "rim": meter.rim.__dict__, "bedrock_calls": agent.nar.log})
    return {"summary": summ, "timeline": timeline, "trace": agent.trace}
