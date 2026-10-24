"""Command line entry point.

    python -m boilguard.cli simulate --seed 7 --liquid milk --heater ceramic --out runs/demo
    python -m boilguard.cli analyze path/to/clip.mp4 --cam-height 450 --heater ceramic
    python -m boilguard.cli camera --device 0 --cam-height 450 --heater induction   # live webcam
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2


def _write(out_dir, r):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(r, f, indent=1, default=str)


def cmd_simulate(a):
    from .monitor import RunOptions, run_sim
    from .sim import random_params
    p = random_params(a.seed, overflow=not a.safe)
    if a.liquid:
        p.liquid = a.liquid
    if a.heater:
        p.heater, p.tau_heat = a.heater, {"induction": 1.2, "ceramic": 4.5, "coil": 6.5}[a.heater]
    if a.steam is not None:
        p.steam_max = a.steam
    p.duration = a.duration
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    r = run_sim(p, RunOptions(policy=a.policy, human=a.cook, video_path=os.path.join(a.out, "annotated.mp4"),
                              seed=a.seed, aws_mirror=a.aws, use_bedrock=a.bedrock))
    _write(a.out, r)
    s = r["summary"]
    print(f"natural spill at {s['natural_overflow_t']} s | overflowed: {s['overflowed']} | "
          f"first alert {s['first_alert_t']} s | first action {s['first_action_t']} s | {time.time() - t0:.1f}s")
    for e in r["trace"]:
        print(f"  {e['t']:6.2f}s  {e['decision']:24s} {e.get('why', '')}")
    print("video:", os.path.join(a.out, "annotated.mp4"))


def cmd_analyze(a):
    from .monitor import run_video
    base = os.path.splitext(a.video)[0]
    r = run_video(a.video, a.cam_height, heater=a.heater, mode=a.mode, out_video=base + "_boilguard.mp4",
                  max_seconds=a.max_seconds, privacy=not a.no_privacy)
    with open(base + "_boilguard.json", "w") as f:
        json.dump(r, f, indent=1, default=str)
    for e in r["trace"]:
        print(f"  {e['t']:6.2f}s  {e['decision']:24s} {e.get('why', '')}")
    print("annotated:", base + "_boilguard.mp4")


def cmd_camera(a):
    """Live loop on a webcam. Actions go to AWS IoT Core if --aws, else they are only printed."""
    from .agent.llm import Narrator
    from .agent.policy import AgentConfig, BoilGuardAgent
    from .agent.tools import AwsActuators, LogActuators, NoHuman
    from .overlay import draw
    from .predict import FoamKalman
    from .vision.surface import SurfaceMeter
    cap = cv2.VideoCapture(a.device)
    frames = []
    while len(frames) < 15:
        ok, f = cap.read()
        if not ok:
            raise SystemExit("camera not readable")
        frames.append(f)
    meter = SurfaceMeter(a.cam_height).calibrate(frames)
    cfg = AgentConfig(mode=a.mode, heater=a.heater)
    act = AwsActuators(device_id=a.device_id) if a.aws else LogActuators()
    agent = BoilGuardAgent(cfg, act, NoHuman(), Narrator())
    kf = FoamKalman(meter.D0)
    t0, next_t, ticker = time.time(), 0.0, []
    meas = fc = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t = time.time() - t0
        if t >= next_t:
            meas = meter.measure(f, t)
            fc = kf.update(t, meas.foam_mm, meas.confidence, meas.fog, lag_s=cfg.lag_s)
            n0 = len(agent.trace)
            next_t = t + 1.0 / agent.step(t, meas, fc, [])
            for e in agent.trace[n0:]:
                ticker.append(f"{e['t']:5.1f}s {e['decision']}")
                print(json.dumps(e, default=str))
        cv2.imshow("BoilGuard", draw(f, meter, meas, fc, agent, None, ticker))
        if cv2.waitKey(1) & 0xFF == 27:
            break


def main():
    ap = argparse.ArgumentParser(prog="boilguard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--liquid", choices=["milk", "pasta_water", "tomato_soup", "coconut_curry"])
    s.add_argument("--heater", choices=["induction", "ceramic", "coil"])
    s.add_argument("--steam", type=float)
    s.add_argument("--safe", action="store_true", help="a pot that would NOT boil over")
    s.add_argument("--cook", choices=["absent", "present", "sample"], default="absent")
    s.add_argument("--policy", default="boilguard",
                   choices=["boilguard", "notify_only", "ask_first", "threshold", "linear", "none"])
    s.add_argument("--duration", type=float, default=60.0)
    s.add_argument("--aws", action="store_true", help="mirror commands to AWS IoT Core")
    s.add_argument("--bedrock", action="store_true", help="use Amazon Bedrock for messages")
    s.add_argument("--out", default="runs/demo")
    s.set_defaults(fn=cmd_simulate)
    v = sub.add_parser("analyze")
    v.add_argument("video")
    v.add_argument("--cam-height", type=float, default=450.0)
    v.add_argument("--heater", default="ceramic")
    v.add_argument("--mode", default="auto_veto")
    v.add_argument("--max-seconds", type=float, default=300.0)
    v.add_argument("--no-privacy", action="store_true")
    v.set_defaults(fn=cmd_analyze)
    c = sub.add_parser("camera")
    c.add_argument("--device", type=int, default=0)
    c.add_argument("--cam-height", type=float, default=450.0)
    c.add_argument("--heater", default="induction")
    c.add_argument("--mode", default="auto_veto")
    c.add_argument("--aws", action="store_true")
    c.add_argument("--device-id", default="kitchen-1")
    c.set_defaults(fn=cmd_camera)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
