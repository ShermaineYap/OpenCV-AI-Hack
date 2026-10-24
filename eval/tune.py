"""Choose the agent's operating point on a DEV set (seeds 2000+/3000+), never on test.

1. Record open-loop measurement timelines at a fixed 8 Hz (vision runs once).
2. Replay the Kalman forecaster + decision rule offline for many settings, and
   re-simulate the physics with the heat reduced at the decision time to see
   whether the boil-over would have been prevented.

    python eval/tune.py --record      # ~5 min on 2 cores
    python eval/tune.py               # grid search, prints the trade-off table
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from boilguard import predict as P  # noqa: E402
from boilguard.agent.policy import HEATER_LAG  # noqa: E402
from boilguard.monitor import RunOptions, run_sim  # noqa: E402
from boilguard.sim import BoilSim, SimParams, SimState, random_params  # noqa: E402

DEV = os.path.join(os.path.dirname(__file__), "..", "results", "dev")


def record(args):
    seed, ovf = args
    out = os.path.join(DEV, f"{seed}_{int(ovf)}.json")
    if os.path.exists(out):
        return
    p = random_params(seed, overflow=ovf, hard=(seed % 4 == 0))
    r = run_sim(p, RunOptions(policy="notify_only", human="absent", seed=seed, fixed_hz=8.0))
    keep = ("t", "foam", "conf", "fog")
    json.dump({"params": r["params"], "D0_est": r["summary"]["D0_est"],
               "timeline": [{k: x[k] for k in keep} for x in r["timeline"]]}, open(out, "w"))


def overflows_with_action(params: dict, t_act):
    p = SimParams(**{k: (tuple(v) if k == "center_jitter" else v) for k, v in params.items()})
    sim = BoilSim.__new__(BoilSim)
    sim.p, sim.s, sim.dt = p, SimState(), 1 / p.fps
    while sim.s.t < p.duration:
        if t_act is not None and sim.s.t >= t_act:
            sim.s.P = 0.35
        sim._physics(sim.dt)
        if sim.s.overflow_time is not None:
            return True
        sim.s.t += sim.dt
    return False


def evaluate(runs, H0, q, theta, fmin):
    P.FoamKalman.H0 = H0
    prev = n_over = fa = n_safe = nm_i = n_nm = 0
    for r in runs:
        lag = HEATER_LAG[r["params"]["heater"]]
        kf = P.FoamKalman(r["D0_est"], q=q, n_samples=200, seed=1)
        t_act, streak = None, 0
        for x in r["timeline"]:
            fc = kf.update(x["t"], x["foam"], x["conf"], x["fog"], lag_s=lag)
            streak = streak + 1 if (fc.p_lag >= theta and fc.fill >= fmin) else 0
            need = 1 if x["conf"] >= 0.4 or fc.eta_p10 <= lag else 2
            if streak >= need:
                t_act = x["t"]
                break
        ratio = r["params"]["hmax_mm"] / r["params"]["freeboard_mm"]
        if ratio > 1:
            n_over += 1
            prev += not overflows_with_action(r["params"], t_act)
        elif ratio < 0.7:
            n_safe += 1
            fa += t_act is not None
        else:
            n_nm += 1
            nm_i += t_act is not None
    return prev / max(n_over, 1), fa / max(n_safe, 1), nm_i / max(n_nm, 1), n_over, n_safe, n_nm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    os.makedirs(DEV, exist_ok=True)
    if a.record:
        jobs = [(2000 + i, True) for i in range(24)] + [(3000 + i, False) for i in range(16)]
        with Pool(a.workers) as pool:
            pool.map(record, jobs)
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(DEV, "*.json")))]
    print(f"{len(runs)} dev runs")
    rows = []
    for H0, q, theta, fmin in itertools.product([3, 6], [0.003, 0.001, 0.0003], [0.5, 0.6, 0.7, 0.8], [0.25, 0.4, 0.55]):
        pr, fa, nm, a1, a2, a3 = evaluate(runs, H0, q, theta, fmin)
        rows.append((pr, fa, nm, H0, q, theta, fmin))
        print(f"H0={H0} q={q} theta={theta} fill>={fmin}: prevented {pr:.2f} (n={a1}) | unneeded cuts {fa:.2f} "
              f"(n={a2}) | near-miss cuts {nm:.2f} (n={a3})", flush=True)
    json.dump(rows, open(os.path.join(DEV, "..", "tuning_grid.json"), "w"))


if __name__ == "__main__":
    main()
