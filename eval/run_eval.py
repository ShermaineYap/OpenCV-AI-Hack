"""BoilGuard evaluation on the held-out synthetic test set.

Runs every scenario under several policies in closed loop and an open-loop pass
for perception / forecast accuracy. Raw per-run JSON goes to results/raw/ (cached,
so the run can be resumed); `eval/report.py` aggregates it into metrics + plots.

    python eval/run_eval.py --n-overflow 40 --n-safe 20 --workers 2

Test seeds (0..) are disjoint from the dev seeds (2000+, 3000+) used for tuning.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from boilguard.monitor import RunOptions, run_sim  # noqa: E402
from boilguard.sim import random_params  # noqa: E402

RAW = os.path.join(os.path.dirname(__file__), "..", "results", "raw")

POLICIES = {
    # name: (policy, human, fixed_hz)
    "none": ("none", "sample", None),
    "threshold_alarm": ("threshold", "sample", None),
    "linear_auto": ("linear", "sample", None),
    "boilguard_notify": ("notify_only", "sample", None),
    "boilguard_auto": ("boilguard", "sample", None),
    "boilguard_auto_absent": ("boilguard", "absent", None),
    "openloop": ("notify_only", "absent", 8.0),
}


def scenarios(n_over, n_safe):
    out = []
    for i in range(n_over):
        out.append((i, True))
    for i in range(n_safe):
        out.append((500 + i, False))
    return out


def job(args):
    seed, overflow, pname = args
    path = os.path.join(RAW, f"{pname}__{seed}.json")
    if os.path.exists(path):
        return path
    policy, human, hz = POLICIES[pname]
    params = random_params(seed, overflow=overflow, hard=(seed % 4 == 0))
    t0 = time.time()
    r = run_sim(params, RunOptions(policy=policy, human=human, seed=seed, fixed_hz=hz))
    r["wall_s"] = time.time() - t0
    r["policy_name"] = pname
    if pname != "openloop":
        r["timeline"] = r["timeline"][::4]          # keep files small
    with open(path, "w") as f:
        json.dump(r, f)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-overflow", type=int, default=40)
    ap.add_argument("--n-safe", type=int, default=20)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--policies", default=",".join(POLICIES))
    a = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)
    jobs = [(s, o, p) for (s, o) in scenarios(a.n_overflow, a.n_safe) for p in a.policies.split(",")]
    jobs.sort(key=lambda j: (j[2] == "none", j[0]))
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(job, jobs), 1):
            if i % 20 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} runs done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
