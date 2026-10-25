import sys, json
sys.path.insert(0, "/home/claude/opencv-ai-hack")
from boilguard.sim import random_params
from boilguard.monitor import run_sim, RunOptions
p = random_params(7, overflow=True, width=1280, height=960)
p.liquid, p.heater, p.tau_heat, p.steam_max, p.duration = "milk", "ceramic", 4.5, 0.35, 42.0
r = run_sim(p, RunOptions(policy="boilguard", human="absent", seed=7, video_path="live_annot.mp4"))
tl = [{k: x[k] for k in ("t", "freeboard", "eta_p50", "plag", "state", "P")} for x in r["timeline"]]
json.dump({"trace": [{"t": e["t"], "decision": e["decision"], "msg": (e.get("msg") or {}).get("message"),
                      "tools": [t["tool"] + (":" + str(t["args"].get("level", t["args"].get("on", ""))) if t["tool"] != "notify" else "") for t in e["tools"]]}
                     for e in r["trace"]], "timeline": tl, "summary": {k: r["summary"][k] for k in ("overflowed", "natural_overflow_t")}},
          open("live_run.json", "w"), indent=1)
for e in r["trace"]: print(e["t"], e["decision"], (e.get("msg") or {}).get("message"))
