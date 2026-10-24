"""Aggregate results/raw/*.json into results/metrics.json, results/summary.csv,
results/eval_table.md and figures.

    python eval/report.py
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")

# reference palette (dataviz skill): categorical slots + text tokens, light surface
C = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "grey": "#9b9a94",
     "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e6e5e0", "bg": "#fcfcfb"}

LABELS = {
    "none": "No system",
    "threshold_alarm": "Rim-threshold alarm + cook",
    "linear_auto": "Linear extrapolation, auto",
    "boilguard_notify": "BoilGuard, notify only + cook",
    "boilguard_auto": "BoilGuard, auto (cancel window) + cook",
    "boilguard_auto_absent": "BoilGuard, auto, cook away",
}
ACTUATING = {"linear_auto", "boilguard_auto", "boilguard_auto_absent"}


def load():
    runs = defaultdict(list)
    for f in glob.glob(os.path.join(RES, "raw", "*.json")):
        with open(f) as fh:
            r = json.load(fh)
        runs[r["policy_name"]].append(r)
    return runs


def cls(r):
    ratio = r["params"]["hmax_mm"] / r["params"]["freeboard_mm"]
    return "overflow" if ratio > 1 else ("near" if ratio >= 0.7 else "safe")


def policy_table(runs):
    rows = []
    for p in LABELS:
        rs = runs.get(p, [])
        if not rs:
            continue
        ov = [r for r in rs if cls(r) == "overflow"]
        sf = [r for r in rs if cls(r) == "safe"]
        nm = [r for r in rs if cls(r) == "near"]
        over_rate = float(np.mean([r["summary"]["overflowed"] for r in ov])) if ov else None

        def intervened(r):
            return bool(r["summary"].get("intervened") or (r["summary"].get("first_action_t") is not None))

        def alerted(r):
            return r["summary"].get("first_alert_t") is not None

        fi = float(np.mean([intervened(r) for r in sf])) if (sf and p in ACTUATING) else None
        fa = float(np.mean([alerted(r) for r in sf])) if (sf and p != "none") else None
        ni = float(np.mean([intervened(r) for r in nm])) if (nm and p in ACTUATING) else None
        leads = [r["summary"]["natural_overflow_t"] - r["summary"]["first_alert_t"] for r in ov
                 if r["summary"].get("first_alert_t") is not None and r["summary"].get("natural_overflow_t")]
        hu = []
        for r in ov:
            nat = r["summary"].get("natural_overflow_t")
            t_hu = next((e["t"] for e in r["trace"] if e.get("decision") == "heads_up"), None)
            if nat and t_hu is not None:
                hu.append(nat - t_hu)
        rows.append({"policy": p, "label": LABELS[p],
                     "median_heads_up_lead_s": float(np.median(hu)) if hu else None, "n_overflow_prone": len(ov), "n_safe": len(sf),
                     "n_near_miss": len(nm), "overflow_rate": over_rate,
                     "false_intervention_rate": fi, "false_alert_rate": fa, "near_miss_intervention_rate": ni,
                     "median_lead_s": float(np.median(leads)) if leads else None,
                     "overflow_ci95": wilson(sum(r["summary"]["overflowed"] for r in ov), len(ov)) if ov else None})
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]


def forecast_and_perception(runs):
    ol = runs.get("openloop", [])
    hz_err = defaultdict(list)
    cover = []
    fb_err, fb_steam, fb_liquid = [], defaultdict(list), defaultdict(list)
    rim_c, rim_r, d0_err = [], [], []
    ms = []
    for r in ol:
        s = r["summary"]
        rim_c.append(s["rim_err"]["center_px"])
        rim_r.append(s["rim_err"]["radius_px"])
        d0_err.append(s["D0_est"] - s["D0_true"])
        for x in r["timeline"]:
            ms.append(x["ms"])
            if not x["overflowed"]:
                e = x["freeboard"] - x["true_freeboard"]
                fb_err.append(e)
                sb = "light (<0.15)" if x["steam"] < 0.15 else ("moderate (0.15-0.35)" if x["steam"] < 0.35 else "heavy (>0.35)")
                fb_steam[sb].append(abs(e))
                fb_liquid[s["liquid"]].append(abs(e))
            te = x.get("true_eta_natural")
            if te is not None and 0.5 < te <= 30 and x["true_foam"] > 0.5:
                pred = x["eta_p50"] if x["eta_p50"] is not None else 60.0
                hz_err[int(round(te))].append(abs(pred - te))
                if x["eta_p10"] is not None:
                    cover.append(x["eta_p10"] <= te)
                else:
                    cover.append(False)
    by_h = {str(h): float(np.median(v)) for h, v in sorted(hz_err.items()) if len(v) >= 5}
    return {
        "mae_by_horizon": by_h,
        "err_iqr_by_horizon": {str(h): [float(np.percentile(v, 25)), float(np.percentile(v, 75))]
                               for h, v in sorted(hz_err.items()) if len(v) >= 5},
        "p10_is_early_rate": float(np.mean(cover)) if cover else None,
    }, {
        "freeboard_mae_mm": float(np.mean(np.abs(fb_err))) if fb_err else None,
        "freeboard_median_abs_mm": float(np.median(np.abs(fb_err))) if fb_err else None,
        "freeboard_bias_mm": float(np.mean(fb_err)) if fb_err else None,
        "freeboard_p90_abs_mm": float(np.percentile(np.abs(fb_err), 90)) if fb_err else None,
        "by_steam": {k: float(np.median(v)) for k, v in fb_steam.items()},
        "by_steam_n": {k: len(v) for k, v in fb_steam.items()},
        "by_liquid": {k: float(np.median(v)) for k, v in fb_liquid.items()},
        "rim_center_err_px_median": float(np.median(rim_c)) if rim_c else None,
        "rim_radius_err_px_median": float(np.median(np.abs(rim_r))) if rim_r else None,
        "rim_ok_rate": float(np.mean([(c < 3) and (abs(rr) < 3) for c, rr in zip(rim_c, rim_r)])) if rim_c else None,
        "calm_freeboard_err_mm_median": float(np.median(np.abs(d0_err))) if d0_err else None,
        "n_scenes": len(ol),
    }, ms


def compute_stats(runs, ms_open):
    auto = runs.get("boilguard_auto", [])
    frames = [r["summary"]["analysed_frames"] for r in auto]
    secs = [r["params"]["duration"] for r in auto]
    hz = float(np.sum(frames) / np.sum(secs)) if auto else None
    return {"vision_ms_mean": float(np.mean(ms_open)) if ms_open else None,
            "vision_ms_p95": float(np.percentile(ms_open, 95)) if ms_open else None,
            "adaptive_analysis_hz_mean": hz, "fixed_rate_hz_baseline": 8.0}


def failures(runs):
    out = []
    for p in ("boilguard_auto", "boilguard_auto_absent"):
        for r in runs.get(p, []):
            s = r["summary"]
            c = cls(r)
            if c == "overflow" and s["overflowed"]:
                out.append({"policy": p, "seed": r["params"]["seed"], "kind": "missed boil-over",
                            "heater": s["heater"], "liquid": s["liquid"], "steam": round(r["params"]["steam_max"], 2),
                            "rim_err_px": round(s["rim_err"]["radius_px"], 1),
                            "first_action_t": s["first_action_t"], "natural_overflow_t": s["natural_overflow_t"]})
            if c == "safe" and (s.get("intervened") or s.get("first_action_t") is not None):
                out.append({"policy": p, "seed": r["params"]["seed"], "kind": "unneeded heat cut",
                            "heater": s["heater"], "liquid": s["liquid"], "steam": round(r["params"]["steam_max"], 2),
                            "rim_err_px": round(s["rim_err"]["radius_px"], 1),
                            "first_action_t": s["first_action_t"], "hmax_ratio": s.get("hmax_ratio")})
    return out


# ------------------------------------------------------------------ figures
def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(C["grid"])
    ax.tick_params(colors=C["ink2"], labelsize=9)
    ax.yaxis.label.set_color(C["ink2"])
    ax.xaxis.label.set_color(C["ink2"])
    ax.title.set_color(C["ink"])


def figures(table, fc, perc, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": C["bg"], "axes.facecolor": C["bg"],
                         "savefig.facecolor": C["bg"]})
    figs = []

    # 1. outcomes: two panels, one measure each
    rows = [r for r in table if r["overflow_rate"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    y = np.arange(len(rows))[::-1]
    vals = [100 * r["overflow_rate"] for r in rows]
    cols = [C["s1"] if r["policy"].startswith("boilguard") else C["grey"] for r in rows]
    ax.barh(y, vals, color=cols, height=0.6)
    for yi, v, r in zip(y, vals, rows):
        ci = r["overflow_ci95"]
        ax.text(v + 1.5, yi, f"{v:.0f}%  (95% CI {100*ci[0]:.0f}-{100*ci[1]:.0f})", va="center", fontsize=9, color=C["ink"])
    ax.set_yticks(y, [r["label"] for r in rows], fontsize=9, color=C["ink"])
    ax.set_xlim(0, 135)
    ax.set_xlabel("pots that boiled over (%)")
    ax.set_title(f"Boil-overs on overflow-prone pots (n={rows[0]['n_overflow_prone']})", fontsize=11, loc="left")
    ax.xaxis.grid(True, color=C["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    ax = axes[1]
    act = [r for r in table if r["false_intervention_rate"] is not None]
    if not act:
        ax.axis("off")
        act = None
    y2 = np.arange(len(act or []))[::-1]
    v2 = [100 * r["false_intervention_rate"] for r in act or []]
    if act: ax.barh(y2, v2, color=[C["s1"] if r["policy"].startswith("boilguard") else C["grey"] for r in act], height=0.6)
    for yi, v in zip(y2, v2):
        ax.text(v + 1.5, yi, f"{v:.0f}%", va="center", fontsize=9, color=C["ink"])
    if act:
        ax.set_yticks(y2, [r["label"] for r in act], fontsize=9, color=C["ink"])
        ax.set_xlim(0, 110)
        ax.set_xlabel("pots where heat was cut unnecessarily (%)")
        ax.set_title(f"Unneeded heat cuts (safe pots, n={act[0]['n_safe']})", fontsize=11, loc="left")
        ax.xaxis.grid(True, color=C["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_outcomes.png"), dpi=150)
    plt.close(fig)
    figs.append({"file": "fig_outcomes.png", "alt": "Bar charts: boil-over rate per policy and unneeded heat cuts"})

    # 2. forecast error vs horizon
    if fc["mae_by_horizon"]:
        h = np.array([int(k) for k in fc["mae_by_horizon"]])
        m = np.array(list(fc["mae_by_horizon"].values()))
        iq = np.array([fc["err_iqr_by_horizon"][str(k)] for k in h])
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.fill_between(h, iq[:, 0], iq[:, 1], color=C["s1"], alpha=0.15, linewidth=0, label="interquartile range")
        ax.plot(h, m, color=C["s1"], linewidth=2, label="median absolute error")
        ax.plot(h, h * 0 + 0, color=C["grid"], linewidth=0.8)
        ax.set_xlabel("true time until the pot would boil over (s)")
        ax.set_ylabel("forecast error (s)")
        ax.set_title("Time-to-overflow forecast error (open loop)", fontsize=11, loc="left")
        ax.invert_xaxis()
        ax.yaxis.grid(True, color=C["grid"], linewidth=0.8)
        ax.legend(frameon=False, fontsize=9, labelcolor=C["ink"])
        _style(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(RES, "fig_forecast_error.png"), dpi=150)
        plt.close(fig)
        figs.append({"file": "fig_forecast_error.png", "alt": "Forecast error shrinks as the boil-over approaches"})

    # 3. perception error by steam
    if perc["by_steam"]:
        order = [k for k in ("light (<0.15)", "moderate (0.15-0.35)", "heavy (>0.35)") if k in perc["by_steam"]]
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        vals = [perc["by_steam"][k] for k in order]
        ax.bar(range(len(order)), vals, color=C["s1"], width=0.55)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.1, f"{v:.1f} mm", ha="center", fontsize=9, color=C["ink"])
        ax.set_xticks(range(len(order)), [f"{k}\nn={perc['by_steam_n'][k]}" for k in order], fontsize=9, color=C["ink"])
        ax.set_ylabel("median |freeboard error| (mm)")
        ax.set_title("Surface measurement error by steam opacity", fontsize=11, loc="left")
        ax.yaxis.grid(True, color=C["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        _style(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(RES, "fig_perception.png"), dpi=150)
        plt.close(fig)
        figs.append({"file": "fig_perception.png", "alt": "Freeboard error by steam level"})

    # 4. example closed-loop timeline: two stacked panels sharing time (no dual axis)
    ex = None
    for r in sorted(runs.get("boilguard_auto_absent", []), key=lambda r: r["params"]["seed"]):
        if cls(r) == "overflow" and not r["summary"]["overflowed"] and r["summary"]["first_action_t"]:
            ex = r
            break
    if ex:
        ol = {r["params"]["seed"]: r for r in runs.get("openloop", [])}.get(ex["params"]["seed"])
        tl = ex["timeline"]
        t = [x["t"] for x in tl]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 5), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
        if ol:
            a1.plot([x["t"] for x in ol["timeline"]], [x["true_freeboard"] for x in ol["timeline"]],
                    color=C["grey"], linewidth=1.5, linestyle="--", label="if nobody acts (true)")
        a1.plot(t, [x["true_freeboard"] for x in tl], color=C["ink2"], linewidth=1.5, label="with BoilGuard (true)")
        a1.plot(t, [x["freeboard"] for x in tl], color=C["s1"], linewidth=2, label="BoilGuard measurement")
        a1.axhline(0, color=C["s2"], linewidth=1)
        a1.text(t[-1], 0.8, "rim", color=C["s2"], fontsize=9, ha="right")
        a1.set_ylabel("surface below rim (mm)")
        a1.legend(frameon=False, fontsize=8.5, labelcolor=C["ink"], loc="lower right", bbox_to_anchor=(1.0, 0.06))
        a1.set_title(f"Closed loop, cook away (test seed {ex['params']['seed']}, {ex['summary']['liquid'].replace('_', ' ')}, "
                     f"{ex['summary']['heater']} hob)", fontsize=11, loc="left")
        a2.plot(t, [x["plag"] for x in tl], color=C["s1"], linewidth=2)
        a2.set_ylabel("P(overflow within\nheat lag + 3 s)")
        a2.set_ylim(0, 1.05)
        a2.set_xlabel("time (s)")
        for e in ex["trace"]:
            if e["decision"] in ("alert_cook", "reduce_heat", "hood_fan_on", "verified_receding", "escalate_heat_off"):
                for a in (a1, a2):
                    a.axvline(e["t"], color=C["s2"] if "heat" in e["decision"] else C["grey"], linewidth=0.8, alpha=0.8)
                a2.text(e["t"] + 0.3, 0.92, e["decision"].replace("_", " "), fontsize=8, color=C["ink2"], rotation=90, va="top")
        if ex["summary"].get("natural_overflow_t"):
            a1.axvline(ex["summary"]["natural_overflow_t"], color=C["s2"], linestyle=":", linewidth=1)
            a1.text(ex["summary"]["natural_overflow_t"] + 0.3, a1.get_ylim()[1] * 0.85, "would have\nspilled", fontsize=8,
                    color=C["s2"])
        for a in (a1, a2):
            a.yaxis.grid(True, color=C["grid"], linewidth=0.8)
            _style(a)
        fig.tight_layout()
        fig.savefig(os.path.join(RES, "fig_example_timeline.png"), dpi=150)
        plt.close(fig)
        figs.append({"file": "fig_example_timeline.png", "alt": "Example run: measurement, forecast probability and actions"})
    return figs


def main():
    runs = load()
    table = policy_table(runs)
    fc, perc, ms = forecast_and_perception(runs)
    comp = compute_stats(runs, ms)
    fails = failures(runs)
    figs = figures(table, fc, perc, runs)
    n_scen = len(runs.get("openloop", []))
    metrics = {
        "description": (f"{n_scen} held-out simulated scenes (seeds disjoint from the tuning set): "
                        "75% overflow-prone, 25% safe simmer; random liquid, hob type, steam, camera height/tilt, "
                        "hands and spoons, camera shake and JPEG noise. The cook is nearby in ~60% of scenes "
                        "(reaction time median 6 s). Same scenes and cooks for every policy."),
        "policies": table, "forecast": fc, "perception": perc, "compute": comp,
        "failures": fails, "figures": figs,
    }
    with open(os.path.join(RES, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    with open(os.path.join(RES, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "seed", "class", "overflowed", "natural_overflow_t", "first_alert_t", "first_action_t",
                    "human_present", "heater", "liquid", "steam_max"])
        for p, rs in runs.items():
            for r in sorted(rs, key=lambda r: r["params"]["seed"]):
                s = r["summary"]
                w.writerow([p, r["params"]["seed"], cls(r), s["overflowed"], s.get("natural_overflow_t"),
                            s.get("first_alert_t"), s.get("first_action_t"), s.get("human_present"),
                            r["params"]["heater"], r["params"]["liquid"], round(r["params"]["steam_max"], 2)])
    fmt = lambda v, pct=True: "—" if v is None else (f"{100 * v:.0f}%" if pct else f"{v:.1f}")
    lines = ["| Policy | Boil-overs (overflow-prone pots) | Unneeded heat cuts (clearly-safe pots) | Heat cuts on near-misses | Warnings on clearly-safe pots | Median warning before spill (s) |",
             "|---|---|---|---|---|---|"]
    for r in table:
        ci = r.get("overflow_ci95")
        ov = fmt(r['overflow_rate']) + (f" (95% CI {100*ci[0]:.0f}-{100*ci[1]:.0f}%)" if ci else "")
        lines.append(f"| {r['label']} | {ov} | {fmt(r['false_intervention_rate'])} | {fmt(r['near_miss_intervention_rate'])} | "
                     f"{fmt(r['false_alert_rate'])} | {fmt(r['median_lead_s'], False)} |")
    n0 = table[0]
    lines.append("")
    lines.append(f"Test set: {n0['n_overflow_prone']} overflow-prone, {n0['n_safe']} clearly-safe and {n0['n_near_miss']} "
                 "near-miss scenes (foam peaks at 70-100% of the rim). Heat cuts on near-misses are arguably correct caution.")
    with open(os.path.join(RES, "eval_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({"forecast": fc["mae_by_horizon"], "p10_early": fc["p10_is_early_rate"]}, indent=0)[:600])
    print(json.dumps(perc, indent=1))
    print(json.dumps(comp, indent=1))
    print(f"{len(fails)} failure cases listed in metrics.json")


if __name__ == "__main__":
    main()
