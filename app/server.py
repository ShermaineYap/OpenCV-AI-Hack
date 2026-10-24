"""BoilGuard web service (runs locally with uvicorn and on AWS Lambda via the
Lambda Web Adapter, same container image).

Endpoints
  GET  /                 web UI
  GET  /health           version info (OpenCV build, region, features)
  POST /api/simulate     run a closed-loop boil-over scenario, returns annotated video + trace
  POST /api/analyze      upload a top-down clip of a pot; dry-run agent decisions
  GET  /api/eval         evaluation metrics bundled with the build
  GET  /runs/<id>/<f>    run artefacts (local mode; on AWS they are presigned S3 URLs)
"""
from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
import threading
import time
import uuid

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from boilguard import __version__
from boilguard.agent.llm import Narrator
from boilguard.monitor import RunOptions, run_sim, run_video
from boilguard.sim import LIQUIDS, random_params

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("BOILGUARD_RUNS_DIR", os.path.join(tempfile.gettempdir(), "boilguard-runs"))
BUCKET = os.environ.get("BOILGUARD_BUCKET")
REGION = os.environ.get("AWS_REGION", "local")
MAX_UPLOAD = 40 * 1024 * 1024
os.makedirs(RUNS, exist_ok=True)

app = FastAPI(title="BoilGuard", version=__version__)
_busy = threading.Semaphore(int(os.environ.get("BOILGUARD_MAX_CONCURRENT", "2")))
_s3 = None


def s3():
    global _s3
    if _s3 is None:
        import boto3
        _s3 = boto3.client("s3", region_name=None if REGION == "local" else REGION)
    return _s3


def emf(metrics: dict, dims: dict):
    """CloudWatch Embedded Metric Format: one structured log line -> metrics, no API calls."""
    rec = {"_aws": {"Timestamp": int(time.time() * 1000), "CloudWatchMetrics": [{
        "Namespace": "BoilGuard", "Dimensions": [list(dims)],
        "Metrics": [{"Name": k, "Unit": u} for k, (_, u) in metrics.items()]}]},
        **dims, **{k: v for k, (v, _) in metrics.items()}}
    print(json.dumps(rec), flush=True)


def publish(run_id: str, files: dict[str, str]) -> dict[str, str]:
    """Store artefacts; return URLs the browser can open."""
    urls = {}
    if BUCKET:
        for name, path in files.items():
            key = f"runs/{run_id}/{name}"
            ctype = "video/mp4" if name.endswith(".mp4") else "application/json"
            s3().upload_file(path, BUCKET, key, ExtraArgs={"ContentType": ctype})
            urls[name] = s3().generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": key},
                                                     ExpiresIn=3600)
    else:
        d = os.path.join(RUNS, run_id)
        os.makedirs(d, exist_ok=True)
        for name, path in files.items():
            dst = os.path.join(d, name)
            if os.path.abspath(path) != os.path.abspath(dst):
                shutil.copy(path, dst)
            urls[name] = f"/runs/{run_id}/{name}"
    return urls


class SimRequest(BaseModel):
    seed: int | None = Field(None, ge=0, le=10_000_000)
    liquid: str | None = None
    heater: str | None = Field(None, pattern="^(induction|ceramic|coil)$")
    steam: float | None = Field(None, ge=0.0, le=0.8)
    overflow_prone: bool = True
    cook: str = Field("absent", pattern="^(present|absent)$")
    mode: str = Field("auto_veto", pattern="^(auto_veto|ask_first|notify_only)$")
    aws_mirror: bool = False
    bedrock: bool = True
    privacy: bool = False


@app.get("/health")
def health():
    return {"ok": True, "service": "boilguard", "version": __version__, "opencv": cv2.__version__,
            "region": REGION, "s3": bool(BUCKET), "iot": bool(os.environ.get("BOILGUARD_IOT_ENABLED")),
            "bedrock": os.environ.get("BOILGUARD_BEDROCK", "0") == "1",
            "threads": cv2.getNumThreads(), "cpu": _cpu_features()}


def _cpu_features():
    feats = []
    for name in ("CPU_AVX2", "CPU_AVX512_SKX", "CPU_NEON", "CPU_SVE"):
        flag = getattr(cv2, name, None)
        if flag is not None and cv2.checkHardwareSupport(flag):
            feats.append(name[4:])
    return feats


@app.post("/api/simulate")
def simulate(req: SimRequest):
    if req.liquid and req.liquid not in LIQUIDS:
        raise HTTPException(400, f"liquid must be one of {list(LIQUIDS)}")
    if not _busy.acquire(timeout=1):
        raise HTTPException(429, "busy, try again in a few seconds")
    try:
        seed = req.seed if req.seed is not None else random.randint(0, 999_999)
        p = random_params(seed, overflow=req.overflow_prone)
        if req.liquid:
            p.liquid = req.liquid
        if req.heater:
            from boilguard.sim import SimParams  # noqa
            tau = {"induction": 1.2, "ceramic": 4.5, "coil": 6.5}[req.heater]
            p.heater, p.tau_heat = req.heater, tau
        if req.steam is not None:
            p.steam_max = req.steam
        p.duration = 60.0
        run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        work = tempfile.mkdtemp(prefix="bg-")
        video = os.path.join(work, "annotated.mp4")
        t0 = time.time()
        opts = RunOptions(policy={"auto_veto": "boilguard"}.get(req.mode, req.mode), human=req.cook,
                          video_path=video, privacy=req.privacy, seed=seed,
                          aws_mirror=req.aws_mirror and bool(os.environ.get("BOILGUARD_IOT_ENABLED")),
                          use_bedrock=req.bedrock and os.environ.get("BOILGUARD_BEDROCK", "0") == "1")
        r = run_sim(p, opts)
        nar = Narrator(use_bedrock=opts.use_bedrock, max_calls=1)
        report, src = nar.incident_report({k: r["summary"].get(k) for k in (
            "overflowed", "overflow_t", "natural_overflow_t", "first_alert_t", "first_action_t", "mode",
            "human_present", "heater", "liquid")}, r["trace"])
        r["report"] = {"text": report, "by": src}
        r["seed"] = seed
        r["elapsed_s"] = round(time.time() - t0, 2)
        jpath = os.path.join(work, "result.json")
        with open(jpath, "w") as f:
            json.dump(r, f)
        urls = publish(run_id, {"annotated.mp4": video, "result.json": jpath})
        s = r["summary"]
        emf({"RunSeconds": (r["elapsed_s"], "Seconds"),
             "VisionMsPerFrame": (s.get("vision_ms_mean") or 0.0, "Milliseconds"),
             "Overflowed": (1 if s["overflowed"] else 0, "Count"),
             "ToolCalls": (s["n_tool_calls"], "Count")}, {"Endpoint": "simulate"})
        shutil.rmtree(work, ignore_errors=True)
        tl = r["timeline"]
        return {"id": run_id, "seed": seed, "summary": s, "trace": r["trace"], "report": r["report"],
                "params": r["params"], "timeline": tl[:: max(1, len(tl) // 400)], "urls": urls,
                "elapsed_s": r["elapsed_s"]}
    finally:
        _busy.release()


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), cam_height_mm: float = Form(450.0),
                  heater: str = Form("ceramic"), mode: str = Form("auto_veto")):
    if heater not in ("induction", "ceramic", "coil", "gas") or mode not in ("auto_veto", "ask_first", "notify_only"):
        raise HTTPException(400, "bad heater/mode")
    if not (100 <= cam_height_mm <= 2000):
        raise HTTPException(400, "cam_height_mm must be 100-2000")
    work = tempfile.mkdtemp(prefix="bg-")
    src = os.path.join(work, "input.mp4")
    size = 0
    with open(src, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD:
                raise HTTPException(413, "max 40 MB")
            f.write(chunk)
    if not _busy.acquire(timeout=1):
        raise HTTPException(429, "busy")
    try:
        run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        video = os.path.join(work, "annotated.mp4")
        t0 = time.time()
        try:
            r = run_video(src, cam_height_mm, heater=heater, mode=mode, out_video=video, max_seconds=90)
        except (ValueError, RuntimeError) as e:
            raise HTTPException(422, str(e))
        r["elapsed_s"] = round(time.time() - t0, 2)
        jpath = os.path.join(work, "result.json")
        with open(jpath, "w") as f:
            json.dump(r, f)
        urls = publish(run_id, {"annotated.mp4": video, "result.json": jpath})
        emf({"RunSeconds": (r["elapsed_s"], "Seconds"),
             "VisionMsPerFrame": (r["summary"].get("vision_ms_mean") or 0.0, "Milliseconds")},
            {"Endpoint": "analyze"})
        return {"id": run_id, "summary": r["summary"], "trace": r["trace"], "timeline": r["timeline"],
                "urls": urls, "elapsed_s": r["elapsed_s"]}
    finally:
        _busy.release()
        shutil.rmtree(work, ignore_errors=True)


@app.get("/api/eval")
def eval_metrics():
    path = os.path.join(ROOT, "..", "results", "metrics.json")
    if not os.path.exists(path):
        raise HTTPException(404, "run eval/run_eval.py and eval/report.py first")
    with open(path) as f:
        return JSONResponse(json.load(f))


@app.get("/runs/{run_id}/{name}")
def run_file(run_id: str, name: str):
    if "/" in run_id or ".." in run_id or name not in ("annotated.mp4", "result.json"):
        raise HTTPException(404)
    path = os.path.join(RUNS, run_id, name)
    if not os.path.exists(path):
        raise HTTPException(404)
    return FileResponse(path, media_type="video/mp4" if name.endswith(".mp4") else "application/json")


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "static")), name="static")
results_dir = os.path.join(ROOT, "..", "results")
if os.path.isdir(results_dir):
    app.mount("/results", StaticFiles(directory=results_dir), name="results")
