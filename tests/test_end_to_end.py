from fastapi.testclient import TestClient

from boilguard.monitor import RunOptions, run_sim
from boilguard.sim import random_params


def test_closed_loop_prevents_boil_over_when_cook_is_away():
    p = random_params(1, overflow=True)
    p.duration = 40.0
    r = run_sim(p, RunOptions(policy="boilguard", human="absent", seed=1))
    s = r["summary"]
    assert s["natural_overflow_t"] is not None          # it would have boiled over
    assert not s["overflowed"]                           # BoilGuard prevented it
    assert any(e["decision"].startswith("reduce_heat") for e in r["trace"])


def test_api_health_and_simulate(tmp_path, monkeypatch):
    monkeypatch.setenv("BOILGUARD_RUNS_DIR", str(tmp_path))
    from app.server import app
    c = TestClient(app)
    h = c.get("/health").json()
    assert h["ok"] and h["opencv"].startswith("5")
    r = c.post("/api/simulate", json={"seed": 1, "cook": "absent", "bedrock": False})
    assert r.status_code == 200, r.text
    d = r.json()
    assert "annotated.mp4" in d["urls"] and d["trace"]
    assert c.get(d["urls"]["annotated.mp4"]).status_code == 200
