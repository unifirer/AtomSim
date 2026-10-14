import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from atomsim.constants import AU_TIME_FS
from atomsim.server.app import create_app
from atomsim.superposition import Component, superposition_grid
from atomsim.systems import get_system

SLOSH = [
    {"n": 1, "l": 0, "m": 0, "weight": 1.0},
    {"n": 2, "l": 1, "m": 0, "weight": 1.0, "phase": 0.5},
]


@pytest.fixture()
def client():
    with TestClient(create_app()) as c:
        yield c


def _wait_done(client, job_id, deadline_s=30.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < deadline_s:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not finish")


def test_superposition_job_end_to_end(client):
    r = client.post(
        "/api/jobs/superposition", json={"components": SLOSH, "resolution": 48}
    )
    assert r.status_code == 200
    job_id = r.json()["id"]
    assert _wait_done(client, job_id)["status"] == "done"

    meta = client.get(f"/api/jobs/{job_id}/meta").json()
    assert meta["kind"] == "superposition"
    assert meta["resolution"] == 48
    assert meta["provenance"]["fidelity"] == "exact"
    assert meta["time_unit_fs"] == pytest.approx(0.02418884, rel=1e-6)

    # coefficients normalized, phase carried through
    c = [complex(k["re"], k["im"]) for k in meta["components"]]
    assert sum(abs(x) ** 2 for x in c) == pytest.approx(1.0)
    assert np.angle(c[1]) == pytest.approx(0.5)

    # one beat, at the hydrogen 1s-2p gap with the reduced mass in it
    mu = get_system("h").mu_ratio.value
    assert meta["components"][1]["energy"]["value"] == pytest.approx(-mu / 8)
    (beat,) = meta["beat_periods"]
    assert beat["unit"] == "fs"
    assert beat["value"] == pytest.approx(2 * np.pi / (0.375 * mu) * AU_TIME_FS)
    assert beat["value"] == pytest.approx(0.405, abs=1e-3)

    # each channel is the engine's grid for that component, bit for bit
    sg = superposition_grid(
        [Component(**k) for k in SLOSH], mu_ratio=mu, resolution=48
    )
    for k in range(2):
        raw = client.get(f"/api/jobs/{job_id}/data?channel=psi{k}").content
        values = np.frombuffer(raw, dtype=np.float32).reshape(48, 48)
        assert np.array_equal(values, sg.psi[k].astype(np.float32))
    assert client.get(f"/api/jobs/{job_id}/data").status_code == 422
    assert client.get(f"/api/jobs/{job_id}/data?channel=psi2").status_code == 422


def test_degenerate_mix_reports_no_beats(client):
    comps = [{"n": 2, "l": 0, "m": 0, "weight": 1}, {"n": 2, "l": 1, "m": 1, "weight": 1}]
    job_id = client.post(
        "/api/jobs/superposition", json={"components": comps, "resolution": 32}
    ).json()["id"]
    assert _wait_done(client, job_id)["status"] == "done"
    assert client.get(f"/api/jobs/{job_id}/meta").json()["beat_periods"] == []


@pytest.mark.parametrize(
    "body",
    [
        {"components": []},
        {"components": [SLOSH[0]] * 5},
        {"components": [SLOSH[0], SLOSH[0]]},
        {"components": [{"n": 1, "l": 0, "m": 0, "weight": 0}]},
        {"components": [{"n": 1, "l": 0, "m": 0, "weight": -1}]},
        {"components": [{"n": 1, "l": 1, "m": 0, "weight": 1}]},
        {"components": SLOSH, "resolution": 4096},
        {"components": SLOSH, "system": "na"},
        {"components": SLOSH, "system": "nonsense"},
    ],
)
def test_bad_requests_are_refused_up_front(client, body):
    assert client.post("/api/jobs/superposition", json=body).status_code in (404, 422)


def test_many_electron_refusal_says_why(client):
    r = client.post("/api/jobs/superposition", json={"components": SLOSH, "system": "na"})
    assert r.status_code == 422
    assert "eigenstate" in r.json()["detail"]
