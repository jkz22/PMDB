import base64, json, os, re, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analytical_benchmarks"))
S = pytest.importorskip("simulate")
V = pytest.importorskip("viewer")
from test_analytical_simulate import disc_lab  # noqa: E402


def unrle(r, n):
    return np.repeat(np.array(r["v"], np.uint8), r["n"])[:n]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("sim")
    r = S.simulate(disc_lab(2), cycles=3, seed=0, snap=True)
    r["traj"]["seed"] = 0
    r["traj"].to_csv(out / "traj_disc.csv", index=False)
    r["within"].to_csv(out / "within_disc.csv", index=False)
    mp = r["maps"]
    labs = np.stack([l for _, l in r["snaps"]])
    np.savez_compressed(out / "frames_disc.npz", labs=labs, m_cycle=[m[0] for m in mp], m_charge=[m[1] for m in mp],
                        m_t=[m[2] for m in mp], li=np.stack([m[3] for m in mp]), s1=np.stack([m[4] for m in mp]),
                        px=S.PX, mb=S.MB)
    end = r["traj"].iloc[-1]
    summ = {"disc": {k: dict(end=float(end[k])) for k in ("retention_pct", "n_cracks")}}
    json.dump(dict(cycles=3, crate=1.0, seeds=1, width_um=12, summary=summ), open(out / "sim.json", "w"))
    old = V.OUT
    V.OUT = str(out)
    try:
        V.main()
    finally:
        V.OUT = old
    return labs, open(out / "viewer_body.html").read()


def test_viewer_is_offline_and_data_roundtrips(built):
    labs, body = built
    assert not re.search(r"https?://|fetch\(|XMLHttpRequest|WebSocket|EventSource|import\(", body)
    assert not re.search(r"__[A-Z]+__", body)
    d = json.loads(re.search(r'<script type="application/json" id="tv-data">(.*?)</script>', body, re.S).group(1).replace("<\\/", "</"))
    s = d["sites"]["disc"]
    assert s["ncyc"] == 3 and len(s["diffs"]) == 3 and sorted(s["maps"]) == ["1", "3"]
    cur = unrle(s["f0"], s["H"] * s["W"]).copy()
    for k, (ix, v) in enumerate(s["diffs"], 1):
        cur[np.frombuffer(base64.b64decode(ix), np.uint32)] = np.frombuffer(base64.b64decode(v), np.uint8)
        assert (cur.reshape(s["H"], s["W"]) == labs[k]).all()
    m = s["maps"]["1"]
    nmask = int(unrle(m["mask"], s["H"] * s["W"] // s["mb"] ** 2).sum())
    assert len(m["t"]) == len(m["li"]) == len(m["s1"]) and nmask > 0
    assert all(len(base64.b64decode(x)) == nmask for x in m["li"] + m["s1"])
    assert all(len(t[0]) == 4 for t in s["traj"].values())
