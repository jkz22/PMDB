"""D23 GNN batch classifier on Modal: Si particles = nodes, graphite-mediated Delaunay links = edges.

    modal run modal_gnn.py

Nodes: Si instances (8-connected) of the v0 BSE segmentation, half res. Node features: log area, eccentricity,
solidity, major/minor axis ratio, sin/cos 2*orientation. Edges: Delaunay neighbours <= 5 um apart whose segment
is < 50 % pore; edge features: length (um), graphite / pore / Si fraction along the segment.
Model: 2-layer GINE (plain torch), mean+max pooling, trained on random 20 um windows, site prediction = mean
softmax over a window grid. LOO over the 31 labelled sites (same protocol as the 21/31 fingerprint), plus one
fit on all 31 for the 3 held-out sites. Writes outputs/gnn/.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
BATCHES = ["Batch_1", "Batch_2", "Batch_3"]
HELDOUT = ["3e122cbj", "fn0mhxef", "xrv9xvzb"]
OF_RECORD = {"3e122cbj": "Batch_1", "fn0mhxef": "Batch_3", "xrv9xvzb": "Batch_2"}  # fingerprint A0, not truth
WIN = 400  # px = 20 um at 50 nm/px

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("numpy==1.26.4", "scipy==1.14.1", "scikit-image==0.25.2", "pandas==2.1.4",
                 "imagecodecs==2025.3.30", "tifffile==2025.5.10")
    .pip_install("torch==2.2.2", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir(ROOT / "cache" / "half", "/cache/half", ignore=["*.csv"])
    .add_local_dir(ROOT / "cache_heldout" / "half", "/cache/heldout", ignore=["*.csv"])
    .add_local_python_source("pmdb")
)
app = modal.App("pmdb-gnn", image=image)


@app.function(cpu=1.0, memory=4096, timeout=1800)
def build_graph(npz: str) -> dict:
    import numpy as np
    from scipy.spatial import Delaunay
    from skimage.measure import label, regionprops

    from pmdb.segment import segment_bse

    bse = np.load(npz)["image"][..., 0].astype(np.float64)
    m = segment_bse(bse, 50.0)
    lab = label(m.si & m.admissible, connectivity=2)
    props = [p for p in regionprops(lab) if p.area >= 4]
    pos = np.array([p.centroid for p in props])
    x = np.array([[np.log(p.area), p.eccentricity, p.solidity,
                   p.minor_axis_length / max(p.major_axis_length, 1e-6),
                   np.sin(2 * p.orientation), np.cos(2 * p.orientation)] for p in props], dtype=np.float32)
    phase = np.zeros(lab.shape, np.uint8)
    phase[m.graphite] = 1
    phase[m.pore] = 2
    phase[m.si] = 3
    tri = Delaunay(pos)
    e = np.vstack([tri.simplices[:, [0, 1]], tri.simplices[:, [1, 2]], tri.simplices[:, [0, 2]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    length = np.linalg.norm(pos[e[:, 0]] - pos[e[:, 1]], axis=1)
    t = np.linspace(0.15, 0.85, 16)[None, :, None]  # skip the endpoints, they sit inside the particles
    pts = np.rint(pos[e[:, 0], None] * (1 - t) + pos[e[:, 1], None] * t).astype(int)
    s = phase[pts[..., 0], pts[..., 1]]
    frac = np.stack([(s == k).mean(1) for k in (1, 2, 3)], 1)  # graphite, pore, si
    keep = (length <= 100) & (frac[:, 1] < 0.5)
    ea = np.column_stack([length * 0.05, frac]).astype(np.float32)[keep]
    return {"x": x, "pos": pos.astype(np.float32), "ei": e[keep].T.astype(np.int64), "ea": ea,
            "shape": lab.shape, "n_nodes": len(props), "n_edges": int(keep.sum())}


def _windows(g, rng=None, n=None):
    """Node index sets for random (rng) or grid (rng=None) WIN x WIN windows."""
    import numpy as np

    H, W = g["shape"]
    if rng is not None:
        tl = np.column_stack([rng.integers(0, max(H - WIN, 1), n), rng.integers(0, max(W - WIN, 1), n)])
    else:
        def starts(n):  # half-overlapping grid whose last window ends at the image edge
            return sorted(set(range(0, max(n - WIN, 0) + 1, WIN // 2)) | {max(n - WIN, 0)})

        tl = np.array([(r, c) for r in starts(H) for c in starts(W)])
    out = []
    for r, c in tl:
        idx = np.where((g["pos"][:, 0] >= r) & (g["pos"][:, 0] < r + WIN) & (g["pos"][:, 1] >= c) & (g["pos"][:, 1] < c + WIN))[0]
        if len(idx) >= 5:
            out.append(idx)
    return out


def _batch(items, mu, sd, emu, esd):
    """items: list of (graph, node idx). Returns torch tensors of the disjoint union (edges both directions)."""
    import numpy as np
    import torch

    xs, eis, eas, bs, off = [], [], [], [], 0
    for b, (g, idx) in enumerate(items):
        remap = -np.ones(g["n_nodes"], np.int64)
        remap[idx] = np.arange(len(idx))
        ei = remap[g["ei"]]
        ok = (ei >= 0).all(0)
        ei, ea = ei[:, ok], g["ea"][ok]
        xs.append((g["x"][idx] - mu) / sd)
        eis.append(np.hstack([ei, ei[::-1]]) + off)
        eas.append(np.vstack([ea, ea]))
        bs.append(np.full(len(idx), b))
        off += len(idx)
    ea = (np.vstack(eas) - emu) / esd
    return (torch.tensor(np.vstack(xs)), torch.tensor(np.hstack(eis)), torch.tensor(ea, dtype=torch.float32),
            torch.tensor(np.hstack(bs)), len(items))


def _model(nf, ef, h=32, nc=3):
    import torch
    from torch import nn

    class GINE(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.e = nn.Linear(ef, i)
            self.mlp = nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.Linear(o, o))
            self.eps = nn.Parameter(torch.zeros(1))

        def forward(self, x, ei, ea):
            msg = torch.relu(x[ei[0]] + self.e(ea))
            agg = torch.zeros_like(x).index_add_(0, ei[1], msg)
            return torch.relu(self.mlp((1 + self.eps) * x + agg))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1, self.c2 = GINE(nf, h), GINE(h, h)
            self.drop = nn.Dropout(0.4)
            self.head = nn.Linear(2 * h, nc)

        def forward(self, x, ei, ea, b, B):
            x = self.c2(self.c1(x, ei, ea), ei, ea)
            idx = b[:, None].expand(-1, x.shape[1])
            mean = torch.zeros(B, x.shape[1]).index_add_(0, b, x) / torch.bincount(b, minlength=B)[:, None].clamp(min=1)
            mx = torch.zeros(B, x.shape[1]).scatter_reduce(0, idx, x, "amax", include_self=False)
            return self.head(self.drop(torch.cat([mean, mx], 1)))

    return Net()


@app.function(cpu=2.0, memory=4096, timeout=3600)
def fit_predict(train: list[dict], y: list[int], test: list[dict], seed: int = 0, steps: int = 400) -> list[list[float]]:
    import numpy as np
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = np.vstack([g["x"] for g in train])
    E = np.vstack([g["ea"] for g in train])
    st = (X.mean(0), X.std(0) + 1e-6, E.mean(0), E.std(0) + 1e-6)
    net = _model(X.shape[1], E.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-3)
    y = np.asarray(y)
    w = torch.tensor(len(y) / (3 * np.bincount(y, minlength=3).clip(min=1)), dtype=torch.float32)
    net.train()
    for _ in range(steps):
        items, ys = [], []
        for i in rng.choice(len(train), 16):  # 16 sites x 4 windows per step
            for idx in _windows(train[i], rng, 4):
                items.append((train[i], idx))
                ys.append(y[i])
        loss = torch.nn.functional.cross_entropy(net(*_batch(items, *st)), torch.tensor(ys), weight=w)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    out = []
    with torch.no_grad():
        for g in test:
            p = torch.softmax(net(*_batch([(g, idx) for idx in _windows(g)], *st)), 1).mean(0)
            out.append(p.tolist())
    return out


@app.local_entrypoint()
def main(seeds: int = 3):
    import numpy as np

    rows = list(csv.DictReader(open(ROOT / "cache" / "half" / "manifest.csv")))
    sites = [(r["batch"], r["site"]) for r in rows]
    paths = [f"/cache/half/{b}__{s}.npz" for b, s in sites] + [f"/cache/heldout/Batch_heldout__{s}.npz" for s in HELDOUT]
    graphs = list(build_graph.map(paths))
    lab, ho = graphs[:len(sites)], graphs[len(sites):]
    y = [BATCHES.index(b) for b, _ in sites]
    print("nodes/site", [g["n_nodes"] for g in graphs], "edges/site", [g["n_edges"] for g in graphs])

    # ponytail: seed ensemble only; no permutation p-value (add 1000-perm .map over shuffled y if the LOO beats 21/31)
    calls = []
    for k in range(seeds):
        for i in range(len(sites)):
            calls.append(([g for j, g in enumerate(lab) if j != i], [v for j, v in enumerate(y) if j != i], [lab[i]], k))
        calls.append((lab, y, ho, k))
    res = list(fit_predict.starmap(calls))
    n = len(sites) + 1
    loo = np.mean([[r[0] for r in res[k * n:k * n + len(sites)]] for k in range(seeds)], 0)
    hop = np.mean([res[k * n + len(sites)] for k in range(seeds)], 0)

    out = ROOT / "outputs" / "gnn"
    out.mkdir(parents=True, exist_ok=True)
    pred = loo.argmax(1)
    y = np.array(y)
    conf = np.zeros((3, 3), int)
    np.add.at(conf, (y, pred), 1)
    recall = conf.diagonal() / conf.sum(1)
    metrics = {"loo_correct": int((pred == y).sum()), "n": len(y), "accuracy": float((pred == y).mean()),
               "balanced_accuracy": float(recall.mean()), "recall": dict(zip(BATCHES, recall.round(3).tolist())),
               "confusion_rows_true_cols_pred": conf.tolist(), "majority_baseline": float(np.bincount(y).max() / len(y)),
               "comparator_fingerprint_A0": "21/31", "seeds": seeds,
               "graph_sizes": {s: [g["n_nodes"], g["n_edges"]] for (_, s), g in zip(sites + [("h", h) for h in HELDOUT], graphs)}}
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    with open(out / "loo_predictions.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["batch", "site", "assigned", "confidence"] + [f"p_{b}" for b in BATCHES])
        for (b, s), p in zip(sites, loo):
            wr.writerow([b, s, BATCHES[p.argmax()], round(p.max(), 3)] + p.round(3).tolist())
    with open(out / "heldout_predictions.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["site", "assigned", "confidence", "fingerprint_A0_of_record", "agrees"] + [f"p_{b}" for b in BATCHES])
        for s, p in zip(HELDOUT, hop):
            a = BATCHES[p.argmax()]
            wr.writerow([s, a, round(p.max(), 3), OF_RECORD[s], a == OF_RECORD[s]] + p.round(3).tolist())
    print(json.dumps({k: metrics[k] for k in ("loo_correct", "accuracy", "balanced_accuracy", "recall", "confusion_rows_true_cols_pred")}, indent=1))
    print(open(out / "heldout_predictions.csv").read())
