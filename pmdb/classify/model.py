"""Two-stage random forest vs Batch_3, LOSO evaluation, metrics, arm selection (classifier plan C8-C12)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline

from pmdb.classify.features import BATCHES, TILE_ID

RF_PARAMS = dict(n_estimators=500, max_features="sqrt", min_samples_leaf=3, max_depth=None,
                 bootstrap=True, class_weight="balanced", random_state=0, n_jobs=1)
N_BOOT = 1000
SEED = 0
SELECT_TIE_BACC = 0.05
ARM_ORDER = ["KPI+FEM", "FEM", "KPI"]
P_COLS = [f"P_{b}" for b in BATCHES]
LEVELS = ["stage1", "stage2", "end_to_end"]


def make_stage_model() -> Pipeline:
    return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("rf", RandomForestClassifier(**RF_PARAMS))])


def _p1(model: Pipeline, X: np.ndarray) -> np.ndarray:
    p = model.predict_proba(X)
    return p[:, list(model.classes_).index(1)]


@dataclass
class TwoStage:
    features: list[str]
    stage1: Pipeline | None = None
    stage2: Pipeline | None = None

    def fit(self, tiles: pd.DataFrame) -> "TwoStage":
        if tiles["heldout"].any():
            raise ValueError("held-out rows must never enter training")
        X = tiles[self.features].to_numpy(float)
        y1 = (tiles["batch"] == "Batch_3").to_numpy().astype(int)
        self.stage1 = make_stage_model().fit(X, y1)
        m = tiles["batch"].isin(["Batch_1", "Batch_2"]).to_numpy()
        y2 = (tiles.loc[m, "batch"] == "Batch_1").to_numpy().astype(int)
        self.stage2 = make_stage_model().fit(X[m], y2)
        return self

    def tile_probs(self, tiles: pd.DataFrame) -> pd.DataFrame:
        X = tiles[self.features].to_numpy(float)
        out = tiles[TILE_ID].copy()
        out["p_b3"] = _p1(self.stage1, X)
        out["q_b1"] = _p1(self.stage2, X)
        return out.reset_index(drop=True)


def combine(p_b3: float, q_b1: float) -> dict:
    P = {"Batch_3": p_b3, "Batch_1": (1 - p_b3) * q_b1, "Batch_2": (1 - p_b3) * (1 - q_b1)}
    pred = "Batch_3" if p_b3 >= 0.5 else ("Batch_1" if q_b1 >= 0.5 else "Batch_2")
    return {"P_Batch_1": P["Batch_1"], "P_Batch_2": P["Batch_2"], "P_Batch_3": P["Batch_3"],
            "predicted": pred, "confidence": P[pred]}


def site_predictions(tile_probs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (batch, site), g in tile_probs.groupby(["batch", "site"], sort=True):
        p, q = float(g["p_b3"].mean()), float(g["q_b1"].mean())
        rows.append({"batch": batch, "site": site, "p_b3": p, "q_b1": q, **combine(p, q),
                     "n_tiles": len(g), "stage1_votes_not_b3": int((g["p_b3"] < 0.5).sum()),
                     "stage2_votes_b1": int((g["q_b1"] >= 0.5).sum())})
    return pd.DataFrame(rows)


def loso(table: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    lab = table[~table["heldout"]].sort_values(TILE_ID).reset_index(drop=True)
    parts = []
    for tr, te in LeaveOneGroupOut().split(lab, groups=lab["site"]):
        model = TwoStage(features).fit(lab.iloc[tr])
        parts.append(model.tile_probs(lab.iloc[te]))
    tp = pd.concat(parts, ignore_index=True).sort_values(TILE_ID).reset_index(drop=True)
    sp = site_predictions(tp)
    sp["correct"] = sp["predicted"] == sp["batch"]
    return sp, tp


def _level_frames(preds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    p = preds
    s1 = pd.DataFrame({"true": np.where(p.batch == "Batch_3", "Batch_3", "not_Batch_3"),
                       "pred": np.where(p.p_b3 >= 0.5, "Batch_3", "not_Batch_3"),
                       "P": list(zip(p.p_b3, 1 - p.p_b3)), "conf": np.maximum(p.p_b3, 1 - p.p_b3)})
    s1.attrs["classes"] = ["Batch_3", "not_Batch_3"]
    q = p[p.batch.isin(["Batch_1", "Batch_2"])]
    s2 = pd.DataFrame({"true": q.batch.to_numpy(), "pred": np.where(q.q_b1 >= 0.5, "Batch_1", "Batch_2"),
                       "P": list(zip(q.q_b1, 1 - q.q_b1)), "conf": np.maximum(q.q_b1, 1 - q.q_b1).to_numpy()})
    s2.attrs["classes"] = ["Batch_1", "Batch_2"]
    e2e = pd.DataFrame({"true": p.batch.to_numpy(), "pred": p.predicted.to_numpy(),
                        "P": list(zip(p.P_Batch_1, p.P_Batch_2, p.P_Batch_3)), "conf": p.confidence.to_numpy()})
    e2e.attrs["classes"] = list(BATCHES)
    return {"stage1": s1, "stage2": s2, "end_to_end": e2e}


def _brier(f: pd.DataFrame) -> float:
    cls = f.attrs["classes"]
    P = np.array(f["P"].tolist(), float)
    Y = np.array([[t == c for c in cls] for t in f["true"]], float)
    return float(((P - Y) ** 2).mean())


def _bootstrap(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(y)
    vals = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            vals.append(balanced_accuracy_score(y[idx], pred[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def level_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level, f in _level_frames(preds).items():
        y, pr = f["true"].to_numpy(), f["pred"].to_numpy()
        lo, hi = _bootstrap(y, pr)
        rows.append({"level": level, "n_sites": len(f), "site_acc": float((y == pr).mean()),
                     "balanced_acc": float(balanced_accuracy_score(y, pr)), "bacc_ci_lo": lo, "bacc_ci_hi": hi,
                     "macro_f1": float(f1_score(y, pr, average="macro")), "brier": _brier(f)})
    return pd.DataFrame(rows)


def confusion(preds: pd.DataFrame) -> pd.DataFrame:
    rows = [{"true": t, "predicted": p, "n": int(((preds.batch == t) & (preds.predicted == p)).sum())}
            for t in BATCHES for p in BATCHES]
    return pd.DataFrame(rows)


def calibration(preds: pd.DataFrame) -> pd.DataFrame:
    bins = {"end_to_end": [0.0, 0.5, 0.7, 0.9, 1.0], "stage1": [0.5, 0.7, 0.9, 1.0],
            "stage2": [0.5, 0.7, 0.9, 1.0]}
    rows = []
    for level, f in _level_frames(preds).items():
        edges = bins[level]
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            last = i == len(edges) - 2
            g = f[(f.conf >= lo) & ((f.conf <= hi) if last else (f.conf < hi))]
            rows.append({"level": level, "bin": f"[{lo:.1f},{hi:.1f}{']' if last else ')'}",
                         "n": len(g), "mean_conf": float(g.conf.mean()) if len(g) else np.nan,
                         "accuracy": float((g.true == g.pred).mean()) if len(g) else np.nan})
    return pd.DataFrame(rows)


def select_arm(metrics: pd.DataFrame) -> str:
    e = metrics[metrics["level"] == "end_to_end"].copy()
    if e.empty:
        raise ValueError("no end_to_end rows")
    best = e["balanced_acc"].max()
    tied = e[e["balanced_acc"] >= best - SELECT_TIE_BACC - 1e-12].copy()
    tied["order"] = tied["arm"].map({a: i for i, a in enumerate(ARM_ORDER)})
    return str(tied.sort_values(["brier", "order"]).iloc[0]["arm"])
