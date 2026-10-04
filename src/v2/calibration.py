"""Calibration and 'does it just lean on Batch_3?' checks for the 3-class batch classifiers.

Pools the out-of-fold crop_predictions.csv of every completed fold of a config (all 31 fields are
test fields exactly once), then per config:
  * ECE / MCE (15 bins) and NLL at crop and field level, before and after temperature scaling;
    T is fitted cross-fold (on the other folds' out-of-fold predictions) so it is never fitted on
    the crops it is applied to;
  * balanced accuracy, per-class recall, predicted-class shares vs the true shares (Batch_3 is
    17/31 = 0.55 of fields; a constant Batch_3 classifier scores 0.55 field accuracy, 0.33 balanced);
  * prior correction: divide probabilities by the training-set class prior and renormalise (a
    classifier that is right for structural reasons keeps its accuracy; one that leans on the
    majority prior loses Batch_3 recall and gains Batch_1/2 recall);
  * over-confidence of wrong field predictions (mean max-probability on wrong vs right fields).
Writes outputs/v2/calibration/cls_calibration.csv and reliability_<group>.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.v2.classify import BATCHES, CLS_RUNS, DEFAULTS
from src.v2.common import OUT

CAL = OUT / "calibration"
KEYS = ("arch", "view", "input", "harmonise", "aug", "dequant")


def _cfg_key(c: dict) -> tuple:
    return tuple(str(c.get(k, DEFAULTS.get(k))) for k in KEYS)


def load_folds(runs_dir: Path = CLS_RUNS) -> dict[tuple, list[tuple[dict, pd.DataFrame]]]:
    out: dict[tuple, list] = {}
    for d in sorted(runs_dir.iterdir()):
        f = d / "crop_predictions.csv"
        if not f.exists() or not (d / "metrics.json").exists():
            continue
        c = json.loads((d / "config.json").read_text())
        if c.get("split", "strat") != "strat" or c.get("n_folds", 5) != 5:
            continue
        out.setdefault(_cfg_key(c), []).append((c, pd.read_csv(f)))
    return out


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> tuple[float, float]:
    conf, pred = p.max(1), p.argmax(1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e, mx = 0.0, 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            gap = abs(acc[m].mean() - conf[m].mean())
            e += m.mean() * gap; mx = max(mx, gap)
    return float(e), float(mx)


def nll(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean())


def fit_temperature(logp: np.ndarray, y: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar
    def loss(t):
        z = logp / t
        z = z - z.max(1, keepdims=True)
        return -(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean()
    return float(minimize_scalar(loss, bounds=(0.05, 20), method="bounded").x)


def apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(p, 1e-9, 1)) / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def prior_correct(p: np.ndarray, prior: np.ndarray) -> np.ndarray:
    q = p / prior[None]
    return q / q.sum(1, keepdims=True)


def field_level(df: pd.DataFrame, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = df.group_id.to_numpy()
    f = pd.DataFrame(P).groupby(g).mean()
    y = pd.Series(df.y_true.to_numpy()).groupby(g).first().loc[f.index].to_numpy()
    return f.to_numpy(), y


def summarise(key: tuple, folds: list[tuple[dict, pd.DataFrame]]) -> dict:
    pcols = [f"p_{b}" for b in BATCHES]
    frames, Pc = [], []
    for k, (c, df) in enumerate(folds):
        df = df.assign(fold=c["fold"])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    P, y = df[pcols].to_numpy(), df.y_true.to_numpy()
    logP = np.log(np.clip(P, 1e-9, 1))
    # cross-fold temperature: fit on the other folds' out-of-fold predictions
    Pt = P.copy(); Ts = []
    for f in df.fold.unique():
        m = df.fold.to_numpy() == f
        t = fit_temperature(logP[~m], y[~m]); Ts.append(t)
        Pt[m] = apply_temperature(P[m], t)
    # training prior per fold = share of training crops (complement of the test fold)
    Pp = P.copy()
    for f in df.fold.unique():
        m = df.fold.to_numpy() == f
        prior = np.bincount(y[~m], minlength=3) / (~m).sum()
        Pp[m] = prior_correct(P[m], prior)
    out = dict(zip(KEYS, key), n_folds=len(folds), n_crops=len(df), n_fields=df.group_id.nunique(),
               temperature_mean=float(np.mean(Ts)), temperature_sd=float(np.std(Ts)))
    for tag, Q in (("", P), ("_temp", Pt), ("_prior", Pp)):
        pred = Q.argmax(1)
        e, mx = ece(Q, y)
        out[f"crop_acc{tag}"] = float((pred == y).mean())
        out[f"crop_ece{tag}"], out[f"crop_mce{tag}"], out[f"crop_nll{tag}"] = e, mx, nll(Q, y)
        F, yf = field_level(df, Q)
        pf = F.argmax(1)
        rec = [float((pf[yf == k] == k).mean()) for k in range(3)]
        out[f"field_acc{tag}"] = float((pf == yf).mean())
        out[f"field_bal_acc{tag}"] = float(np.mean(rec))
        for k, b in enumerate(BATCHES):
            out[f"field_recall_{b}{tag}"] = rec[k]
            out[f"field_pred_share_{b}{tag}"] = float((pf == k).mean())
        ef, mxf = ece(F, yf, n_bins=5)
        out[f"field_ece{tag}"], out[f"field_nll{tag}"] = ef, nll(F, yf)
        conf = F.max(1)
        out[f"field_conf_right{tag}"] = float(conf[pf == yf].mean()) if (pf == yf).any() else np.nan
        out[f"field_conf_wrong{tag}"] = float(conf[pf != yf].mean()) if (pf != yf).any() else np.nan
    out["field_true_share_Batch_3"] = float((yf == 2).mean())
    out["const_batch3_field_acc"] = float((yf == 2).mean())
    return out


def reliability_plot(groups: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, n_bins: int = 10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(groups), figsize=(3.2 * len(groups), 3.4), squeeze=False)
    bins = np.linspace(0, 1, n_bins + 1)
    for a, (name, (P, y)) in zip(ax[0], groups.items()):
        conf, acc = P.max(1), (P.argmax(1) == y).astype(float)
        xs, ys, ns = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 5:
                xs.append(conf[m].mean()); ys.append(acc[m].mean()); ns.append(m.sum())
        a.plot([0, 1], [0, 1], "k--", lw=0.8)
        a.scatter(xs, ys, s=10 + 200 * np.array(ns) / max(ns), alpha=0.7)
        a.plot(xs, ys, lw=1)
        e, _ = ece(P, y)
        a.set_title(f"{name}\nECE={e:.3f}", fontsize=9); a.set_xlabel("confidence"); a.set_xlim(0.3, 1); a.set_ylim(0, 1)
    ax[0][0].set_ylabel("accuracy")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    CAL.mkdir(parents=True, exist_ok=True)
    folds = load_folds()
    rows = [summarise(k, v) for k, v in folds.items() if len(v) == 5]
    df = pd.DataFrame(rows).sort_values(["arch", "view", "harmonise", "aug"])
    df.to_csv(CAL / "cls_calibration.csv", index=False)
    # reliability diagrams for the stack ResNet-18 across input variants
    pcols = [f"p_{b}" for b in BATCHES]
    groups = {}
    for k, v in folds.items():
        if len(v) == 5 and k[0] == "resnet18_imnet" and k[1] == "stack" and k[4] == "aug1" and k[5] in ("0", "0.0"):
            d = pd.concat([x[1] for x in v]); groups[k[3]] = (d[pcols].to_numpy(), d.y_true.to_numpy())
    if groups:
        reliability_plot(groups, CAL / "reliability_resnet18_stack.png")
    cols = ["arch", "view", "harmonise", "aug", "field_acc", "field_bal_acc", "field_recall_Batch_3", "field_pred_share_Batch_3",
            "field_acc_prior", "field_bal_acc_prior", "field_recall_Batch_3_prior", "crop_ece", "crop_ece_temp", "temperature_mean",
            "field_conf_right", "field_conf_wrong"]
    print(df[cols].round(3).to_string(index=False))
    return df


if __name__ == "__main__":
    main()
