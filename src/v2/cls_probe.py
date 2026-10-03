"""CPU batch-classification baselines (3 classes, split by field code), no GPU training:

* every saved embedding (outputs/v2/runs/*/embeddings_eval.npz): logistic regression and kNN
* KPI features only (Kevin's crop KPIs, all columns) -> what microstructure alone gives
* imaging statistics only (p1/p99/noise/sharpness/GMM) -> what brightness/focus alone gives

Leave-one-field-out CV over all 31 fields; crop-level and field-level (mean probability) accuracy.
Writes outputs/v2/cls_probe.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.v2.common import OUT
from src.v2.kpi_adapter import ALL_COLS, crop_kpi_frame
from src.v2.evaluate import IMG_STATS
from src.v2.latent_audit import load

BATCHES = ("Batch_1", "Batch_2", "Batch_3")


def probe(X: np.ndarray, meta: pd.DataFrame, name: str, source: str) -> list[dict]:
    y = meta.batch.map(BATCHES.index).to_numpy()
    ok = ~np.isnan(X).any(1)
    X, y, g = X[ok], y[ok], meta.group_id.to_numpy()[ok]
    rows = []
    for clf_name, clf in (("logreg", LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")),
                          ("knn5", KNeighborsClassifier(5))):
        pipe = make_pipeline(StandardScaler(), clf)
        P = cross_val_predict(pipe, X, y, groups=g, cv=LeaveOneGroupOut(), method="predict_proba")
        pred = P.argmax(1)
        fld = pd.DataFrame(P).groupby(g).mean()
        yf = pd.Series(y).groupby(g).first().loc[fld.index].to_numpy()
        pf = fld.to_numpy().argmax(1)
        rows.append(dict(source=source, name=name, clf=clf_name, n_crops=len(y), n_fields=len(fld),
                         crop_acc=float((pred == y).mean()), crop_f1=float(f1_score(y, pred, average="macro")),
                         field_acc=float((pf == yf).mean()), field_f1=float(f1_score(yf, pf, average="macro")),
                         **{f"field_acc_{b}": float((pf[yf == k] == k).mean()) for k, b in enumerate(BATCHES) if (yf == k).any()}))
    return rows


def imaging_features(meta: pd.DataFrame) -> np.ndarray:
    ist = pd.read_csv(OUT / "imaging_stats" / "per_crop.csv")
    X = meta[["group_id", "y", "x"]].copy()
    for c in (0, 1, 2):
        s = ist[ist.channel == c][["group_id", "y_half", "x_half", *IMG_STATS]]
        s.columns = ["group_id", "y", "x", *[f"{k}_c{c}" for k in IMG_STATS]]
        X = X.merge(s, on=["group_id", "y", "x"], how="left")
    return X.drop(columns=["group_id", "y", "x"]).to_numpy(float)


def main(pair: tuple[str, str] | None = None):
    """pair=('Batch_1','Batch_2'): restrict to two batches (binary, chance 0.5) -> cls_probe_<b1>_<b2>.csv"""
    rows = []
    kp = crop_kpi_frame("eval", ALL_COLS)
    meta0 = kp[["group_id", "y", "x"]].copy(); meta0["batch"] = meta0.group_id.str.split("/").str[0]
    if pair:
        keep = meta0.batch.isin(pair).to_numpy()
        kp, meta0 = kp[keep].reset_index(drop=True), meta0[keep].reset_index(drop=True)
    gated = ["frac_si", "frac_graphite", "frac_pore", "K01_si_frac_adm", "K04_agglom_frac"]
    rows += probe(kp[gated].to_numpy(float), meta0, "kpi_gated5", "features")
    k_all = kp[list(ALL_COLS)].copy()
    k_all = k_all.fillna(k_all.median())  # K03 is undefined on most crops; impute so all rows are usable
    rows += probe(k_all.to_numpy(float), meta0, "kpi_all10_imputed", "features")
    rows += probe(imaging_features(meta0), meta0, "imaging_stats", "features")
    for mp in sorted(Path(OUT / "runs").glob("*/embeddings_eval.npz")):
        c = json.loads((mp.parent / "config.json").read_text())
        E, meta = load(mp.parent)
        if pair:
            keep = meta.batch.isin(pair).to_numpy()
            E, meta = E[keep], meta[keep].reset_index(drop=True)
        tag = f"{c['family']}|{c.get('view')}|{c.get('input')}|{c.get('train_set')}|{c.get('aug')}|kpi={c.get('kpi_set', 'gated')}"
        rows += probe(E, meta, tag, "embedding:" + mp.parent.name)
    df = pd.DataFrame(rows).sort_values("field_acc", ascending=False)
    df.to_csv(OUT / ("cls_probe.csv" if not pair else f"cls_probe_{pair[0]}_{pair[1]}.csv"), index=False)
    print(df.head(25).to_string())
    return df


if __name__ == "__main__":
    import sys
    main(tuple(sys.argv[1:3]) if len(sys.argv) == 3 else None)
