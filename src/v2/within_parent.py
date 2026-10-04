"""Parent-image view of the off-detector: the 31 fields are crops of 13 parent images that are split across
batches (outputs/parent_groups.csv from main). Within a mixed parent the acquisition is shared, so any batch
difference there is the designed signal; between parents it is confounded with the session."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from src.v2.common import OUT

PAR = pd.read_csv(OUT / "parent_groups.csv")
IM = pd.read_csv(OUT / "imaging_stats" / "per_image.csv").query("view == 'BSE' and channel == 0")[["site", "p1", "p99", "noise_sigma", "sharpness"]]
KP = pd.read_csv(OUT / "kpis" / "crop_kpis_eval.csv")
KP["site"] = KP.group_id.str.split("/").str[1]
KPf = KP.groupby("site")[["frac_si", "frac_graphite", "frac_pore", "K01_si_frac_adm", "K04_agglom_frac"]].mean().reset_index()
LAT = pd.read_csv(OUT / "cluster" / "latent_coords.csv").rename(columns={"field": "site"})[["site", "PC1", "PC2", "PC3"]]
OFF = {r: pd.read_csv(OUT / "off" / f"fields_{r}_resnet18_imnet.csv").assign(site=lambda d: d.group_id.str.split("/").str[1])[["site", "p_off"]].rename(columns={"p_off": f"p_off_{r}"}) for r in ("raw", "extreme")}
D = PAR.merge(IM, on="site", how="left").merge(KPf, on="site", how="left").merge(LAT, on="site", how="left")
for r in OFF.values(): D = D.merge(r, on="site", how="left")
D["off"] = (D.batch != "Batch_3").astype(int)
lab = D[D.batch != "Batch_heldout"].copy()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
mix = lab.groupby("parent_id").batch.nunique(); mixed = mix[mix > 1].index.tolist()
print("parents:", lab.parent_id.nunique(), "mixed:", mixed)
cols = ["parent_id", "batch", "site", "p1", "noise_sigma", "sharpness", "frac_si", "frac_pore", "K04_agglom_frac", "PC1", "PC2", "p_off_raw", "p_off_extreme"]
print("\n== fields of the mixed parents (same acquisition, different batch label) ==")
print(D[D.parent_id.isin(mixed)].sort_values(["parent_id", "batch"])[cols].round(3).to_string(index=False))
print("\n== pure parents ==")
print(lab[~lab.parent_id.isin(mixed)].groupby(["parent_id", "batch"]).agg(n=("site", "count"), noise=("noise_sigma", "mean"), sharp=("sharpness", "mean"), p1=("p1", "mean"), si=("frac_si", "mean"), p_off_raw=("p_off_raw", "mean"), p_off_ext=("p_off_extreme", "mean")).round(3).to_string())
# within-parent contrasts: value minus parent mean, pooled over mixed parents, off vs B3
lab["pm"] = lab.parent_id
feats = ["p1", "p99", "noise_sigma", "sharpness", "frac_si", "frac_graphite", "frac_pore", "K01_si_frac_adm", "K04_agglom_frac", "PC1", "PC2", "PC3", "p_off_raw", "p_off_extreme"]
W = lab[lab.parent_id.isin(mixed)].copy()
for f in feats: W[f + "_c"] = W[f] - W.groupby("parent_id")[f].transform("mean")
sd = lab[feats].std()
print("\n== within-mixed-parent contrast (B3 minus off, in pooled field SD) ==")
rows = []
for f in feats:
    a, b = W.loc[W.off == 0, f + "_c"], W.loc[W.off == 1, f + "_c"]
    rows.append(dict(feature=f, n_B3=len(a), n_off=len(b), gap_sd=round((a.mean() - b.mean()) / sd[f], 2)))
print(pd.DataFrame(rows).to_string(index=False))
# leave-one-parent-out probes
def lopo(X, y, groups):
    p = np.zeros(len(y))
    for g in np.unique(groups):
        te = groups == g
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")).fit(X[~te], y[~te])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p
print("\n== leave-one-parent-out (13 folds) vs leave-one-field-out, B3 vs off ==")
sets = {"noise+sharpness": ["noise_sigma", "sharpness"], "p1+p99": ["p1", "p99"], "5 KPIs": ["frac_si", "frac_graphite", "frac_pore", "K01_si_frac_adm", "K04_agglom_frac"], "latent PC1-3": ["PC1", "PC2", "PC3"], "all imaging": ["p1", "p99", "noise_sigma", "sharpness"]}
y = lab.off.to_numpy()
for name, fs in sets.items():
    X = lab[fs].to_numpy()
    pp = lopo(X, y, lab.parent_id.to_numpy()); pf = lopo(X, y, lab.site.to_numpy())
    print(f"{name:16s} LOPO acc {((pp > .5) == y).mean():.2f} auroc {roc_auc_score(y, pp):.2f} | LOFO acc {((pf > .5) == y).mean():.2f} auroc {roc_auc_score(y, pf):.2f}")
for r in ("raw", "extreme"):
    p = lab[f"p_off_{r}"].to_numpy(); print(f"CNN {r:8s} (grouped 5-fold, not parent-grouped) acc {((p > .5) == y).mean():.2f} auroc {roc_auc_score(y, p):.2f}")
print("\n== CNN out-of-fold accuracy, mixed vs pure parents ==")
for r in ("raw", "extreme"):
    for nm, sub in (("mixed", lab[lab.parent_id.isin(mixed)]), ("pure", lab[~lab.parent_id.isin(mixed)])):
        print(f"{r:8s} {nm:5s} n={len(sub):2d} acc {(((sub[f'p_off_{r}'] > .5).astype(int)) == sub.off).mean():.2f}")
D.to_csv(OUT / "parents" / "fields_by_parent.csv", index=False)
