"""Feature importance for the baseline-free QC: which KPIs separate batches, and which make sites unusual.

1. Univariate (primary, model-free): eta^2 = share of site-to-site variance explained by batch (on ranks),
   Kruskal-Wallis p (Holm over all KPIs), largest leave-one-batch-out |Cohen's d|.
2. Site level: robust z of every site on every KPI vs all other sites (heatmap).
3. Multivariate cross-check: random forest predicting batch from all KPIs, stratified CV balanced accuracy
   vs a label-shuffle null. If it does not clearly beat the null, model-based importances are not meaningful.
"""
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from compare import KPI_INFO, KPIS, holm

d = pd.read_csv("site_kpis.csv"); res = json.load(open("compare.json")); B = sorted(d.batch.unique())
rows = []
for k in KPIS:
    r = stats.rankdata(d[k]); g = [r[d.batch.values == b] for b in B]
    ssb = sum(len(x) * (x.mean() - r.mean()) ** 2 for x in g); eta2 = ssb / ((r - r.mean()) ** 2).sum()
    kw = stats.kruskal(*[d[k][d.batch == b] for b in B]).pvalue
    loo = {b: next(x for x in res["loo"][b]["rows"] if x["kpi"] == k)["d"] for b in B}
    bb = max(loo, key=lambda b: abs(loo[b]))
    rows.append(dict(kpi=k, name=KPI_INFO[k][0], eta2=eta2, kw_p=kw, max_abs_d=abs(loo[bb]), d_signed=loo[bb], batch=bb))
U = pd.DataFrame(rows); U["kw_p_holm"] = holm(U.kw_p.values); U = U.sort_values("eta2", ascending=False)

med = {k: d[k].median() for k in KPIS}
Z = pd.DataFrame({k: [(d[k][i] - d[k].drop(i).median()) / (1.4826 * stats.median_abs_deviation(d[k].drop(i)) + 1e-12)
                      for i in d.index] for k in KPIS}, index=d.batch.str.replace("Batch_", "B") + " " + d.site)

X = ((d[KPIS] - d[KPIS].mean()) / d[KPIS].std()).values; y = d.batch.values
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
def cv_bacc(yy):
    s = []
    for tr, te in cv.split(X, yy):
        m = RandomForestClassifier(200, class_weight="balanced", random_state=0, n_jobs=4).fit(X[tr], yy[tr])
        s.append(balanced_accuracy_score(yy[te], m.predict(X[te])))
    return float(np.mean(s))
acc = cv_bacc(y); rng = np.random.default_rng(0)
null = np.array([cv_bacc(rng.permutation(y)) for _ in range(30)])
p_model = float((1 + (null >= acc).sum()) / (1 + len(null)))

out = dict(univariate=U.to_dict("records"),
           model=dict(cv_balanced_accuracy=acc, chance=1 / len(B), null_mean=float(null.mean()),
                      null_95=float(np.percentile(null, 95)), p=p_model),
           site_top={s: Z.loc[s].abs().sort_values(ascending=False).head(4).round(1).to_dict()
                     for s in Z.index if (Z.loc[s].abs() > 3.5).sum() >= 2})
json.dump(out, open("importance.json", "w"), indent=1, default=float)
U.to_csv("importance.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(19, 7.5), gridspec_kw=dict(width_ratios=[1, 1.5]))
col = {"Batch_1": "#2f6fdf", "Batch_2": "#e08a1e", "Batch_3": "#c23b3b"}
a = ax[0]; yy = np.arange(len(U))[::-1]
a.barh(yy, U.eta2, color=[col[b] for b in U.batch])
for yi, (_, r) in zip(yy, U.iterrows()):
    a.text(r.eta2 + .005, yi, f"d={r.d_signed:+.1f} ({r.batch.replace('Batch_', 'B')})  p_holm={r.kw_p_holm:.2f}", va="center", fontsize=7)
a.set_yticks(yy); a.set_yticklabels(U.name, fontsize=8); a.set_xlim(0, max(.45, U.eta2.max() * 1.6))
a.set_xlabel("η² = share of site-to-site variance explained by batch (ranks)")
a.set_title("KPI importance for separating batches\n(colour = batch that differs most from the rest)", fontsize=10)
a = ax[1]; Zc = Z.clip(-8, 8).sort_index()
im = a.imshow(Zc.values, cmap="RdBu_r", vmin=-8, vmax=8, aspect="auto")
a.set_xticks(range(len(KPIS))); a.set_xticklabels([KPI_INFO[k][0][:28] for k in KPIS], rotation=70, ha="right", fontsize=7)
a.set_yticks(range(len(Zc))); a.set_yticklabels(Zc.index, fontsize=7)
for i, j in zip(*np.where(np.abs(Zc.values) > 3.5)): a.text(j, i, "●", ha="center", va="center", fontsize=6, color="k")
plt.colorbar(im, ax=a, fraction=.03, label="robust z vs all other sites (● = |z|>3.5)")
a.set_title("Which KPIs make each site unusual", fontsize=10)
fig.tight_layout(); fig.savefig("fig_importance.png", dpi=75)

print(U[["name", "eta2", "kw_p", "kw_p_holm", "d_signed", "batch"]].round(3).to_string(index=False))
print("RF:", out["model"]); print(json.dumps(out["site_top"], indent=0))
