"""Assign unlabelled sites to a batch with the analytical KPIs only (site_kpis.csv -> heldout_kpis.csv).

Model: per-KPI Gaussian per batch (batch mean, pooled within-batch SD), equal priors, top-k KPIs by
Kruskal-Wallis chosen on the training sites only. Validated by leave-one-site-out (LOSO) on the 31 labelled
sites, with a label-shuffle null. A 3-NN on the same z-scores is a second opinion. Imaging-only features
(grey-level fingerprint) are scored separately so material and acquisition evidence never mix.
"""
import json, sys, numpy as np, pandas as pd
from scipy import stats
from compare import KPIS, KPI_INFO

B = ["Batch_1", "Batch_2", "Batch_3"]
KS = [3, 5, 10, 20]
RNG = np.random.default_rng(0)


def fit(X, y, k):
    p = [stats.kruskal(*[X[y == b, j] for b in B]).pvalue for j in range(X.shape[1])]
    sel = np.argsort(p)[:k]
    mu = np.array([X[y == b][:, sel].mean(0) for b in B])
    sd = np.sqrt(sum(((X[y == b][:, sel] - mu[i]) ** 2).sum(0) for i, b in enumerate(B)) / (len(y) - len(B))) + 1e-12
    return sel, mu, sd


def loglik(m, x):
    sel, mu, sd = m
    return stats.norm.logpdf((x[sel] - mu) / sd).sum(1) - np.log(sd).sum()  # per batch


def post(ll, T=1.0):
    z = ll / T; z = np.exp(z - z.max()); return z / z.sum()


def knn(Xtr, ytr, sd, x, k=3):
    d = np.sqrt((((Xtr - x) / sd) ** 2).sum(1)); nn = np.argsort(d)[:k]
    return np.array([(ytr[nn] == b).mean() for b in B]), nn


def loso(X, y, k):
    LL = []
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        LL.append(loglik(fit(X[tr], y[tr], k), X[i]))
    return np.array(LL)


def score(LL, y):
    pred = np.array(B)[LL.argmax(1)]
    rec = [np.mean(pred[y == b] == b) for b in B]
    return float(np.mean(pred == y)), float(np.mean(rec)), pred


def best_T(LL, y):
    yi = np.array([B.index(v) for v in y]); Ts = np.geomspace(0.5, 200, 60)
    nll = [-np.mean([np.log(post(l, T)[j] + 1e-12) for l, j in zip(LL, yi)]) for T in Ts]
    return float(Ts[int(np.argmin(nll))]), float(min(nll))


def evaluate(X, y, name, nperm=200):
    out = {}
    for k in [k for k in KS if k <= X.shape[1]]:
        LL = loso(X, y, k); acc, bal, pred = score(LL, y)
        null = [score(loso(X, yp, k), yp)[1] for yp in (RNG.permutation(y) for _ in range(nperm))]
        T, nll = best_T(LL, y)
        out[k] = dict(acc=acc, bal_acc=bal, p_perm=float((1 + sum(n >= bal for n in null)) / (1 + nperm)),
                      null_bal_95=float(np.percentile(null, 95)), T=T, nll=nll, chance_nll=float(np.log(3)),
                      confusion=pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="pred")).to_dict())
        print(f"{name:9s} k={k:2d}  LOSO acc {acc:.2f}  balanced {bal:.2f}  (null 95% {out[k]['null_bal_95']:.2f}, p={out[k]['p_perm']:.3f})"
              f"  calibrated T={T:.1f} log-loss {nll:.2f} vs chance {np.log(3):.2f}", flush=True)
    return out


def assign(X, y, Xh, sites, k, T, feats, info, Xh_raw):
    m = fit(X, y, k); sel, mu, sd = m; res = []
    for s, x in zip(sites, Xh):
        ll = loglik(m, x); p = post(ll, T); kp, nn = knn(X, y, np.std(X, 0) + 1e-12, x)
        z = (x[sel] - mu) / sd  # per batch
        top = np.argsort(-np.abs(z[B.index("Batch_3")]))
        res.append(dict(site=s, assigned=B[int(p.argmax())], p={b: float(v) for b, v in zip(B, p)},
                        p_uncalibrated={b: float(v) for b, v in zip(B, post(ll))}, knn3={b: float(v) for b, v in zip(B, kp)},
                        kpis=[dict(kpi=feats[sel[j]], label=info.get(feats[sel[j]], feats[sel[j]]), value=float(x[sel[j]]),
                                   **{f"mean_{b}": float(mu[i, j]) for i, b in enumerate(B)},
                                   **{f"z_{b}": float(z[i, j]) for i, b in enumerate(B)}) for j in top]))
    return res


if __name__ == "__main__":
    lab = pd.read_csv("site_kpis.csv"); hel = pd.read_csv("heldout_kpis.csv")
    y = lab.batch.values; out = {}
    X = lab[KPIS].fillna(lab[KPIS].median()).values; Xh = hel[KPIS].fillna(lab[KPIS].median()).values
    print("material KPIs (analytical pipeline):")
    ev = evaluate(X, y, "material")
    k = min(ev, key=lambda k: ev[k]["nll"])  # chosen by LOSO log-loss
    out["material"] = dict(eval=ev, k=k, sites=assign(X, y, Xh, hel.site, k, ev[k]["T"], KPIS, {q: v[0] for q, v in KPI_INFO.items()}, Xh))
    if len(sys.argv) > 1:  # optional imaging-fingerprint table (outputs/clean/summary.csv + held-out version)
        F = ["BSE_D", "BSE_G", "BSE_si_graphite", "BSE_grey_step", "Inlens_grey_step", "BSE_sigma_e", "BSE_noise_sigma_g", "height"]
        a = pd.read_csv(sys.argv[1]).set_index("site").loc[lab.site]; h = pd.read_csv(sys.argv[2]).set_index("site").loc[hel.site]
        print("imaging fingerprint (acquisition, not material):")
        ev = evaluate(a[F].values.astype(float), y, "imaging")
        k = min(ev, key=lambda k: ev[k]["nll"])
        out["imaging"] = dict(eval=ev, k=k, sites=assign(a[F].values.astype(float), y, h[F].values.astype(float), hel.site, k,
                                                          ev[k]["T"], F, {}, None))
    if "imaging" in out:  # independent-evidence product (assumes material and acquisition evidence are independent)
        out["combined"] = dict(sites=[])
        for m, i in zip(out["material"]["sites"], out["imaging"]["sites"]):
            q = np.array([m["p"][b] * i["p"][b] for b in B]); q /= q.sum()
            out["combined"]["sites"].append(dict(site=m["site"], assigned=B[int(q.argmax())], p=dict(zip(B, q.tolist()))))
        print("\ncombined (material x imaging):")
        for r in out["combined"]["sites"]:
            print(f"  {r['site']}: {r['assigned']}  p=" + " ".join(f"{b[-1]}:{v:.2f}" for b, v in r["p"].items()))
    for kind, o in [(k, v) for k, v in out.items() if k != "combined"]:
        print(f"\n{kind}: k={o['k']}  T={o['eval'][o['k']]['T']:.1f}")
        for r in o["sites"]:
            print(f"  {r['site']}: {r['assigned']}  p=" + " ".join(f"{b[-1]}:{v:.2f}" for b, v in r["p"].items())
                  + "  3-NN " + " ".join(f"{b[-1]}:{v:.2f}" for b, v in r["knn3"].items()))
            for q in r["kpis"][:5]:
                print(f"     {q['kpi']:24s} {q['value']:.4g}  means B1 {q['mean_Batch_1']:.4g} B2 {q['mean_Batch_2']:.4g} B3 {q['mean_Batch_3']:.4g}"
                      f"  z vs B1/B2/B3 {q['z_Batch_1']:+.1f}/{q['z_Batch_2']:+.1f}/{q['z_Batch_3']:+.1f}")
    json.dump(out, open("heldout_assign.json", "w"), indent=1, default=str)
