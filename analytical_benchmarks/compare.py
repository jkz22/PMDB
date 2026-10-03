import collections
"""Baseline-free batch comparison on per-site KPIs (site_kpis.csv).

For each batch B: B vs all other sites pooled (leave-one-batch-out), plus all pairs.
 - diff of means, Welch 95% CI, permutation p (site labels shuffled), Holm-adjusted per batch
 - effect size d = diff / pooled within-batch site SD
 - spread test: is B less uniform than the rest (permutation on |x - median|)
Verdict per KPI: outlier (p_holm<0.01 & |d|>=0.8), investigate (p_holm<0.05 or spread p<0.01), consistent.
Also: per-site robust z flags, imaging-confound checks.
"""
import json, sys, itertools, numpy as np, pandas as pd
from scipy import stats

KPI_INFO = {
    "porosity": ("Porosity (area fraction)", "calendering / coating density"),
    "si_frac": ("Si area fraction", "formulation (Si:graphite ratio)"),
    "interface_um_per_um2": ("Phase-boundary density (µm/µm²)", "particle size / breakage / mixing"),
    "porosity__ls_um": ("Pore patch size, GP lengthscale (µm)", "pore clustering / uniformity"),
    "si_frac__ls_um": ("Si patch size, GP lengthscale (µm)", "Si dispersion / agglomeration"),
    "si_d50_um": ("Si particle median diameter (µm)", "Si milling / supplier PSD"),
    "si_d90_um": ("Si particle d90 (µm)", "coarse Si / agglomerates"),
    "si_count_per_1000um2": ("Si particles per 1000 µm²", "Si loading / dispersion"),
    "si_solidity": ("Si particle solidity (0-1)", "Si particle shape / fracture"),
    "si_d10_um": ("Si particle d10 (µm)", "fine Si fraction / milling"),
    "si_psd_span": ("Si size span (d90-d10)/d50", "breadth of the Si size distribution"),
    "si_aspect_ratio": ("Si aspect ratio (median)", "elongated / flake-like / broken particles"),
    "si_circularity": ("Si circularity (median, 0-1)", "particle roundness vs angularity"),
    "si_nn_um": ("Si nearest-neighbour distance (µm)", "Si spacing / dispersion"),
    "si_clark_evans": ("Si Clark-Evans ratio (<1 clustered)", "Si agglomeration vs random"),
    "si_cluster_size": ("Si cluster size (particles, gap <1 µm)", "Si agglomeration"),
    "si_cracked_frac": ("Cracked/irregular Si fraction", "particle fracture in processing"),
    "si_anomalous_per_1000um2": ("Anomalous Si per 1000 µm² (>8 µm or AR>3)", "oversized / elongated particles"),
    "inlens_texture": ("Inlens fine texture (relative)", "surface roughness / binder & carbon black"),
    "inlens_edge_binder": ("Inlens edge strength in graphite/binder (relative)", "binder / carbon-black texture"),
}
KPIS = list(KPI_INFO)
RNG = np.random.default_rng(0)
NPERM = 20000


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def perm_test(a, b, stat):
    x = np.r_[a, b]; n = len(a); obs = stat(a, b); cnt = 0
    for _ in range(NPERM):
        RNG.shuffle(x); cnt += abs(stat(x[:n], x[n:])) >= abs(obs) - 1e-12
    return obs, (cnt + 1) / (NPERM + 1)


def mean_diff(a, b): return a.mean() - b.mean()
def spread_diff(a, b): return np.mean(np.abs(a - np.median(a))) - np.mean(np.abs(b - np.median(b)))


def pooled_sd(d, k):
    g = [x for x in (v[k].dropna().values for _, v in d.groupby("batch")) if len(x) > 1]
    return np.sqrt(sum(((x - x.mean()) ** 2).sum() for x in g) / sum(len(x) - 1 for x in g))


def compare(a, b, sd):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]   # sites without this KPI (e.g. <3 Si particles) are left out
    if len(a) < 2 or len(b) < 2:                # too few sites with this KPI: report it, never flag it
        nan = float("nan")
        return dict(mean_a=a.mean() if len(a) else nan, mean_b=b.mean() if len(b) else nan, diff=nan, ci95=[nan, nan],
                    d=nan, p=1.0, spread_diff=nan, p_spread=1.0, insufficient=True)
    diff, p = perm_test(a, b, mean_diff)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    dof = se ** 4 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    t = stats.t.ppf(0.975, dof)
    sp, p_sp = perm_test(a, b, spread_diff)
    return dict(mean_a=a.mean(), mean_b=b.mean(), diff=diff, ci95=[diff - t * se, diff + t * se],
                d=diff / sd, p=p, spread_diff=sp, p_spread=p_sp)


def verdict_of(r):
    if r["p_holm"] < 0.01 and abs(r["d"]) >= 0.8: return "outlier"
    if r["p_holm"] < 0.05 or (r["p_spread"] < 0.01 and r["spread_diff"] > 0): return "investigate"
    return "consistent"


def site_flags(d):
    rows = []
    for k in KPIS:
        for i, r in d.iterrows():
            if pd.isna(r[k]): continue
            rest = d.drop(i)[k].dropna(); med = rest.median(); mad = 1.4826 * np.median(np.abs(rest - med))
            z = (r[k] - med) / mad if mad > 0 else 0
            if abs(z) > 3.5: rows.append(dict(batch=r.batch, site=r.site, kpi=k, value=r[k], robust_z=z))
    return rows


BATCH_COLOURS = ["#2f6fdf", "#e08a1e", "#c23b3b", "#3a9a5b", "#8e5bc2", "#7f7f7f", "#17becf", "#bcbd22"]
def batch_colours(batches):
    """Colour per batch in the given order; the first four match the original figures, then the list cycles."""
    return {b: BATCH_COLOURS[i % len(BATCH_COLOURS)] for i, b in enumerate(batches)}

def n_cv_splits(y, max_splits=5):
    """Stratified CV folds the smallest batch allows (callers skip CV below 2)."""
    return int(min(max_splits, pd.Series(y).value_counts().min()))

def hotspot_sites(res, defaults, tile_kpis):
    """Sites for spots.py: the fixed defaults first, then every other site flagged high on a mappable tile KPI
    or unusual (>=2 flags) in compare.json. KPI = its highest positive tile-KPI flag, else si_frac."""
    seen = {(b, s) for b, s, _ in defaults}; cand = {}
    for f in res["site_flags"]:
        key = (f["batch"], f["site"])
        if key in seen or f["kpi"] not in tile_kpis or f["robust_z"] <= 0: continue
        if key not in cand or f["robust_z"] > cand[key][1]: cand[key] = (f["kpi"], f["robust_z"])
    for b, v in res["loo"].items():
        for s in v.get("unusual_sites", []):
            if (b, s) not in seen: cand.setdefault((b, s), ("si_frac", 0.0))
    return list(defaults) + [(b, s, cand[(b, s)][0]) for b, s in sorted(cand)]


def confounds(d):
    out = []
    off = d.bse_black > 10; se = d.se_detector == "SE"
    for k in KPIS:
        for name, mask in [("BSE black-level offset (>10)", off), ("SE instead of ETD detector", se)]:
            if mask.sum() >= 3:
                p = stats.mannwhitneyu(d[k][mask], d[k][~mask]).pvalue
                out.append(dict(kpi=k, confound=name, n=int(mask.sum()), median_with=d[k][mask].median(),
                                median_without=d[k][~mask].median(), p=p))
        rho, p = stats.spearmanr(d[k], d.si_peak)
        out.append(dict(kpi=k, confound="BSE Si/graphite contrast (si_peak)", n=len(d), rho=rho, p=p))
    return out


def run(d):
    batches = sorted(d.batch.unique()); sds = {k: pooled_sd(d, k) for k in KPIS}
    res = {"loo": {}, "pairs": {}, "pooled_site_sd": sds}
    for B in batches:
        rows = []
        for k in KPIS:
            r = compare(d[d.batch == B][k].values, d[d.batch != B][k].values, sds[k]); r["kpi"] = k; rows.append(r)
        for r, ph in zip(rows, holm([r["p"] for r in rows])): r["p_holm"] = ph; r["verdict"] = verdict_of(r)
        rows.sort(key=lambda r: r["p"])
        order = {"consistent": 0, "investigate": 1, "outlier": 2}
        res["loo"][B] = dict(n_sites=int((d.batch == B).sum()), rows=rows,
                            verdict=max((r["verdict"] for r in rows), key=order.get))
    for A, B in itertools.combinations(batches, 2):
        rows = []
        for k in KPIS:
            r = compare(d[d.batch == A][k].values, d[d.batch == B][k].values, sds[k]); r["kpi"] = k; rows.append(r)
        for r, ph in zip(rows, holm([r["p"] for r in rows])): r["p_holm"] = ph; r["verdict"] = verdict_of(r)
        res["pairs"][f"{A} vs {B}"] = rows
    res["site_flags"] = site_flags(d); res["confounds"] = confounds(d)
    for B, v in res["loo"].items():  # a sub-population of odd sites also needs a human look
        n_flags = collections.Counter(f["site"] for f in res["site_flags"] if f["batch"] == B)
        v["flagged_sites"] = sorted(n_flags)
        # with ~20 KPIs single flags occur by chance; a site is "unusual" only if flagged on >=2 KPIs
        odd = sorted(s for s, n in n_flags.items() if n >= 2)
        v["unusual_sites"] = odd
        if len(odd) >= 2 and v["verdict"] == "consistent":
            v["verdict"] = "investigate"
    return res


def explain(res):
    L = []
    for B, v in res["loo"].items():
        L.append(f"\n{B} ({v['n_sites']} sites) vs all other sites -> {v['verdict'].upper()}  flagged sites: {v['flagged_sites']}")
        for r in v["rows"]:
            L.append(f"  {r['kpi']:22s} {r['mean_a']:.4g} vs {r['mean_b']:.4g}  diff {r['diff']:+.3g} "
                     f"[{r['ci95'][0]:+.3g}, {r['ci95'][1]:+.3g}]  d={r['d']:+.2f}  p={r['p']:.3f} p_holm={r['p_holm']:.3f}"
                     f"  spread p={r['p_spread']:.3f}  -> {r['verdict']}")
    L.append("\nSite flags (|robust z| > 3.5):")
    for f in res["site_flags"]: L.append(f"  {f['batch']} {f['site']} {f['kpi']} = {f['value']:.4g} (z={f['robust_z']:+.1f})")
    L.append("\nConfound checks (p<0.05 shown):")
    for c in res["confounds"]:
        if c["p"] < 0.05: L.append(f"  {c['kpi']} ~ {c['confound']}: p={c['p']:.3f} " + (f"rho={c['rho']:+.2f}" if "rho" in c else f"median {c['median_with']:.4g} vs {c['median_without']:.4g}"))
    return "\n".join(L)


if __name__ == "__main__":
    d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "site_kpis.csv")
    res = run(d)
    print(explain(res))
    json.dump(res, open("compare.json", "w"), indent=1, default=float)
