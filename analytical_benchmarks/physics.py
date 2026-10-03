"""Physics-based estimates per tile and per site, from the phase fractions and Si particle sizes.

Textbook constants, not measurements. Bright phase assumed to be crystalline Si; an SiOx scenario
is reported alongside because the bright phase is not confirmed by EDS. Binder/carbon is counted
as graphite (BSE cannot separate them), so capacity is slightly overestimated.

Per tile (then one GP per site, as for the other tile KPIs):
  spec_capacity  mAh/g of solids = (Q_si*rho_si*f_si + Q_gr*rho_gr*f_gr) / (rho_si*f_si + rho_gr*f_gr)
  vol_capacity   mAh/cm3 of coating = Q_si*rho_si*f_si + Q_gr*rho_gr*f_gr
  swelling       added volume at full lithiation / coating volume = E_si*f_si + E_gr*f_gr
  net_expansion  swelling - porosity: volume the pores cannot absorb (-> electrode thickening)
Per site:
  swell_to_pore  swelling / porosity (>1: pores cannot absorb the expansion); __err ignores the swelling-porosity covariance (approximate)
  diff_time_rel  (d90 / median d90 of all sites)^2: Li diffusion time t ~ r^2/D of the coarse particles
Writes physics.csv, physics.json, fig_physics.png.
"""
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from multiprocessing import Pool
from kpis import gp_mean
from compare import compare, holm, verdict_of, pooled_sd, batch_colours

Q_GR, RHO_GR, E_GR = 372.0, 2.26, 0.10           # graphite LiC6: mAh/g, g/cm3, volume expansion
SCEN = {"si": (3579.0, 2.33, 2.80),               # crystalline Si -> Li15Si4
        "siox": (1600.0, 2.20, 1.60)}             # SiOx (x~1), mid literature values
MIN_SOLID = 0.5                                   # tiles with less solid are skipped for mAh/g
KPI_INFO = {
    "spec_capacity": ("Specific capacity (mAh/g solids)", "energy per gram; Si:graphite ratio"),
    "vol_capacity": ("Volumetric capacity (mAh/cm³ coating)", "energy per volume; Si ratio and density"),
    "swelling": ("Swelling at full charge (fraction of volume)", "expansion load on the electrode"),
    "net_expansion": ("Expansion not absorbed by pores (fraction)", "electrode thickening / stress"),
    "swell_to_pore": ("Swelling ÷ porosity", ">1: pores cannot absorb the expansion"),
    "diff_time_rel": ("Diffusion time of d90 particles (relative)", "rate capability; t ~ r²"),
}
KPIS = list(KPI_INFO)


def tile_physics(Y, scen="si"):
    q, rho, e = SCEN[scen]; por, fsi, fgr = Y[:, 0], Y[:, 1], Y[:, 2]
    mass = rho * fsi + RHO_GR * fgr; vol = q * rho * fsi + Q_GR * RHO_GR * fgr
    spec = np.where(1 - por >= MIN_SOLID, vol / np.maximum(mass, 1e-9), np.nan)
    sw = e * fsi + E_GR * fgr
    return dict(spec_capacity=spec, vol_capacity=vol, swelling=sw, net_expansion=sw - por)


def run(r):
    z = np.load(f"tiles/{r['batch']}__{r['site']}.npz"); X, Y = z["X"], z["Y"]
    out = dict(batch=r["batch"], site=r["site"])
    for scen in SCEN:
        T = tile_physics(Y, scen); sfx = "" if scen == "si" else "__siox"
        for k, y in T.items():
            if scen != "si" and k == "net_expansion": continue
            ok = ~np.isnan(y)
            for a, v in gp_mean(X[ok], y[ok]).items():
                if scen == "si" or a in ("mean", "err"):
                    out[f"{k}{sfx}" if a == "mean" else f"{k}{sfx}__{a}"] = v
        p, ep = r["porosity"], r["porosity__err"]; s, es = out[f"swelling{sfx}"], out[f"swelling{sfx}__err"]
        out[f"swell_to_pore{sfx}"] = s / p
        out[f"swell_to_pore{sfx}__err"] = s / p * np.hypot(es / s, ep / p)   # delta method treating swelling and porosity errors as independent; both come from the same tiles (graphite = 1 - pore - Si), so the covariance is ignored and this error is approximate (can be too small or too large)
    d = z["p_d"] if "p_d" in z else np.array([])
    rng = np.random.default_rng(0)
    out["d90_boot"] = [float(np.percentile(rng.choice(d, len(d)), 90)) for _ in range(500)] if len(d) >= 3 else []
    return out


def site_flags(d):
    rows = []
    for k in KPIS:
        for i, r in d.iterrows():
            if pd.isna(r[k]): continue
            rest = d.drop(i)[k].dropna(); med = rest.median(); mad = 1.4826 * np.median(np.abs(rest - med))
            zz = (r[k] - med) / mad if mad > 0 else 0
            if abs(zz) > 3.5: rows.append(dict(batch=r.batch, site=r.site, kpi=k, value=r[k], robust_z=zz))
    return rows


if __name__ == "__main__":
    S = pd.read_csv("site_kpis.csv")
    with Pool(8) as pool: res = pool.map(run, S.to_dict("records"))
    med90 = S.si_d90_um.median()
    for o, (_, r) in zip(res, S.iterrows()):
        b = np.array(o.pop("d90_boot"))
        o["diff_time_rel"] = (r.si_d90_um / med90) ** 2
        o["diff_time_rel__ci95"] = list((np.percentile(b, [2.5, 97.5]) / med90) ** 2) if len(b) else [np.nan, np.nan]
        o["diff_time_rel__err"] = float(np.std(b / med90) * 2 * r.si_d90_um / med90) if len(b) else np.nan
    d = pd.DataFrame(res)
    d.drop(columns=["diff_time_rel__ci95"]).to_csv("physics.csv", index=False)

    B = sorted(d.batch.unique()); sds = {k: pooled_sd(d, k) for k in KPIS}; out = {"loo": {}, "constants": {
        "graphite": dict(Q_mAh_g=Q_GR, rho_g_cm3=RHO_GR, expansion=E_GR),
        **{k: dict(Q_mAh_g=q, rho_g_cm3=rho, expansion=e) for k, (q, rho, e) in SCEN.items()}}}
    for b in B:
        rows = []
        for k in KPIS:
            x = compare(d[d.batch == b][k].values, d[d.batch != b][k].values, sds[k]); x["kpi"] = k; rows.append(x)
        for x, ph in zip(rows, holm([x["p"] for x in rows])): x["p_holm"] = ph; x["verdict"] = verdict_of(x)
        out["loo"][b] = rows
    out["site_flags"] = site_flags(d)
    out["batch_means"] = d.groupby("batch")[KPIS + ["spec_capacity__siox", "swelling__siox", "swell_to_pore__siox"]].mean().to_dict("index")
    json.dump(out, open("physics.json", "w"), indent=1, default=float)

    for b in B:
        print(b, "  ".join(f"{k}={d[d.batch == b][k].mean():.3g}" for k in KPIS))
    for x in sorted((x for b in B for x in out["loo"][b]), key=lambda x: x["p_holm"])[:4]:
        print(f"  {x['kpi']}: d={x['d']:+.2f} p_holm={x['p_holm']:.3f} -> {x['verdict']}")
    for f in out["site_flags"]: print(f"  flag {f['batch']} {f['site']} {f['kpi']}={f['value']:.3g} z={f['robust_z']:+.1f}")

    # --- figure: one dot per site + tile maps for flagged vs typical sites
    col = batch_colours(B)
    fig = plt.figure(figsize=(20, 13)); gs = fig.add_gridspec(4, 6, height_ratios=[1, 1, 1, 1])
    for n, k in enumerate(KPIS):
        a = fig.add_subplot(gs[0, n])
        for i, b in enumerate(B):
            s = d[d.batch == b]; xx = i + np.random.default_rng(i).uniform(-.12, .12, len(s))
            a.errorbar(xx, s[k], yerr=1.96 * s[f"{k}__err"], fmt="o", color=col[b], alpha=.8, ms=5, capsize=2)
            a.hlines(s[k].mean(), i - .3, i + .3, color="k", lw=2)
            for xi, (_, rr) in zip(xx, s.iterrows()):
                if rr.site in ("5n1q8atc", "4ih2ggld"): a.annotate(rr.site, (xi + .05, rr[k]), fontsize=7)
        if k == "swell_to_pore": a.axhline(1, color="grey", ls="--", lw=1)
        a.set_xticks(range(len(B))); a.set_xticklabels([b.replace("Batch_", "B") for b in B])
        a.set_title(KPI_INFO[k][0], fontsize=9); a.grid(alpha=.3)
    maps = [("Batch_1", "5n1q8atc", "flagged"), ("Batch_1", "4ih2ggld", "flagged"), ("Batch_2", "epqdaau9", "typical")]
    lims = {}
    for row, (b, s, tag) in enumerate(maps, 1):
        z = np.load(f"tiles/{b}__{s}.npz"); X, Y = z["X"], z["Y"]; T = tile_physics(Y)
        rr = np.unique(X[:, 0]); cc = np.unique(X[:, 1]); ext = [0, cc.max() + 1.6, rr.max() + 1.6, 0]
        for j, (k, cmap, norm) in enumerate([("vol_capacity", "magma", matplotlib.colors.LogNorm(500, 6000)),
                                             ("net_expansion", "RdBu_r", matplotlib.colors.TwoSlopeNorm(0, -0.5, 2.0))]):
            a = fig.add_subplot(gs[row, 3 * j:3 * j + 3])
            im = a.imshow(T[k].reshape(len(rr), len(cc)), cmap=cmap, norm=norm, extent=ext, interpolation="nearest")
            a.set_title(f"{b} {s} ({tag}): {KPI_INFO[k][0]}, mean {d[d.site == s][k].iloc[0]:.3g}", fontsize=9)
            a.set_xlabel("µm"); plt.colorbar(im, ax=a, fraction=.015)
    fig.suptitle("Physics estimates. Top: one dot per site (error bar = 95%). Below: per 3.2 µm tile. "
                 "Net expansion > 0 (red) = Si/graphite swelling the local pores cannot absorb.", fontsize=11)
    fig.tight_layout(); fig.savefig("fig_physics.png", dpi=70)
