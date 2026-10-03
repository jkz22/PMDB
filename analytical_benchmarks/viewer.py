"""Builds the interactive trajectory viewer from simulate.py outputs (sim/frames_*.npz, traj_*.csv, within_*.csv).

Writes sim/viewer_body.html. If sim/trajectory_viewer.html exists (compiled once with the artifact kit),
the content between the VIEWER markers is replaced in place, so the file stays one offline HTML page.
"""
import base64, json, os, re, numpy as np, pandas as pd

OUT = "sim"
KPI = [("retention_pct", "Capacity retention", "%"), ("thickness_charged_pct", "Swelling when charged", "%"),
       ("n_cracks", "Cracks (cumulative)", ""), ("sei_frac", "SEI area fraction", ""),
       ("si_active_frac", "Active Si area fraction", ""), ("si_d90_um", "Si d90", "µm"),
       ("si_d50_um", "Si d50", "µm"), ("si_fragments_per_1000um2", "Si fragments <1 µm per 1000 µm²", ""),
       ("thickness_irrev_pct", "Irreversible thickening", "%"), ("porosity", "Porosity", ""),
       ("si_utilisation", "Si used per cycle (0-1)", ""), ("sigma1_p95_si", "Tensile stress in Si, p95", "GPa"),
       ("capacity_mAh_cm3", "Capacity", "mAh/cm³"), ("li_lost_sei_mAh_cm3", "Li lost to SEI", "mAh/cm³"),
       ("si_solidity", "Si solidity", ""), ("si_count_per_1000um2", "Si particles ≥1 µm per 1000 µm²", "")]


def rle(a):
    f = a.ravel(); st = np.r_[0, np.flatnonzero(np.diff(f)) + 1]
    return dict(v=f[st].tolist(), n=np.diff(np.r_[st, f.size]).tolist())


def b64(a): return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def blocks(a, mb):
    H, W = a.shape; return a[:H // mb * mb, :W // mb * mb].reshape(H // mb, mb, W // mb, mb).mean((1, 3))


def site_data(name):
    z = np.load(f"{OUT}/frames_{name}.npz"); labs = z["labs"]; mb = int(z["mb"])
    tr = pd.read_csv(f"{OUT}/traj_{name}.csv"); wi = pd.read_csv(f"{OUT}/within_{name}.csv")
    g = tr.groupby("cycle"); traj = {}
    for k, _, _ in KPI:
        traj[k] = [[None if not np.isfinite(x) else round(float(x), 5) for x in s] for s in (g[k].mean(), g[k].min(), g[k].max())]
    diffs = []
    for k in range(1, len(labs)):
        d = np.flatnonzero(labs[k] != labs[k - 1]); diffs.append([b64(d.astype(np.uint32)), b64(labs[k].ravel()[d].astype(np.uint8))])
    maps = {}
    for cyc in sorted(set(z["m_cycle"].tolist())):
        sel = np.flatnonzero(z["m_cycle"] == cyc); lab = labs[cyc - 1]
        si = blocks((lab == 2).astype(float), mb); mask = si > 0.5
        li = [np.clip(blocks(z["li"][i].astype(float), mb)[mask] / si[mask], 0, 1) for i in sel]
        s1 = [z["s1"][i].astype(float).reshape(mask.shape)[mask] for i in sel]
        w = wi[wi.cycle == cyc]
        maps[str(cyc)] = dict(mask=rle(mask.astype(np.uint8)), bg=rle(np.where(lab[::mb, ::mb] == 0, 0, 1).astype(np.uint8)),
                              li=[b64((x * 255).round().astype(np.uint8)) for x in li], s1=s1,
                              t=w.t_min.round(1).tolist(), charge=(w.phase == "charge").tolist(),
                              thick=w.thickness_pct.round(3).tolist(), si_li=w.si_li.round(4).tolist(),
                              sig=w.sigma1_p95_si.round(3).tolist(), pore=w.pore_closure_pct.round(3).tolist())
    return dict(H=labs.shape[1], W=labs.shape[2], mb=mb, px=float(z["px"]), f0=rle(labs[0]), diffs=diffs,
                ncyc=len(labs) - 1, traj=traj, maps=maps)


def main():
    meta = json.load(open(f"{OUT}/sim.json")); names = list(meta["summary"])
    sites = {n: site_data(n) for n in names}
    allsig = np.concatenate([np.concatenate(m["s1"]) for s in sites.values() for m in s["maps"].values()])
    vmax = float(np.percentile(allsig[allsig > 0], 99))
    for s in sites.values():
        for m in s["maps"].values():
            m["s1"] = [b64((np.clip(x, 0, vmax) / vmax * 255).round().astype(np.uint8)) for x in m["s1"]]
    S = meta["summary"]
    flag = [n for n in names if n in ("5n1q8atc", "4ih2ggld")]; typ = [n for n in names if n not in flag]
    data = dict(sites=sites, names=names, kpi=KPI, smax=round(vmax, 2), A=flag[0] if flag else names[0],
                B=typ[0] if typ else names[-1], flagged=flag, cycles=meta["cycles"], crate=meta["crate"], seeds=meta["seeds"])

    def m(n, k): return S[n][k]["end"]
    lead = ""
    if flag and typ:
        lead = (f"After {meta['cycles']} simulated cycles, the flagged Batch_1 sites keep {min(m(n, 'retention_pct') for n in flag):.1f}–"
                f"{max(m(n, 'retention_pct') for n in flag):.1f}% of their capacity vs {min(m(n, 'retention_pct') for n in typ):.1f}–"
                f"{max(m(n, 'retention_pct') for n in typ):.1f}% for the typical sites. They crack {min(m(n, 'n_cracks') for n in flag):.0f}–"
                f"{max(m(n, 'n_cracks') for n in flag):.0f} times vs {min(m(n, 'n_cracks') for n in typ):.0f}–{max(m(n, 'n_cracks') for n in typ):.0f}, "
                "because they contain more and larger Si particles.")
    opts = "".join(f'<option value="{k}">{lab}{" (" + u + ")" if u else ""}</option>' for k, lab, u in KPI)
    sopts = "".join(f'<option value="{n}">{n}{" (flagged)" if n in flag else ""}</option>' for n in names)
    body = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer_template.html")).read()
    body = (body.replace("__LEAD__", lead).replace("__KPI_OPTIONS__", opts).replace("__SITE_OPTIONS__", sopts)
            .replace("__CYCLES__", str(meta["cycles"])).replace("__CRATE__", f"{meta['crate']:g}")
            .replace("__SEEDS__", str(meta["seeds"])).replace("__WIDTH__", f"{meta['width_um']:g}")
            .replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")))
    body = "<!--VIEWER_START-->" + body + "<!--VIEWER_END-->"
    open(f"{OUT}/viewer_body.html", "w").write(body)
    fp = f"{OUT}/trajectory_viewer.html"
    if os.path.exists(fp):
        h = open(fp).read()
        h = re.sub(r"<!--VIEWER_START-->.*<!--VIEWER_END-->", lambda _: body, h, flags=re.S)
        open(fp, "w").write(h); print("updated", fp)
    print("body", len(body) / 1e6, "MB")


if __name__ == "__main__":
    main()
