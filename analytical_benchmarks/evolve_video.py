"""Realistic evolution video of one `simulate.py --render` run: the real BSE image, deformed by the simulated
displacement field.   python3 evolve_video.py <name> [--dir sim] [--fps 24] [--jobs 8] [--preview N]

Reads <dir>/render_<name>.npz, frames_<name>.npz, traj_<name>.csv, within_<name>.csv; writes <dir>/evolve_<name>.mp4.
Real vs model: the grey texture is the real BSE crop, moved pixel by pixel by the finite-element displacement field
(true scale, no exaggeration), plus the model's irreversible thickening (crack volume) as a uniform vertical
strain. New voids (cracks), SEI and the darkening of Li-filled Si are painted by the model.
"""
import argparse, os, shutil, subprocess, tempfile
from multiprocessing import Pool
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import ndimage as ndi
from skimage import measure

G = {}
DARK = 0.55                      # fraction of the Si-graphite BSE contrast lost when Si is fully lithiated (model)
NORM = TwoSlopeNorm(vmin=-20, vcenter=0, vmax=60); CM = plt.get_cmap("RdBu_r")
ZW, PAD = 140, 100               # zoom window and source padding, sim px (0.1 µm)


def load_bse(src, nm, x0, H, W):
    import imagecodecs, tifffile  # noqa: F401
    im = tifffile.imread(src); im = (im[..., 0] if im.ndim == 3 else im).astype(np.float32)
    if im.max() > 255: im = im / im.max() * 255
    k = 100.0 / nm
    if abs(k - round(k)) > 1e-6:
        from skimage.transform import rescale; im = rescale(im, 4 / k, preserve_range=True).astype(np.float32); k = 4
    k = int(round(k)); return im[:H * k, x0 * k:(x0 + W) * k], k


def fm(v, f): return "–" if pd.isna(v) else f.format(v)


def up(a, q): return np.repeat(np.repeat(a, q, 0), q, 1)


def soft(m, q): return np.clip((ndi.gaussian_filter(up(m.astype(np.float32), q), 0.4 * q) - 0.25) / 0.5, 0, 1)


def compose(img, q, sl, lab, c):
    """Reference-frame texture: real BSE + model-painted voids, SEI and Li darkening (region sl of the sim grid)."""
    l0 = G["lab0"][sl]; lab = lab[sl]; c = c[sl].astype(np.float32); L = G["lev"]
    rng = np.random.default_rng(1); nz = ndi.gaussian_filter(rng.normal(0, 1, img.shape), 0.7); nz /= nz.std()
    T = img.copy()
    a = soft((lab == 0) & (l0 != 0), q); T += a * (L["pore"] + L["pore_sd"] * nz - T)
    a = soft(lab == 3, q); T += a * (L["sei"] + 0.6 * L["pore_sd"] * nz - T)
    a = soft((lab == 2) | (lab == 4), q); cs = ndi.gaussian_filter(up(c, q), 0.5 * q)
    return T - a * np.clip(cs, 0, 1) * DARK * np.maximum(T - L["gr"], 0)


def warp(T, q, r0, c0, u, vol, orow, ocol, oh, ow):
    """Pull the reference texture T (origin r0, c0 in sim px, q px per sim px) into a deformed-frame window."""
    ny, nx = G["Hm"], G["Wm"]; mb = G["mb"]
    dc = u[0::2].astype(np.float32).reshape(ny + 1, nx + 1) / G["px"]
    dr = -u[1::2].astype(np.float32).reshape(ny + 1, nx + 1) / G["px"]
    Yr, Yc = np.meshgrid(orow + (np.arange(oh) + 0.5) / q, ocol + (np.arange(ow) + 0.5) / q, indexing="ij")
    Xr, Xc = Yr.copy(), Yc.copy()
    for _ in range(6):
        p = [Xr / mb, Xc / mb]
        Xr = Yr - ndi.map_coordinates(dr, p, order=1, mode="nearest"); Xc = Yc - ndi.map_coordinates(dc, p, order=1, mode="nearest")
    out = ndi.map_coordinates(T, [(Xr - r0) * q - 0.5, (Xc - c0) * q - 0.5], order=1, mode="constant", cval=np.nan)
    st = ndi.map_coordinates(vol.astype(np.float32).reshape(ny, nx), [Xr / mb - 0.5, Xc / mb - 0.5], order=1, mode="nearest") * 100
    out[Xr < 0] = np.nan; st[np.isnan(out)] = np.nan
    return out, st


def gray_rgb(g):
    g = np.nan_to_num(g, nan=G["lev"]["bg"]); return np.repeat(np.clip(g / 255, 0, 1)[..., None], 3, 2)


def strain_rgb(g, st):
    base = gray_rgb(g); col = CM(NORM(np.nan_to_num(st)))[..., :3]
    a = np.where(np.isnan(st), 0, 0.1 + 0.6 * np.clip(np.abs(np.nan_to_num(st)) / 15, 0, 1))[..., None]
    return base * (1 - a) + col * a


def frame_state(f):
    return f["lab"], f["c"], f["u"], f["vol"]


def render(i):
    f = G["frames"][i]; path = f"{G['tmp']}/f{i:04d}.png"
    if f["kind"] == "title": return title(path)
    lab, c, u, vol = frame_state(f); H, W, M = G["H"], G["W"], G["margin"]
    T = compose(G["img2"], 2, np.s_[:, :], lab, c)
    g, st = warp(T, 2, 0, 0, u, vol, -M, 0, (H + M) * 2, W * 2)
    # zoom: follow the largest Si particle
    zy, zx = G["zc"]; ny, nx = G["Hm"], G["Wm"]; mb = G["mb"]
    p = [[zy / mb], [zx / mb]]
    dzy = -ndi.map_coordinates(u[1::2].astype(np.float32).reshape(ny + 1, nx + 1), p, order=1)[0] / G["px"]
    dzx = ndi.map_coordinates(u[0::2].astype(np.float32).reshape(ny + 1, nx + 1), p, order=1)[0] / G["px"]
    def zoom(lab, c, u, vol, cy, cx):
        r0 = int(np.clip(cy - ZW / 2, -M, H - ZW)); q0 = int(np.clip(cx - ZW / 2, 0, W - ZW))
        a0, a1 = max(0, r0 - PAD), min(H, r0 + ZW + PAD); b0, b1 = max(0, q0 - PAD), min(W, q0 + ZW + PAD)
        k = G["k"]; Tz = compose(G["img"][a0 * k:a1 * k, b0 * k:b1 * k], k, np.s_[a0:a1, b0:b1], lab, c)
        return warp(Tz, k, a0, b0, u, vol, r0, q0, ZW * k, ZW * k)[0], (r0, q0)
    z0, (r00, q00) = zoom(G["lab0"], np.zeros_like(c), np.zeros_like(u), np.zeros_like(vol), zy, zx)
    z1, (r01, q01) = zoom(lab, c, u, vol, zy + dzy, zx + dzx)
    draw(path, f, g, st, z0, z1, (r00, q00), (r01, q01))


def title(path):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#111")
    fig.text(0.5, 0.62, f"{G['name']}: how this electrode evolves over {G['N']} cycles", color="w", ha="center", fontsize=34, weight="bold")
    fig.text(0.5, 0.52, "Real BSE image, deformed by the simulated expansion of Si and graphite (true scale)", color="#ddd", ha="center", fontsize=20)
    fig.text(0.5, 0.44, f"{G['crate']:g}C constant-current charge/discharge  ·  {G['W'] * G['px']:.0f} × {G['H'] * G['px']:.0f} µm crop  ·  1 random run",
             color="#bbb", ha="center", fontsize=17)
    fig.text(0.5, 0.30, "Illustrative model with literature parameters, not calibrated to cycling data.\n"
             "Grey texture = real BSE pixels moved by the finite-element displacement field.\n"
             "Cracks (new voids), SEI and darkening of Li-filled Si are painted by the model.",
             color="#f0b860", ha="center", fontsize=15, linespacing=1.6)
    fig.savefig(path, facecolor=fig.get_facecolor()); plt.close(fig)


def draw(path, f, g, st, z0, z1, o0, o1):
    H, W, M, px = G["H"], G["W"], G["margin"], G["px"]; n = f["cycle"]; tr = G["tr"]
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#111")
    fig.text(0.012, 0.965, f"{G['name']}  ·  simulated evolution of the real BSE image", color="w", fontsize=20, weight="bold", va="top")
    fig.text(0.012, 0.925, f["status"], color="#7fd3ff", fontsize=19, va="top", weight="bold")
    fig.text(0.988, 0.965, "illustrative model, uncalibrated  ·  texture real, deformation simulated (true scale)",
             color="#f0b860", fontsize=12, ha="right", va="top")
    ext = [0, W * px, H * px, -M * px]
    for k, (img, ttl) in enumerate([(gray_rgb(g), "Real BSE texture, deformed by the simulated expansion"),
                                     (strain_rgb(g, st), "Same, coloured by local area change (%)")]):
        ax = fig.add_axes([0.012 + k * 0.268, 0.33, 0.262, 0.53]); ax.imshow(img, extent=ext, interpolation="antialiased")
        ax.axhline(0, color="#7fd3ff", lw=1, ls=(0, (4, 4)), alpha=0.8)
        ax.text(0.5, -M * px + 0.4, "original top surface ↓" if M else "", color="#7fd3ff", fontsize=9, va="top")
        ax.set_title(ttl, color="w", fontsize=13); ax.tick_params(colors="#aaa", labelsize=9)
        ax.set_xlabel("µm", color="#aaa", fontsize=10); ax.set_facecolor("#111")
        for s in ax.spines.values(): s.set_color("#444")
        for zr, zc, col in ((o1[0], o1[1], "#ffd24a"),):
            ax.add_patch(plt.Rectangle((zc * px, zr * px), ZW * px, ZW * px, fill=False, ec=col, lw=1.2))
    cax = fig.add_axes([0.548, 0.36, 0.006, 0.46]); cb = plt.colorbar(plt.cm.ScalarMappable(NORM, CM), cax=cax)
    cb.ax.tick_params(colors="#ccc", labelsize=9)
    fig.text(0.551, 0.835, "area change (%)\nred = swollen\nblue = squeezed", color="#ccc", fontsize=9, ha="center", va="bottom")
    for k, (img, o, ttl) in enumerate([(z0, o0, "Zoom, start (cycle 0, no Li)"), (z1, o1, "Zoom, now (same particle)")]):
        ax = fig.add_axes([0.6 + k * 0.198, 0.43, 0.19, 0.43])
        ax.imshow(gray_rgb(img), extent=[o[1] * px, (o[1] + ZW) * px, (o[0] + ZW) * px, o[0] * px], interpolation="antialiased")
        ax.set_title(ttl, color="#ffd24a" if k else "w", fontsize=13); ax.tick_params(colors="#aaa", labelsize=9)
        for s in ax.spines.values(): s.set_color("#ffd24a" if k else "#444")
    # live numbers
    row = tr.loc[n]; act = (f["lab"] == 2)
    vals = [("Thickness change now", f"{f['thick']:+.1f} %"), ("Li in active Si now", f"{f['c'][act].astype(float).mean():.2f}" if act.any() else "–"),
            ("Capacity retention", fm(row.retention_pct, "{:.1f} %")), ("Cracks (cumulative)", fm(row.n_cracks, "{:.0f}")),
            ("SEI area fraction", fm(row.sei_frac, "{:.3f}")), ("Si d90", fm(row.si_d90_um, "{:.1f} µm"))]
    for j, (k_, v) in enumerate(vals):
        x = 0.605 + (j % 3) * 0.13; y = 0.385 - (j // 3) * 0.045
        fig.text(x, y, k_, color="#aaa", fontsize=11); fig.text(x, y - 0.022, v, color="w", fontsize=15, weight="bold")
    # charts
    wi = G["wi"]; rc = G["rcycles"]; cmap = plt.get_cmap("viridis")
    def style(ax, ttl, xl):
        ax.set_facecolor("#1b1b1b"); ax.tick_params(colors="#aaa", labelsize=9); ax.set_title(ttl, color="w", fontsize=11)
        ax.set_xlabel(xl, color="#aaa", fontsize=9); ax.grid(color="#333", lw=0.5)
        for s in ax.spines.values(): s.set_color("#444")
    for j, (col, ttl) in enumerate([("thickness_pct", "Thickness change in a cycle (%)"), ("si_li", "Li fraction in Si in a cycle")]):
        ax = fig.add_axes([0.04 + j * 0.195, 0.06, 0.165, 0.2]); style(ax, ttl, "time in cycle (min)")
        for cc in rc:
            if cc > n or (cc == n and f["kind"] == "tl"): continue
            w = wi[wi.cycle == cc]
            if cc == n: w = w[w.t_min <= f["t_min"] + 1e-6]
            ax.plot(w.t_min, w[col], color=cmap(cc / G["N"]), lw=2 if cc == n else 1.2, label=f"cycle {cc}")
            if cc == n and len(w): ax.plot(w.t_min.iloc[-1], w[col].iloc[-1], "o", color="w", ms=5)
        ax.set_xlim(0, 2 * G["T"]); ax.set_ylim(*G["lim"][col]); ax.axvline(G["T"], color="#666", lw=0.8, ls=":")
        if j == 0: ax.legend(fontsize=8, facecolor="#1b1b1b", labelcolor="w", edgecolor="#444", loc="upper right")
    t = tr.loc[:n]
    for j, (cols, ttl) in enumerate([(["retention_pct"], "Capacity retention (%)"),
                                     (["thickness_charged_pct", "thickness_discharged_pct"], "Thickness: charged / discharged (%)\nincl. permanent thickening"),
                                     (["n_cracks"], "Cracks (cumulative)")]):
        ax = fig.add_axes([0.43 + j * 0.195, 0.06, 0.165, 0.2]); style(ax, ttl, "cycle")
        for cl, colr in zip(cols, ("#7fd3ff", "#ff9966")):
            ax.plot(t.index, t[cl], color=colr, lw=1.6); ax.plot(n, t[cl].iloc[-1], "o", color="w", ms=4)
        ax.set_xlim(0, G["N"]); ax.set_ylim(*G["lim"][cols[0]])
    fig.savefig(path, facecolor=fig.get_facecolor()); plt.close(fig)


def build(a):
    R = dict(np.load(f"{a.dir}/render_{a.name}.npz")); Z = np.load(f"{a.dir}/frames_{a.name}.npz"); labs = Z["labs"]
    tr = pd.read_csv(f"{a.dir}/traj_{a.name}.csv"); tr = tr[tr.seed == tr.seed.min()].set_index("cycle")
    wi = pd.read_csv(f"{a.dir}/within_{a.name}.csv")
    H, W, x0, mb, px = int(R["H"]), int(R["W"]), int(R["x0"]), int(R["mb"]), float(R["px"])
    img, k = load_bse(str(R["src"]), float(R["nm_per_px"]), x0, H, W)
    assert img.shape == (H * k, W * k), (img.shape, H, W, k)
    img2 = img.reshape(H * 2, k // 2, W * 2, k // 2).mean((1, 3)) if k % 2 == 0 else ndi.zoom(img, 2 / k, order=1)
    lab0 = labs[0]; l0u = up(lab0, k); lev = dict(pore=np.median(img[l0u == 0]), gr=np.median(img[l0u == 1]), pore_sd=img[l0u == 0].std())
    lev["sei"] = lev["pore"] + 0.55 * (lev["gr"] - lev["pore"]); lev["bg"] = 8.0
    L = measure.label(lab0 == 2); rp = max(measure.regionprops(L), key=lambda r: r.area); zc = rp.centroid
    T = 3600 / float(R["crate"]) / 60; Hm, Wm = H // mb, W // mb
    top = lambda u: float(u[1::2].astype(np.float32).reshape(Hm + 1, Wm + 1)[0].mean())
    uy_max = max(float(R["r_u"][:, 1::2].astype(np.float32).max()), float(R["tl_u"][:, 1::2].astype(np.float32).max())) \
        + float(tr.thickness_irrev_pct.max()) / 100 * H * px
    margin = int(np.ceil(uy_max / px)) + 15
    st = lambda **kw: dict(kind="frame", **kw)
    fr = [dict(kind="title", hold=3.0),
          st(cycle=0, lab=lab0, c=np.zeros((H, W), np.float16), u=np.zeros_like(R["r_u"][0]), vol=np.zeros_like(R["r_vol"][0]),
             t_min=0, status="Cycle 0  ·  as imaged (no Li)", hold=1.5)]
    rcyc = R["r_cycle"]; tlc = list(R["tl_cycle"]); rc = sorted(set(rcyc.tolist())); last = rc[-1]
    def detailed(n):
        for i in np.flatnonzero(rcyc == n):
            ch = bool(R["r_charge"][i]); tm = float(R["r_t"][i]) / 60 + (0 if ch else T)
            fr.append(st(cycle=n, lab=labs[n - 1], c=R["r_c"][i], u=R["r_u"][i], vol=R["r_vol"][i], t_min=tm,
                         status=f"Cycle {n}  ·  {'charging' if ch else 'discharging'}  ·  {tm:.0f} min of {2 * T:.0f}",
                         hold=0.12 if n in (1, last) else 0.125))
    prev = 0
    for n in rc:
        tlrange = [m for m in range(prev + 1, n) if m not in rc]
        for m in tlrange:
            i = tlc.index(m)
            fr.append(st(cycle=m, lab=labs[m], c=R["tl_c"][i], u=R["tl_u"][i], vol=R["tl_vol"][i], t_min=2 * T,
                         status=f"Fast-forward, cycles {tlrange[0]}–{tlrange[-1]}  ·  cycle {m}, after discharge", hold=0.08))
        detailed(n); prev = n
    i = tlc.index(last)
    fr.append(st(cycle=last, lab=labs[last], c=R["tl_c"][i], u=R["tl_u"][i], vol=R["tl_vol"][i], t_min=2 * T,
                 status=f"After {last} cycles  ·  discharged: permanent changes", hold=3.0))
    # irreversible thickening (crack volume, the model's thickness_irrev_pct) as a uniform vertical strain
    irr = tr.thickness_irrev_pct.fillna(0) / 100; yrow = 1 - np.arange(Hm + 1)[:, None] / Hm
    for f in fr:
        if f["kind"] != "frame": continue
        e = float(irr.loc[f["cycle"] - 1 if f["status"].startswith("Cycle ") and f["cycle"] > 0 else f["cycle"]])
        u = f["u"].astype(np.float32).copy(); u[1::2] += (e * H * px * np.repeat(yrow, Wm + 1, 1)).ravel(); f["u"] = u
        f["thick"] = 100 * top(u) / (H * px)
    tr = tr.copy(); wi = wi.copy()
    for c in ("thickness_charged_pct", "thickness_discharged_pct"): tr[c] = tr[c] + 100 * irr
    wi["thickness_pct"] = wi.thickness_pct + 100 * irr.reindex(wi.cycle - 1).fillna(0).values
    for f in fr:
        if f.get("status", "").startswith("Fast"): f["kind"] = "tl"
    def lim(c, lo=None):
        v = pd.concat([wi[c]] if c in wi else [tr[c]]); a_, b_ = float(v.min()), float(v.max()); d = (b_ - a_) * 0.08 + 1e-9
        return (a_ - d if lo is None else lo, b_ + d)
    G.update(frames=fr, H=H, W=W, mb=mb, px=px, Hm=Hm, Wm=Wm, k=k, img=img, img2=img2, lab0=lab0, lev=lev, zc=zc, tr=tr, wi=wi,
             margin=margin, name=a.name, N=int(tr.index.max()), crate=float(R["crate"]), T=T, rcycles=rc,
             lim=dict(thickness_pct=lim("thickness_pct", 0), si_li=(0, 1), retention_pct=lim("retention_pct"),
                      thickness_charged_pct=(0, max(tr.thickness_charged_pct.max(), tr.thickness_discharged_pct.max()) * 1.1),
                      n_cracks=lim("n_cracks", 0)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--dir", default="sim")
    ap.add_argument("--fps", type=int, default=24); ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--preview", type=int, nargs="*", help="only render these frame indices to <dir>/evolve_preview_<i>.png")
    a = ap.parse_args(); build(a); fr = G["frames"]
    G["tmp"] = tempfile.mkdtemp(); idx = a.preview if a.preview is not None else range(len(fr))
    with Pool(a.jobs) as p: p.map(render, idx, chunksize=1)
    if a.preview is not None:
        for i in idx: shutil.copy(f"{G['tmp']}/f{i:04d}.png", f"{a.dir}/evolve_preview_{i}.png")
        print(len(fr), "frames;", [(i, fr[i].get("status")) for i in idx]); return
    lst = f"{G['tmp']}/list.txt"
    with open(lst, "w") as fh:
        for i, f in enumerate(fr): fh.write(f"file '{G['tmp']}/f{i:04d}.png'\nduration {f['hold']}\n")
        fh.write(f"file '{G['tmp']}/f{len(fr) - 1:04d}.png'\n")
    out = f"{a.dir}/evolve_{a.name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-r", str(a.fps), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", out], check=True)
    shutil.rmtree(G["tmp"]); print(out, len(fr), "frames")


if __name__ == "__main__":
    main()
