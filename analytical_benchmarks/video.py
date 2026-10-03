"""MP4 video of one simulate.py run: python3 video.py <name> [--dir sim] [--fps 8].

Reads <dir>/frames_<name>.npz, traj_<name>.csv and within_<name>.csv (seed 0) and writes <dir>/video_<name>.mp4:
title card -> Li and stress maps through the first charge/discharge -> microstructure and KPIs cycle by cycle
-> Li and stress maps through the last charge/discharge. Needs ffmpeg on PATH.
"""
import argparse, os, shutil, subprocess, tempfile
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from simulate import CMAP, PX, MB

CH = np.array([[250, 225, 40], [40, 190, 90], [220, 30, 30]], np.uint8)
NAMES = ["pore / crack", "graphite + binder", "active Si", "SEI", "inactive Si"]
KP = [("retention_pct", "Capacity retention (%)"), ("thickness_charged_pct", "Swelling when charged (%)"),
      ("n_cracks", "Cracks (cumulative)"), ("sei_frac", "SEI area fraction")]
WK = [("si_li", "Li fraction in Si"), ("thickness_pct", "Thickness change (%)"),
      ("sigma1_p95_si", "Tensile stress in Si, p95 (GPa)"), ("pore_closure_pct", "Pore volume closed (%)")]


def change_rgb(l0, l):
    o = (CMAP[l0].astype(float) * 0.35 + 150).astype(np.uint8)
    for k, v in enumerate((0, 3, 4)):
        o[(l != l0) & (l == v)] = CH[k]
    return o


def bg_rgb(lab):
    return np.where((lab == 0)[..., None], 245, 175).astype(np.uint8) * np.ones(3, np.uint8)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--dir", default="sim")
    ap.add_argument("--fps", type=int, default=8); a = ap.parse_args()
    z = np.load(f"{a.dir}/frames_{a.name}.npz"); labs = z["labs"]
    tr = pd.read_csv(f"{a.dir}/traj_{a.name}.csv"); tr = tr[tr.seed == tr.seed.min()].set_index("cycle")
    wi = pd.read_csv(f"{a.dir}/within_{a.name}.csv")
    N, (H, W) = len(labs) - 1, labs.shape[1:]; ext = [0, W * PX, H * PX, 0]; Hm, Wm = H // MB, W // MB
    s1 = z["s1"].astype(float); cyc = z["m_cycle"]
    sim_e = [np.kron((labs[c - 1][:Hm * MB, :Wm * MB] != 0).reshape(Hm, MB, Wm, MB).mean((1, 3)) > 0.5, np.ones((MB, MB), bool)) for c in cyc]
    vmax = float(np.percentile(np.concatenate([np.kron(s.reshape(Hm, Wm), np.ones((MB, MB)))[m] for s, m in zip(s1, sim_e)]), 99))

    tmp = tempfile.mkdtemp(); frames = []
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)

    def save(hold=1):
        p = f"{tmp}/f{len(frames):05d}.png"; fig.savefig(p, dpi=100); frames.extend([p] * hold)

    def header(sub):
        fig.text(0.02, 0.965, f"{a.name}: simulated cycling (illustrative model, uncalibrated literature parameters)", fontsize=17, weight="bold")
        fig.text(0.02, 0.935, sub, fontsize=12, color="0.3")

    # title card
    fig.clf(); fig.text(0.5, 0.62, f"Simulated cycling of {a.name}", ha="center", fontsize=34, weight="bold")
    fig.text(0.5, 0.52, f"{W * PX:.0f} × {H * PX:.0f} µm crop of the BSE segmentation · {N} cycles at 1C · 1 random run", ha="center", fontsize=18)
    fig.text(0.5, 0.42, "Li diffusion in Si → Si (+280% vol.) and graphite (+10%) swelling → 2-D finite-element stress\n"
             "→ stress-driven cracking → SEI growth → isolated Si fragments → KPIs recomputed every cycle",
             ha="center", fontsize=14, color="0.25", linespacing=1.6)
    fig.text(0.5, 0.28, "Illustrative model with uncalibrated parameters: use it to compare microstructures, not as a lifetime prediction.",
             ha="center", fontsize=13, color="0.45", style="italic")
    save(hold=3 * a.fps)

    def within(c):
        idx = np.flatnonzero(cyc == c); w = wi[wi.cycle == c].reset_index(drop=True); lab = labs[c - 1]
        for j, i in enumerate(idx):
            fig.clf(); ph = "charging" if z["m_charge"][i] else "discharging"
            header(f"Inside cycle {c}: {ph}, t = {w.t_min[j]:.0f} min. " + (
                "Li enters each Si particle from its surface; the swelling Si pushes on the graphite/binder around it (pores in white)."
                if z["m_charge"][i] else "Li leaves each Si particle through its surface; the shrinking outer shell is pulled into tension (pores in white)."))
            gs = fig.add_gridspec(2, 4, left=0.04, right=0.955, top=0.9, bottom=0.07, hspace=0.32, wspace=0.28, height_ratios=[1.5, 1])
            li = np.where(lab == 2, z["li"][i].astype(float), np.nan)
            st = np.where(sim_e[i], np.kron(s1[i].reshape(Hm, Wm), np.ones((MB, MB))), np.nan)
            for k, (m, cm, vm, t, cl) in enumerate([(li, "viridis", 1, "Li fraction in Si (0 empty, 1 full)", "Li fraction"),
                                                   (st, "inferno", vmax, "Max. tensile stress in the solid (GPa, elastic, relative)", "GPa")]):
                ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2]); ax.imshow(bg_rgb(lab), extent=ext)
                h = ax.imshow(m, extent=ext, cmap=cm, vmin=0, vmax=vm, interpolation="nearest"); ax.set_title(t, fontsize=13)
                ax.set_xlabel("µm"); fig.colorbar(h, ax=ax, fraction=0.035, label=cl)
            for k, (col, t) in enumerate(WK):
                ax = fig.add_subplot(gs[1, k]); x = np.r_[0, w.t_min]; y = np.r_[0 if col in ("si_li", "thickness_pct") else np.nan, w[col]]
                ax.plot(x, y, color="0.75", lw=1.5); ax.plot(x[:j + 2], y[:j + 2], "o-", color="C3", lw=2.2)
                ax.axvspan(0, w.t_min[w.phase == "charge"].max(), color="0.93", zorder=0)
                ax.set_title(f"{t}: {w[col][j]:.2f}", fontsize=12); ax.set_xlabel("time (min) · grey = charge"); ax.grid(alpha=0.3)
            save(hold=max(1, a.fps * 3 // 4))

    within(int(cyc.min()))

    # cycle-by-cycle
    for n in range(N + 1):
        fig.clf(); r = tr.loc[n]
        header(f"Cycle {n} of {N}. Cracks open in the largest, most stressed Si particles; SEI grows on fresh Si surface; small cut-off pieces stop working.")
        gs = fig.add_gridspec(2, 5, left=0.04, right=0.985, top=0.9, bottom=0.07, hspace=0.3, wspace=0.3, height_ratios=[1.55, 1])
        ax = fig.add_subplot(gs[0, :2]); ax.imshow(CMAP[labs[n]], extent=ext, interpolation="nearest"); ax.set_title(f"Microstructure at cycle {n}", fontsize=13); ax.set_xlabel("µm")
        ax.legend(handles=[Patch(color=CMAP[i] / 255, label=NAMES[i]) for i in range(5)], loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=5, fontsize=10, frameon=False)
        ax = fig.add_subplot(gs[0, 2:4]); ax.imshow(change_rgb(labs[0], labs[n]), extent=ext, interpolation="nearest"); ax.set_title("Changes since cycle 0", fontsize=13); ax.set_xlabel("µm")
        ax.legend(handles=[Patch(color=CH[0] / 255, label="new crack / pore"), Patch(color=CH[1] / 255, label="new SEI"),
                           Patch(color=CH[2] / 255, label="Si became inactive"), Patch(color=(0.75, 0.75, 0.75), label="unchanged")],
                  loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4, fontsize=10, frameon=False)
        ax = fig.add_subplot(gs[0, 4]); ax.axis("off")
        f = lambda k, fm: "–" if not np.isfinite(r[k]) else fm.format(r[k])
        rows = [("Cycle", f"{n}"), ("Capacity retention", f("retention_pct", "{:.1f} %")), ("Swelling when charged", f("thickness_charged_pct", "{:.1f} %")),
                ("Irreversible thickening", f("thickness_irrev_pct", "{:.2f} %")), ("Cracks", f("n_cracks", "{:.0f}")),
                ("SEI area fraction", f("sei_frac", "{:.3f}")), ("Active Si fraction", f"{r.si_active_frac:.3f}"),
                ("Porosity", f"{r.porosity:.3f}"), ("Si d90", f"{r.si_d90_um:.2f} µm"), ("Si fragments /1000 µm²", f"{r.si_fragments_per_1000um2:.1f}")]
        for i, (k, v) in enumerate(rows):
            ax.text(0, 0.97 - i * 0.1, k, fontsize=12.5, color="0.35", transform=ax.transAxes)
            ax.text(1, 0.97 - i * 0.1, v, fontsize=14, weight="bold", ha="right", transform=ax.transAxes)
        for k, (col, t) in enumerate(KP):
            ax = fig.add_subplot(gs[1, k])
            y = tr[col]; ax.plot(y.index, y, color="0.8", lw=1.5); ax.plot(y.index[:n + 1], y.iloc[:n + 1], color="C3", lw=2.4)
            if np.isfinite(y.iloc[n]): ax.plot(n, y.iloc[n], "o", color="C3", ms=8)
            ax.axvline(n, color="0.5", lw=0.8); ax.set_title(t, fontsize=12); ax.set_xlabel("cycle"); ax.grid(alpha=0.3); ax.set_xlim(0, N)
        ax = fig.add_subplot(gs[1, 4]); y = tr["si_d90_um"]
        ax.plot(y.index, y, color="0.8", lw=1.5); ax.plot(y.index[:n + 1], y.iloc[:n + 1], color="C0", lw=2.4); ax.plot(n, y.iloc[n], "o", color="C0", ms=8)
        ax.axvline(n, color="0.5", lw=0.8); ax.set_title("Si d90 (µm)", fontsize=12); ax.set_xlabel("cycle"); ax.grid(alpha=0.3); ax.set_xlim(0, N)
        save(hold=a.fps * 2 if n in (0, N) else 1)

    within(int(cyc.max()))
    plt.close(fig)
    lst = f"{tmp}/list.txt"
    open(lst, "w").write("".join(f"file '{p}'\nduration {1 / a.fps}\n" for p in frames) + f"file '{frames[-1]}'\n")
    out = f"{a.dir}/video_{a.name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-r", str(a.fps),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", out], check=True)
    shutil.rmtree(tmp); print(out, f"{len(frames) / a.fps:.1f} s")


if __name__ == "__main__":
    main()
