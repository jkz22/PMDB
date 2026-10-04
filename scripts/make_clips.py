"""Short pitch / demo animations (MP4 + GIF, 1920x1080, ~7-9 s each).

    python scripts/make_clips.py            # all clips
    python scripts/make_clips.py confound   # one clip: confound | depth | fem | verdict

Everything is drawn from committed outputs and the half-resolution cache;
results land in outputs/clips/.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pmdb import fingerprint as fp  # noqa: E402
from pmdb.io import load_site  # noqa: E402
from pmdb.kpis.fields import band_profile  # noqa: E402
from pmdb.segment import segment_bse  # noqa: E402

OUT = ROOT / "outputs" / "clips"
FPS = 30
BATCHES = ["Batch_1", "Batch_2", "Batch_3"]
COL = {"Batch_1": "#3d8bfd", "Batch_2": "#ff7a45", "Batch_3": "#22c38e"}
BG, FG, DIM, RED = "#0d1117", "#f0f3f6", "#8b949e", "#ff4d5e"
LABEL = {"Batch_1": "Batch 1", "Batch_2": "Batch 2", "Batch_3": "Batch 3"}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
    "axes.edgecolor": "#30363d", "font.family": "DejaVu Sans", "font.size": 16,
})


def ease(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3 - 2 * t)


def ramp(t: float, t0: float, t1: float) -> float:
    return ease((t - t0) / (t1 - t0))


def new_fig():
    fig = plt.figure(figsize=(16, 9), dpi=120)
    return fig


def render(name: str, fig, update, seconds: float) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mp4 = OUT / f"{name}.mp4"
    writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=6000,
                          extra_args=["-pix_fmt", "yuv420p", "-preset", "medium"])
    n = int(round(seconds * FPS))
    with writer.saving(fig, str(mp4), dpi=120):
        for i in range(n):
            update(i / FPS)
            writer.grab_frame(facecolor=BG)
    plt.close(fig)
    gif = OUT / f"{name}.gif"
    vf = "fps=15,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf", vf, str(gif)], check=True)
    print(f"wrote {mp4.relative_to(ROOT)} and {gif.name}")


def card(ax, x, y, w, h, colour, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                       transform=ax.transAxes, fc="#161b22", ec=colour, lw=3, alpha=alpha)
    ax.add_patch(p)
    return p


# ---------------------------------------------------------------------------
# shared data
# ---------------------------------------------------------------------------

def features() -> pd.DataFrame:
    fdir = ROOT / "outputs" / "kpis"
    return fp.read_feature_inputs(fdir / "curves.csv", fdir / "tile_kpis.csv")


BANDS = [f"si_depth_rel_band{i}" for i in range(5)]


def representatives(X: pd.DataFrame) -> dict[str, str]:
    """Per batch: the correctly-classified (LOO) site closest to the batch median depth profile."""
    loo = pd.read_csv(ROOT / "outputs/fingerprint/loo_predictions.csv", dtype=str).set_index(["batch", "site"])
    reps = {}
    for b in BATCHES:
        xb = X.xs(b, level="batch")[BANDS]
        med = xb.median()
        d = ((xb - med) ** 2).sum(axis=1)
        ok = [s for s in d.index if loo.loc[(b, s), "assigned"] == b]
        reps[b] = d.loc[ok].idxmin()
    return reps


# ---------------------------------------------------------------------------
# clip 1: the confound flip
# ---------------------------------------------------------------------------

def clip_confound() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import demo_confound as dc

    S = dc.load_bse_stats()
    y = S.index.get_level_values("batch").to_numpy()
    batch, site = "Batch_1", "4ih2ggld"
    x0 = S.loc[(batch, site)].to_numpy()

    def intensity_pred(o: int) -> str:
        # any offset >= 1 grey level lifts every pixel off 0, so nothing clips to black
        x = np.array([x0[0] + o, x0[1] + o, x0[2] if o == 0 else 0.0])
        return dc.nearest_centroid(S, y, x)

    raw = load_site(batch, site, resolution="half", normalise="none").image[..., 0]
    bse = raw.astype(np.float64)
    m0 = segment_bse(bse, 50.0).si
    m7 = segment_bse(np.clip(bse + 7, 0, 255), 50.0).si
    same = float((m0 == m7).mean())
    print(f"confound: Si map identical on {same:.4%} of pixels; flip at "
          f"+{next(o for o in range(8) if intensity_pred(o) != batch)}")

    h = raw.shape[0]
    crop = raw[:, 1200:1200 + int(h * 1.15)].astype(np.float64)

    fig = new_fig()
    fig.text(0.05, 0.90, "Same electrode. Slightly brighter microscope.", fontsize=40, weight="bold")
    sub = fig.text(0.05, 0.845, f"{LABEL[batch]} site {site}: add a flat grey-level offset, change nothing about the material",
                   fontsize=19, color=DIM)
    ax = fig.add_axes([0.05, 0.12, 0.42, 0.68])
    ax.set_axis_off()
    im = ax.imshow(crop, cmap="gray", vmin=0, vmax=150, interpolation="lanczos")
    off_txt = fig.text(0.26, 0.055, "", fontsize=30, weight="bold", ha="center")

    A = fig.add_axes([0.52, 0.12, 0.44, 0.68])
    A.set_axis_off()
    A.set_xlim(0, 1)
    A.set_ylim(0, 1)
    c1 = card(A, 0.02, 0.55, 0.96, 0.40, COL[batch])
    A.text(0.07, 0.86, "Intensity model", fontsize=24, weight="bold", transform=A.transAxes)
    A.text(0.07, 0.79, "reads grey levels  -  71% LOO accuracy", fontsize=16, color=DIM, transform=A.transAxes)
    v1 = A.text(0.07, 0.63, "", fontsize=38, weight="bold", transform=A.transAxes)
    card(A, 0.02, 0.05, 0.96, 0.40, COL[batch])
    A.text(0.07, 0.36, "Fingerprint model", fontsize=24, weight="bold", transform=A.transAxes)
    A.text(0.07, 0.29, "reads Si arrangement  -  blind to grey-level offsets", fontsize=16, color=DIM,
           transform=A.transAxes)
    A.text(0.07, 0.13, f"-> {LABEL[batch]}", fontsize=38, weight="bold", color=COL[batch], transform=A.transAxes)
    A.text(0.93, 0.13, f"Si map {same:.0%} identical", fontsize=16, color=DIM, ha="right", transform=A.transAxes)
    flip_note = A.text(0.07, 0.575, "", fontsize=16, color=RED, transform=A.transAxes)
    punch = fig.text(0.5, 0.035, "It learned the microscope, not the material.", fontsize=30, weight="bold",
                     ha="center", color=RED, alpha=0)

    def update(t):
        o = 0 if t < 1.2 else (1 if t < 2.6 else 1 + int(6 * ramp(t, 2.6, 4.4)))
        im.set_data(np.clip(crop + o, 0, 255))
        off_txt.set_text(f"black level  +{o} / 255")
        p = intensity_pred(o)
        v1.set_text(f"-> {LABEL[p]}")
        v1.set_color(COL[p] if p == batch else RED)
        flip_note.set_text("" if p == batch else "flipped at +1: no pixel clips to black any more")
        c1.set_edgecolor(COL[p] if p == batch else RED)
        a = ramp(t, 5.0, 5.6)
        punch.set_alpha(a)
        off_txt.set_alpha(1 - a)

    render("01_confound_flip", fig, update, 7.5)


# ---------------------------------------------------------------------------
# clip 2: depth-profile sweep
# ---------------------------------------------------------------------------

def clip_depth() -> None:
    X = features()
    reps = representatives(X)
    tag = {"Batch_1": "-  bottom-heavy", "Batch_2": "-  mid-depth depleted", "Batch_3": "-  uniform (supplier baseline)"}
    print("depth reps:", reps)

    fig = new_fig()
    fig.text(0.05, 0.935, "Same ingredients. Different arrangement.", fontsize=40, weight="bold")
    fig.text(0.05, 0.893, "Scanning each coating from top to bottom: where does the silicon sit?", fontsize=19,
             color=DIM)
    depth = np.linspace(0, 1, 400)
    centres = (np.arange(5) + 0.5) / 5
    rows = []
    for r, b in enumerate(BATCHES):
        y0 = 0.615 - r * 0.275
        s = reps[b]
        raw = load_site(b, s, resolution="half", normalise="none").image[..., 0]
        m = segment_bse(raw.astype(np.float64), 50.0)
        h = raw.shape[0]
        w = raw.shape[1]
        c0 = (raw.shape[1] - w) // 2
        g = raw[:, c0:c0 + w].astype(np.float32) / 255.0
        si = m.si[:, c0:c0 + w]
        prof_site = band_profile(si, m.fraction_space[:, c0:c0 + w], len(BANDS))
        prof_site = prof_site / np.nanmean(prof_site)
        print(f"  {s}: strip vs KPI profile max |diff| =",
              f"{np.nanmax(np.abs(prof_site - X.loc[(b, s), BANDS].to_numpy(dtype=float))):.3f}")
        rgb = np.repeat(g[..., None], 3, axis=2) * 0.9
        colour = np.array(matplotlib.colors.to_rgb(COL[b]), dtype=np.float32)
        lit = rgb.copy()
        lit[si] = colour
        dimmed = rgb * 0.45
        dimmed[si] = colour * 0.35 + 0.1
        ax = fig.add_axes([0.05, y0, 0.60, 0.225])
        ax.set_axis_off()
        img = ax.imshow(dimmed, interpolation="lanczos", aspect="auto")
        line = ax.axhline(0, color="white", lw=2.5, alpha=0)
        fig.text(0.05, y0 + 0.232, f"{LABEL[b]}", fontsize=22, weight="bold", color=COL[b])
        lab = fig.text(0.145, y0 + 0.232, tag[b], fontsize=22, weight="bold", color=FG, alpha=0)
        fig.text(0.65, y0 + 0.232, f"site {s}", fontsize=15, color=DIM, ha="right")

        P = fig.add_axes([0.70, y0, 0.25, 0.225])
        P.set_xlim(0.3, 1.8)
        P.set_ylim(1, 0)
        P.axvline(1.0, color="#30363d", lw=1.5, ls="--")
        P.set_yticks([0, 1], ["top", "bottom"])
        P.set_xticks([0.5, 1.0, 1.5])
        if r == 2:
            P.set_xlabel("Si fraction / site mean")
        if r == 0:
            P.text(1.8, -0.1, "thick: this site   thin: batch median", fontsize=12, color=DIM, ha="right",
                   va="bottom", transform=P.transData)
        for sp in ("top", "right"):
            P.spines[sp].set_visible(False)
        prof_med = X.xs(b, level="batch")[BANDS].median().to_numpy(dtype=float)
        fs = np.interp(depth, centres, prof_site)
        fm = np.interp(depth, centres, prof_med)
        lm, = P.plot([], [], color=COL[b], lw=1.5, alpha=0.55)
        ls, = P.plot([], [], color=COL[b], lw=4.5)
        dot, = P.plot([], [], "o", color="white", ms=9)
        rows.append(dict(img=img, lit=lit, dimmed=dimmed, line=line, h=h, ls=ls, lm=lm, dot=dot, fs=fs, fm=fm,
                         lab=lab))

    def update(t):
        d = ramp(t, 0.6, 5.2)
        for R in rows:
            cut = int(round(d * R["h"]))
            frame = R["dimmed"].copy()
            frame[:cut] = R["lit"][:cut]
            R["img"].set_data(frame)
            R["line"].set_ydata([cut, cut])
            R["line"].set_alpha(0.9 if 0 < d < 1 else 0)
            k = max(1, int(round(d * (len(depth) - 1))) + 1)
            R["ls"].set_data(R["fs"][:k], depth[:k])
            R["lm"].set_data(R["fm"][:k], depth[:k])
            R["dot"].set_data([R["fs"][k - 1]], [depth[k - 1]])
            R["dot"].set_alpha(1 if d < 1 else 0)
            R["lab"].set_alpha(ramp(t, 5.3, 5.9))

    render("02_depth_sweep", fig, update, 8.0)


# ---------------------------------------------------------------------------
# clip 3: FEM swelling triptych
# ---------------------------------------------------------------------------

def _longest_run(flags: np.ndarray) -> tuple[int, int]:
    best, start, run = (0, 0), None, 0
    for i, f in enumerate(np.append(flags, False)):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def fem_box(first: np.ndarray, last: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box of the microstructure in a FEM GIF frame (drops title, axes and colourbar)."""
    def content(f):
        return f.min(axis=2) < 0.85
    m0, m1 = content(first), content(last)
    c0, c1 = _longest_run(m1.mean(axis=0) > 0.5)
    r0a, r1a = _longest_run(m0[:, c0:c1].mean(axis=1) > 0.5)
    r0b, _ = _longest_run(m1[:, c0:c1].mean(axis=1) > 0.5)
    top = max(0, min(r0a, r0b) - 6)
    return top, r1a, c0 + 2, c1 - 2


def clip_fem() -> None:
    X = features()
    reps = representatives(X)
    curves = pd.read_csv(ROOT / "outputs/fem/site_curves.csv", dtype={"site": str})
    curves = curves[(curves["orientation"] == "bottom") & (~curves["heldout"])]
    med = curves.groupby(["batch", "frame"])["swelling"].median().unstack(0)
    frames = {}
    for b in BATCHES:
        gif = Image.open(ROOT / f"outputs/fem/gifs/{b}__{reps[b]}.gif")
        fr = []
        for i in range(gif.n_frames):
            gif.seek(i)
            fr.append(np.asarray(gif.convert("RGB"), dtype=np.float32) / 255.0)
        r0, r1, c0, c1 = fem_box(fr[0], fr[-1])
        frames[b] = [f[r0:r1, c0:c1] for f in fr]
    nf = len(frames[BATCHES[0]])
    site_sw = {b: curves[(curves["batch"] == b) & (curves["site"] == reps[b])].set_index("frame")["swelling"]
               for b in BATCHES}

    fig = new_fig()
    fig.text(0.05, 0.915, "Charging: silicon swells ~3x in volume", fontsize=38, weight="bold")
    fig.text(0.05, 0.865, "2D finite-element lithiation on the real segmented microstructure, one site per batch",
             fontsize=19, color=DIM)
    ims, txt = [], []
    for c, b in enumerate(BATCHES):
        ax = fig.add_axes([0.04 + c * 0.315, 0.36, 0.30, 0.44])
        ax.set_axis_off()
        ims.append(ax.imshow(frames[b][0], interpolation="lanczos", aspect="auto"))
        fig.text(0.04 + c * 0.315, 0.815, LABEL[b], fontsize=22, weight="bold", color=COL[b])
        txt.append(fig.text(0.34 + c * 0.315, 0.815, "", fontsize=20, weight="bold", ha="right"))
    soc = fig.text(0.5, 0.30, "", fontsize=26, weight="bold", ha="center")
    P = fig.add_axes([0.08, 0.075, 0.55, 0.19])
    P.set_xlim(0, 100)
    P.set_ylim(0, 16)
    P.set_xlabel("state of charge (%)")
    P.set_ylabel("swelling (%)")
    for sp in ("top", "right"):
        P.spines[sp].set_visible(False)
    socx = med.index.to_numpy() / (nf - 1) * 100
    lines = {b: P.plot([], [], color=COL[b], lw=4, label=f"{LABEL[b]} median")[0] for b in BATCHES}
    P.legend(loc="upper left", frameon=False, fontsize=14)
    punch = fig.text(0.66, 0.15, "All batches swell ~13%.\nChemistry is the same -\nthe arrangement is not.",
                     fontsize=24, weight="bold", alpha=0, va="center")

    def update(t):
        f = (nf - 1) * ramp(t, 0.6, 5.0)
        i0 = int(np.floor(f))
        i1 = min(i0 + 1, nf - 1)
        a = f - i0
        for c, b in enumerate(BATCHES):
            ims[c].set_data(frames[b][i0] * (1 - a) + frames[b][i1] * a)
            sw = np.interp(f, site_sw[b].index.to_numpy(), site_sw[b].to_numpy()) * 100
            txt[c].set_text(f"+{sw:.1f}%")
            k = np.searchsorted(socx, f / (nf - 1) * 100, side="right")
            xs = np.append(socx[:k], f / (nf - 1) * 100)
            ys = np.append(med[b].to_numpy()[:k], np.interp(f, med.index.to_numpy(), med[b].to_numpy())) * 100
            lines[b].set_data(xs, ys)
        soc.set_text(f"SOC {f / (nf - 1) * 100:.0f}%")
        punch.set_alpha(ramp(t, 5.3, 5.9))

    render("03_fem_swelling", fig, update, 7.5)


# ---------------------------------------------------------------------------
# clip 4: held-out verdicts + drifting shipment rejected
# ---------------------------------------------------------------------------

def clip_verdict() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import demo_reject as dr

    X = features()
    y = pd.Series(X.index.get_level_values("batch"), index=X.index)
    model = fp.fit(X, y)
    H = pd.read_csv(ROOT / "outputs/fingerprint/heldout_features.csv", dtype={"site": str}).set_index(["batch", "site"])
    HP = pd.read_csv(ROOT / "outputs/fingerprint/heldout_predictions.csv", dtype={"site": str}).set_index("site")
    p3 = fp.conformal_p(model, X)["Batch_3"]
    is_b3 = X.index.get_level_values("batch") == "Batch_3"
    base_idx = p3[is_b3].idxmax()
    base = X.loc[base_idx]
    ts = np.linspace(0, 1.5, 91)
    probes = pd.DataFrame([dr.drift(base, t) for t in ts], index=pd.MultiIndex.from_tuples(
        [("Batch_N", f"d{i}") for i in range(len(ts))], names=["batch", "site"]))
    pred = fp.predict(model, probes)
    fx, fy = "si_depth_slope", "si_depth_mid_dip"

    fig = new_fig()
    title = fig.text(0.05, 0.915, "Three unknown sites. Three calls.", fontsize=40, weight="bold")
    subt = fig.text(0.05, 0.865, "Each dot = one labelled site (2 of the 16 fingerprint features shown)", fontsize=19,
                    color=DIM)
    ax = fig.add_axes([0.07, 0.10, 0.52, 0.72])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("Si depth slope  (bottom - top)")
    ax.set_ylabel("mid-depth dip")
    for b in BATCHES:
        xb = X.xs(b, level="batch")
        ax.scatter(xb[fx], xb[fy], s=140, color=COL[b], alpha=0.85, edgecolor="none", label=LABEL[b])
    ax.legend(loc="lower left", frameon=False, fontsize=16)
    allx = np.concatenate([X[fx], H[fx], probes[fx]])
    ally = np.concatenate([X[fy], H[fy], probes[fy]])
    pad = 0.12
    ax.set_xlim(allx.min() - pad, allx.max() + pad)
    ax.set_ylim(ally.min() - pad, ally.max() + pad)

    R = fig.add_axes([0.63, 0.10, 0.34, 0.72])
    R.set_axis_off()
    R.set_xlim(0, 1)
    R.set_ylim(0, 1)
    hold_order = ["fn0mhxef", "xrv9xvzb", "3e122cbj"]
    note = {"fn0mhxef": "Si in a band only Batch 3 occupies",
            "xrv9xvzb": "mid-depth dip inside the Batch 2 cluster",
            "3e122cbj": "stable call, typical of every batch"}
    hold_art = []
    for k, s in enumerate(hold_order):
        b = HP.loc[s, "assigned"]
        hx, hy = H.loc[("Batch_heldout", s), fx], H.loc[("Batch_heldout", s), fy]
        mk = ax.scatter([hx], [hy], s=520, marker="*", color="white", edgecolor=COL[b], lw=2.5, zorder=5, alpha=0)
        lb = ax.annotate(s, (hx, hy), xytext=(12, -26 if s == "3e122cbj" else 10), textcoords="offset points",
                         fontsize=15, weight="bold", alpha=0)
        yy = 0.70 - k * 0.31
        cd = card(R, 0.02, yy, 0.96, 0.26, COL[b], alpha=0)
        t1 = R.text(0.07, yy + 0.17, f"{s}  ->  {LABEL[b]}", fontsize=24, weight="bold", color=COL[b], alpha=0)
        t2 = R.text(0.07, yy + 0.07, note[s], fontsize=15, color=DIM, alpha=0)
        hold_art.append((0.4 + 0.75 * k, [mk, lb, cd, t1, t2]))

    trail, = ax.plot([], [], color=RED, lw=2.5, ls=":", alpha=0)
    star = ax.scatter([], [], s=700, marker="*", color=RED, edgecolor="white", lw=1.5, zorder=6)
    dcard = card(R, 0.02, 0.32, 0.96, 0.36, RED, alpha=0)
    d1 = R.text(0.07, 0.57, "", fontsize=18, color=DIM, alpha=0)
    d2 = R.text(0.07, 0.43, "", fontsize=32, weight="bold", alpha=0)
    d3 = R.text(0.07, 0.36, "", fontsize=16, color=DIM, alpha=0)
    T2 = 4.0

    def update(t):
        a_out = 1 - ramp(t, T2 - 0.4, T2)
        for t0, arts in hold_art:
            a = ramp(t, t0, t0 + 0.35)
            for j, art in enumerate(arts):
                aa = a if j < 2 else a * a_out
                art.set_alpha(aa * (0.35 if (j < 2 and t > T2) else 1))
        if t < T2:
            return
        title.set_text("A new shipment drifts. The model says no.")
        subt.set_text(f"Start from the most typical Batch 3 site ({base_idx[1]}), let Si sink toward the bottom")
        s = ramp(t, T2 + 0.4, T2 + 4.0)
        i = int(round(s * (len(ts) - 1)))
        trail.set_data(probes[fx].iloc[:i + 1], probes[fy].iloc[:i + 1])
        trail.set_alpha(0.8)
        star.set_offsets([[probes[fx].iloc[i], probes[fy].iloc[i]]])
        r = pred.iloc[i]
        for art in (dcard, d1, d2, d3):
            art.set_alpha(ramp(t, T2, T2 + 0.3))
        d1.set_text(f"drift {ts[i]:.2f}")
        if r["ood"]:
            d2.set_text("REJECT\nout of distribution")
            d2.set_color(RED)
            d2.set_position((0.07, 0.40))
            dcard.set_edgecolor(RED)
            d3.set_text("")
        else:
            d2.set_text(f"accept as {LABEL[r['assigned']]}")
            d2.set_color(COL[r["assigned"]])
            d2.set_position((0.07, 0.45))
            dcard.set_edgecolor(COL[r["assigned"]])
            d3.set_text(f"credibility {r['credibility']:.2f}")

    render("04_verdict_reject", fig, update, 9.0)


CLIPS = {"confound": clip_confound, "depth": clip_depth, "fem": clip_fem, "verdict": clip_verdict}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(CLIPS):
        CLIPS[name]()
