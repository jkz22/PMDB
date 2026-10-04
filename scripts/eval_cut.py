"""Evaluate the experimental CUT/FastCUT translation (``scripts/modal_cut.py``) against its own input.

The generator was trained on hybrid-corrected BSE tiles of the four strong-session sites (domain A) vs all
other labelled sites (domain B). This script compares, per site, the translated BSE with the hybrid BSE it
was fed: phase fractions and object counts (fixed thresholds taken from the *input*, so a change means the
network moved pixels across phase boundaries), pixel-wise phase-label change (hallucination / erasure proxy),
SSIM, texture (hf_ratio, noise sigma, edge sigma) and the strong-vs-rest texture gap. Clean reference sites
and held-out sites, if present in the npz, measure identity drift (a safe translator must leave domain-B
images alone). Nothing here is used by the modelling routes; CUT stays experimental.

    python scripts/eval_cut.py --prepare-inputs             # writes translate_inputs.npz (strong + refs + held-out)
    python scripts/eval_cut.py --translated translated_latest.npz --out outputs/harmonisation_shift/cut
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import ndimage  # noqa: E402
from skimage.metrics import structural_similarity  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from pmdb import clean as C  # noqa: E402
from pmdb import harmonise as H  # noqa: E402
from pmdb import harmonise_ext as X  # noqa: E402
from pmdb.io import get_cache_root, list_clean_sites  # noqa: E402
from pmdb.segment import V0_PARAMS, segment_bse  # noqa: E402

_spec = importlib.util.spec_from_file_location("eval_ext", REPO_ROOT / "scripts" / "eval_harmonise_ext.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

STRONG = ("71vgq3fw", "kbdh4tri", "tuy3zymq", "x7u69zsw")
REF_SITES = ("4ih2ggld", "i9jiqjwl", "vc2whyaq")   # clean sites of Batch 1 / 2 / 3 (domain B: identity check)
HELDOUT = ("3e122cbj", "fn0mhxef", "xrv9xvzb")


def hybrid_bse(batch: str, site: str) -> tuple[np.ndarray, np.ndarray]:
    """Hybrid-LUT-corrected BSE uint8 and its uint16 clean mask at 50 nm/px (the CUT input domain)."""
    held = batch.lower().startswith("batch_heldout")
    root = REPO_ROOT / "cache_heldout" if held else get_cache_root()
    g, m = X.load_half_raw(batch, site, "BSE")
    raw = np.clip(np.round(g), 0, 255).astype(np.uint8)
    lut = H.load_lut(root, "hybrid", batch, site)
    return lut[0][raw], m


def all_sites() -> list[tuple[str, str]]:
    lab = list_clean_sites()
    sites = list(zip(lab["batch"], lab["site"]))
    return sites + [("Batch_heldout", s) for s in HELDOUT]


def prepare_inputs(out: Path) -> None:
    arrs = {}
    for b, s in all_sites():
        if s in STRONG or s in REF_SITES or s in HELDOUT:
            arrs[s], _ = hybrid_bse(b, s)
    np.savez_compressed(out, **arrs)
    print("wrote", out, sorted(arrs))


def phases_fixed(bse: np.ndarray, valid: np.ndarray, t_pore: float, t_si: float) -> tuple[np.ndarray, np.ndarray]:
    """`segment_bse` with its two grey thresholds fixed (same smoothing, closing, hole filling, small-object and
    artefact rules), so that input and translated output are labelled by one and the same rule; with the input's
    own thresholds this reproduces `segment_bse(input)` exactly, hence an identity translation scores zero change."""
    from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects

    p = V0_PARAMS
    px = lambda um2: int(np.ceil(um2 / (E.NM / 1000.0) ** 2))  # noqa: E731
    g = ndimage.gaussian_filter(np.asarray(E._fill_invalid(bse, valid), dtype=np.float64), sigma=p["gauss_sigma_px"])
    dark = g < t_pore
    si = (g > t_si) & ~dark
    si = binary_closing(si, disk(p["si_closing_radius_px"]))
    si = ndimage.binary_fill_holes(si)
    si = remove_small_objects(si, min_size=px(p["si_min_area_um2"]))
    pore_all = dark & ~si
    artefact = remove_small_objects(pore_all, min_size=px(p["artefact_min_area_um2"]))
    return pore_all & ~artefact, si


def _objects(mask: np.ndarray, valid: np.ndarray) -> int:
    return int(ndimage.label(mask & valid)[1])


def site_row(batch: str, site: str, cut: np.ndarray) -> dict:
    inp, msk = hybrid_bse(batch, site)
    if cut.shape != inp.shape:
        raise ValueError(f"{site}: translated {cut.shape} vs input {inp.shape}")
    valid = C.valid_for_stats(msk)
    v0 = C.valid_for_kpis(msk)
    prm = segment_bse(E._fill_invalid(inp, valid), E.NM).params  # thresholds from the input only
    pore_i, si_i = phases_fixed(inp, valid, prm["T_pore"], prm["T_si"])
    pore_o, si_o = phases_fixed(cut, valid, prm["T_pore"], prm["T_si"])
    lab_i = pore_i.astype(np.int8) + 2 * si_i
    lab_o = pore_o.astype(np.int8) + 2 * si_o
    row = {
        "batch": batch, "site": site,
        "group": "heldout" if site in HELDOUT else ("strong" if site in STRONG else "ref"),
        "f_pore_in": float(pore_i[v0].mean()), "f_pore_out": float(pore_o[v0].mean()),
        "f_si_in": float(si_i[v0].mean()), "f_si_out": float(si_o[v0].mean()),
        "n_pore_in": _objects(pore_i, v0), "n_pore_out": _objects(pore_o, v0),
        "n_si_in": _objects(si_i, v0), "n_si_out": _objects(si_o, v0),
        "label_change_frac": float((lab_i != lab_o)[v0].mean()),
        "mean_abs_change": float(np.abs(cut[v0].astype(np.int16) - inp[v0].astype(np.int16)).mean()),
        "ssim": float(structural_similarity(inp, cut, data_range=255)),
        "mean_in": float(inp[valid].mean()), "mean_out": float(cut[valid].mean()),
        "p1_in": float(np.percentile(inp[valid], 1)), "p1_out": float(np.percentile(cut[valid], 1)),
    }
    for tag, im in (("in", inp), ("out", cut)):
        row.update({f"{k}_{tag}": v for k, v in E._texture(im, valid).items()})
    return row


def _fig(rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]], out: Path) -> None:
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 3.2 * len(rows)))
    axes = np.atleast_2d(axes)
    for ax_row, (site, inp, cut, valid) in zip(axes, rows):
        h, w = inp.shape
        sl = (slice(h // 2 - 128, h // 2 + 128), slice(w // 2 - 256, w // 2 + 256))
        d = cut[sl].astype(np.int16) - inp[sl].astype(np.int16)
        for ax, im, t, kw in ((ax_row[0], inp[sl], f"{site} hybrid input", dict(cmap="gray", vmin=0, vmax=255)),
                              (ax_row[1], cut[sl], "CUT output", dict(cmap="gray", vmin=0, vmax=255)),
                              (ax_row[2], d, "difference (grey)", dict(cmap="RdBu_r", vmin=-30, vmax=30))):
            ax.imshow(im, **kw)
            ax.set_title(t, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--translated", default=None, help="npz written by modal_cut.py translate (key = site)")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "harmonisation_shift" / "cut"))
    ap.add_argument("--prepare-inputs", default=None, nargs="?", const="translate_inputs.npz")
    args = ap.parse_args()
    if args.prepare_inputs:
        prepare_inputs(Path(args.prepare_inputs))
        if not args.translated:
            return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(args.translated)
    batch_of = {s: b for b, s in all_sites()}
    rows, figrows = [], []
    for site in z.files:
        b = batch_of[site]
        r = site_row(b, site, z[site])
        rows.append(r)
        inp, msk = hybrid_bse(b, site)
        figrows.append((site, inp, z[site], C.valid_for_stats(msk)))
        print(site, r["group"], f"ssim={r['ssim']:.3f} label_change={r['label_change_frac']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "site_metrics.csv", index=False)
    _fig(figrows[:6], out / "examples.png")
    summ = {}
    for grp, g in df.groupby("group"):
        summ[grp] = {
            "n": int(len(g)),
            "ssim_mean": float(g.ssim.mean()),
            "label_change_frac_mean": float(g.label_change_frac.mean()),
            "mean_abs_change": float(g.mean_abs_change.mean()),
            "d_f_pore_pp": float(100 * (g.f_pore_out - g.f_pore_in).mean()),
            "d_f_si_pp": float(100 * (g.f_si_out - g.f_si_in).mean()),
            "d_n_pore_rel": float(((g.n_pore_out - g.n_pore_in) / g.n_pore_in.clip(lower=1)).mean()),
            "d_n_si_rel": float(((g.n_si_out - g.n_si_in) / g.n_si_in.clip(lower=1)).mean()),
            "hf_ratio_in": float(g.hf_ratio_in.mean()), "hf_ratio_out": float(g.hf_ratio_out.mean()),
            "noise_sigma_in": float(g.noise_sigma_in.mean()), "noise_sigma_out": float(g.noise_sigma_out.mean()),
            "d_mean_grey": float((g.mean_out - g.mean_in).mean()),
        }
    if {"strong", "ref"} <= set(summ):
        summ["strong_minus_ref_hf_ratio_in"] = summ["strong"]["hf_ratio_in"] - summ["ref"]["hf_ratio_in"]
        summ["strong_minus_ref_hf_ratio_out"] = summ["strong"]["hf_ratio_out"] - summ["ref"]["hf_ratio_out"]
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
