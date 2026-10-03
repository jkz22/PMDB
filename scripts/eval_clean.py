#!/usr/bin/env python3
"""Acceptance checks for the physical clean pipeline (brief §7) on a finished build.

    python scripts/eval_clean.py --out outputs/clean [--heldout outputs/clean_heldout]

Writes ``<out>/eval/`` : site_residuals.csv, fingerprint.csv, material.csv, acceptance.json,
acceptance.md and figures.  Nothing here changes the build outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pmdb.clean as C  # noqa: E402
from scripts.build_clean import REFERENCE_SITES  # noqa: E402

STRONG = ("71vgq3fw", "kbdh4tri", "tuy3zymq", "x7u69zsw")
MILD = ("9luzk4jm", "hzumfsms", "ufdvpb81")
Z_PORE_THR, Z_SI_THR = 0.5, 1.45  # fixed thresholds in z units (pore 0, graphite 1, Si ≈ 1.7–2.6)


def group_of(site: str) -> str:
    if site in STRONG:
        return "strong"
    if site in MILD:
        return "mild"
    if site in REFERENCE_SITES:
        return "reference"
    return "clean"


def _hsm_blocks(z: np.ndarray, sel: np.ndarray, block: int = 128) -> np.ndarray:
    b = C.block_medians(z, sel, block_px=block, min_cover=0.2)
    return b["median"].to_numpy()


def site_residuals(site_dir: Path, params: dict, rng: np.random.Generator) -> dict:
    """Post-normalisation anchors, flatness, GMM component means and fixed-threshold fractions."""
    out = {"batch": params["batch"], "site": params["site"]}
    for kind in ("norm", "harm"):
        if not (site_dir / f"BSE_{kind}.tif").exists():
            continue
        z, m = C.read_site(site_dir, "BSE", kind)
        v_stat, v_kpi = C.valid_for_stats(m), C.valid_for_kpis(m)
        ph = C.provisional_phases(z, v_kpi)
        # pores keep their 0-clipped pixels (valid for KPIs, not for fits): on Batch 1/2 sites with D <= 0 up to
        # half of the pore pixels sit at the clip, and dropping them would leave only the bright pore tail
        pore = z[ph.pore & v_kpi]
        graph = z[ph.graphite & v_stat]
        si = z[ph.si & v_stat]
        blocks = _hsm_blocks(z, ph.graphite & v_stat)
        sub = z[v_stat][:: max(1, int(v_stat.sum()) // 300_000)]
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(3, random_state=0).fit(sub[:, None])
        means = np.sort(gmm.means_.ravel())
        out.update({
            f"{kind}_pore_median": float(np.median(pore)), f"{kind}_pore_mode": C._robust_mode(pore[:: max(1, pore.size // 200_000)]),
            f"{kind}_graphite_hsm": C._robust_mode(graph[:: max(1, graph.size // 200_000)]), f"{kind}_graphite_median": float(np.median(graph)),
            f"{kind}_si_median": float(np.median(si)) if si.size else np.nan,
            f"{kind}_flatness_rel_sd": float(np.std(blocks) / np.median(blocks)), f"{kind}_flatness_ptp_rel": float(np.ptp(blocks) / np.median(blocks)),
            f"{kind}_gmm_mu0": means[0], f"{kind}_gmm_mu1": means[1], f"{kind}_gmm_mu2": means[2],
            f"{kind}_frac_pore_fixed": float(((z < Z_PORE_THR) & v_kpi).sum() / v_kpi.sum()),
            f"{kind}_frac_si_fixed": float(((z > Z_SI_THR) & v_kpi).sum() / v_kpi.sum()),
            f"{kind}_frac_pore_otsu": float((ph.pore & v_kpi).sum() / v_kpi.sum()),
            f"{kind}_frac_si_otsu": float((ph.si & v_kpi).sum() / v_kpi.sum()),
            f"{kind}_otsu_thr_pore": ph.thresholds[0], f"{kind}_otsu_thr_si": ph.thresholds[1],
            f"{kind}_p01": float(np.percentile(sub, 1)), f"{kind}_p50": float(np.percentile(sub, 50)), f"{kind}_p99": float(np.percentile(sub, 99)),
            f"{kind}_valid_frac": float(v_kpi.mean()),
        })
        if kind == "harm":
            for d in ("Inlens", "SE_type"):
                zz, mm = C.read_site(site_dir, d, "harm")
                vv = C.valid_for_stats(mm)
                s2 = zz[vv][:: max(1, int(vv.sum()) // 300_000)]
                out.update({f"harm_{d}_p01": float(np.percentile(s2, 1)), f"harm_{d}_p50": float(np.percentile(s2, 50)), f"harm_{d}_p99": float(np.percentile(s2, 99))})
    # raw-domain percentiles from the source TIFF (imaging fingerprint "before")
    for d, p in params["paths"].items():
        raw, _ = C.read_rgb_tiff(p)
        rr = raw[8:-8, 8:-8].ravel()[::50]
        out.update({f"raw_{d}_p01": float(np.percentile(rr, 1)), f"raw_{d}_p50": float(np.percentile(rr, 50)), f"raw_{d}_p99": float(np.percentile(rr, 99)),
                    f"raw_{d}_frac0": float((rr == 0).mean()), f"raw_{d}_frac255": float((rr == 255).mean())})
    return out


def loso_accuracy(X: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_perm: int = 200) -> dict:
    X = np.nan_to_num(np.asarray(X, dtype=float))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acc = float((cross_val_predict(clf, X, y, cv=LeaveOneOut()) == y).mean())
        null = []
        for _ in range(n_perm):
            yp = rng.permutation(y)
            null.append(float((cross_val_predict(clf, X, yp, cv=LeaveOneOut()) == yp).mean()))
    null = np.asarray(null)
    return {"accuracy": acc, "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
            "p_value": float((np.sum(null >= acc) + 1) / (n_perm + 1)), "chance": float(np.bincount(y).max() / y.size)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/clean")
    ap.add_argument("--heldout", default=None)
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()
    out = Path(args.out)
    ev = out / "eval"
    ev.mkdir(exist_ok=True)
    rng = np.random.default_rng(C.SEED)
    summary = pd.read_csv(out / "summary.csv")
    summary["group"] = summary.site.map(group_of)
    params = {r.site: json.loads((out / r.batch / r.site / "params.json").read_text()) for r in summary.itertuples()}
    targets = json.loads((out / "targets.json").read_text()) if (out / "targets.json").exists() else None

    # ---- per-site residuals (streams the full-resolution outputs) --------------------------------
    rows = []
    for r in summary.itertuples():
        rows.append(site_residuals(out / r.batch / r.site, params[r.site], rng))
        print(f"  residuals {r.batch}/{r.site}", flush=True)
    res = pd.DataFrame(rows).merge(summary, on=["batch", "site"])
    res.to_csv(ev / "site_residuals.csv", index=False)

    acc: dict = {"n_sites": int(len(res)), "code_version": C.CODE_VERSION}

    # 7.3 known cases -----------------------------------------------------------------------------
    known = {
        "ufdvpb81_marker_cols_masked": int(res.loc[res.site == "ufdvpb81", "n_marker_cols"].iloc[0]) >= 4,
        "strong_sites_BSE_D_ge_15": bool((res.loc[res.site.isin(STRONG), "BSE_D"] >= 15).all()),
        "strong_sites_no_zero_clipping": bool((res.loc[res.site.isin(STRONG), "BSE_D_frac_zero"] < 0.001).all()),
        "hzumfsms_crack_flagged_not_patched": bool(res.loc[res.site == "hzumfsms", "flag_crack"].iloc[0]) and float(res.loc[res.site == "hzumfsms", "BSE_mask_crack"].iloc[0]) == 0.0,
        "reference_sites_unflagged": bool(~res.loc[res.site.isin(REFERENCE_SITES), ["flag_crack", "flag_pore_band_top", "flag_pore_band_bottom"]].any(axis=None)),
        "masked_beyond_border_median_lt_2pct": float(res["BSE_masked_beyond_border"].median()) < 0.02,
    }
    acc["known_cases"] = known

    # 7.4 normalisation residuals --------------------------------------------------------------------
    d_before = res["BSE_D"] / res["BSE_G"]  # dark level in graphite units before (what z removes)
    acc["normalisation"] = {
        "pore_median_z_max_abs": float(res["norm_pore_median"].abs().max()),
        "pore_median_z_sd_across_sites": float(res["norm_pore_median"].std()),
        "graphite_hsm_z_max_abs_dev": float((res["norm_graphite_hsm"] - 1).abs().max()),
        "graphite_hsm_z_sd_across_sites": float(res["norm_graphite_hsm"].std()),
        "dark_level_rel_sd_before": float(d_before.std()),
        "strong_minus_clean_pore_z_after": float(res.loc[res.group == "strong", "norm_pore_median"].mean() - res.loc[res.group == "clean", "norm_pore_median"].mean()),
        "strong_minus_clean_dark_rel_before": float(d_before[res.group == "strong"].mean() - d_before[res.group == "clean"].mean()),
        "gmm_mu_sd_across_sites": {f"mu{k}": float(res[f"norm_gmm_mu{k}"].std()) for k in range(3)},
        "pass_pore_within_0p05": bool((res["norm_pore_median"].abs() < 0.05).all()),
        "pass_graphite_within_0p02": bool(((res["norm_graphite_hsm"] - 1).abs() < 0.02).all()),
    }
    # 7.5 flat-field flatness ------------------------------------------------------------------------
    acc["flatness"] = {
        "block_rel_sd_max": float(res["norm_flatness_rel_sd"].max()), "block_rel_sd_median": float(res["norm_flatness_rel_sd"].median()),
        "G_ptp_rel_max": float(res["BSE_G_ptp_rel"].max()),
        "pass_block_rel_sd_lt_0p03": bool((res["norm_flatness_rel_sd"] < 0.03).all()),
    }

    # 7.6 imaging fingerprint: LOSO batch classifier on acquisition-only features ----------------------
    y = pd.factorize(res["batch"])[0]
    feats_before = [f"{d}_{k}" for d in C.DETECTORS for k in ("D", "G", "sigma_e", "noise_sigma_g", "grey_levels")] + [f"raw_{d}_{k}" for d in C.DETECTORS for k in ("p01", "p50", "p99", "frac0")]
    feats_after_acq = [f"{d}_{k}" for d in C.DETECTORS for k in ("sigma_e_after", "noise_sigma_g_after")] if "BSE_sigma_e_after" in res else []
    feats_after_int = ["harm_p01", "harm_p50", "harm_p99"] + [f"harm_{d}_{k}" for d in ("Inlens", "SE_type") for k in ("p01", "p50", "p99")] if "harm_p01" in res else []
    feats_before_int = [f"raw_{d}_{k}" for d in C.DETECTORS for k in ("p01", "p50", "p99")]
    fp = {"before_acquisition_and_percentiles": loso_accuracy(res[feats_before].to_numpy(), y, rng, args.n_perm),
          "before_intensity_percentiles": loso_accuracy(res[feats_before_int].to_numpy(), y, rng, args.n_perm)}
    if feats_after_acq:
        fp["after_acquisition_only"] = loso_accuracy(res[feats_after_acq].to_numpy(), y, rng, args.n_perm)
    if feats_after_int:
        fp["after_intensity_percentiles"] = loso_accuracy(res[feats_after_int].to_numpy(), y, rng, args.n_perm)
        fp["after_all"] = loso_accuracy(res[feats_after_acq + feats_after_int].to_numpy(), y, rng, args.n_perm)
    # strong-vs-rest separability (the artefact the earlier LUT pipeline targeted)
    ys = (res.group == "strong").astype(int).to_numpy()
    fp["strong_vs_rest_before"] = loso_accuracy(res[feats_before].to_numpy(), ys, rng, args.n_perm)
    if feats_after_acq:
        fp["strong_vs_rest_after"] = loso_accuracy(res[feats_after_acq + feats_after_int].to_numpy(), ys, rng, args.n_perm)
    acc["fingerprint"] = fp
    pd.DataFrame(fp).T.to_csv(ev / "fingerprint.csv")

    # 7.7 material KPI preservation ------------------------------------------------------------------
    mat = res[["batch", "site", "group", "porosity", "si_fraction", "norm_frac_pore_fixed", "norm_frac_si_fixed", "norm_frac_pore_otsu", "norm_frac_si_otsu",
               "BSE_si_graphite", "largest_pore_um2", "largest_pore_span_um", "porosity_top8", "porosity_bottom8", "flag_crack", "flag_pore_band_top", "flag_pore_band_bottom"]].copy()
    if "harm_frac_pore_fixed" in res:
        mat["harm_frac_pore_fixed"] = res["harm_frac_pore_fixed"]
        mat["harm_frac_si_fixed"] = res["harm_frac_si_fixed"]
    mat.to_csv(ev / "material.csv", index=False)
    acc["material"] = {
        "porosity_otsu_vs_fixed_corr": float(np.corrcoef(mat["porosity"], mat["norm_frac_pore_fixed"])[0, 1]),
        "porosity_otsu_vs_fixed_mean_abs_diff": float((mat["porosity"] - mat["norm_frac_pore_fixed"]).abs().mean()),
        "si_otsu_vs_fixed_corr": float(np.corrcoef(mat["si_fraction"], mat["norm_frac_si_fixed"])[0, 1]),
        "si_otsu_vs_fixed_mean_abs_diff": float((mat["si_fraction"] - mat["norm_frac_si_fixed"]).abs().mean()),
        "si_graphite_ratio_by_group": {g: float(v) for g, v in mat.groupby("group")["BSE_si_graphite"].median().items()},
        "si_graphite_ratio_sd_across_sites": float(mat["BSE_si_graphite"].std()),
        "flagged_crack_sites": mat.loc[mat.flag_crack, "site"].tolist(),
        "flagged_pore_band_sites": mat.loc[mat.flag_pore_band_top | mat.flag_pore_band_bottom, "site"].tolist(),
    }
    if "harm_frac_pore_fixed" in mat:
        acc["material"]["porosity_norm_vs_harm_max_abs_diff"] = float((mat["norm_frac_pore_fixed"] - mat["harm_frac_pore_fixed"]).abs().max())
        acc["material"]["si_norm_vs_harm_max_abs_diff"] = float((mat["norm_frac_si_fixed"] - mat["harm_frac_si_fixed"]).abs().max())

    # 7.8 scan bands / charging / FOV ------------------------------------------------------------------
    acc["patching"] = {
        "sites_with_band_events": int((res["n_band_events"] > 0).sum()), "bands_corrected_total": int(res["n_bands_corrected"].sum()),
        "bands_masked_total": int(res["n_bands_masked"].sum()), "band_rows_corrected_frac_median": float(res["BSE_mask_band_corrected"].median()),
        "sites_with_local_charging": int((res["n_charging_local"] > 0).sum()), "sites_with_broad_charging": int(res["charging_broad"].sum()),
        "sites_with_collector": int(res["collector_found"].sum()), "sites_with_free_surface": int(res["free_surface_found"].sum()),
        "masked_beyond_border_max": {d: float(res[f"{d}_masked_beyond_border"].max()) for d in C.DETECTORS},
        "clip_high_frac_max": {d: float(res[f"{d}_mask_clip_high"].max()) for d in C.DETECTORS},
        "clip_low_frac_max": {d: float(res[f"{d}_mask_clip_low"].max()) for d in C.DETECTORS},
    }

    # 4.6/4.7 harmonisation verification ------------------------------------------------------------------
    if targets:
        h = {}
        for d in C.DETECTORS:
            blurred = res[res[f"{d}_sigma_k"] > 0]
            h[d] = {"sigma_t": targets["sigma_t"][d], "n_blurred": int(len(blurred)),
                    "sigma_e_after_max_abs_dev": float((blurred[f"{d}_sigma_e_after"] - targets["sigma_t"][d]).abs().max()) if len(blurred) else 0.0,
                    "resolution_outliers": res.loc[res[f"{d}_resolution_outlier"], "site"].tolist(),
                    "n_noise_added": int(res[f"{d}_noise_added"].sum()), "noise_within_10pct_all": bool(res[f"{d}_noise_within_10pct"].all()),
                    "noise_sigma_g_after_cv": float(res[f"{d}_noise_sigma_g_after"].std() / res[f"{d}_noise_sigma_g_after"].mean()),
                    "noise_sigma_g_before_cv": float(res[f"{d}_noise_sigma_g"].std() / res[f"{d}_noise_sigma_g"].mean()),
                    "sigma_e_before_cv": float(res[f"{d}_sigma_e"].std() / res[f"{d}_sigma_e"].mean()),
                    "sigma_e_after_cv": float(res[f"{d}_sigma_e_after"].std() / res[f"{d}_sigma_e_after"].mean())}
        acc["harmonisation"] = h

    # held-out ------------------------------------------------------------------------------------------
    if args.heldout and Path(args.heldout, "summary.csv").exists():
        ho = pd.read_csv(Path(args.heldout) / "summary.csv")
        acc["heldout"] = {"n_sites": int(len(ho)), "sites": ho.site.tolist(),
                          "BSE_D": {s: float(v) for s, v in zip(ho.site, ho.BSE_D)}, "BSE_G": {s: float(v) for s, v in zip(ho.site, ho.BSE_G)},
                          "sigma_k_BSE": {s: float(v) for s, v in zip(ho.site, ho.get("BSE_sigma_k", pd.Series([np.nan] * len(ho))))},
                          "fit_on_heldout": "no — targets.json and material thresholds come from the labelled run"}

    (ev / "acceptance.json").write_text(json.dumps(acc, indent=1, default=float))
    _figures(res, ev, out)
    _acceptance_md(acc, ev)
    print(f"wrote {ev}")


def _acceptance_md(acc: dict, ev: Path) -> None:
    lines = ["# Acceptance numbers", ""]

    def walk(d, depth=0):
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"{'  ' * depth}- **{k}**")
                walk(v, depth + 1)
            else:
                vv = f"{v:.4g}" if isinstance(v, float) else v
                lines.append(f"{'  ' * depth}- {k}: {vv}")

    walk(acc)
    (ev / "acceptance.md").write_text("\n".join(lines) + "\n")


def _figures(res: pd.DataFrame, ev: Path, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Batch_1": "C0", "Batch_2": "C1", "Batch_3": "C2"}
    order = res.sort_values(["batch", "BSE_D"]).reset_index(drop=True)
    # anchors
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    for i, d in enumerate(C.DETECTORS):
        for j, k in enumerate(("D", "G")):
            ax = axes[i, j]
            ax.bar(range(len(order)), order[f"{d}_{k}"], color=[colors[b] for b in order.batch])
            ax.set_ylabel(f"{d} {k} (DN)")
            for n, r in enumerate(order.itertuples()):
                if r.group in ("strong", "mild", "reference"):
                    ax.text(n, order[f"{d}_{k}"].iloc[n], r.group[0].upper(), ha="center", va="bottom", fontsize=7)
    axes[-1, 0].set_xticks(range(len(order)))
    axes[-1, 0].set_xticklabels(order.site, rotation=90, fontsize=7)
    axes[-1, 1].set_xticks(range(len(order)))
    axes[-1, 1].set_xticklabels(order.site, rotation=90, fontsize=7)
    fig.suptitle("Per-site physical anchors from the raw images (S = strong, M = mild, R = reference)")
    fig.tight_layout()
    fig.savefig(ev / "fig_anchors.png", dpi=90)
    plt.close(fig)
    # residuals after normalisation
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].bar(range(len(order)), order["norm_pore_median"], color=[colors[b] for b in order.batch])
    axes[0].set_title("pore median z after normalisation (target 0)")
    axes[1].bar(range(len(order)), order["norm_graphite_hsm"] - 1, color=[colors[b] for b in order.batch])
    axes[1].set_title("graphite mode z − 1 after normalisation (target 0)")
    axes[2].bar(range(len(order)), order["BSE_si_graphite"], color=[colors[b] for b in order.batch])
    axes[2].set_title("Si / graphite ratio (material KPI, not forced)")
    for ax in axes:
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order.site, rotation=90, fontsize=6)
    fig.tight_layout()
    fig.savefig(ev / "fig_residuals.png", dpi=90)
    plt.close(fig)
    # fingerprint before/after
    if "BSE_sigma_e_after" in res:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, (a, b, t) in zip(axes, [("BSE_sigma_e", "BSE_sigma_e_after", "ESF sigma BSE (px)"), ("BSE_noise_sigma_g", "BSE_noise_sigma_g_after", "noise SD at graphite, BSE (z)")]):
            for g, mk in (("strong", "s"), ("mild", "^"), ("reference", "*"), ("clean", "o")):
                sel = res.group == g
                ax.scatter(res.loc[sel, a], res.loc[sel, b], marker=mk, label=g, c=[colors[x] for x in res.loc[sel, "batch"]])
            lim = [min(res[a].min(), res[b].min()) * 0.9, max(res[a].max(), res[b].max()) * 1.1]
            ax.plot(lim, lim, "k:")
            ax.set_xlabel("before")
            ax.set_ylabel("after harmonisation")
            ax.set_title(t)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(ev / "fig_harmonisation.png", dpi=90)
        plt.close(fig)
    # material preservation
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (a, b, t) in zip(axes, [("porosity", "norm_frac_pore_fixed", "porosity: per-site Otsu (raw) vs fixed z<0.5"), ("si_fraction", "norm_frac_si_fixed", "Si fraction: per-site Otsu vs fixed z>1.45")]):
        ax.scatter(res[a], res[b], c=[colors[x] for x in res.batch])
        for r in res.itertuples():
            if r.group != "clean":
                ax.annotate(r.site, (getattr(r, a), getattr(r, b)), fontsize=6)
        lim = [0, max(res[a].max(), res[b].max()) * 1.1]
        ax.plot(lim, lim, "k:")
        ax.set_title(t, fontsize=10)
    fig.tight_layout()
    fig.savefig(ev / "fig_material.png", dpi=90)
    plt.close(fig)
    # visual montage: strong / mild / reference / Batch_1 crops
    picks = [s for s in ("71vgq3fw", "hzumfsms", "ptg8lmto", "4ih2ggld") if s in set(res.site)]
    kinds = ["raw", "norm"] + (["harm"] if "BSE_sigma_e_after" in res else [])
    fig, axes = plt.subplots(len(picks), len(kinds), figsize=(5 * len(kinds), 3.2 * len(picks)))
    axes = np.atleast_2d(axes)
    for i, s in enumerate(picks):
        r = res[res.site == s].iloc[0]
        sd = out / r.batch / s
        p = json.loads((sd / "params.json").read_text())
        y0, x0 = p["shape"][0] // 2 - 300, p["shape"][1] // 2 - 500
        for j, k in enumerate(kinds):
            if k == "raw":
                img = C.read_rgb_tiff(p["paths"]["BSE"])[0][y0: y0 + 600, x0: x0 + 1000]
                axes[i, j].imshow(img, cmap="gray", vmin=0, vmax=255)
            else:
                z, _ = C.read_site(sd, "BSE", k)
                axes[i, j].imshow(z[y0: y0 + 600, x0: x0 + 1000], cmap="gray", vmin=-0.2, vmax=2.6)
            axes[i, j].set_title(f"{s} ({r.group}) {k}", fontsize=9)
            axes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(ev / "fig_montage.png", dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    main()
