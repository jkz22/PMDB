"""'Is this image off, and why?' -- evaluation and explanation of the binary off-detectors (labels='off':
Batch_3 = supplier baseline vs Batch_1+2 = 'off'), one block per harmonisation route and architecture.

For every configuration (out-of-fold crop predictions of the 5 stratified grouped folds):
* metrics: field / crop accuracy, balanced accuracy, recall per class, predicted 'off' share, AUROC,
  crop + field ECE, cross-fold temperature, confidence on right vs wrong fields;
* per-field explanation: P(off) raw and temperature-scaled, fold the field was tested in, KPI deviations
  (z vs the Batch_3 fields' mean/SD), BSE imaging-stat deviations (p1, noise_sigma, sharpness, p99; z vs
  Batch_3), occlusion / Grad-CAM phase shares (from attribution.csv) and a plain-language 'why' sentence;
* the 3 held-out sites: every fold model (none saw them) scores the site, giving P(off) mean +- SD, with
  the same KPI / imaging deviation table.
Outputs under outputs/v2/off/: summary.csv, fields_<route>_<arch>.csv, heldout.csv, REPORT.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.v2.calibration import apply_temperature, ece, fit_temperature, nll
from src.v2.classify import CLS_RUNS, Classifier, OFF_CLASSES, classes_of, full_cfg
from src.v2.common import OUT, harm_method
from src.v2.sae_ablate import HELDOUT, _route_array, crop_imaging_stats, heldout_crops

OFF_OUT = OUT / "off"
KPIS = ["frac_si", "frac_graphite", "frac_pore", "K01_si_frac_adm", "K04_agglom_frac"]
KPI_NAME = {"frac_si": "Si fraction", "frac_graphite": "graphite fraction", "frac_pore": "pore fraction",
            "K01_si_frac_adm": "K01 Si admixture", "K04_agglom_frac": "K04 Si agglomerate fraction"}
IMG = ["p1", "p99", "noise_sigma", "sharpness"]
IMG_NAME = {"p1": "BSE black level (p1)", "p99": "BSE bright level (p99)", "noise_sigma": "BSE noise",
            "sharpness": "BSE sharpness"}
ROUTE = {("raw", "none"): "raw", ("raw", "hybrid"): "hybrid (normal)", ("naive", "none"): "naive",
         ("extreme", "none"): "extreme", ("raw", "nyul"): "nyul", ("raw", "basic"): "basic"}
ROUTE_NOTE = {
    "raw": "no harmonisation: grey level, gain, noise, focus and clipping cues all available",
    "hybrid (normal)": "PR #16 LUT: black level / gain removed; noise, focus, clipping, grey-step untouched",
    "naive": "phase-only pixels, per-crop p1-p99 rescale, sigma=0.1 noise: grey level, gain and fine texture swamped",
    "extreme": "naive + 1 px Gaussian blur before the noise: additionally removes the focus / sharpness cue",
    "nyul": "PR #37 N4ITK bias field + Nyul-Udupa histogram standardisation (non-linear, per-site landmarks)",
    "basic": "PR #37 BaSiC flat/dark field + per-image baseline (offset only, gain untouched)"}
Z_FLAG = 2.0


def route_of(c: dict) -> str:
    return ROUTE.get((c.get("input", "raw"), harm_method(c.get("harmonise"))), f"{c.get('input')}/{c.get('harmonise')}")


# ----------------------------------------------------------------------------------------------- references
def kpi_reference() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    crops = pd.concat([pd.read_csv(OUT / "kpis" / f"crop_kpis_{k}.csv") for k in ("train", "eval")], ignore_index=True)
    f = crops.drop_duplicates(["group_id", "y", "x"]).groupby("group_id")[KPIS].mean()  # field KPI = mean over its crops
    b3 = f[f.index.str.startswith("Batch_3")]
    return f, b3.mean(), b3.std(ddof=1)


def imaging_reference() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    im = pd.read_csv(OUT / "imaging_stats" / "per_image.csv")
    im = im[im.detector == "BSE"].set_index("group_id")[IMG]
    b3 = im[im.index.str.startswith("Batch_3")]
    return im, b3.mean(), b3.std(ddof=1)


def heldout_imaging(site: str) -> dict:
    im = _route_array(site, "none")[0][..., 0].astype(np.float32)
    p1, p99 = np.percentile(im, [1, 99])
    st = crop_imaging_stats(im[:1024, :1024])
    return {"p1": float(p1), "p99": float(p99), "noise_sigma": st["noise_sigma"], "sharpness": st["sharpness"]}


def heldout_kpis() -> pd.DataFrame:
    """Held-out field KPIs on the same scale as the labelled reference: mean over the non-overlapping
    256-px eval-grid crops (K04 agglomerate fraction is scale dependent, so whole-field values from
    predict_heldout_kpi.csv are not comparable with crop means)."""
    cache = OFF_OUT / "heldout_crop_kpis.csv"
    if cache.exists():
        crops = pd.read_csv(cache)
    else:
        from src.v2 import kpi_adapter as K
        from src.v2.common import CROP, NM_HALF, grid
        rows = []
        for site in HELDOUT:
            bse = _route_array(site, "none")[0][..., 0]
            m = K.segment(bse, NM_HALF)
            for r in K.crop_kpis(m, NM_HALF, f"Batch_heldout/{site}", grid(*bse.shape, CROP, CROP), CROP):
                rows.append({"group_id": f"Batch_heldout/{site}", **r})
        crops = pd.DataFrame(rows)
        OFF_OUT.mkdir(parents=True, exist_ok=True)
        crops.to_csv(cache, index=False)
    return crops.groupby("group_id")[KPIS].mean()


# ----------------------------------------------------------------------------------------------- runs
def load_binary_runs(runs_dir: Path = CLS_RUNS) -> dict[tuple[str, str], list[tuple[dict, pd.DataFrame, Path]]]:
    out: dict[tuple[str, str], list] = {}
    for d in sorted(runs_dir.iterdir()):
        f = d / "crop_predictions.csv"
        if not f.exists() or not (d / "config.json").exists():
            continue
        c = json.loads((d / "config.json").read_text())
        if c.get("labels") != "off" or c.get("view", "stack") != "stack":
            continue
        out.setdefault((route_of(c), c["arch"]), []).append((c, pd.read_csv(f), d))
    return out


def field_table(folds: list[tuple[dict, pd.DataFrame, Path]]) -> tuple[pd.DataFrame, dict]:
    pcols = [f"p_{b}" for b in OFF_CLASSES]
    df = pd.concat([d.assign(fold=c["fold"]) for c, d, _ in folds], ignore_index=True)
    P, y = df[pcols].to_numpy(), df.y_true.to_numpy()
    logP = np.log(np.clip(P, 1e-9, 1))
    Pt, Ts = P.copy(), []
    for f in sorted(df.fold.unique()):
        m = df.fold.to_numpy() == f
        t = fit_temperature(logP[~m], y[~m]); Ts.append(t)
        Pt[m] = apply_temperature(P[m], t)
    df["p_off"], df["p_off_temp"] = P[:, 0], Pt[:, 0]
    fld = df.groupby("group_id").agg(batch=("batch", "first"), y_true=("y_true", "first"), fold=("fold", "first"),
                                     n_crops=("p_off", "size"), p_off=("p_off", "mean"), p_off_temp=("p_off_temp", "mean"),
                                     p_off_sd=("p_off", "std"), frac_crops_off=("p_off", lambda s: float((s > 0.5).mean())))
    fld["pred_off"] = fld.p_off > 0.5
    fld["true_off"] = fld.y_true == 0
    fld["correct"] = fld.pred_off == fld.true_off
    yf = fld.true_off.to_numpy().astype(int); pf = fld.pred_off.to_numpy().astype(int)
    e_c, _ = ece(P, y); e_ct, _ = ece(Pt, y)
    Ff = np.stack([fld.p_off.to_numpy(), 1 - fld.p_off.to_numpy()], 1)
    e_f, _ = ece(Ff, 1 - yf, n_bins=5)
    conf = np.maximum(fld.p_off, 1 - fld.p_off)
    met = dict(n_folds=len(folds), n_fields=len(fld), n_crops=len(df),
               crop_acc=float(((P[:, 0] > 0.5) == (y == 0)).mean()), crop_auroc=float(roc_auc_score(y == 0, P[:, 0])),
               crop_ece=e_c, crop_ece_temp=e_ct, crop_nll=nll(P, y), crop_nll_temp=nll(Pt, y),
               temperature=float(np.mean(Ts)), temperature_sd=float(np.std(Ts)),
               field_acc=float((pf == yf).mean()),
               recall_off=float(pf[yf == 1].mean()), recall_Batch_3=float(1 - pf[yf == 0].mean()),
               field_auroc=float(roc_auc_score(yf, fld.p_off)), field_ece=e_f,
               pred_share_off=float(pf.mean()), true_share_off=float(yf.mean()),
               conf_right=float(conf[pf == yf].mean()), conf_wrong=float(conf[pf != yf].mean()) if (pf != yf).any() else np.nan,
               recall_Batch_1=float(fld[fld.batch == "Batch_1"].pred_off.mean()),
               recall_Batch_2=float(fld[fld.batch == "Batch_2"].pred_off.mean()))
    met["bal_acc"] = 0.5 * (met["recall_off"] + met["recall_Batch_3"])
    met["const_Batch_3_acc"] = 1 - met["true_share_off"]
    return fld, met


def attribution_by_field(folds) -> pd.DataFrame:
    frames = [pd.read_csv(d / "attribution.csv") for _, _, d in folds if (d / "attribution.csv").exists()]
    if not frames:
        return pd.DataFrame()
    a = pd.concat(frames, ignore_index=True)
    cols = [c for c in a.columns if c.startswith(("occ_ratio_", "occ_share_", "occ_corr_", "cam_ratio_"))]
    return a.groupby("group_id")[cols].median()


# ----------------------------------------------------------------------------------------------- explanations
def deviations(vals: pd.Series, mu: pd.Series, sd: pd.Series) -> pd.Series:
    return (vals - mu) / sd.replace(0, np.nan)


def why(row: pd.Series, zk: pd.Series, zi: pd.Series, attr: pd.Series | None, route: str) -> str:
    p = row.p_off_temp
    verdict = "OFF" if row.pred_off else "BASELINE-LIKE"
    s = [f"{verdict} (P(off) = {row.p_off:.2f}, calibrated {p:.2f}; {row.frac_crops_off:.0%} of {int(row.n_crops)} crops vote off)."]
    kflag = zk[zk.abs() >= Z_FLAG].sort_values(key=np.abs, ascending=False)
    iflag = zi[zi.abs() >= Z_FLAG].sort_values(key=np.abs, ascending=False)
    if len(kflag):
        s.append("Material deviates from Batch_3: " + ", ".join(f"{KPI_NAME[k]} {z:+.1f} SD" for k, z in kflag.items()) + ".")
    else:
        s.append("All five KPIs within 2 SD of the Batch_3 fields.")
    if len(iflag):
        s.append("Imaging deviates from Batch_3: " + ", ".join(f"{IMG_NAME[k]} {z:+.1f} SD" for k, z in iflag.items()) + ".")
    else:
        s.append("BSE black level / noise / sharpness within 2 SD of Batch_3.")
    if attr is not None and len(attr):
        r = {k: attr.get(f"occ_ratio_{k}", np.nan) for k in ("si", "graphite", "pore")}
        top = max(r, key=lambda k: (r[k] if np.isfinite(r[k]) else -1))
        s.append(f"Occlusion evidence is spread over the phases (Si x{r['si']:.1f}, graphite x{r['graphite']:.1f}, "
                 f"pore x{r['pore']:.1f} relative to area; most concentrated on {top}).")
    if row.pred_off:
        if route in ("extreme", "naive") and not len(iflag):
            s.append("Flagged on a route that removes grey level, gain, noise and focus, with no imaging deviation: "
                     "the 'off' call rests on phase geometry / morphology.")
        elif route in ("extreme", "naive"):
            s.append("Flagged on a route that removes grey level, gain, noise and focus; the imaging deviation listed "
                     "cannot be what the model sees, so the call rests on phase geometry / morphology.")
        elif len(iflag) and not len(kflag):
            s.append("Only imaging statistics deviate: on this route the 'off' call is most plausibly an imaging-session "
                     "fingerprint, not material -- compare with the extreme route.")
        elif len(kflag):
            s.append("KPI deviation present: the 'off' call has a material basis.")
    elif row.true_off:
        s.append("Missed: a Batch_1/2 field that looks like the Batch_3 baseline to this model.")
    return " ".join(s)


def explain_config(key, folds, kref, iref, hk, hi) -> tuple[pd.DataFrame, dict]:
    route, arch = key
    fld, met = field_table(folds)
    attr = attribution_by_field(folds)
    kf, kmu, ksd = kref; imf, imu, isd = iref
    rows = []
    for gid, r in fld.iterrows():
        zk = deviations(kf.loc[gid], kmu, ksd) if gid in kf.index else pd.Series(np.nan, index=KPIS)
        zi = deviations(imf.loc[gid], imu, isd) if gid in imf.index else pd.Series(np.nan, index=IMG)
        a = attr.loc[gid] if len(attr) and gid in attr.index else None
        rec = dict(group_id=gid, **r.to_dict(), **{f"z_{k}": v for k, v in zk.items()}, **{f"z_{k}": v for k, v in zi.items()})
        if a is not None:
            rec.update({k: a[k] for k in a.index if k.startswith("occ_ratio_")})
        rec["why"] = why(r, zk, zi, a, route)
        rows.append(rec)
    out = pd.DataFrame(rows)
    met.update(route=route, arch=arch, n_attributed=int(len(attr)))
    return out, met


# ----------------------------------------------------------------------------------------------- held-out
def heldout_scores(folds, dev) -> pd.DataFrame:
    c0 = full_cfg(folds[0][0])
    X, meta = heldout_crops(c0)
    xt = torch.from_numpy(X).permute(0, 3, 1, 2).float()
    probs = []
    for c, _, d in folds:
        m = Classifier(c["arch"], n_cls=len(classes_of(c)))
        m.load_state_dict(torch.load(d / "final.pt", map_location="cpu"))
        m = m.to(dev).eval()
        with torch.no_grad():
            p = torch.cat([torch.softmax(m(b.to(dev)).float(), 1).cpu() for b in xt.split(32)]).numpy()
        probs.append(p[:, 0])
    P = np.stack(probs)  # folds x crops
    meta["p_off"] = P.mean(0)
    rows = []
    for s in HELDOUT:
        m = (meta.site == s).to_numpy()
        per_fold = P[:, m].mean(1)
        rows.append(dict(site=s, n_crops=int(m.sum()), p_off=float(per_fold.mean()), p_off_fold_sd=float(per_fold.std()),
                         p_off_min=float(per_fold.min()), p_off_max=float(per_fold.max()),
                         frac_crops_off=float((P[:, m].mean(0) > 0.5).mean())))
    return pd.DataFrame(rows)


def heldout_why(r: pd.Series, zk: pd.Series, zi: pd.Series, route: str) -> str:
    verdict = "OFF" if r.p_off > 0.5 else "BASELINE-LIKE"
    s = [f"{verdict}: P(off) = {r.p_off:.2f} (fold models {r.p_off_min:.2f}-{r.p_off_max:.2f}), {r.frac_crops_off:.0%} of crops vote off."]
    kflag = zk[zk.abs() >= Z_FLAG].sort_values(key=np.abs, ascending=False)
    iflag = zi[zi.abs() >= Z_FLAG].sort_values(key=np.abs, ascending=False)
    s.append(("KPIs off Batch_3: " + ", ".join(f"{KPI_NAME[k]} {z:+.1f} SD" for k, z in kflag.items()) + ".") if len(kflag)
             else "All five KPIs within 2 SD of Batch_3.")
    s.append(("Imaging off Batch_3: " + ", ".join(f"{IMG_NAME[k]} {z:+.1f} SD" for k, z in iflag.items()) + ".") if len(iflag)
             else "BSE imaging statistics within 2 SD of Batch_3.")
    if r.p_off_max - r.p_off_min > 0.4:
        s.append("Fold models disagree strongly: treat as uncertain.")
    return " ".join(s)


# ----------------------------------------------------------------------------------------------- report
def main(with_heldout: bool = True, reuse_heldout: bool = False):
    OFF_OUT.mkdir(parents=True, exist_ok=True)
    reuse = None
    if reuse_heldout and (OFF_OUT / "heldout.csv").exists():  # keep the (slow) held-out model scores, redo z-scores / text
        reuse = pd.read_csv(OFF_OUT / "heldout.csv")
        reuse = reuse[[c for c in reuse.columns if not c.startswith("z_") and c != "why"]]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    runs = load_binary_runs()
    kref, iref = kpi_reference(), imaging_reference()
    hk = heldout_kpis()
    hi = pd.DataFrame({s: heldout_imaging(s) for s in HELDOUT}).T
    summ, held, md = [], [], ["# Off-detector: Batch_3 baseline vs Batch_1+2 ('off')\n",
                             "Binary classifiers, stack view, stratified grouped 5-fold; every number is out-of-fold. "
                             "Explanations: KPI and BSE-imaging z-scores against the 17 Batch_3 fields, occlusion phase "
                             "shares, temperature-scaled confidence.\n"]
    for key in sorted(runs):
        folds = runs[key]
        if len(folds) < 5:
            print("skipping incomplete", key, len(folds), file=sys.stderr)
            continue
        out, met = explain_config(key, folds, kref, iref, hk, hi)
        tag = f"{key[0].split(' ')[0]}_{key[1]}"
        out.to_csv(OFF_OUT / f"fields_{tag}.csv", index=False)
        summ.append(met)
        if with_heldout:
            try:
                h = reuse[(reuse.route == key[0]) & (reuse.arch == key[1])].drop(columns=["route", "arch"]) if reuse is not None else None
                if h is None or h.empty:
                    h = heldout_scores(folds, dev)
                for _, r in h.iterrows():
                    gid = f"Batch_heldout/{r.site}"
                    zk = deviations(hk.loc[gid], kref[1], kref[2]) if gid in hk.index else pd.Series(np.nan, index=KPIS)
                    zi = deviations(hi.loc[r.site].astype(float), iref[1], iref[2])
                    held.append(dict(route=key[0], arch=key[1], **r.to_dict(), **{f"z_{k}": v for k, v in zk.items()},
                                     **{f"z_{k}": v for k, v in zi.items()}, why=heldout_why(r, zk, zi, key[0])))
            except Exception as e:  # noqa: BLE001
                print("held-out scoring failed for", key, e, file=sys.stderr)
    S = pd.DataFrame(summ)
    S.to_csv(OFF_OUT / "summary.csv", index=False)
    H = pd.DataFrame(held)
    if len(H):
        H.to_csv(OFF_OUT / "heldout.csv", index=False)
    md.append("## Leaderboard (field level)\n")
    cols = ["route", "arch", "field_acc", "bal_acc", "recall_off", "recall_Batch_1", "recall_Batch_2", "recall_Batch_3",
            "pred_share_off", "field_auroc", "crop_auroc", "crop_ece", "crop_ece_temp", "temperature", "conf_right", "conf_wrong"]
    md.append(S[cols].round(3).to_markdown(index=False) + "\n")
    md.append(f"Constant 'Batch_3' predictor: field acc {S.const_Batch_3_acc.iloc[0]:.2f}, balanced acc 0.50. True 'off' share "
              f"{S.true_share_off.iloc[0]:.2f}.\n")
    for route in S.route.unique():
        md.append(f"\n## Route `{route}` -- {ROUTE_NOTE.get(route, '')}\n")
        for arch in S[S.route == route].arch:
            tag = f"{route.split(' ')[0]}_{arch}"
            f = pd.read_csv(OFF_OUT / f"fields_{tag}.csv")
            md.append(f"### {arch}\n")
            md.append("Per-field verdicts (sorted by P(off)):\n")
            show = f.sort_values("p_off", ascending=False)[["group_id", "p_off", "p_off_temp", "correct", "why"]]
            md.append(show.round(2).to_markdown(index=False) + "\n")
    if len(H):
        md.append("\n## Held-out sites (all fold models, none trained on these sites)\n")
        md.append(H[["route", "arch", "site", "p_off", "p_off_fold_sd", "frac_crops_off", "why"]].round(2).to_markdown(index=False) + "\n")
    (OFF_OUT / "REPORT.md").write_text("\n".join(md))
    print(S[cols].round(3).to_string())
    if len(H):
        print(H[["route", "arch", "site", "p_off", "p_off_fold_sd"]].round(3).to_string())


if __name__ == "__main__":
    main(with_heldout="--no-heldout" not in sys.argv, reuse_heldout="--reuse-heldout" in sys.argv)
