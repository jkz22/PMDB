"""Patch-level MicroNet embedding + calibrated kNN batch classifier on Modal.

One-off LUT upload (affine2 LUTs are not in the volume yet):
    modal volume put pmdb-data cache/harmonised/affine2/luts.npz /harmonised/affine2/luts.npz
    modal volume put pmdb-data cache_heldout/harmonised/affine2/luts.npz /heldout/harmonised/affine2/luts.npz

Run:
    modal run modal_patch_mil.py --smoke      # 3 sites/batch + 1 held-out, 20 permutations
    modal run modal_patch_mil.py              # full: 31 labelled + 3 held-out, 1000 permutations
    modal run modal_patch_mil.py --mode eval  # re-run only a stage (embed | distances | eval | lopo | all)
    modal run modal_patch_mil.py --mode lopo     # leave-one-parent-out eval + ensemble + final held-out calls
    modal run modal_patch_mil.py --mode heldout  # embed only missing held-out sites, then distances, eval, lopo
    modal run modal_patch_mil.py --mode menu     # 34-site (31 + 3 held-out truths) LOPO model menu + frozen selection
    modal run modal_test_prep.py::main           # prerequisite of --mode test: preprocess data_test/ on Modal
    modal run modal_patch_mil.py --mode probe    # supervised linear probe, LOPO, 34 sites
    modal run modal_patch_mil.py --mode test     # embed + score new organiser test sites with the frozen selection

See docs/patch_mil.md.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

from pmdb.patch_embed import WEIGHTS_URL

APP_NAME = "pmdb-patch-mil"
WEIGHTS_PATH = "/weights/resnet50_micronet_v1.1.pth.tar"
HARMONISE = "affine2"
SMOKE_PER_BATCH = 3
SMOKE_HELDOUT = ("3e122cbj",)
SMOKE_N_PERM = 20

METHOD = {
    "backbone": "torchvision resnet50, NASA MicroNet v1.1 (frozen)",
    "weights_url": WEIGHTS_URL,
    "layers": ["layer2", "layer3"],
    "patch_px": 224,
    "nm_per_px": 50.0,
    "channels": ["BSE", "Inlens"],
    "harmonise": "affine2",
    "normalise": "fixed",
    "k_nn": 3,
    "distance": "cosine",
    "pool": "top-10% mean (headline); mean (secondary)",
    "calibration": "ECDF of training-batch leave-site-out distances, leave-one-bank-site-out matched",
    "softmax_tau": 0.1,
    "anomalous_u": 0.95,
}

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.4.1", "torchvision==0.19.1",
        "numpy==1.26.4", "scipy==1.14.1", "scikit-image==0.25.2", "pandas==2.1.4",
        "scikit-learn==1.5.2", "Pillow==12.3.0", "tifffile==2025.5.10", "imagecodecs==2025.3.30", "matplotlib==3.9.2",
    )
    .run_commands(
        "mkdir -p /weights && python -c \"import urllib.request; "
        f"urllib.request.urlretrieve('{WEIGHTS_URL}', '{WEIGHTS_PATH}')\"",
        f"python -c \"import os; s = os.path.getsize('{WEIGHTS_PATH}'); assert s == 102546991, s\"",
    )
    .env({"PMDB_CACHE": "/data"})
    .add_local_python_source("pmdb")
)
data_vol = modal.Volume.from_name("pmdb-data").read_only()
out_vol = modal.Volume.from_name("pmdb-patch-out", create_if_missing=True)
app = modal.App(APP_NAME, image=image)


@app.function(gpu="T4", volumes={"/data": data_vol, "/out": out_vol}, memory=8192, timeout=1800)
def embed_sites(sites: list[tuple[str, str]], tag: str, skip_existing: bool = False) -> list[dict]:
    import numpy as np
    import scipy.ndimage
    import torch

    from pmdb.io import load_site
    from pmdb.patch_embed import build_backbone, patch_features
    from pmdb.patch_mil import combine_channels, grid_coords

    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = False
    model = build_backbone(WEIGHTS_PATH, "cuda")
    print("MicroNet weights loaded cleanly (only fc.* missing, no unexpected keys)")
    emb_dir = Path(f"/out/{tag}/emb")
    emb_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for batch, site in sites:
        if skip_existing and (emb_dir / f"{batch}__{site}.npz").exists():
            print("skip", batch, site)
            continue
        t0 = time.time()
        cache_root = {"Batch_heldout": "/data/heldout", "Batch_test": "/data/test"}.get(batch, "/data")
        s = load_site(batch, site, resolution="half", normalise="fixed", harmonise=HARMONISE,
                      cache_root=cache_root)
        img = s.image
        h, w = img.shape[:2]
        coords = grid_coords(h, w)
        feat = combine_channels(patch_features(model, img[..., 0], coords, "cuda"),
                                patch_features(model, img[..., 1], coords, "cuda")).astype(np.float16)
        bse_hf = float(np.std(scipy.ndimage.laplace(img[..., 0].astype(np.float64))))
        np.savez(emb_dir / f"{batch}__{site}.npz", feat=feat, coords=coords, batch=np.array(batch),
                 site=np.array(site), shape=np.array([h, w]), bse_hf=np.array(bse_hf))
        rows.append({"batch": batch, "site": site, "n_patches": int(len(coords)), "height": int(h),
                     "width": int(w), "bse_hf": bse_hf, "elapsed_s": round(time.time() - t0, 2)})
        print(rows[-1])
    out_vol.commit()
    return rows


def _load_emb(tag: str, batch: str, site: str):
    import numpy as np

    z = np.load(f"/out/{tag}/emb/{batch}__{site}.npz", allow_pickle=False)
    return z["feat"].astype(np.float32), z["coords"], float(z["bse_hf"])


@app.function(volumes={"/out": out_vol}, cpu=4.0, memory=8192, timeout=1800)
def compute_distances(labelled: list[tuple[str, str]], heldout: list[str], tag: str,
                      heldout_batch: str = "Batch_heldout", out_name: str = "distances.npz") -> dict:
    import numpy as np

    from pmdb.patch_mil import site_distance_matrix

    t0 = time.time()
    out_vol.reload()
    feats, coords, bse_hf, ps = [], [], [], []
    for i, (b, s) in enumerate(labelled):
        f, c, h = _load_emb(tag, b, s)
        feats.append(f)
        coords.append(c)
        bse_hf.append(h)
        ps.append(np.full(len(f), i))
    Q = np.concatenate(feats)
    patch_site = np.concatenate(ps)
    D = site_distance_matrix(Q, patch_site, feats)

    hfeats, hcoords, hbse, hps = [], [], [], []
    for j, s in enumerate(heldout):
        f, c, h = _load_emb(tag, heldout_batch, s)
        hfeats.append(f)
        hcoords.append(c)
        hbse.append(h)
        hps.append(np.full(len(f), j))
    if heldout:
        Qh = np.concatenate(hfeats)
        Dh = site_distance_matrix(Qh, np.full(len(Qh), -1), feats)
        psh, coords_h, bse_h = np.concatenate(hps), np.concatenate(hcoords), np.array(hbse)
    else:
        Dh, psh, coords_h, bse_h = np.zeros((0, len(labelled))), np.zeros(0, int), np.zeros((0, 2), int), np.zeros(0)
    np.savez(f"/out/{tag}/{out_name}", D=D, Dh=Dh, patch_site=patch_site,
             patch_site_h=psh, coords=np.concatenate(coords),
             coords_h=coords_h,
             site_batch=np.array([b for b, _ in labelled]), site_id=np.array([s for _, s in labelled]),
             heldout_id=np.array(heldout), bse_hf=np.array(bse_hf), bse_hf_h=bse_h)
    out_vol.commit()
    return {"D_shape": list(D.shape), "Dh_shape": list(Dh.shape),
            "n_inf_per_row_ok": bool((np.isinf(D).sum(axis=1) == 1).all()),
            "elapsed_s": round(time.time() - t0, 2)}


@app.function(volumes={"/data": data_vol, "/out": out_vol}, cpu=2.0, memory=8192, timeout=3600)
def evaluate(tag: str, n_perm: int, seed: int = 0) -> dict:
    import numpy as np
    import pandas as pd

    from pmdb.io import load_site
    from pmdb.patch_mil import BATCHES, build_outputs, render_heldout_figure

    t0 = time.time()
    out_vol.reload()
    z = np.load(f"/out/{tag}/distances.npz", allow_pickle=False)
    site_batch = [str(x) for x in z["site_batch"]]
    sites = pd.DataFrame({"batch": site_batch, "site": [str(x) for x in z["site_id"]]})
    heldout = [str(x) for x in z["heldout_id"]]
    site_labels = np.array([BATCHES.index(b) for b in site_batch])
    res = build_outputs(z["D"], z["patch_site"], site_labels, sites, z["coords"], z["bse_hf"],
                        z["Dh"], z["patch_site_h"], heldout, z["coords_h"], z["bse_hf_h"],
                        n_perm=n_perm, seed=seed)
    res["evaluation"]["method"] = METHOD
    held = res["heldout_predictions"]
    ps = res["patch_scores"]
    figures = {}
    for _, r in held.iterrows():
        site = r["site"]
        s = load_site("Batch_heldout", site, resolution="half", normalise="fixed", harmonise=HARMONISE,
                      cache_root="/data/heldout")
        sel = ps[(ps["split"] == "heldout") & (ps["site"] == site)]
        coords = sel[["row", "col", "y0", "x0"]].to_numpy()
        figures[site] = render_heldout_figure(
            s.image[..., 0], s.image[..., 1], coords, sel["u_Batch_3"].to_numpy(),
            title=f"{site}: assigned {r['assigned']}, anomaly vs Batch 3 = {r['anomaly_vs_B3']:.2f}")
    return {"evaluation": res["evaluation"],
            "loo_predictions": res["loo_predictions"].to_dict("records"),
            "heldout_predictions": held.to_dict("records"),
            "patch_scores": ps.to_dict("records"),
            "figures": figures, "elapsed_s": round(time.time() - t0, 2)}


@app.function(volumes={"/out": out_vol}, cpu=4.0, memory=8192, timeout=3600)
def evaluate_lopo(tag: str, n_perm: int, parents: list[dict], fp_features: list[dict],
                  fp_heldout: list[dict], seed: int = 0) -> dict:
    import numpy as np
    import pandas as pd

    from pmdb.patch_lopo import build_lopo_outputs
    from pmdb.patch_mil import BATCHES

    t0 = time.time()
    out_vol.reload()
    z = np.load(f"/out/{tag}/distances.npz", allow_pickle=False)
    site_batch = [str(x) for x in z["site_batch"]]
    sites = pd.DataFrame({"batch": site_batch, "site": [str(x) for x in z["site_id"]]})
    heldout = [str(x) for x in z["heldout_id"]]
    site_labels = np.array([BATCHES.index(b) for b in site_batch])
    X = pd.DataFrame(fp_features).set_index(["batch", "site"])
    H = pd.DataFrame(fp_heldout).set_index(["batch", "site"])
    res = build_lopo_outputs(z["D"], z["patch_site"], site_labels, sites, pd.DataFrame(parents), X, H,
                             z["Dh"], z["patch_site_h"], heldout, z["coords_h"], n_perm=n_perm, seed=seed)
    return {"evaluation": res["evaluation"],
            "lopo_predictions": res["lopo_predictions"].to_dict("records"),
            "final_heldout": res["final_heldout"].to_dict("records"),
            "elapsed_s": round(time.time() - t0, 2)}


@app.function(volumes={"/out": out_vol}, cpu=4.0, memory=8192, timeout=3600)
def evaluate_menu(tag: str, dist_name: str, labels: list[dict], parents: list[dict], fp_lab: list[dict],
                  fp_test: list[dict], test_batch: str, selection: dict | None) -> dict:
    import numpy as np
    import pandas as pd

    from pmdb import batch_menu as bm
    from pmdb.patch_mil import BATCHES

    t0 = time.time()
    out_vol.reload()
    z = np.load(f"/out/{tag}/{dist_name}", allow_pickle=False)
    lab = pd.DataFrame(labels)
    assert [str(x) for x in z["site_batch"]] == list(lab["batch"])
    assert [str(x) for x in z["site_id"]] == list(lab["site"])
    keys = list(zip(lab["batch"], lab["site"]))
    site_labels = np.array([BATCHES.index(b) for b in lab["label"]])
    par = pd.DataFrame(parents).drop_duplicates(["batch", "site"]).set_index(["batch", "site"])["parent_id"]
    lab_par = par.reindex(pd.MultiIndex.from_tuples(keys))
    assert not lab_par.isna().any()
    groups, uniques = pd.factorize(lab_par.to_numpy())
    X = pd.DataFrame(fp_lab).set_index(["batch", "site"]).loc[keys]
    y = pd.Series([BATCHES[c] for c in site_labels], index=X.index)
    test_sites = [str(x) for x in z["heldout_id"]]
    test_keys = [(test_batch, s) for s in test_sites]
    if test_sites:
        H = pd.DataFrame(fp_test).set_index(["batch", "site"]).loc[test_keys][list(X.columns)]
        t_par = [par.get(k) for k in test_keys]
        t_par_ids = [p if p is not None else f"new_{k[1]}" for p, k in zip(t_par, test_keys)]
        code_of = {u: i for i, u in enumerate(uniques)}
        test_codes = np.array([code_of.get(p, -1) if p is not None else -1 for p in t_par])
        pool_par = np.concatenate([lab_par.to_numpy(), t_par_ids])
        pool = pd.concat([X, H])
    else:
        H, t_par_ids, test_codes = X.iloc[:0], [], np.zeros(0, int)
        pool_par, pool = lab_par.to_numpy(), X
    Xc_all, sing_all = bm.centre_by_parent(pool, pd.Series(pool_par))
    n = len(X)
    Xc, Hc = Xc_all.iloc[:n], Xc_all.iloc[n:]
    singleton, singleton_h = sing_all.to_numpy()[:n], sing_all.to_numpy()[n:]
    menu_df, sc = bm.menu_cv(z["D"], z["patch_site"], site_labels, groups, X, Xc, y, singleton)
    summary = bm.menu_summary(menu_df, site_labels)
    if selection is None:
        selection = bm.select_option(summary)
    test_pred = []
    if test_sites:
        test_pred = bm.predict_test(z["D"], z["patch_site"], site_labels, groups, X, Xc, y, singleton, menu_df, sc,
                                    z["Dh"], z["patch_site_h"], test_keys, test_codes, H, Hc, singleton_h,
                                    z["coords_h"], selection, test_parent_ids=t_par_ids).to_dict("records")
    menu_df.insert(0, "parent_id", lab_par.to_numpy())
    return {"summary": summary, "selection": selection, "menu_predictions": menu_df.to_dict("records"),
            "test_predictions": test_pred, "elapsed_s": round(time.time() - t0, 2)}


@app.function(volumes={"/out": out_vol}, cpu=4.0, memory=16384, timeout=1800)
def probe_lopo(tag: str, labels: list[dict], parents: list[dict], menu: list[dict], n_perm: int = 200, centre: str = "none") -> dict:
    import numpy as np
    import pandas as pd

    from pmdb import patch_probe as pp

    t0 = time.time()
    out_vol.reload()
    X, sb, sp = {}, {}, {}
    par = pd.DataFrame(parents).drop_duplicates(["batch", "site"]).set_index(["batch", "site"])["parent_id"]
    for r in labels:
        f, _, _ = _load_emb(tag, r["batch"], r["site"])
        X[r["site"]] = f
        sb[r["site"]] = r["label"]
        sp[r["site"]] = par[(r["batch"], r["site"])]
    assert len(X) == 34
    if centre == "parent":
        X = pp.centre_by_parent(X, sp)
    df = pp.lopo_probe(X, sb, sp)
    m = pd.DataFrame(menu)
    pe = [f"p_ens_{b}" for b in pp.BATCHES]
    ens = m.set_index("site").loc[df["site"]]
    assert (ens["true"].to_numpy() == df["true"].to_numpy()).all()
    ens_df = df[["parent_id", "site", "true"]].copy()
    pens = ens[pe].to_numpy()
    for i, b in enumerate(pp.BATCHES):
        ens_df[f"p_{b}"] = pens[:, i]
    ens_df["call"] = [pp.BATCHES[i] for i in pens.argmax(1)]
    comb = ens_df.copy()
    pc = (pens + df[[f"p_{b}" for b in pp.BATCHES]].to_numpy()) / 2
    for i, b in enumerate(pp.BATCHES):
        comb[f"p_{b}"] = pc[:, i]
    comb["call"] = [pp.BATCHES[i] for i in pc.argmax(1)]
    held = ["3e122cbj", "fn0mhxef", "xrv9xvzb"]
    out = {"metrics": {"probe": pp.site_metrics(df), "probe+ensemble": pp.site_metrics(comb),
                       "ensemble": pp.site_metrics(ens_df)},
           "heldout": {k: d[d["site"].isin(held)].drop(columns="parent_id").to_dict("records")
                       for k, d in (("probe", df), ("probe+ensemble", comb), ("ensemble", ens_df))}}
    out["predictions"] = df.assign(
        **{f"ens_p_{b}": ens_df[f"p_{b}"] for b in pp.BATCHES},
        **{f"comb_p_{b}": comb[f"p_{b}"] for b in pp.BATCHES}, comb_call=comb["call"], ens_call=ens_df["call"]
    ).to_dict("records")
    out["permutation"] = pp.permutation_test(X, sb, sp, n_perm=n_perm)
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


@app.local_entrypoint()
def main(mode: str = "all", smoke: bool = False, n_perm: int = 1000, centre: str = "none"):
    import pandas as pd

    from pmdb.parents import write_parent_groups
    from pmdb.patch_mil import fingerprint_comparison

    if mode not in ("embed", "distances", "eval", "lopo", "heldout", "all", "menu", "test", "probe"):
        raise SystemExit(f"unknown mode {mode!r}; use embed | distances | eval | lopo | heldout | all | menu | test | probe")
    root = Path(__file__).resolve().parent
    if mode == "probe":
        _probe(root, n_perm, centre)
        return
    if mode in ("menu", "test"):
        _menu_or_test(mode, root)
        return
    man = pd.read_csv(root / "cache" / "half" / "manifest.csv")
    labelled = [(str(b), str(s)) for b, s in zip(man["batch"], man["site"])]
    if smoke:
        counts: dict[str, int] = {}
        keep = []
        for b, s in labelled:
            if counts.get(b, 0) < SMOKE_PER_BATCH:
                keep.append((b, s))
                counts[b] = counts.get(b, 0) + 1
        labelled, heldout, n_perm, tag = keep, list(SMOKE_HELDOUT), SMOKE_N_PERM, "smoke"
        out_dir = root / "outputs" / "modal" / "patch_mil_smoke"
    else:
        hman = pd.read_csv(root / "cache_heldout" / "half" / "manifest.csv", dtype={"site": str})
        heldout, tag = [str(x) for x in hman["site"]], "full"
        out_dir = root / "outputs" / "patch_mil"

    if mode in ("embed", "all"):
        t0 = time.time()
        rows = embed_sites.remote(labelled + [("Batch_heldout", s) for s in heldout], tag)
        print(f"embed: {len(rows)} sites, {sum(r['n_patches'] for r in rows)} patches, "
              f"{time.time() - t0:.0f}s wall")
    if mode == "heldout":
        t0 = time.time()
        rows = embed_sites.remote([("Batch_heldout", s) for s in heldout], tag, True)
        print(f"embed (held-out, skip existing): {len(rows)} new sites, {time.time() - t0:.0f}s wall")
    if mode in ("distances", "all", "heldout"):
        t0 = time.time()
        print("distances:", compute_distances.remote(labelled, heldout, tag), f"{time.time() - t0:.0f}s wall")
    if mode in ("eval", "all", "heldout"):
        t0 = time.time()
        res = evaluate.remote(tag, n_perm)
        ev = res["evaluation"]
        fp = json.loads((root / "outputs" / "fingerprint" / "evaluation.json").read_text())
        ev["comparison"] = fingerprint_comparison(fp, ev)
        _atomic_write(out_dir / "evaluation.json", json.dumps(ev, indent=2).encode())
        for name in ("loo_predictions", "heldout_predictions", "patch_scores"):
            _atomic_write(out_dir / f"{name}.csv", pd.DataFrame(res[name]).to_csv(index=False).encode())
        for site, png in res["figures"].items():
            _atomic_write(out_dir / "figures" / f"heldout_{site}.png", png)
        sd = ev["shortcut_diagnostic"]
        assigns = {r["site"]: f"{r['assigned']} ({r['confidence']:.2f})" for r in res["heldout_predictions"]}
        print(f"eval ({time.time() - t0:.0f}s wall, remote {res['elapsed_s']}s): "
              f"LOO acc {ev['loo']['accuracy']:.3f}, bal acc {ev['loo']['balanced_accuracy']:.3f}, "
              f"perm p {ev['permutation']['p_value']:.4f}, shortcut rho {sd['spearman_rho']:.3f} "
              f"flag={sd['flag']}, held-out {assigns}")
    if mode in ("lopo", "all", "heldout"):
        if smoke:
            print("lopo: skipped for --smoke (3 sites/batch cannot give >= 2 groups per batch per fold)")
            return
        t0 = time.time()
        parents = write_parent_groups()
        fpf = pd.read_csv(root / "outputs" / "fingerprint" / "features.csv", dtype={"site": str})
        fph = pd.read_csv(root / "outputs" / "fingerprint" / "heldout_features.csv", dtype={"site": str})
        res = evaluate_lopo.remote(tag, n_perm, parents.to_dict("records"), fpf.to_dict("records"),
                                   fph.to_dict("records"))
        ev = res["evaluation"]
        _atomic_write(out_dir / "lopo_evaluation.json", json.dumps(ev, indent=2).encode())
        _atomic_write(out_dir / "lopo_predictions.csv",
                      pd.DataFrame(res["lopo_predictions"]).to_csv(index=False).encode())
        _atomic_write(out_dir / "final_heldout_predictions.csv",
                      pd.DataFrame(res["final_heldout"]).to_csv(index=False).encode())
        for k in ("patch", "fingerprint", "ensemble"):
            print(f"lopo {k}: LOSO acc {ev['loso'][k]['accuracy']:.3f}, LOPO acc {ev['lopo'][k]['accuracy']:.3f}, "
                  f"perm p {ev['permutation'][k]['p_value']:.4f}")
        print(f"rubric ensemble_flag {ev['rubric_expected_score']['ensemble_flag']:.3f}; "
              f"wall {time.time() - t0:.0f}s, remote {res['elapsed_s']}s")
        for r in res["final_heldout"]:
            print(f"  {r['site']} -> {r['assigned']} ({r['confidence_flag']})")


def _probe(root: Path, n_perm: int = 200, centre: str = "none") -> None:
    import pandas as pd

    from pmdb.batch_menu import labelled_sites
    from pmdb.parents import parent_groups

    t0 = time.time()
    lab = labelled_sites()
    menu = pd.read_csv(root / "outputs" / "menu" / "menu_predictions.csv", dtype={"site": str})
    res = probe_lopo.remote("full", lab.to_dict("records"), parent_groups().to_dict("records"),
                            menu.to_dict("records"), n_perm, centre=centre)
    out = root / "outputs" / ("patch_probe_centred" if centre == "parent" else "patch_probe")
    ev = {k: res[k] for k in ("metrics", "heldout", "permutation", "elapsed_s")}
    _atomic_write(out / "evaluation.json", json.dumps(ev, indent=2).encode())
    _atomic_write(out / "predictions.csv", pd.DataFrame(res["predictions"]).to_csv(index=False).encode())
    for k, m in res["metrics"].items():
        print(f"{k:16s} acc {m['accuracy']:.3f} bal {m['balanced_accuracy']:.3f} f1 {m['macro_f1']:.3f} "
              f"all-high {m['rubric_all_high']:.3f} flag {m['rubric_flag_pmax_ge_0.5']:.3f}")
    print(f"perm {res['permutation']}; wall {time.time() - t0:.0f}s, remote {res['elapsed_s']}s")


def _menu_or_test(mode: str, root: Path) -> None:
    import pandas as pd

    from pmdb import batch_menu as bm
    from pmdb.batch_menu import feature_table, labelled_sites
    from pmdb.parents import parent_groups, write_parent_groups
    from pmdb.within_parent import signal_sentence

    t0 = time.time()
    lab = labelled_sites()
    labelled34 = [(b, s) for b, s in zip(lab["batch"], lab["site"])]
    lab_rec = lab.to_dict("records")
    fp_lab = feature_table(labelled34).reset_index()
    sel_path = root / "outputs" / "menu" / "selection.json"
    if mode == "menu":
        parents = write_parent_groups()
        print("distances:", compute_distances.remote(labelled34, [], "full", "Batch_test", "distances_r3.npz"))
        res = evaluate_menu.remote("full", "distances_r3.npz", lab_rec, parents.to_dict("records"),
                                   fp_lab.to_dict("records"), [], "Batch_test", None)
        summ, sel = res["summary"], res["selection"]
        out = root / "outputs" / "menu"
        _atomic_write(out / "menu_evaluation.json",
                      json.dumps({"summary": summ, "selection": sel, "n_sites": 34}, indent=2).encode())
        _atomic_write(out / "menu_predictions.csv", pd.DataFrame(res["menu_predictions"]).to_csv(index=False).encode())
        _atomic_write(sel_path, json.dumps({**sel, "selected_on": "LOPO, 34 labelled sites, r3 run"},
                                           indent=2).encode())
        for o in bm.OPTIONS:
            r = summ[o]
            print(f"{o:20s} acc {r['accuracy']:.3f} bal {r['balanced_accuracy']:.3f} rubric {r['rubric']:.3f} "
                  f"(SE {r['rubric_se']:.3f}) all-high {r['rubric_all_high']:.3f} n_high {r['n_high']}")
        print("selection:", sel, f"wall {time.time() - t0:.0f}s, remote {res['elapsed_s']}s")
        return
    cm = root / "cache_test" / "half" / "manifest.csv"
    if not cm.exists():
        raise SystemExit("cache_test/half/manifest.csv missing: run `modal run modal_test_prep.py::main` first")
    test_sites = [str(x) for x in pd.read_csv(cm, dtype={"site": str})["site"]]
    if not sel_path.exists():
        raise SystemExit("outputs/menu/selection.json missing: run --mode menu first")
    selection = json.loads(sel_path.read_text())
    ft = root / "outputs" / "test" / "features.csv"
    if not ft.exists():
        raise SystemExit("outputs/test/features.csv missing: run `modal run modal_test_prep.py::main` first")
    fp_test = pd.read_csv(ft, dtype={"batch": str, "site": str})
    if set(test_sites) - set(fp_test["site"]):
        raise SystemExit("test sites missing from outputs/test/features.csv: run `modal run modal_test_prep.py::main`")
    parents = parent_groups()
    embed_sites.remote([("Batch_test", s) for s in test_sites], "full", True)
    print("distances:", compute_distances.remote(labelled34, test_sites, "full", "Batch_test", "distances_r3.npz"))
    res = evaluate_menu.remote("full", "distances_r3.npz", lab_rec, parents.to_dict("records"),
                               fp_lab.to_dict("records"), fp_test.to_dict("records"), "Batch_test", selection)
    sig = signal_sentence(pd.read_csv(root / "outputs" / "within_parent" / "summary.csv"))
    preds = pd.DataFrame(res["test_predictions"])
    preds["explanation"] = [e.replace(f" (see outputs/patch_mil/figures/heldout_{s}.png)", "") + (" " + sig if sig else "")
                            for s, e in zip(preds["site"], preds["explanation"])]
    out = root / "outputs" / "test"
    _atomic_write(out / "menu_evaluation.json",
                  json.dumps({"summary": res["summary"], "selection": selection}, indent=2).encode())
    # every menu option per site; the final call (selected model fem_a1) and submission.md come from
    # scripts/score_test_all.py
    _atomic_write(out / "menu_predictions.csv", preds.to_csv(index=False).encode())
    for r in preds.itertuples():
        print(f"{r.site} -> {r.assigned} ({r.confidence})")
    print(f"test: wall {time.time() - t0:.0f}s, remote {res['elapsed_s']}s")
