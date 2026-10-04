"""Aggregate outputs/v2/sae_ablation/<hash>/ into outputs/v2/sae_ablation/REPORT_tables.md."""
from __future__ import annotations

import json

import pandas as pd

from src.v2.common import OUT

ABL = OUT / "sae_ablation"
NAMES = {"489232073a90": "VAE-C raw", "863083fb14d9": "VAE-C hybrid", "fac1e9fbbfe4": "VAE-C clean norm",
         "9ebbdbfa63f1": "VAE-C clean harm", "a5bd3c5bd3c0": "VAE-C hybrid phase-inpaint", "883e58f22bcf": "VAE-C hybrid phase-weight",
         "6beb3d6c9de9": "DINO-FT clean norm", "2fb5ff30ff3e": "DINO-FT clean harm",
         "b6422cae3813": "MAE-adapted clean norm", "cebd80163f5a": "MAE-adapted clean harm"}
VARS = ["orig", "sae_recon", "abl_imaging_pure", "abl_imaging_strong", "abl_imaging", "abl_imaging+mixed", "abl_random_matched", "abl_material"]


def main():
    v = pd.concat([pd.read_csv(p) for p in ABL.glob("*/variants.csv")])
    h = pd.concat([pd.read_csv(p) for p in ABL.glob("*/heldout.csv")])
    v["model"] = v.run.map(NAMES).fillna(v.run)
    h["model"] = h.run.map(NAMES).fillna(h.run)
    v.to_csv(ABL / "variants_all.csv", index=False)
    h.to_csv(ABL / "heldout_all.csv", index=False)
    md = ["# SAE imaging-feature ablation: tables\n"]
    md.append("## Batch probe (LOFO logistic regression on the embedding), KPI probe and imaging-stat probe per variant\n")
    md.append("`n_feat` features zeroed carrying `mass` of the SAE activation mass. bal_acc chance = 0.33. kpi_r2 = mean LOFO ridge R2 over the 5 gated KPIs; imaging_r2 = mean over p1 / noise / sharpness x 3 detectors.\n")
    for run, g in v.groupby("run", sort=False):
        g = g.set_index("variant").loc[[x for x in VARS if x in g.variant.values]]
        md.append(f"### {NAMES.get(run, run)} (`{run}`)\n")
        t = g[["n_feat", "mass_frac", "field_acc", "bal_acc", "recall_Batch_1", "recall_Batch_2", "recall_Batch_3", "share_Batch_3", "kpi_r2", "r2_frac_si", "r2_frac_pore", "imaging_r2", "r2_p1", "r2_noise_sigma", "r2_sharpness"]]
        t.columns = ["n_feat", "mass", "field acc", "bal acc", "rec B1", "rec B2", "rec B3", "pred share B3", "KPI R2", "R2 Si", "R2 pore", "imaging R2", "R2 p1", "R2 noise", "R2 sharp"]
        md.append(t.round(3).to_markdown() + "\n")
    md.append("## Held-out sites: P(batch) from a probe fitted on all 31 labelled fields, before / after ablation\n")
    for run, g in h.groupby("run", sort=False):
        md.append(f"### {NAMES.get(run, run)} (`{run}`)\n")
        t = g.pivot(index="site", columns="variant", values=["pred", "P_Batch_1", "P_Batch_2", "P_Batch_3"])
        t = t.reindex(columns=[c for c in ["orig", "abl_imaging_pure", "abl_imaging", "abl_imaging+mixed"] if c in g.variant.values], level=1)
        md.append(t.round(2).to_markdown() + "\n")
    s = []
    for p in ABL.glob("*/summary.json"):
        j = json.load(open(p))
        s.append(dict(model=NAMES.get(j["run"], j["run"]), r2_recon=round(j["r2_recon"], 3), n_imaging=j["n_imaging"], mass_imaging=round(j["mass_imaging"], 3), n_flips=j["n_flips"]))
    md.insert(1, "## SAE fit\n\n" + pd.DataFrame(s).to_markdown(index=False) + "\n")
    (ABL / "REPORT_tables.md").write_text("\n".join(md))
    print("\n".join(md)[:6000])


if __name__ == "__main__":
    main()
