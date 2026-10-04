"""Collect the parent-fold shift sweeps (shift_cls, spechyb_cls) from the Modal volume into one table."""
import json, os, subprocess, sys
import numpy as np, pandas as pd
from src.v2.classify import cfg_hash
pd.set_option("display.width", 250)
rows = []
for mode in ("shift_cls", "spechyb_cls", "micronet_cls"):
    for s in json.load(open(f"outputs/v2/{mode}_specs.json")):
        h = cfg_hash(s); d = f"outputs/v2/cls_runs/{h}"; os.makedirs(d, exist_ok=True)
        if not os.path.exists(f"{d}/metrics.json"):
            subprocess.run(["modal", "volume", "get", "pmdb-v2", f"cls_runs/{h}/metrics.json", f"{d}/metrics.json", "--force"], capture_output=True)
        if not os.path.exists(f"{d}/metrics.json"): continue
        m = json.load(open(f"{d}/metrics.json"))
        r = dict(arch=m["arch"][:6], labels=m["labels"], harm=m["harmonise"], fold=m["fold"])
        if m["n_test_fields"] == 0:
            r.update(correct=m["heldout_correct"], **{f"{x['site'][:2]}_p_true": round(x[f"p_{x['truth']}"], 2) for x in m["heldout"] if x["correct"] is not None})
        else:
            cm = np.array(m["field_confusion"]); r.update(n=m["n_test_fields"], field_acc=m["field_acc"], cm=cm.tolist())
        rows.append(r)
df = pd.DataFrame(rows); df.to_csv("outputs/v2/off/shift_sweep.csv", index=False)
f = df[df.fold >= 0]
if len(f):
    for (arch, lab, harm), g in f.groupby(["arch", "labels", "harm"]):
        cm = np.sum([np.array(c) for c in g.cm], axis=0); rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
        cls = {"off": ["B1+2", "B3"], "batch": ["B1", "B2", "B3"], "b12": ["B1", "B2"]}[lab]
        print(f"{arch} {lab:6s} {harm:9s} folds {len(g)}  n {cm.sum():2d}  acc {np.trace(cm)/cm.sum():.2f}  bal {rec.mean():.2f}  recall " + " ".join(f"{c}={r:.2f}" for c, r in zip(cls, rec)))
print(df[df.fold == -1].drop(columns=[c for c in df.columns if c in ("n", "field_acc", "cm")]).to_string(index=False))
