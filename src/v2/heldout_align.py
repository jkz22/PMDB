"""Held-out alignment: every saved classifier family (fold models averaged) scores the 3 held-out sites;
compared with the organisers' labels. Writes outputs/v2/heldout_align.csv."""
from __future__ import annotations

import glob
import json
import sys

import numpy as np
import pandas as pd
import torch

from src.v2.classify import Classifier, classes_of
from src.v2.common import OUT
from src.v2.sae_ablate import heldout_crops

TRUTH = {"3e122cbj": "Batch_2", "fn0mhxef": "Batch_1", "xrv9xvzb": "Batch_3"}


def main(archs=("resnet18_imnet", "dinov2_ft", "effb4_micronet", "effb4_imnet")):
    groups: dict[tuple, list] = {}
    for p in sorted(glob.glob(str(OUT / "cls_runs" / "*" / "final.pt"))):
        cfg = json.load(open(p.replace("final.pt", "config.json")))
        if cfg["arch"] not in archs:
            continue
        k = (cfg.get("labels", "batch"), cfg["arch"], cfg["input"], str(cfg["harmonise"]), cfg["view"], cfg.get("exclude", ""))
        groups.setdefault(k, []).append((p, cfg))
    rows = []
    dev = "cpu"
    for k in sorted(groups, key=lambda t: (archs.index(t[1]), t)):
        runs = groups[k]; cfg = runs[0][1]; cls = classes_of(cfg)
        try:
            X, meta = heldout_crops(cfg)
        except FileNotFoundError as e:
            print('skip', k, e.filename, flush=True); continue
        P = np.zeros((len(X), len(cls)))
        for p, c in runs:
            m = Classifier(c["arch"], n_cls=len(cls)); m.load_state_dict(torch.load(p, map_location=dev)); m.eval()
            with torch.no_grad():
                for i in range(0, len(X), 32):
                    xb = torch.from_numpy(X[i:i + 32]).permute(0, 3, 1, 2).float()
                    P[i:i + 32] += torch.softmax(m(xb), 1).numpy()
        P /= len(runs)
        for site in sorted(TRUTH):
            mm = (meta.site == site).to_numpy()
            pf = P[mm].mean(0); pred = cls[int(pf.argmax())]
            truth = TRUTH[site] if cfg.get("labels", "batch") == "batch" else ("Batch_3" if TRUTH[site] == "Batch_3" else "Batch_1+2")
            rows.append(dict(labels=k[0], arch=k[1], input=k[2], harmonise=k[3], view=k[4], exclude=bool(k[5]), n_models=len(runs), site=site,
                             truth=truth, pred=pred, correct=pred == truth, **{f"P_{c}": round(float(v), 3) for c, v in zip(cls, pf)}))
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(OUT / "heldout_align.csv", index=False)


if __name__ == "__main__":
    main(*(tuple(sys.argv[1].split(",")),) if len(sys.argv) > 1 else ())
