"""Save 4-colour segmentation overlays for three images per batch into
qc/segmentation/ for eyeballing before any KPI code runs on the masks.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import ROOT, list_batches, load, load_config, normalise
from src.segment import overlay, segment


def main() -> None:
    cfg = load_config()
    out_dir = ROOT / "qc" / "segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    ncfg = cfg["normalise"]

    for batch, paths in list_batches(cfg).items():
        for path in paths[:3]:
            norm, _, _ = normalise(load(path), ncfg["p_low"], ncfg["p_high"])
            masks = segment(norm, cfg)
            fracs = {k: v.mean() for k, v in masks.items()}
            print(f"{batch} {path.stem}: "
                  + ", ".join(f"{k}={v:.3f}" for k, v in fracs.items()))

            # side-by-side on a crop, full-width overlay below
            fig, axes = plt.subplots(3, 1, figsize=(16, 12))
            axes[0].imshow(norm[:, :2400], cmap="gray", vmin=0, vmax=255)
            axes[0].set_title(f"{batch}/{path.stem} normalised (left 2400 px)")
            axes[1].imshow(overlay(masks)[:, :2400])
            axes[1].set_title("segmentation (black=pore grey=graphite yellow=bright red=rim)")
            axes[2].imshow(overlay(masks))
            axes[2].set_title("full width")
            for a in axes:
                a.axis("off")
            fig.tight_layout()
            fig.savefig(out_dir / f"{batch}_{path.stem}.png", dpi=90)
            plt.close(fig)


if __name__ == "__main__":
    main()
