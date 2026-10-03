"""Random-colour graphite instance overlays for the same three images per
batch as the segmentation QC. Eyeball: flakes should separate where they
touch, without shattering single flakes into fragments.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.instances import graphite_instances
from src.io import ROOT, list_batches, load, load_config, normalise
from src.segment import segment


def main() -> None:
    cfg = load_config()
    rng = np.random.default_rng(cfg["seed"])
    out_dir = ROOT / "qc" / "instances"
    out_dir.mkdir(parents=True, exist_ok=True)
    ncfg = cfg["normalise"]

    for batch, paths in list_batches(cfg).items():
        for path in paths[:3]:
            norm, _, _ = normalise(load(path), ncfg["p_low"], ncfg["p_high"])
            masks = segment(norm, cfg)
            labels, border = graphite_instances(masks, cfg)
            n = labels.max()
            print(f"{batch} {path.stem}: {n} graphite instances, "
                  f"{len(border)} touch border")

            lut = np.vstack([[0, 0, 0], rng.integers(50, 255, (n, 3))]).astype(np.uint8)
            rgb = lut[labels]
            fig, axes = plt.subplots(2, 1, figsize=(16, 9))
            axes[0].imshow(norm[:, :2400], cmap="gray", vmin=0, vmax=255)
            axes[0].set_title(f"{batch}/{path.stem} normalised (left 2400 px)")
            axes[1].imshow(rgb[:, :2400])
            axes[1].set_title(f"graphite instances ({n} total, {len(border)} on border)")
            for a in axes:
                a.axis("off")
            fig.tight_layout()
            fig.savefig(out_dir / f"{batch}_{path.stem}.png", dpi=90)
            plt.close(fig)


if __name__ == "__main__":
    main()
