"""First-task QC: load every image, assert identical shape/dtype, print
per-image mean/std, and save one overlaid intensity histogram per batch.

Run before any KPI code. If magnification or contrast differ between
images this script is where it must show up.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import ROOT, list_batches, load, load_config


def main() -> None:
    cfg = load_config()
    batches = list_batches(cfg)
    qc_dir = ROOT / "qc"
    qc_dir.mkdir(exist_ok=True)

    shapes: dict[tuple, list[str]] = {}
    print(f"{'batch':<10} {'image':<24} {'shape':<14} {'dtype':<6} {'mean':>7} {'std':>7}")
    for batch, paths in batches.items():
        fig, ax = plt.subplots(figsize=(10, 4))
        for path in paths:
            img = load(path)
            shapes.setdefault((img.shape, str(img.dtype)), []).append(path.name)
            print(f"{batch:<10} {path.stem:<24} {str(img.shape):<14} {img.dtype} "
                  f"{img.mean():7.2f} {img.std():7.2f}")
            hist, _ = np.histogram(img, bins=256, range=(0, 256), density=True)
            ax.plot(np.arange(256), hist, alpha=0.5, lw=0.8)
        ax.set(title=f"{batch}: raw intensity histograms ({len(paths)} images)",
               xlabel="intensity", ylabel="density")
        fig.tight_layout()
        fig.savefig(qc_dir / f"hist_{batch}.png", dpi=120)
        plt.close(fig)

    print()
    if len(shapes) == 1:
        (shape, dtype), names = next(iter(shapes.items()))
        print(f"OK: all {len(names)} images share shape {shape} and dtype {dtype}")
    else:
        print("WARNING: images differ in shape/dtype - STOP and flag before any KPI work:")
        for (shape, dtype), names in shapes.items():
            print(f"  {shape} {dtype}: {len(names)} images, e.g. {names[0]}")

    for batch, paths in batches.items():
        print(f"{batch}: {len(paths)} {cfg['data']['detector']} images -> qc/hist_{batch}.png")


if __name__ == "__main__":
    main()
