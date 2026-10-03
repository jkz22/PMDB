"""Segmentation check figure: BSE crop next to the phase overlay for 6 sites across batches.
Includes a Batch_3 black-level-offset site (71vgq3fw) and an SE-detector site (x77cy643)."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from pmdb.io import load_site
from seg import segment, overlay

SITES = [("Batch_1", "4ih2ggld"), ("Batch_1", "5n1q8atc"), ("Batch_2", "epqdaau9"),
         ("Batch_3", "71vgq3fw"), ("Batch_3", "x77cy643"), ("Batch_3", "hzumfsms")]
fig, ax = plt.subplots(len(SITES), 2, figsize=(11.2, 16.8))
for r, (b, s) in enumerate(SITES):
    site = load_site(b, s, resolution="half", normalise="none")
    bse = site.image[..., 0]; lab, info = segment(bse)
    crop = (slice(0, 400), slice(0, 600))
    ax[r, 0].imshow(bse[crop], cmap="gray", vmin=0, vmax=255); ax[r, 0].set_title(f"{b} {s} BSE ({site.se_detector})", fontsize=9)
    ax[r, 1].imshow(overlay(bse, lab)[crop]); ax[r, 1].set_title(f"blue pore, orange Si (thr {info['si_thr']:.2f})", fontsize=9)
    for a in ax[r]: a.axis("off")
fig.tight_layout(); fig.savefig("overlays.png", dpi=100)
im = Image.open("overlays.png").convert("RGB"); w = 1000
im.resize((w, int(im.size[1] * w / im.size[0])), Image.LANCZOS).save("overlays_small.jpg", quality=82)
