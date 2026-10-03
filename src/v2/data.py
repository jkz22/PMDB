"""Crop datasets, detector views and grouped field splits (Phase 1/3).

All crops from one field (every detector) share ``group_id = batch/site``; splits are by
group only. Cache arrays (50 nm/px uint8, [BSE, Inlens, SE_type]) are held in RAM.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.v2.common import CROP, SEED, STRIDE_TRAIN, grid, load_half_raw, manifest

VIEWS = {"stack": (0, 1, 2), "BSE": (0, 0, 0), "Inlens": (1, 1, 1), "SE_type": (2, 2, 2)}
BASELINE = "Batch_1"
HELDOUT_FRAC = 0.2


def heldout_split(m: pd.DataFrame | None = None, frac: float = HELDOUT_FRAC, seed: int = SEED) -> pd.DataFrame:
    """Mark ~frac of fields per batch (at least one) as held-out screening fields."""
    m = manifest() if m is None else m.copy()
    rng = np.random.default_rng(seed)
    m["heldout"] = False
    for _, g in m.groupby("batch"):
        k = max(1, int(round(frac * len(g))))
        m.loc[rng.choice(g.index.to_numpy(), k, replace=False), "heldout"] = True
    return m


def grouped_folds(groups: np.ndarray, k: int = 5, seed: int = SEED) -> np.ndarray:
    """Fold index per row; every row of a group gets the same fold."""
    u = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(len(u))
    fold_of = {g: perm[i] % k for i, g in enumerate(u)}
    return np.array([fold_of[g] for g in groups])


def normalise_percentile(img: np.ndarray) -> np.ndarray:
    """Teammate normalisation (pmdb.io): per image & channel, p0.5->0, p99.5->1, clip."""
    x = img.astype(np.float32)
    lo, hi = np.percentile(x.reshape(-1, x.shape[-1]), [0.5, 99.5], axis=0)
    return np.clip((x - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def harmonise_params(per_image: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-image, per-channel linear gain/offset mapping the image's 3-GMM means onto the
    reference (median over all images of that view) by least squares. Weights untouched."""
    from src.v2.common import OUT
    p = pd.read_csv(OUT / "imaging_stats" / "per_image.csv") if per_image is None else per_image
    mus = ["gmm_mu0", "gmm_mu1", "gmm_mu2"]
    ref = p.groupby("view")[mus].median()
    rows = []
    for r in p.itertuples(index=False):
        x, y = np.array([getattr(r, k) for k in mus]), ref.loc[r.view].to_numpy()
        gain, off = np.polyfit(x, y, 1)
        rows.append(dict(group_id=r.group_id, channel=r.channel, gain=gain, offset=off))
    return pd.DataFrame(rows)


class FieldStore:
    """All fields in RAM as float32 [0,1] (raw: /255; norm: teammate percentile).
    ``harmonise`` applies the GMM-mean linear harmonisation to raw counts first."""

    def __init__(self, fields: pd.DataFrame, input_mode: str = "raw", harmonise: bool = False):
        assert input_mode in ("raw", "norm")
        self.fields = fields.reset_index(drop=True)
        hp = harmonise_params().set_index(["group_id", "channel"]) if harmonise else None
        self.images = []
        for b, s, gid in self.fields[["batch", "site", "group_id"]].itertuples(index=False, name=None):
            a = load_half_raw(b, s).astype(np.float32)
            if hp is not None:
                for c in range(3):
                    g, o = hp.loc[(gid, c), ["gain", "offset"]]
                    a[..., c] = np.clip(g * a[..., c] + o, 0, 255)
            self.images.append(normalise_percentile(a) if input_mode == "norm" else a / 255.0)


class CropDataset(Dataset):
    def __init__(self, store: FieldStore, view: str = "stack", stride: int = STRIDE_TRAIN,
                 size: int = CROP, random_offset: bool = False, kpis: pd.DataFrame | None = None,
                 kpi_cols: tuple[str, ...] = ()):
        self.store, self.ch, self.size, self.random_offset = store, list(VIEWS[view]), size, random_offset
        self.index = [(i, y, x) for i, im in enumerate(store.images) for (y, x) in grid(*im.shape[:2], size, stride)]
        self.group = np.array([store.fields.group_id[i] for i, _, _ in self.index])
        self.batch = np.array([store.fields.batch[i] for i, _, _ in self.index])
        self.kpi = None
        if kpis is not None and kpi_cols:
            k = kpis.set_index(["group_id", "y", "x"])[list(kpi_cols)]
            self.kpi = np.stack([k.loc[(g, y, x)].to_numpy(np.float32) for g, (_, y, x) in zip(self.group, self.index)])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, j):
        i, y, x = self.index[j]
        im = self.store.images[i]
        if self.random_offset:  # fixed-scale random crop within +-stride/2, never resized
            h, w = im.shape[:2]
            g = torch.randint(-STRIDE_TRAIN // 2, STRIDE_TRAIN // 2 + 1, (2,))
            y = int(np.clip(y + g[0], 0, h - self.size)); x = int(np.clip(x + g[1], 0, w - self.size))
        c = torch.from_numpy(np.ascontiguousarray(im[y:y + self.size, x:x + self.size][..., self.ch])).permute(2, 0, 1)
        out = {"x": c, "idx": j}
        if self.kpi is not None:
            out["kpi"] = torch.from_numpy(self.kpi[j])
        return out
