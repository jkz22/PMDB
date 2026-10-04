"""Crop datasets, detector views and grouped field splits (Phase 1/3).

All crops from one field (every detector) share ``group_id = batch/site``; splits are by
group only. Cache arrays (50 nm/px uint8, [BSE, Inlens, SE_type]) are held in RAM.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.v2.common import CLEAN_METHODS, CROP, SEED, STRIDE_TRAIN, grid, harm_method, load_half_clean, load_half_raw, manifest

VIEWS = {"stack": (0, 1, 2), "BSE": (0, 0, 0), "Inlens": (1, 1, 1), "SE_type": (2, 2, 2)}
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


def stratified_group_folds(groups: np.ndarray, labels: np.ndarray, k: int, seed: int = SEED) -> np.ndarray:
    """Fold id per row: fields of each class are shuffled and dealt round-robin to the k folds,
    so every fold holds >=1 field of every batch (7/7/17 fields -> 1-2 / 1-2 / 3-4 per fold)."""
    rng = np.random.default_rng(seed)
    fold_of = {}
    for lab in sorted(set(labels)):
        fields = sorted(set(groups[labels == lab]))
        rng.shuffle(fields)
        for j, f in enumerate(fields):
            fold_of[f] = (j + rng.integers(k)) % k if j == 0 else (fold_of[fields[0]] + j) % k
    return np.array([fold_of[g] for g in groups])


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
    return np.clip((x - lo) / np.maximum(hi - lo, 1e-6), 0, 1).astype(np.float32)


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
    ``harmonise``: False/'none' raw counts; True/'gmm' this pipeline's GMM-mean linear map applied to
    raw counts; or a pmdb.harmonise LUT method ('hybrid', 'affine2', 'histmatch', ...) read from
    cache/harmonised/<method>/half (PR #16)."""

    def __init__(self, fields: pd.DataFrame, input_mode: str = "raw", harmonise=False, phase: bool = False):
        assert input_mode in ("raw", "norm", "naive")
        self.fields = fields.reset_index(drop=True)
        self.naive = input_mode == "naive"
        self.harm = harm_method(harmonise)
        hp = harmonise_params().set_index(["group_id", "channel"]) if self.harm == "gmm" else None
        self.images, self.valid, self.phase = [], [], ([] if phase else None)
        for b, s, gid in self.fields[["batch", "site", "group_id"]].itertuples(index=False, name=None):
            if self.harm in CLEAN_METHODS:
                a, v = load_half_clean(b, s, self.harm)
                a = a.astype(np.float32)
                self.valid.append(v.all(-1))
            else:
                a = load_half_raw(b, s, self.harm).astype(np.float32)
                self.valid.append(None)
            if hp is not None:
                for c in range(3):
                    g, o = hp.loc[(gid, c), ["gain", "offset"]]
                    a[..., c] = np.clip(g * a[..., c] + o, 0, 255)
            if phase or self.naive:
                lab = phase_labels(a[..., 0])
                if phase:
                    self.phase.append(lab)
                if self.naive:  # keep only pore / Si / graphite pixels; everything else -> 0 (pore level)
                    a[lab == 0] = 0.0
            self.images.append(normalise_percentile(a) if input_mode == "norm" else a / 255.0)


NAIVE_SIGMA = 0.1  # ~25 grey levels of Gaussian noise on the per-crop p1-p99 rescaled [0,1] image


def naive_transform(x: torch.Tensor, gen: torch.Generator, sigma: float = NAIVE_SIGMA) -> torch.Tensor:
    """Naive harmonisation baseline, applied per crop and channel at train *and* test time: rescale to the
    crop's own p1-p99 (clipped to [0,1], ~min-max) and add N(0, sigma) noise so that grey level, gain and
    the fine noise/sharpness texture are all swamped. ``x``: (B, C, H, W) float in [0,1]."""
    flat = x.flatten(2)
    q = torch.quantile(flat, torch.tensor([0.01, 0.99], device=x.device, dtype=x.dtype), dim=2)
    lo, hi = q[0][..., None, None], q[1][..., None, None]
    y = ((x - lo) / (hi - lo).clamp_min(1e-3)).clamp(0, 1)
    return y + sigma * torch.randn(y.shape, generator=gen, device=y.device, dtype=y.dtype)


PHASE_SI, PHASE_GRAPHITE, PHASE_PORE = 1, 2, 3  # 0 = artefact / unassigned


def phase_labels(bse: np.ndarray) -> np.ndarray:
    """Teammate KPI segmentation (pmdb.segment at the pinned KPI commit) of a whole BSE field on the grey
    scale the model sees (harmonised / clean), as a uint8 label map: 1 Si, 2 graphite, 3 pore, 0 other."""
    from src.v2.kpi_adapter import segment
    m = segment(bse, 50.0)
    lab = np.zeros(bse.shape, np.uint8)
    lab[m.graphite] = PHASE_GRAPHITE
    lab[m.pore] = PHASE_PORE
    lab[m.si] = PHASE_SI
    return lab


class GPUCropLoader:
    """Device-resident replacement for DataLoader(CropDataset, shuffle=True, drop_last=True): every
    field is uploaded once and crops are sliced on the device, so no CPU workers or host copies.
    Same sampling as CropDataset (epoch-wise shuffle, +-stride/2 fixed-scale offsets, no resize)."""

    def __init__(self, ds: "CropDataset", batch_size: int, dev, generator: torch.Generator):
        self.ds, self.bs, self.dev, self.g = ds, batch_size, dev, generator
        self.images = [torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))).to(dev) for im in ds.store.images]
        self.ch = torch.tensor(ds.ch, device=dev)
        self.idx = torch.tensor(ds.index, dtype=torch.long)
        self.hw = torch.tensor([im.shape[:2] for im in ds.store.images], dtype=torch.long)
        self.kpi = torch.from_numpy(ds.kpi).to(dev) if ds.kpi is not None else None
        self.phase = [torch.from_numpy(p).to(dev) for p in ds.store.phase] if getattr(ds.store, "phase", None) else None
        self.naive = getattr(ds.store, "naive", False)
        if self.naive:
            self.ng = torch.Generator(device=dev)
            self.ng.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=generator)))

    def __len__(self):
        return len(self.ds) // self.bs

    def __iter__(self):
        S = self.ds.size
        perm = torch.randperm(len(self.ds), generator=self.g)
        for k in range(len(self)):
            j = perm[k * self.bs:(k + 1) * self.bs]
            i, y, x = self.idx[j].T
            if self.ds.random_offset:
                o = torch.randint(-STRIDE_TRAIN // 2, STRIDE_TRAIN // 2 + 1, (2, len(j)), generator=self.g)
                h, w = self.hw[i].T
                y = torch.minimum((y + o[0]).clamp_min(0), h - S); x = torch.minimum((x + o[1]).clamp_min(0), w - S)
            xs = torch.stack([self.images[a][:, b:b + S, c:c + S] for a, b, c in zip(i.tolist(), y.tolist(), x.tolist())])
            out = {"x": xs.index_select(1, self.ch), "idx": j.to(self.dev)}
            if self.naive:
                out["x"] = naive_transform(out["x"], self.ng)
            if self.kpi is not None:
                out["kpi"] = self.kpi[j.to(self.dev)]
            if self.phase is not None:
                out["phase"] = torch.stack([self.phase[a][b:b + S, c:c + S] for a, b, c in zip(i.tolist(), y.tolist(), x.tolist())])
            yield out


MIN_VALID = 0.9  # clean route: drop crops with < 90 % valid pixels (border/collector/free surface/charging)


def _valid_frac(valid, y, x, size):
    return 1.0 if valid is None else float(valid[y:y + size, x:x + size].mean())


class CropDataset(Dataset):
    def __init__(self, store: FieldStore, view: str = "stack", stride: int = STRIDE_TRAIN,
                 size: int = CROP, random_offset: bool = False, kpis: pd.DataFrame | None = None,
                 kpi_cols: tuple[str, ...] = (), min_valid: float = MIN_VALID):
        self.store, self.ch, self.size, self.random_offset = store, list(VIEWS[view]), size, random_offset
        valid = getattr(store, "valid", None) or [None] * len(store.images)
        self.index = [(i, y, x) for i, im in enumerate(store.images) for (y, x) in grid(*im.shape[:2], size, stride)
                      if _valid_frac(valid[i], y, x, size) >= min_valid]
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
        if getattr(self.store, "naive", False):  # deterministic per-crop noise at evaluation time
            c = naive_transform(c[None], torch.Generator().manual_seed(1_000_003 * j + 17))[0]
        out = {"x": c, "idx": j}
        if self.kpi is not None:
            out["kpi"] = torch.from_numpy(self.kpi[j])
        if getattr(self.store, "phase", None):
            out["phase"] = torch.from_numpy(np.ascontiguousarray(self.store.phase[i][y:y + self.size, x:x + self.size]))
        return out
