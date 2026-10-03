import numpy as np
import pandas as pd
import torch

from src.v2.data import CropDataset, GPUCropLoader


class _Store:
    def __init__(self):
        rng = np.random.default_rng(0)
        self.images = [rng.random((300, 520, 3), dtype=np.float32), rng.random((400, 300, 3), dtype=np.float32)]
        self.fields = pd.DataFrame({"group_id": ["B/a", "B/b"], "batch": ["B", "B"]})


def test_gpu_loader_matches_dataset():
    ds = CropDataset(_Store(), "BSE", stride=128)
    dl = GPUCropLoader(ds, 2, torch.device("cpu"), torch.Generator().manual_seed(0))
    n = 0
    for b in dl:
        assert b["x"].shape == (2, 3, 256, 256)
        for k, j in enumerate(b["idx"].tolist()):
            assert torch.equal(b["x"][k], ds[j]["x"])
        n += 1
    assert n == len(ds) // 2


def test_gpu_loader_random_offset_in_bounds():
    ds = CropDataset(_Store(), "stack", stride=128, random_offset=True)
    dl = GPUCropLoader(ds, 3, torch.device("cpu"), torch.Generator().manual_seed(1))
    for _ in range(5):
        for b in dl:
            assert b["x"].shape == (3, 3, 256, 256)
