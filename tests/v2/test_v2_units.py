import numpy as np
import pandas as pd
import torch

from src.v2.augment import ForwardRanges, augment, black_level_lift, patch_mask
from src.v2.common import grid
from src.v2.data import VIEWS, grouped_folds, heldout_split
from src.v2.evaluate import selection_score
from src.v2.models import VAE
from src.v2.rungrid import stage_a, stage_b, train_cfg
from src.v2.train import cfg_hash


def _manifest():
    rows = [dict(batch=b, site=f"{b}_{i}", group_id=f"{b}/{b}_{i}") for b, n in (("Batch_1", 7), ("Batch_2", 7), ("Batch_3", 17)) for i in range(n)]
    return pd.DataFrame(rows)


def test_grid_eval_non_overlapping():
    g = grid(600, 1000, 256, 256)
    assert all(y + 256 <= 600 and x + 256 <= 1000 for y, x in g)
    assert len(set(g)) == len(g) == 2 * 3


def test_heldout_per_batch_and_folds_grouped():
    m = heldout_split(_manifest())
    assert m.groupby("batch").heldout.sum().to_dict() == {"Batch_1": 1, "Batch_2": 1, "Batch_3": 3}
    groups = np.repeat(m.group_id.to_numpy(), 4)
    f = grouped_folds(groups, 5)
    assert pd.DataFrame({"g": groups, "f": f}).groupby("g").f.nunique().max() == 1


def test_views_single_detector_replicated():
    assert VIEWS["BSE"] == (0, 0, 0) and VIEWS["stack"] == (0, 1, 2)


def test_augment_shapes_and_range():
    x = torch.rand(4, 3, 64, 64)
    for lvl in ("aug0", "aug1", "aug2"):
        y = augment(x, lvl, ForwardRanges(ksize=7), torch.Generator().manual_seed(0))
        assert y.shape == x.shape and float(y.min()) >= 0 and float(y.max()) <= 1
    xm, m = patch_mask(x, 0.5, 16, torch.Generator().manual_seed(0))
    assert abs(float(m.mean()) - 0.5) < 0.2
    assert torch.all(black_level_lift(torch.zeros(1, 3, 8, 8), 0.1) == 0.1)


def test_vae_missing_kpi_is_masked_not_filled():
    torch.manual_seed(0)
    v = VAE("C", n_kpi=5)
    kpi = torch.rand(2, 5); kpi[:, 4] = float("nan")
    loss, logs = v.loss({"x": torch.rand(2, 3, 256, 256), "kpi": kpi})
    assert torch.isfinite(loss) and np.isfinite(logs["aux"])


def test_run_grid_fixed_design():
    a, b = stage_a(), stage_b()
    assert len(a) == 6 and len(b) == 46
    assert not any(r["family"] == "dino_ft" and r.get("aug") == "aug0" for r in b)
    assert len({cfg_hash(train_cfg(r)) for r in a + b}) == len(a + b)


def test_selection_rank_directions():
    lb = pd.DataFrame(dict(kpi_r2=[0.9, 0.1], img_r2=[0.0, 0.5], image_id_ratio=[1.0, 3.0], lift_shift=[0.1, 0.5],
                           recon_kpi_err=[np.nan, np.nan]))
    s = selection_score(lb)
    assert s[0] < s[1]
