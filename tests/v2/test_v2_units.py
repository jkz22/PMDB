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


def test_selection_gates_runs_without_kpi_signal():
    lb = pd.DataFrame({"kpi_r2": [-0.1, 0.4], "img_r2": [0.0, 0.3], "image_id_ratio": [1.0, 2.0],
                       "lift_shift": [0.0, 0.5], "recon_kpi_err": [np.nan, np.nan]})
    s = selection_score(lb)
    assert s[1] < s[0]


def test_phase_mask_weights():
    """KPI-phase-driven VAE loss: 'inpaint' hides one minority phase (dilated) and splits the loss half/half
    between hidden and visible pixels; 'weight' keeps the input and gives each phase a third of the loss."""
    import torch
    from src.v2.data import PHASE_GRAPHITE, PHASE_PORE, PHASE_SI
    from src.v2.models import VAE
    torch.manual_seed(0)
    x = torch.rand(2, 3, 64, 64)
    ph = torch.full((2, 64, 64), PHASE_GRAPHITE, dtype=torch.uint8)
    ph[:, :16, :16] = PHASE_SI
    ph[:, 40:, 40:] = PHASE_PORE
    ph[1, :16, :16] = PHASE_GRAPHITE  # crop 1 has no Si: the inpaint mask must fall back to pores
    xin, w = VAE("A", phase_mask="inpaint")._phase_weights(x, ph)
    hidden = (xin == 0).all(1)
    assert hidden[0].float().mean() > 0.05 and hidden[1].float().mean() > 0.05
    assert torch.allclose(w.mean((1, 2, 3)), torch.ones(2), atol=1e-3)
    assert torch.allclose(w[:, 0][hidden].sum() / w.numel(), torch.tensor(0.5), atol=1e-3)
    xin, w = VAE("A", phase_mask="weight")._phase_weights(x, ph)
    assert torch.equal(xin, x)
    assert abs(w[0, 0][ph[0] == PHASE_SI].sum() / w[0].numel() - 1 / 3) < 1e-3
    assert abs(w[0, 0][ph[0] == PHASE_PORE].sum() / w[0].numel() - 1 / 3) < 1e-3
    assert abs(w[1].mean() - 2 / 3) < 1e-3  # a missing phase simply contributes nothing


def test_naive_transform_rescales_and_is_deterministic():
    import torch
    from src.v2.data import NAIVE_SIGMA, naive_transform

    x = torch.rand(2, 3, 64, 64) * 0.3 + 0.2  # low-contrast crops on different grey ranges
    x[1] = x[1] * 0.5 + 0.4
    g = torch.Generator().manual_seed(0)
    y = naive_transform(x, g)
    y2 = naive_transform(x, torch.Generator().manual_seed(0))
    assert torch.equal(y, y2)
    clean = naive_transform(x, torch.Generator().manual_seed(0), sigma=0.0)
    for b in range(2):
        q = torch.quantile(clean[b].flatten(1), torch.tensor([0.01, 0.99]), dim=1)
        assert torch.allclose(q[0], torch.zeros(3), atol=0.02) and torch.allclose(q[1], torch.ones(3), atol=0.02)
    assert abs(float((y - clean).std()) - NAIVE_SIGMA) < 0.01


def test_extreme_blur_and_binary_labels():
    import pandas as pd
    import torch
    from src.v2.classify import _labels, cfg_hash, classes_of
    from src.v2.data import naive_transform

    x = torch.rand(1, 3, 64, 64)
    g = torch.Generator().manual_seed(0)
    sharp = naive_transform(x, g, sigma=0.0)
    soft = naive_transform(x, g, sigma=0.0, blur=1.0)
    assert soft.shape == x.shape and float(soft.std()) < 0.5 * float(sharp.std())
    f = pd.DataFrame({"batch": ["Batch_1", "Batch_2", "Batch_3"]})
    assert _labels(f, {"labels": "off"}).tolist() == [0, 0, 1] and _labels(f).tolist() == [0, 1, 2]
    assert classes_of({"labels": "off"}) == ("Batch_1+2", "Batch_3")
    base = dict(task="cls", arch="resnet18_imnet", view="stack", harmonise="hybrid", fold=0)
    assert cfg_hash(base) == cfg_hash({**base, "labels": "batch"}) != cfg_hash({**base, "labels": "off"})


import pytest as _pytest


@_pytest.mark.data
def test_ext_harmonised_arrays_align_with_cache_grid():
    """Imported nyul/basic arrays (pmdb.harmonise_ext) keep the 4-px full-res border; the loader trims
    2 px/side so crops index the same pixels as cache/half (and the KPI / imaging-stat tables)."""
    import numpy as np
    import pytest
    from src.v2.common import REPO, load_half_raw

    p = REPO / "cache" / "harmonised_ext" / "basic" / "half" / "Batch_1__4ih2ggld.npz"
    if not p.exists():
        pytest.skip("imported harmonisation arrays not built (scripts/build_harmonise_ext.py)")
    raw = load_half_raw("Batch_1", "4ih2ggld", harm="none").astype(float)
    for h in ("basic", "nyul"):
        a = load_half_raw("Batch_1", "4ih2ggld", harm=h)
        assert a.shape == raw.shape and a.dtype == np.uint8
    b = load_half_raw("Batch_1", "4ih2ggld", harm="basic").astype(float)
    win = (slice(200, 800), slice(500, 3000), 0)
    assert np.corrcoef(b[win].ravel(), raw[win].ravel())[0, 1] > 0.99
