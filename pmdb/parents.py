"""Parent-image grouping of microscopy sites (crops of the same parent electrode image).

parent_id = full-res height + SE detector + BSE grey-level step. Used ONLY to exclude
sibling crops from training banks / CV folds, never as evidence. See docs/patch_mil.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

_PAIRS = (
    ("cache/half/manifest.csv", "outputs/clean/summary.csv"),
    ("cache_heldout/half/manifest.csv", "outputs/clean_heldout/summary.csv"),
)


def parent_id(height: int, se_detector: str, bse_grey_step: int) -> str:
    return f"h{int(height)}_{se_detector}_s{int(bse_grey_step)}"


def parent_groups(root: Path | str | None = None, include_heldout: bool = True) -> pd.DataFrame:
    """Columns [batch, site, parent_id, height, se_detector, bse_grey_step]; labelled rows first
    (cache/half/manifest.csv order), then held-out rows (cache_heldout/half/manifest.csv order)."""
    root = Path(root) if root is not None else ROOT
    frames = []
    for man_rel, sum_rel in _PAIRS[: 2 if include_heldout else 1]:
        man = pd.read_csv(root / man_rel, dtype={"site": str})[["batch", "site", "se_detector"]]
        summ = pd.read_csv(root / sum_rel, dtype={"site": str})[["batch", "site", "height", "BSE_grey_step"]]
        df = man.merge(summ, on=["batch", "site"], how="left")
        missing = df.loc[df["height"].isna(), "site"].tolist()
        if missing:
            raise ValueError(f"no clean summary for {missing}; run scripts/build_clean.py for them first "
                             "(held-out: python scripts/build_clean.py --data-root data_heldout "
                             "--out outputs/clean_heldout --targets outputs/clean/targets.json --workers 3)")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True).rename(columns={"BSE_grey_step": "bse_grey_step"})
    out["height"] = out["height"].astype(int)
    out["bse_grey_step"] = out["bse_grey_step"].astype(int)
    out["parent_id"] = [parent_id(h, d, s) for h, d, s in
                        zip(out["height"], out["se_detector"], out["bse_grey_step"])]
    return out[["batch", "site", "parent_id", "height", "se_detector", "bse_grey_step"]]


def write_parent_groups(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else ROOT / "outputs" / "parent_groups.csv"
    df = parent_groups()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
