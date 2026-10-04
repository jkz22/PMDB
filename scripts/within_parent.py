"""Within-parent contrast analysis on the 34 labelled sites. Writes outputs/within_parent/{contrasts,summary}.csv
and within_parent.png. Run: python scripts/within_parent.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pmdb import within_parent as wp  # noqa: E402
from pmdb.batch_menu import feature_table, labelled_sites  # noqa: E402
from pmdb.parents import parent_groups  # noqa: E402


def main() -> None:
    lab = labelled_sites()
    keys = list(zip(lab.batch, lab.site))
    F = feature_table(keys)
    pg = parent_groups(include_test=False).drop_duplicates(["batch", "site"]).set_index(["batch", "site"])
    parent = pg.loc[keys, "parent_id"]
    C = wp.parent_contrasts(F, lab["label"], parent)
    S = wp.summarise_contrasts(C, F.std(ddof=1))
    out = ROOT / "outputs" / "within_parent"
    out.mkdir(parents=True, exist_ok=True)
    C.to_csv(out / "contrasts.csv", index=False)
    S.to_csv(out / "summary.csv", index=False)

    feats = list(F.columns)
    pairs = [f"{a}-{b}" for a, b in wp.PAIRS]
    M = np.full((len(feats), 3), np.nan)
    ann = [[""] * 3 for _ in feats]
    for r in S.itertuples():
        i, j = feats.index(r.feature), pairs.index(r.pair)
        M[i, j] = r.median_diff_sd
        ann[i][j] = f"{'+' if r.n_pos >= r.n_neg else '-'}{max(r.n_pos, r.n_neg)}/{r.n_parents}"
    lim = np.nanmax(np.abs(M))
    fig, ax = plt.subplots(figsize=(6.5, 7))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(3), pairs)
    ax.set_yticks(range(len(feats)), feats)
    for i in range(len(feats)):
        for j in range(3):
            ax.text(j, i, ann[i][j], ha="center", va="center", fontsize=7)
    ax.set_title("Median within-parent difference (first batch minus second), in feature SD\n"
                 "cell text: majority sign, parents with that sign / parents")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "within_parent.png", dpi=130)

    print(S.groupby("pair")["n_parents"].max().to_dict())
    print(S[S["is_signal"]].to_string())
    print("sentence:", wp.signal_sentence(S) or "(no signal rows)")


if __name__ == "__main__":
    main()
