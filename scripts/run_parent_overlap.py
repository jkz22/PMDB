"""Verify the parent-image hypothesis from pixels.

main's `pmdb/parents.py` infers that sites sharing (full-res height, SE detector, BSE grey
step) are crops of one parent electrode image. That key is really an *acquisition* key: two
different electrodes imaged in the same session could share it. This script asks the pixels:

1. Phase correlation (BSE, 1/4-scale, zero-padded): do two same-key sites overlap? A real
   overlap gives a sharp translation peak; unrelated images give a flat correlation surface.
2. Edge adjacency: Pearson r between the last column of A and the first column of B (and
   vice versa) compared with the null of r between random non-adjacent columns of the same
   two images. Adjacent crops of a continuous image share neighbouring pixels (r >> 0).

Different-height pairs (necessarily different parents) serve as the negative control, cropped
to the common height. Writes outputs/parents/overlap_pairs.csv and overlap_summary.md.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pmdb.io import load_site  # noqa: E402

OUT = ROOT / "outputs" / "parents"
DS = 4           # extra downsampling of half-res (-> 200 nm/px)
EDGE_W = 4       # columns averaged for the edge test (half-res px)
N_NULL_COLS = 200
RNG = np.random.default_rng(0)


def bse(batch: str, site: str) -> np.ndarray:
    kw = {"data_root": ROOT / "data_heldout", "cache_root": ROOT / "cache_heldout"} if batch == "Batch_heldout" else {}
    img = load_site(batch, site, resolution="half", normalise="none", **kw).image[..., 0].astype(np.float32)
    return img


def small(img: np.ndarray) -> np.ndarray:
    h, w = (img.shape[0] // DS) * DS, (img.shape[1] // DS) * DS
    s = img[:h, :w].reshape(h // DS, DS, w // DS, DS).mean(axis=(1, 3))
    s = (s - s.mean()) / (s.std() + 1e-9)
    win = np.outer(np.hanning(s.shape[0]), np.hanning(s.shape[1]))
    return s * win


def phase_corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int, int]:
    """Peak-to-sidelobe ratio of the zero-padded phase correlation and the peak shift."""
    H, W = a.shape[0] + b.shape[0], a.shape[1] + b.shape[1]
    Fa, Fb = np.fft.rfft2(a, (H, W)), np.fft.rfft2(b, (H, W))
    R = Fa * np.conj(Fb)
    R /= np.abs(R) + 1e-9
    c = np.fft.irfft2(R, (H, W))
    pk = np.unravel_index(np.argmax(c), c.shape)
    peak = c[pk]
    mask = np.ones_like(c, bool)
    y0, x0 = pk
    mask[max(0, y0 - 3):y0 + 4, max(0, x0 - 3):x0 + 4] = False
    side = c[mask]
    psr = (peak - side.mean()) / (side.std() + 1e-12)
    dy = y0 if y0 <= H // 2 else y0 - H
    dx = x0 if x0 <= W // 2 else x0 - W
    return float(psr), float(peak), int(dy * DS), int(dx * DS)


def edge_r(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """r(last cols of a, first cols of b), r(last of b, first of a), and the null 95th pct."""
    h = min(a.shape[0], b.shape[0])
    a, b = a[:h], b[:h]
    ra = np.corrcoef(a[:, -EDGE_W:].mean(1), b[:, :EDGE_W].mean(1))[0, 1]
    rb = np.corrcoef(b[:, -EDGE_W:].mean(1), a[:, :EDGE_W].mean(1))[0, 1]
    null = []
    for _ in range(N_NULL_COLS):
        i = RNG.integers(EDGE_W, a.shape[1] - EDGE_W)
        j = RNG.integers(EDGE_W, b.shape[1] - EDGE_W)
        null.append(np.corrcoef(a[:, i:i + EDGE_W].mean(1), b[:, j:j + EDGE_W].mean(1))[0, 1])
    return float(ra), float(rb), float(np.percentile(null, 95))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pg = pd.read_csv(ROOT / "outputs" / "parent_groups.csv", dtype=str)
    imgs = {r.site: bse(r.batch, r.site) for r in pg.itertuples()}
    sm = {s: small(v) for s, v in imgs.items()}
    key = dict(zip(pg.site, pg.parent_id))
    rows = []
    for a, b in itertools.combinations(pg.site, 2):
        same = key[a] == key[b]
        ia, ib = imgs[a], imgs[b]
        h = min(ia.shape[0], ib.shape[0])
        psr, peak, dy, dx = phase_corr(sm[a][: h // DS], sm[b][: h // DS])
        ra, rb, r95 = edge_r(ia, ib)
        rows.append(dict(site_a=a, site_b=b, parent_a=key[a], parent_b=key[b], same_key=same,
                         psr=psr, peak=peak, shift_z_px=dy, shift_x_px=dx,
                         edge_r_ab=ra, edge_r_ba=rb, edge_r_null95=r95,
                         edge_adjacent=max(ra, rb) > max(0.5, r95)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "overlap_pairs.csv", index=False)

    same, diff = df[df.same_key], df[~df.same_key]
    thr = np.percentile(diff.psr, 99)
    lines = ["# Parent-image hypothesis: pixel check", "",
             f"{len(same)} same-key pairs, {len(diff)} different-key pairs (negative control).", "",
             "| statistic | same key (median, max) | different key (median, 99th pct, max) |", "|---|---|---|",
             f"| phase-corr peak-to-sidelobe | {same.psr.median():.1f}, {same.psr.max():.1f} | {diff.psr.median():.1f}, {thr:.1f}, {diff.psr.max():.1f} |",
             f"| edge Pearson r (best direction) | {same[['edge_r_ab','edge_r_ba']].max(1).median():.2f}, {same[['edge_r_ab','edge_r_ba']].max(1).max():.2f} | {diff[['edge_r_ab','edge_r_ba']].max(1).median():.2f}, {np.percentile(diff[['edge_r_ab','edge_r_ba']].max(1), 99):.2f}, {diff[['edge_r_ab','edge_r_ba']].max(1).max():.2f} |",
             "", f"Same-key pairs with phase-corr PSR above the control 99th percentile: {(same.psr > thr).sum()} / {len(same)}.",
             f"Same-key pairs passing the edge-adjacency test: {int(same.edge_adjacent.sum())} / {len(same)}; control: {int(diff.edge_adjacent.sum())} / {len(diff)}.", "",
             "## Same-key pairs", "", "| a | b | key | PSR | shift z,x (half px) | edge r ab / ba | null95 |", "|---|---|---|---|---|---|---|"]
    for r in same.sort_values("psr", ascending=False).itertuples():
        lines.append(f"| {r.site_a} | {r.site_b} | {r.parent_a} | {r.psr:.1f} | {r.shift_z_px},{r.shift_x_px} | {r.edge_r_ab:.2f} / {r.edge_r_ba:.2f} | {r.edge_r_null95:.2f} |")
    lines += ["", "## Top control pairs (different key) by PSR", "", "| a | b | PSR | edge r max |", "|---|---|---|---|"]
    for r in diff.sort_values("psr", ascending=False).head(8).itertuples():
        lines.append(f"| {r.site_a} | {r.site_b} | {r.psr:.1f} | {max(r.edge_r_ab, r.edge_r_ba):.2f} |")
    lines += ["", "## Reconstructed tile order within each parent (left -> right along x)", "",
              "Direction from the edge test: r(last cols of a, first cols of b) high means a is left of b. "
              "`weak` = r above the null 95th percentile but below 0.5; `|` = adjacency not detected (gap or missing crop).", ""]
    lab = dict(zip(pg.site, pg.batch))
    truth = pd.read_csv(ROOT / "outputs" / "heldout_labels.csv", dtype=str).set_index("site")["batch"]
    lab.update(truth.to_dict())
    chains = []
    for key, g in pg.groupby("parent_id", sort=True):
        sites = list(g.site)
        right = {}
        for r in same[(same.parent_a == key)].itertuples():
            for a, b, val in ((r.site_a, r.site_b, r.edge_r_ab), (r.site_b, r.site_a, r.edge_r_ba)):
                if val > r.edge_r_null95 and val > 0.25 and (a not in right or val > right[a][1]):
                    right[a] = (b, val)
        has_left = {b for b, _ in right.values()}
        order, used = [], set()
        for start in [s for s in sites if s not in has_left] + sites:
            if start in used:
                continue
            seg, cur = [], start
            while cur is not None and cur not in used:
                seg.append(cur); used.add(cur)
                nxt = right.get(cur)
                if nxt is None:
                    cur = None
                else:
                    seg.append("weak" if nxt[1] <= 0.5 else "")
                    cur = nxt[0]
            order.append(seg)
        def fmt(seg):
            out = []
            for i, s in enumerate(seg):
                if s in ("", "weak"):
                    out.append(f" -{s}- " if s else " - ")
                else:
                    out.append(f"{s} ({lab[s][-1]})")
            return "".join(out)
        chains.append(f"- `{key}`: " + "  |  ".join(fmt(seg) for seg in order))
    lines += chains
    (OUT / "chains.md").write_text("\n".join(chains) + "\n")
    (OUT / "overlap_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
