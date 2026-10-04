"""The single pre-registered evaluation of the extended fingerprint features.

Implements docs/eval-plan-oct4.md exactly: the merged model's 16 features plus
10 pre-specified additions from outputs/overnight/features/, evaluated once
(LOO + 2000-permutation test, seed 0). Adopt iff LOO >= 0.742 AND p <= 0.005.

    .venv/Scripts/python scripts/eval_extended_features.py

Writes to outputs/overnight/eval_extended/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pmdb import fingerprint as fp  # noqa: E402

ADOPT_LOO = 0.742
ADOPT_P = 0.005
N_PERM = 2000
SEED = 0
MIN_TILES = 8


def rel_profile(curves: pd.DataFrame, curve: str) -> pd.DataFrame:
    """(batch, site) x band table of the curve, normalised by the site mean."""
    band = curves[curves["curve"] == curve].pivot_table(
        index=["batch", "site"], columns="x", values="value")
    return band.div(band.mean(axis=1), axis=0)


def nan_tile_std(tiles: pd.DataFrame, col: str) -> pd.Series:
    def f(s: pd.Series) -> float:
        v = s.dropna()
        return float(v.std()) if len(v) >= MIN_TILES else float("nan")
    return tiles.groupby(["batch", "site"])[col].apply(f)


def extended_features(kpis_dir: Path, overnight_dir: Path,
                      sites: pd.MultiIndex) -> pd.DataFrame:
    """The 16 merged features + the 10 pre-registered additions, for `sites`."""
    base = fp.read_feature_inputs(kpis_dir / "curves.csv", kpis_dir / "tile_kpis.csv")

    rich = pd.read_csv(overnight_dir / "curves_rich.csv", dtype={"batch": str, "site": str})
    scal = pd.read_csv(overnight_dir / "site_scalars.csv",
                       dtype={"batch": str, "site": str}).set_index(["batch", "site"])
    tiles = pd.read_csv(overnight_dir / "tile_kpis_rich.csv", dtype={"batch": str, "site": str})
    t16 = tiles[tiles["n_tiles"] == 16]

    si15 = rel_profile(rich, "band15_si_frac")
    g15 = rel_profile(rich, "band15_graphite_frac")
    p15 = rel_profile(rich, "band15_pore_frac")

    add = pd.DataFrame(index=si15.index)
    add["si15_top3_rel"] = si15.iloc[:, 0:3].mean(axis=1)
    add["si15_bottom3_rel"] = si15.iloc[:, 12:15].mean(axis=1)
    add["si15_mid5_rel"] = si15.iloc[:, 5:10].mean(axis=1)
    add["si15_roughness"] = si15.diff(axis=1).std(axis=1)
    add["acl_depth_si_um"] = scal["acl_depth_si_um"]
    add["acl_lateral_si_um"] = scal["acl_lateral_si_um"]
    add["graphite15_slope"] = g15.iloc[:, 12:15].mean(axis=1) - g15.iloc[:, 0:3].mean(axis=1)
    add["pore15_slope"] = p15.iloc[:, 12:15].mean(axis=1) - p15.iloc[:, 0:3].mean(axis=1)
    add["k15_contact_tilestd16"] = nan_tile_std(t16, "K15_si_graphite_contact_frac")
    add["k01_si_frac_tilestd16"] = nan_tile_std(t16, "K01_si_frac_adm")

    out = base.join(add.loc[base.index.intersection(add.index)])
    return out.loc[sites] if sites is not None else out


def main() -> int:
    t0 = time.time()
    out = ROOT / "outputs" / "overnight" / "eval_extended"
    out.mkdir(parents=True, exist_ok=True)
    overnight = ROOT / "outputs" / "overnight" / "features"

    X = extended_features(ROOT / "outputs" / "kpis", overnight, None)
    y = pd.Series(X.index.get_level_values("batch"), index=X.index, name="batch")
    print(f"{len(X)} labelled sites, {X.shape[1]} features "
          f"(16 merged + {X.shape[1] - 16} pre-registered)", flush=True)
    assert X.shape[1] == 26, "feature count must match the pre-registration"
    X.to_csv(out / "features_extended.csv")

    loo_pred, metrics = fp.loo_evaluate(X, y)
    loo_pred.to_csv(out / "loo_predictions.csv")
    print(f"LOO accuracy: {metrics['accuracy']:.4f} (rule: adopt iff >= {ADOPT_LOO})")
    for bt, row in metrics["confusion"].items():
        print(f"  {bt}: {row}")

    perm = fp.permutation_test(X, y, n_perm=N_PERM, seed=SEED)
    print(f"permutation: p = {perm['p_value']:.4f} (rule: adopt iff <= {ADOPT_P})")

    adopt = metrics["accuracy"] >= ADOPT_LOO and perm["p_value"] <= ADOPT_P

    H = extended_features(ROOT / "outputs" / "heldout" / "kpis", overnight, None)
    H = H[H.index.get_level_values("batch") == "Batch_heldout"]
    H.to_csv(out / "heldout_features_extended.csv")
    model = fp.fit(X, y)
    pred = fp.predict(model, H)
    pred.to_csv(out / "heldout_predictions_extended.csv")
    print("\nheld-out under extended set (for the record):")
    print(pred[["assigned", "credibility", "confidence", "ood"]].to_string())

    decision = {
        "plan": "docs/eval-plan-oct4.md",
        "n_features": int(X.shape[1]),
        "loo": metrics,
        "permutation": perm,
        "rule": {"loo_min": ADOPT_LOO, "p_max": ADOPT_P},
        "adopt": bool(adopt),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (out / "evaluation.json").write_text(json.dumps(decision, indent=2))
    print(f"\nDECISION: {'ADOPT extended set' if adopt else 'KEEP merged 16-feature model'}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
