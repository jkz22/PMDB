"""Two-stage random-forest batch classifier vs Batch_3 (D18): KPI / FEM / KPI+FEM ablation.

    python scripts/classify_batches.py kpi-tiles [--jobs 8]
    python scripts/classify_batches.py run [--fem-tiles outputs/fem/tile_curves.csv] [--require-fem]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pmdb.classify.explain import explanation_text, site_means, zscores_vs_b3  # noqa: E402
from pmdb.classify.features import (  # noqa: E402
    BATCHES, N_FEM_TILES, build_arm_table, feature_catalogue, fem_tile_features, kpi_feature_columns,
    kpi_tile_columns, load_fem_tile_curves, load_kpi_tiles6)
from pmdb.classify.kpi_tiles import site_kpi_tiles6  # noqa: E402
from pmdb.classify.model import (  # noqa: E402
    ARM_ORDER, LEVELS, N_BOOT, RF_PARAMS, SELECT_TIE_BACC, TwoStage, calibration, confusion, level_metrics,
    loso, select_arm, site_predictions)
from tile_signal_checks import md_table  # noqa: E402

DEFAULT_OUT = ROOT / "outputs" / "classifier"
DEFAULT_DOCS = ROOT / "docs" / "classifier" / "results.md"
DEFAULT_FEM = ROOT / "outputs" / "fem" / "tile_curves.csv"
KPI_TILES = DEFAULT_OUT / "kpi_tiles6.csv"
CHECK3 = ROOT / "outputs" / "pooling_checks" / "check3_metrics.csv"
FMT = "%.6g"


def all_sites() -> list[tuple[str, str, str | None]]:
    lab = pd.read_csv(ROOT / "cache" / "half" / "manifest.csv")
    held = pd.read_csv(ROOT / "cache_heldout" / "half" / "manifest.csv")
    sites = [(str(b), str(s), None) for b, s in zip(lab["batch"], lab["site"])]
    sites += [(str(b), str(s), str(ROOT / "cache_heldout")) for b, s in zip(held["batch"], held["site"])]
    return sorted(sites, key=lambda x: (x[0], x[1]))


def _worker(args):
    batch, site, cache_root = args
    t0 = time.time()
    try:
        return batch, site, site_kpi_tiles6(batch, site, cache_root), None, time.time() - t0
    except Exception as e:  # reported, never filled
        return batch, site, None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}", time.time() - t0


def cmd_kpi_tiles(args) -> int:
    sites = all_sites()
    rows, failed = [], False
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for batch, site, res, err, sec in ex.map(_worker, sites):
            if err:
                print(f"FAILED {batch}/{site}: {err}", file=sys.stderr)
                failed = True
            else:
                print(f"{batch}/{site}: {sec:.0f}s")
                rows.extend(res)
    if failed:
        return 1
    cols = ["batch", "site", "heldout", "tile", "tile_x0_um", "tile_x1_um", *kpi_tile_columns(), "nan_reason"]
    df = pd.DataFrame(rows)[cols].sort_values(["batch", "site", "tile"]).reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.10g")
    print(f"wrote {out}: {len(df)} rows")
    nan = df[kpi_tile_columns()].isna().sum()
    print("per-column NaN counts:")
    print(nan.to_string())
    return 0


# --------------------------------------------------------------------------------------- run

def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _rank_importance(model: TwoStage, features: list[str]) -> pd.DataFrame:
    rows = []
    for stage, m in (("stage1", model.stage1), ("stage2", model.stage2)):
        imp = m.named_steps["rf"].feature_importances_
        order = sorted(range(len(features)), key=lambda i: (-imp[i], features[i]))
        rank = {i: r + 1 for r, i in enumerate(order)}
        rows += [{"stage": stage, "feature": f, "importance": float(imp[i]), "rank": rank[i]}
                 for i, f in enumerate(features)]
    return pd.DataFrame(rows)


def plot_confusion(conf: dict[str, pd.DataFrame], metrics: pd.DataFrame, path: Path) -> None:
    arms = list(conf)
    fig, axes = plt.subplots(1, len(arms), figsize=(4.2 * len(arms), 4), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        m = conf[arm].pivot(index="true", columns="predicted", values="n").loc[BATCHES, BATCHES].to_numpy()
        ax.imshow(m, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, int(m[i, j]), ha="center", va="center")
        ax.set_xticks(range(3), BATCHES, rotation=30)
        ax.set_yticks(range(3), BATCHES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        r = metrics[(metrics.arm == arm) & (metrics.level == "end_to_end")].iloc[0]
        ax.set_title(f"{arm}: bacc {r.balanced_acc:.2f} [{r.bacc_ci_lo:.2f}, {r.bacc_ci_hi:.2f}]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(imp: pd.DataFrame, arm: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, stage in zip(axes, ("stage1", "stage2")):
        g = imp[(imp.arm == arm) & (imp.stage == stage)].sort_values("rank").head(10)[::-1]
        ax.barh(g.feature, g.importance)
        ax.set_title(f"{arm} {stage} ({'Batch_3 vs not' if stage == 'stage1' else 'Batch_1 vs Batch_2'})")
        ax.set_xlabel("MDI importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_report(fig_prefix: str, ctx: dict) -> str:
    arms, final = ctx["arms"], ctx["final"]
    metrics, held = ctx["metrics"], ctx["heldout"]
    provisional = set(arms) == {"KPI"}
    L = ["# Batch classifier: two-stage random forest vs Batch_3 (D18)", ""]
    L.append(f"Status: {'provisional (FEM arms not run): ' + final if provisional else 'final arm: ' + final}")
    L += ["", "## Inputs", ""]
    L.append(f"- KPI tiles: `outputs/classifier/kpi_tiles6.csv` (md5 `{ctx['md5_kpi']}`)")
    L.append(f"- FEM tiles: {('`' + ctx['fem_label'] + '` (md5 `' + ctx['md5_fem'] + '`)') if ctx['md5_fem'] else 'not available'}")
    L.append(f"- Arms run: {', '.join(arms)}")
    L.append("- Feature counts: " + "; ".join(f"{a} {len(ctx['features'][a])}" for a in arms))
    L.append(f"- KPI zero-variance columns dropped: {ctx['kpi_dropped'] or 'none'}")
    L += ["", f"## Held-out predictions (arm: {final})", ""]
    f = held[(held.arm == final)][["site", "predicted", "confidence", "p_b3", "q_b1", "n_tiles",
                                   "stage1_votes_not_b3", "stage2_votes_b1"]]
    L.append(md_table(f))
    L.append("")
    for e in held[held.arm == final].sort_values("site").explanation:
        L.append(f"- {e}")
    L += ["", "All arms, predicted (confidence):", ""]
    piv = held.assign(cell=held.predicted + " (" + held.confidence.map(lambda v: f"{v:.2f}") + ")").pivot(
        index="site", columns="arm", values="cell")[arms].reset_index()
    L.append(md_table(piv))
    L += ["", "## LOSO ablation (31 labelled sites)", ""]
    L.append(md_table(metrics[["arm", "level", "n_sites", "site_acc", "balanced_acc", "bacc_ci_lo",
                               "bacc_ci_hi", "macro_f1", "brier"]]))
    L += ["", "Arm selection rule (fixed before FEM results): take the maximum end-to-end LOSO balanced accuracy; "
          f"arms within {SELECT_TIE_BACC} of it are tied; among tied arms the lowest end-to-end Brier wins; exact "
          "Brier ties go by the order KPI+FEM, FEM, KPI. If FEM tables are absent only the KPI arm runs and the "
          "result is provisional.", ""]
    L.append(f"Applied: selected arm = **{final}**.")
    L += ["", f"![confusion]({fig_prefix}fig_confusion.png)", "", f"End-to-end confusion ({final}, rows true, "
          "columns predicted):", ""]
    c = ctx["confusion"][final].pivot(index="true", columns="predicted", values="n").loc[BATCHES, BATCHES]
    L.append(md_table(c.reset_index()))
    L += ["", "## Calibration", ""]
    cal = ctx["calibration"]
    L.append(md_table(cal[cal.arm == final][["level", "bin", "n", "mean_conf", "accuracy"]]))
    L += ["", f"![importance]({fig_prefix}fig_importance.png)", "", f"Top-5 MDI importance per stage ({final}):", ""]
    imp = ctx["importance"]
    L.append(md_table(imp[(imp.arm == final) & (imp["rank"] <= 5)].sort_values(["stage", "rank"])[
        ["stage", "rank", "feature", "importance"]]))
    L += ["", "## External reference (check 3, not an arm)", ""]
    if ctx["check3"] is not None:
        L.append(md_table(ctx["check3"]))
        L += ["", "Flat logistic on 4-tile KPIs; not the same task structure."]
    L += ["", "Caveat: 31 labelled sites: differences below ~0.1 balanced accuracy are within the bootstrap CI.", ""]
    return "\n".join(L)


def cmd_run(args) -> int:
    out = Path(args.out)
    fem_path = Path(args.fem_tiles)
    have_fem = fem_path.exists()
    if args.require_fem and not have_fem:
        print(f"--require-fem: {fem_path} not found", file=sys.stderr)
        return 2
    kpi = load_kpi_tiles6(KPI_TILES)
    fem = None
    if have_fem:
        fem = fem_tile_features(load_fem_tile_curves(fem_path))
        if set(fem["site"]) != set(kpi["site"]):
            raise ValueError("FEM sites differ from kpi_tiles6.csv sites: "
                             f"missing {sorted(set(kpi['site']) - set(fem['site']))}, "
                             f"extra {sorted(set(fem['site']) - set(kpi['site']))}")
    else:
        print("FEM tables not found: KPI arm only")
    arms = [a for a in ARM_ORDER if a == "KPI" or have_fem]
    out.mkdir(parents=True, exist_ok=True)

    site_rows, tile_rows, met, conf, cal, imps, hsite, htile = [], [], [], [], [], [], [], []
    feats_by_arm, tables, models = {}, {}, {}
    for arm in arms:
        table, feats = build_arm_table(arm, kpi, fem)
        feats_by_arm[arm], tables[arm] = feats, table
        sp, tp = loso(table, feats)
        site_rows.append(sp.assign(arm=arm))
        tile_rows.append(tp.assign(arm=arm))
        met.append(level_metrics(sp).assign(arm=arm))
        conf.append(confusion(sp).assign(arm=arm))
        cal.append(calibration(sp).assign(arm=arm))
        lab, held = table[~table["heldout"]], table[table["heldout"]]
        assert not lab["heldout"].any() and held["heldout"].all() and len(held) == 3 * N_FEM_TILES
        model = TwoStage(feats).fit(lab)
        models[arm] = model
        imps.append(_rank_importance(model, feats).assign(arm=arm))
        htp = model.tile_probs(held)
        hsite.append(site_predictions(htp).assign(arm=arm))
        htile.append(htp.assign(arm=arm))

    metrics = pd.concat(met, ignore_index=True)
    metrics["arm"] = pd.Categorical(metrics["arm"], arms)
    metrics["level"] = pd.Categorical(metrics["level"], LEVELS)
    metrics = metrics.sort_values(["arm", "level"]).reset_index(drop=True)
    metrics["arm"], metrics["level"] = metrics["arm"].astype(str), metrics["level"].astype(str)
    metrics = metrics[["arm", "level", "n_sites", "site_acc", "balanced_acc", "bacc_ci_lo", "bacc_ci_hi",
                       "macro_f1", "brier"]]
    final = select_arm(metrics)
    importance = pd.concat(imps, ignore_index=True)[["arm", "stage", "feature", "importance", "rank"]]

    # explanation (final arm)
    ftab, ffeats = tables[final], feats_by_arm[final]
    imp1 = importance[(importance.arm == final) & (importance.stage == "stage1")].set_index("feature")["importance"]
    z = zscores_vs_b3(site_means(ftab, ffeats), ffeats, imp1.to_dict())
    meanings = dict(zip(*feature_catalogue(ffeats)[["feature", "meaning"]].T.values))
    hp = pd.concat(hsite, ignore_index=True).drop(columns="batch")
    hp["final"] = hp["arm"] == final
    expl = []
    for r in hp.itertuples():
        expl.append(explanation_text(r.site, r._asdict(), z, meanings) if r.final else "")
    hp["explanation"] = expl
    hp["arm"] = pd.Categorical(hp["arm"], arms)
    hp = hp.sort_values(["arm", "site"]).reset_index(drop=True)
    hp["arm"] = hp["arm"].astype(str)
    hp = hp[["arm", "final", "site", "predicted", "confidence", "p_b3", "q_b1", "P_Batch_1", "P_Batch_2",
             "P_Batch_3", "n_tiles", "stage1_votes_not_b3", "stage2_votes_b1", "explanation"]]

    def order(df):
        df = df.copy()
        df["arm"] = pd.Categorical(df["arm"], arms)
        return df

    ls = order(pd.concat(site_rows, ignore_index=True)).sort_values(["arm", "batch", "site"])
    ls["arm"] = ls["arm"].astype(str)
    ls = ls[["arm", "batch", "site", "p_b3", "q_b1", "P_Batch_1", "P_Batch_2", "P_Batch_3", "predicted",
             "confidence", "correct", "n_tiles", "stage1_votes_not_b3", "stage2_votes_b1"]]
    lt = order(pd.concat(tile_rows, ignore_index=True)).sort_values(["arm", "batch", "site", "tile"])
    lt["arm"] = lt["arm"].astype(str)
    lt = lt[["arm", "batch", "site", "tile", "p_b3", "q_b1"]]
    ht = order(pd.concat(htile, ignore_index=True)).sort_values(["arm", "site", "tile"])
    ht["arm"] = ht["arm"].astype(str)
    ht["stage1_vote_not_b3"] = ht["p_b3"] < 0.5
    ht["stage2_vote_b1"] = ht["q_b1"] >= 0.5
    ht = ht[["arm", "site", "tile", "p_b3", "q_b1", "stage1_vote_not_b3", "stage2_vote_b1"]]
    conf_df = order(pd.concat(conf, ignore_index=True)).sort_values(["arm"], kind="stable")
    conf_df["arm"] = conf_df["arm"].astype(str)
    cal_df = order(pd.concat(cal, ignore_index=True)).sort_values(["arm"], kind="stable")
    cal_df["arm"] = cal_df["arm"].astype(str)
    conf_df, cal_df = conf_df[["arm", "true", "predicted", "n"]], cal_df[["arm", "level", "bin", "n", "mean_conf", "accuracy"]]
    zout = z.assign(arm=final)[["arm", "site", "feature", "site_value", "b3_mean", "b3_std", "z",
                                "importance_stage1", "score", "rank"]].sort_values(["site", "rank", "feature"])
    all_feats = list(dict.fromkeys(f for a in arms for f in feats_by_arm[a]))
    cat = feature_catalogue(all_feats)

    kw = dict(index=False, float_format=FMT)
    cat.to_csv(out / "feature_catalogue.csv", **kw)
    ls.to_csv(out / "loso_site_predictions.csv", **kw)
    lt.to_csv(out / "loso_tile_predictions.csv", **kw)
    metrics.to_csv(out / "metrics.csv", **kw)
    conf_df.to_csv(out / "confusion.csv", **kw)
    cal_df.to_csv(out / "calibration.csv", **kw)
    importance.to_csv(out / "importance.csv", **kw)
    hp.to_csv(out / "heldout_predictions.csv", **kw)
    ht.to_csv(out / "heldout_tiles.csv", **kw)
    zout.to_csv(out / "heldout_zscores.csv", **kw)
    plot_confusion({a: conf_df[conf_df.arm == a] for a in arms}, metrics, out / "fig_confusion.png")
    plot_importance(importance, final, out / "fig_importance.png")

    kpi_all = kpi_tile_columns()
    kpi_used = kpi_feature_columns(kpi)
    check3 = None
    if CHECK3.exists():
        c3 = pd.read_csv(CHECK3)
        check3 = c3[c3.rule.isin(["R1_mean_prob", "R4_site_kpis"]) & (c3.task == "multiclass")][
            ["rule", "task", "balanced_acc", "bacc_ci_lo", "bacc_ci_hi"]]
    ctx = dict(arms=arms, final=final, metrics=metrics, heldout=hp, features=feats_by_arm,
               md5_kpi=md5(KPI_TILES), md5_fem=md5(fem_path) if have_fem else "",
               fem_label=str(fem_path.relative_to(ROOT)) if fem_path.is_relative_to(ROOT) else str(fem_path),
               kpi_dropped=", ".join(c for c in kpi_all if c not in kpi_used),
               confusion={a: conf_df[conf_df.arm == a] for a in arms}, calibration=cal_df,
               importance=importance, check3=check3)
    (out / "report.md").write_text(render_report("", ctx))
    docs = Path(args.docs)
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(render_report("../../outputs/classifier/", ctx))

    log = {"git_commit": git_commit(), "argv": sys.argv[1:], "sklearn": sklearn.__version__,
           "numpy": np.__version__, "pandas": pd.__version__, "rf_params": RF_PARAMS, "n_boot": N_BOOT,
           "inputs_md5": {"kpi_tiles6.csv": md5(KPI_TILES), **({"fem_tiles": md5(fem_path)} if have_fem else {})},
           "arms_run": arms, "selected_arm": final}
    (out / "run_log.json").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")

    print(f"final arm: {final}")
    print(metrics[metrics.level.isin(LEVELS)].to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("kpi-tiles")
    p1.add_argument("--jobs", type=int, default=8)
    p1.add_argument("--out", default=str(KPI_TILES))
    p2 = sub.add_parser("run")
    p2.add_argument("--fem-tiles", default=str(DEFAULT_FEM))
    p2.add_argument("--require-fem", action="store_true")
    p2.add_argument("--out", default=str(DEFAULT_OUT))
    p2.add_argument("--docs", default=str(DEFAULT_DOCS))
    args = ap.parse_args()
    return cmd_kpi_tiles(args) if args.cmd == "kpi-tiles" else cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
