"""Leave-one-site-out outlier test on site-level KPIs (spec 003 preliminary).

Pure functions, no I/O. Every site is scored against the other sites with a
robust-standardised, family-wise PCA feature, a classical Mahalanobis distance and a
Hotelling prediction F-test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

VERDICT_FAMILIES: dict[str, tuple[int, dict[str, int]]] = {
    "loading": (1, {"K01_si_frac_adm": 1}),
    "clustering": (
        3,
        {
            "K02_si_density_per_1000um2": -1,
            "K03_ecd_d50_um": 0,
            "K03_ecd_d90_um": 1,
            "K03_ecd_max_um": 1,
            "K04_agglom_frac": 1,
            "K04_n_clusters_per_1000um2": 0,
            "K05_voronoi_sigma": 1,
            "K05_voronoi_sigma_z": 1,
            "K05_local_af_cv": 1,
            "K06_cluster_region_frac": 1,
            "K06_void_region_frac": 1,
            "K07_R_rl": -1,
            "K07_R_csr": -1,
            "K08_pcf_excess_max_x": 1,
            "K08_pcf_rpeak_x_um": 0,
            "K08_pcf_excess_max_z": 1,
            "K08_pcf_rpeak_z_um": 0,
            "K09_mst_m_norm": -1,
            "K09_mst_sigma_norm": 1,
        },
    ),
    "localisation": (
        2,
        {
            "K10_cv_w10": 1,
            "K10_cv_slope": 1,
            "K11_lacey_w10": -1,
            "K12_depth_maxdev": 1,
            "K12_depth_absslope": 1,
            "K13_lateral_cv": 1,
            "K14_empty_p50_um": 1,
            "K14_empty_p95_um": 1,
        },
    ),
    "contact": (1, {"K15_si_graphite_contact_frac": -1}),
}

DIAGNOSTIC_FAMILIES: dict[str, list[str]] = {
    "process": [
        "D01_graphite_frac_mean",
        "D01_graphite_band_maxdev",
        "D02_graphite_orient_circsd_deg",
        "D03_graphite_ecd_d50_um",
        "D03_graphite_aspect_median",
        "D04_porosity_mean",
        "D04_porosity_band_maxdev",
        "D05_pore_size_d50_um",
        "D06_pore_euler_per_1000um2",
    ],
    "artefact": ["A01_large_void_frac", "A02_curtaining_index", "A03_height_um"],
}

EXCLUDED_COLUMNS = {
    "K05_voronoi_sigma_null_mean": "null reference, not a KPI",
    "K16_si_graphite_dist_median_um": "identically 0 at all sites",
}

N_FEATURES = sum(k for k, _ in VERDICT_FAMILIES.values())
CLIP = 5.0


def _robust_fit(ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(ref, dtype=float)
    med = np.median(ref, axis=0)
    scale = 1.4826 * np.median(np.abs(ref - med), axis=0)
    for j in np.where(scale == 0)[0]:
        scale[j] = np.std(ref[:, j], ddof=1)
        if scale[j] == 0:
            raise ValueError(f"zero scale in reference for column index {j}")
    return med, scale


def robust_z(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    med, scale = _robust_fit(ref)
    return np.clip((np.asarray(x, dtype=float) - med) / scale, -CLIP, CLIP)


@dataclass
class _FamilyModel:
    cols: list[str]
    med: np.ndarray
    scale: np.ndarray
    mean: np.ndarray
    comps: np.ndarray  # (k, p)


@dataclass
class Model:
    families: dict[str, _FamilyModel]
    mean: np.ndarray
    cov_inv: np.ndarray
    n: int
    diag: dict[str, tuple[np.ndarray, np.ndarray]]


def _fz(fm: _FamilyModel, x: np.ndarray) -> np.ndarray:
    return np.clip((x - fm.med) / fm.scale, -CLIP, CLIP)


def _features(model: Model, x_df: pd.DataFrame) -> np.ndarray:
    parts = []
    for fm in model.families.values():
        z = _fz(fm, x_df[fm.cols].to_numpy(dtype=float))
        parts.append((z - fm.mean) @ fm.comps.T)
    return np.hstack(parts)


def fit_model(ref_df: pd.DataFrame) -> Model:
    families: dict[str, _FamilyModel] = {}
    feats = []
    for name, (k, signs) in VERDICT_FAMILIES.items():
        cols = list(signs)
        if len(cols) < k:
            raise ValueError(f"family {name} has {len(cols)} columns < k={k}")
        ref = ref_df[cols].to_numpy(dtype=float)
        med, scale = _robust_fit(ref)
        z = np.clip((ref - med) / scale, -CLIP, CLIP)
        mean = z.mean(axis=0)
        _, _, vt = np.linalg.svd(z - mean, full_matrices=False)
        comps = vt[:k].copy()
        sv = np.array([signs[c] for c in cols], dtype=float)
        for i in range(k):
            if comps[i] @ sv < 0:
                comps[i] = -comps[i]
        fm = _FamilyModel(cols, med, scale, mean, comps)
        families[name] = fm
        feats.append((z - mean) @ comps.T)
    f = np.hstack(feats)
    cov = np.cov(f, rowvar=False, ddof=1)
    if np.linalg.cond(cov) > 1e12:
        raise ValueError("singular feature covariance in fold")
    diag = {n: _robust_fit(ref_df[c].to_numpy(dtype=float)) for n, c in DIAGNOSTIC_FAMILIES.items()}
    return Model(families, f.mean(axis=0), np.linalg.inv(cov), len(ref_df), diag)


def score(model: Model, x_df: pd.DataFrame) -> tuple[float, float, int]:
    """Return (distance d, Hotelling prediction F statistic, n_ref)."""
    v = _features(model, x_df)[0] - model.mean
    t2 = float(v @ model.cov_inv @ v)
    n, p = model.n, N_FEATURES
    f_stat = n * (n - p) / (p * (n + 1) * (n - 1)) * t2
    return float(np.sqrt(t2)), f_stat, n


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def _attribution(model: Model, row: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    comps, zs = {}, {}
    for name, (_, signs) in VERDICT_FAMILIES.items():
        fm = model.families[name]
        z = _fz(fm, row[fm.cols].to_numpy(dtype=float))[0]
        zs.update(dict(zip(fm.cols, z)))
        sv = np.array([signs[c] for c in fm.cols], dtype=float)
        m = sv != 0
        comps[f"comp_{name}"] = float(np.mean(sv[m] * z[m]))
    idx = {}
    for name, cols in DIAGNOSTIC_FAMILIES.items():
        med, scale = model.diag[name]
        z = np.clip((row[cols].to_numpy(dtype=float)[0] - med) / scale, -CLIP, CLIP)
        idx[f"idx_{name}"] = float(np.sqrt(np.mean(z**2)))
    return {**comps, **idx}, zs


def _loso(df: pd.DataFrame):
    """Per-site LOSO fold results: (model, s, F, n_ref, cal_d, cal_F, p_conf, p_param)."""
    out = []
    for i in range(len(df)):
        ref = df.drop(index=i).reset_index(drop=True)
        model = fit_model(ref)
        d, f_stat, n_ref = score(model, df.iloc[[i]])
        cal = np.array(
            [score(fit_model(ref.drop(index=j).reset_index(drop=True)), ref.iloc[[j]]) for j in range(n_ref)]
        )
        p_conf = (1 + np.sum(cal[:, 0] >= d)) / (n_ref + 1)
        p_param = float(stats.f.sf(f_stat, N_FEATURES, n_ref - N_FEATURES))
        out.append((model, d, f_stat, n_ref, cal[:, 0], cal[:, 1], p_conf, p_param))
    return out


def loso_scores(df: pd.DataFrame) -> pd.DataFrame:
    res = _loso(df.reset_index(drop=True))
    sc = pd.DataFrame(
        {"d": [r[1] for r in res], "p_conformal": [r[6] for r in res], "p_param": [r[7] for r in res]}
    )
    sc["q_bh"] = bh_qvalues(sc["p_param"].to_numpy())
    return sc


def run_pipeline(df: pd.DataFrame, n_boot: int = 500):
    df = df.reset_index(drop=True)
    n = len(df)
    rows, contrib, cal_f = [], [], []
    boot_rng = np.random.default_rng(0)
    res = _loso(df)
    for i, (model, s, _, n_ref, cal_d, cal_fs, p_conf, p_param) in enumerate(res):
        ref = df.drop(index=i).reset_index(drop=True)
        x = df.iloc[[i]]
        cal_f.append(cal_fs)
        boots = []
        for _ in range(n_boot):
            idx = boot_rng.integers(0, n_ref, n_ref)
            boots.append(score(fit_model(ref.iloc[idx].reset_index(drop=True)), x)[0])
        d_lo, d_hi = np.percentile(boots, [5, 95])
        attr, zs = _attribution(model, x)
        top = sorted(zs, key=lambda k: -abs(zs[k]))[:3]
        rows.append(
            {
                "batch": df.loc[i, "batch"],
                "site": df.loc[i, "site"],
                "se_detector": df.loc[i, "se_detector"],
                "d": s,
                "d_lo": d_lo,
                "d_hi": d_hi,
                "p_conformal": p_conf,
                "p_param": p_param,
                **attr,
                "top_kpis": "; ".join(f"{k}:{zs[k]:+.1f}" for k in top),
            }
        )
        contrib += [{"site": df.loc[i, "site"], "kpi": k, "z": v} for k, v in zs.items()]
    sc = pd.DataFrame(rows)
    sc["q_bh"] = bh_qvalues(sc["p_param"].to_numpy())
    sc["rank"] = sc["d"].rank(ascending=False, method="min").astype(int)
    sc = sc.sort_values("d", ascending=False).reset_index(drop=True)
    cols = [
        "batch", "site", "se_detector", "d", "d_lo", "d_hi", "p_conformal", "p_param", "q_bh", "rank",
        "comp_loading", "comp_clustering", "comp_localisation", "comp_contact",
        "idx_process", "idx_artefact", "top_kpis",
    ]
    sc = sc[cols]

    # batch level
    batch_of = df["batch"].to_numpy()
    nlp = -np.log(df["site"].map(dict(zip(sc["site"], sc["p_param"]))).to_numpy())
    labels = sorted(set(batch_of))
    t_obs = {b: float(nlp[batch_of == b].mean()) for b in labels}
    rng = np.random.default_rng(1)
    n_perm = 2000
    perm_t = {b: np.empty(n_perm) for b in labels}
    for r in range(n_perm):
        pl = rng.permutation(batch_of)
        for b in labels:
            perm_t[b][r] = nlp[pl == b].mean()
    batch_df = pd.DataFrame(
        [
            {
                "batch": b,
                "n_sites": int((batch_of == b).sum()),
                "T": t_obs[b],
                "p_perm": (1 + np.sum(perm_t[b] >= t_obs[b])) / (n_perm + 1),
            }
            for b in labels
        ]
    )

    n_fit = n - 2
    pooled = np.concatenate(cal_f)
    u = stats.f.cdf(pooled, N_FEATURES, n_fit - N_FEATURES)
    ks = stats.kstest(u, "uniform")
    log = {
        "n_sites": n,
        "n_boot": n_boot,
        "families": {k: {"k": v[0], "columns": list(v[1])} for k, v in VERDICT_FAMILIES.items()},
        "diagnostic_families": DIAGNOSTIC_FAMILIES,
        "excluded_columns": EXCLUDED_COLUMNS,
        "seeds": {"bootstrap_rng": 0, "batch_permutation_rng": 1},
        "n_perm": n_perm,
        "calibration_check": {
            "n_pooled": int(pooled.size),
            "n_fit": n_fit,
            "ks_stat": float(ks.statistic),
            "ks_p": float(ks.pvalue),
            "frac_u_gt_0.95": float(np.mean(u > 0.95)),
            "frac_u_gt_0.99": float(np.mean(u > 0.99)),
        },
    }
    return sc, pd.DataFrame(contrib), batch_df, log
