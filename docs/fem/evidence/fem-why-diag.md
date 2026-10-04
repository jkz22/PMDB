## Question
Gather diagnostic evidence on why FEM-simulation features did not improve batch classification (items 1-5 of the spawn prompt: input integrity, univariate batch signal, redundancy with KPIs, saturation of stress features, model sensitivity). Labelled sites only (7/7/17); held-out rows are excluded from every statistic.

## Search log
- S1 Read pmdb/classify/features.py, model.py, fem-build.md section 2.1, run_log.json head, metrics.csv: done.
- S2 Ran scripts pre.py, t1.py, t1b.py (integrity), t2.py (univariate, redundancy, partial, saturation), t5.py (sensitivity). Full source is appended at the end of this file; copies in <local scratch file> Interpreter: repo default python (pandas 2.1.4, numpy 1.26.4). statsmodels is not installed; OLS done with numpy lstsq.

## Evidence

### 1. Integrity (t1.py, t1b.py)
- tile_curves.csv: 6732 rows x 93 cols; orientation counts bottom 2244 / top 2244 / sym 2244; heldout True 594 rows. load_fem_tile_curves (sym only) returns 2244 rows, 34 sites, 0 duplicate (site,tile,frame). Labelled curated table: 186 rows (31 sites x 6), 0 duplicate (batch,site,tile); 7/7/17 sites per batch.
- Convergence: sym rows with converged False = 0; sites with failed_at_s < 1: none in bottom, top or sym. Rows with any NaN among the 16 curated features: 0.
- Per-feature (labelled tile rows, n=186): NaN count 0 for all 16. Distinct values / range / std:
  - fem_swell_50 186 / 0.0728-0.1533 / 0.0140; fem_swell_100 186 / 0.111-0.2044 / 0.0161
  - fem_swell_slope_early 186 / 0.0862-0.3897 / 0.053; fem_swell_slope_late 186 / 0.0925-0.1252 / 0.00568
  - fem_surface_rough_100 186 / 0.00132-0.0264 / 0.00364; fem_pore_left_100 186 / 0.4785-1.104 / 0.1035
  - fem_pore_closed_frac_100 186 / 0.00226-0.238 / 0.0399
  - fem_first_closure_s: 2 distinct values (0.1 and 0.2), std 0.0226
  - fem_vm_si_p95_100 186 / 21357.8-29700 MPa / 1503; fem_vm_binder_p95_100 186 / 416-1838 / 242; fem_vm_gr_p95_100 186 / 7169-22350 / 2905
  - fem_si_yield_frac_100: 1 distinct value (=1.0 on all 186 rows), constant
  - fem_p_si_mean_100 186 / 14225-20620 / 1069
  - fem_sxx_mean_100: only 31 distinct values (-5466 to -2070, std 639), i.e. one value per site, identical across the 6 tiles of a site
  - fem_J_si_mean_100 185 / 1.60-1.778 / 0.0299; fem_band_vm_maxdev_100 186 / 0.183-1.449 / 0.217
- Join: outer merge KPI(labelled) x FEM curated on (batch,site,tile): both=186, left_only=0, right_only=18 (3 held-out sites x 6 tiles, FEM only). max |tile_x0_um diff| = 0.0 and max |tile_x1_um diff| = 0.0 between kpi_tiles6.csv and FEM. KPI: 6 tiles per site for all 31 labelled sites, 0 duplicates.
- Tile example (first site): tiles 0..5 x0-x1 um: 20.0-42.4, 42.4-64.9, 64.9-87.4, 87.4-109.8, 109.8-132.2, 132.2-154.7.
- fem_grid_slices vs FEM tile_x0/x1_um: with W = list_sites() 'width'//2 (manifest width is the full-res 25 nm width: 6960/6996/7000 for the 31 sites), x_um = slice*0.05, max abs difference over 31 sites = 0.2 um (both x0 and x1). With W = manifest width (no //2) the difference is 146-175 um.
- sym handling: max|sym - (bottom+top)/2| for swelling over all (site,tile,frame) = 7e-07; bottom vs top swelling at s=1 (tile level): Pearson r = 0.878, max |bottom-top| = 0.0286.

### 2. Univariate batch signal (t2.py)
Site level, n=31. Curated = mean of the site's 6 tile values; raw = site_curves.csv sym rows at frames 5 and 10, 81 metrics each (162 columns); KPI = 43 numeric site_kpis.csv columns. Cliff's delta = mean sign(a-b); positive = first-named batch larger.
- Tests with valid KW p (non-constant, no NaN): curated 15, raw 160, KPI 42. Count p<0.05 / p<0.01: curated 2/0; raw 34/8; KPI 0/0. Excluded as constant/NaN: fem_si_yield_frac_100, si_yield_frac@s0.5, si_yield_frac@s1.0, K16_si_graphite_dist_median_um. Several raw columns are duplicates of each other (vm_si_p50_MPa = q50_vm_si; J_si_mean and p_si_mean_MPa give identical p). No multiplicity correction applied.
- Top 15 by KW p (distinct rows; all FEM raw; none curated, none KPI):
  1 q5_vm_si@s1.0 p=0.00104 d13=+0.80 d23=+0.76 d12=+0.18
  2 q25_vm_si@s0.5 p=0.00177 d13=+0.75 d23=+0.75 d12=+0.31
  3 q25_vm_si@s1.0 p=0.00206 d13=+0.76 d23=+0.71 d12=+0.22
  4 q50_vm_si@s0.5 (= vm_si_p50_MPa@s0.5) p=0.00627 d13=+0.73 d23=+0.60 d12=+0.22
  5 q5_vm_si@s0.5 p=0.00636 d13=+0.70 d23=+0.65 d12=+0.06
  6 q25_p_si@s0.5 p=0.00907 d13=-0.76 d23=-0.48 d12=-0.27
  7 q75_J_si@s0.5 p=0.00907 d13=+0.76 d23=+0.48 d12=+0.27
  8 q75_J_si@s1.0 p=0.01055 d13=+0.75 d23=+0.43 d12=+0.43
  9 q25_p_si@s1.0 p=0.01055 d13=-0.75 d23=-0.43 d12=-0.43
  10 q25_vm_binder@s0.5 p=0.01246 d13=+0.76 d23=+0.19 d12=+0.63
  11 p_si_mean_MPa@s1.0 p=0.01258 d13=-0.68 d23=-0.53 d12=-0.35
  12 J_si_mean@s1.0 p=0.01258 d13=+0.68 d23=+0.53 d12=+0.35
  13 vm_si_p50_MPa@s1.0 / q50_vm_si@s1.0 p=0.01311 d13=+0.71 d23=+0.48 d12=+0.31
  The printed top-15 list (with duplicates as separate rows) has 15 FEM raw entries; ordered by max|delta| it adds q25_vm_binder@s1.0 (d13 +0.71, p 0.0208) and q75_J_pore@s1.0 (d13 +0.70, p 0.036). Full table with all columns: <local scratch file>
- Curated FEM features by p (site mean of tiles; d13/d23/d12): fem_J_si_mean_100 p=0.0212 (+0.63/+0.51/+0.31); fem_p_si_mean_100 p=0.0212 (signs reversed); fem_surface_rough_100 0.057 (+0.51/+0.50/+0.14); fem_swell_slope_late 0.090 (+0.53/+0.31/+0.39); fem_swell_slope_early 0.178 (+0.45/-0.06/+0.51); fem_swell_50 0.194 (+0.43/-0.06/+0.51); fem_swell_100 0.217 (+0.41/+0.01/+0.51); fem_vm_si_p95_100 0.333 (+0.34/+0.28/+0.06); fem_pore_closed_frac_100 0.443; fem_sxx_mean_100 0.518; fem_band_vm_maxdev_100 0.569; fem_first_closure_s 0.569; fem_vm_gr_p95_100 0.601; fem_vm_binder_p95_100 0.988; fem_pore_left_100 0.997.
- KPI by p (site level): K12_depth_maxdev 0.065 (d13 +0.08, d23 +0.61, d12 -0.51); K15_si_graphite_contact_frac 0.070 (-0.56/-0.36/-0.27); K08_pcf_rpeak_x_um 0.075; K08_pcf_excess_max_x 0.075; K07_R_csr 0.085; K14_empty_p50_um 0.086; K06_void_region_frac 0.107; K07_R_rl 0.121; K05_local_af_cv 0.131; K05_voronoi_sigma 0.152. K01_si_frac_adm KW p = 0.377.
- Medians of site means per batch (B1 / B2 / B3): vm_si_p95 24554 / 24482 / 24234; p_si_mean 17238 / 17657 / 18048; vm_binder_p95 920 / 956 / 875; vm_gr_p95 13767 / 13402 / 13310; sxx_mean -2989 / -2783 / -2854; band_vm_maxdev 0.468 / 0.580 / 0.523; J_si_mean 1.694 / 1.683 / 1.672.

### 3. Redundancy (t2.py; Spearman, site level, n=31, vs the NaN-free non-constant KPI columns of site_kpis.csv)
Max |rho| of each curated FEM feature with any KPI:
- fem_swell_50 +0.903 K01_si_frac_adm; fem_swell_100 +0.891 K01; fem_swell_slope_early +0.909 K01; fem_swell_slope_late -0.763 D01_graphite_frac_mean; fem_surface_rough_100 -0.599 D01_graphite_frac_mean; fem_pore_left_100 -0.560 K01; fem_pore_closed_frac_100 +0.675 K01; fem_first_closure_s -0.560 K01; fem_vm_si_p95_100 -0.631 K09_mst_m_norm; fem_si_yield_frac_100 n/a (constant); fem_p_si_mean_100 +0.632 K15_si_graphite_contact_frac; fem_vm_binder_p95_100 +0.671 K01; fem_vm_gr_p95_100 +0.825 K01; fem_sxx_mean_100 -0.926 K01; fem_J_si_mean_100 -0.632 K15_si_graphite_contact_frac; fem_band_vm_maxdev_100 +0.575 K10_cv_w10.
- rho with K01 specifically: swell_50 0.903, swell_100 0.891, slope_early 0.909, slope_late 0.672, rough 0.421, pore_left -0.560, pore_closed 0.675, first_closure -0.560, vm_si_p95 0.550, p_si_mean -0.410, vm_binder 0.671, vm_gr 0.825, sxx -0.926, J_si_mean 0.410, band_vm_maxdev -0.513.
- Partial check (site swelling at s=1, sym; same numbers for site_curves swelling@s1.0 and curated fem_swell_100 site mean): KW p by batch = 0.2166. OLS on K01: R^2 = 0.971, slope 0.559. Residual KW p = 0.699; residual means B1 +0.00097, B2 -0.00003, B3 -0.00039; residual Cliff's delta B1-B3 +0.193, B2-B3 +0.160, B1-B2 +0.061; rank-regression residual KW p = 0.336.

### 4. Saturation (t2.py)
- si_yield_frac: equals 1 on every labelled site for every frame 1..10 (min = max = 1 per frame), 0 at frame 0. sigmaY(u=0.8) from the plan P9 formula = 637.5 MPa. vm_si_p95_MPa at s=1 site level: min 23414.9, median 24496.7, max 27445.9, std 1032; q50_vm_si@s1 range 16190-18867; q99_vm_si@s1 range 28147-34079 MPa.
- Site-level (mean of tiles) CV (std/|median|): vm_si_p95 0.039 (min 23577, 25th pct 23907, std 959); p_si_mean 0.034 (min 16381, 25th pct 17559, std 614); J_si_mean 0.010 (min 1.644, std 0.017); vm_binder_p95 0.137 (min 731, std 126); vm_gr_p95 0.131 (min 10769, std 1742); sxx_mean 0.227 (min -5466, std 648); band_vm_maxdev 0.178 (std 0.093).
- Tile-level CV: vm_si_p95 0.061 (std 1503), p_si_mean 0.060, J_si_mean 0.018, vm_binder 0.265, vm_gr 0.216, sxx 0.224, band_vm_maxdev 0.442. Mean within-site SD of tile vm_si_p95 = 1181 MPa; between-site SD of site means = 959 MPa.

### 5. Model sensitivity (t5.py)
FEM arm, labelled tile rows, repo loso()/level_metrics()/TwoStage with `pmdb.classify.model.make_stage_model` swapped for variants. Stress group (7) = vm_si_p95, si_yield_frac, p_si_mean, vm_binder_p95, vm_gr_p95, sxx_mean, band_vm_maxdev; non-stress (9) = swell_50, swell_100, swell_slope_early, swell_slope_late, surface_rough, pore_left, pore_closed_frac, first_closure_s, J_si_mean. LR = SimpleImputer(median) + StandardScaler + LogisticRegression(C=1, class_weight=balanced, max_iter=2000). Site-level = mean of the 6 tile features per site, 31 rows, one tile each. Balanced accuracy:

| variant | n feat | stage1 | stage2 | end-to-end | e2e site acc |
|---|---|---|---|---|---|
| baseline tile RF, all 16 (reproduces metrics.csv FEM 0.626/0.357/0.370) | 16 | 0.626 | 0.357 | 0.370 | 0.516 |
| (a) tile RF, non-stress | 9 | 0.584 | 0.286 | 0.437 | 0.581 |
| (b) tile LR, all 16 | 16 | 0.697 | 0.571 | 0.417 | 0.548 |
| (c) site RF, all 16 | 16 | 0.639 | 0.643 | 0.473 | 0.548 |
| (a)+(b) tile LR, non-stress | 9 | 0.639 | 0.571 | 0.473 | 0.548 |
| (a)+(c) site RF, non-stress | 9 | 0.668 | 0.571 | 0.445 | 0.548 |
| (b)+(c) site LR, all 16 | 16 | 0.645 | 0.571 | 0.454 | 0.516 |
| (a)+(b)+(c) site LR, non-stress | 9 | 0.616 | 0.286 | 0.387 | 0.452 |
| ref: KPI tile RF (repo, 15 feats) | 15 | 0.548 | 0.357 | 0.437 | 0.581 |

Repo metrics.csv end-to-end balanced accuracy: KPI 0.437, FEM 0.370, KPI+FEM 0.350; FEM e2e bootstrap CI 0.250-0.513; n=31 sites (stage 2 n=14). Each variant is a single LOSO run (random_state 0), no repeats, no CIs computed for the variants.

## Not found
- Constant curated features: only fem_si_yield_frac_100 (fem_first_closure_s has 2 distinct values; fem_sxx_mean_100 is per-site constant). No NaNs, no duplicates, no join or tile-grid misalignment found (S2).

## Not checked
- KPI+FEM arm sensitivity, multiplicity correction, seed repeats of sensitivity runs, tile-level KW (pseudo-replication), per-frame curves other than s=0.5 and 1.0, bottom/top orientation-specific features. Stress magnitudes (~24 GPa von Mises in Si) were not cross-checked against material parameters beyond sigmaY. Top-15 contains duplicated metrics (see section 2).

## Code
All run from the repo root with the repo default python. Files: <local scratch file>,t1.py,t1b.py,t2.py,t5.py}; contents appended below.

### pre.py
```python
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from pmdb.classify.features import *
raw=pd.read_csv('outputs/fem/tile_curves.csv')
cur=load_fem_tile_curves('outputs/fem/tile_curves.csv')
fem=fem_tile_features(cur)
kpi=load_kpi_tiles6('outputs/classifier/kpi_tiles6.csv')
```

### t1.py
```python
exec(open('<local scratch file>').read())
print(raw.orientation.value_counts().to_dict(), raw.shape, raw.heldout.value_counts().to_dict())
print('sym rows',len(cur),'sites',cur.site.nunique(),'dups key',cur.duplicated(['site','tile','frame']).sum())
lab=fem[~fem.heldout]; print('labelled fem tile rows',len(lab), 'dup',lab.duplicated(TILE_ID).sum())
print(lab.groupby('batch').site.nunique().to_dict())
print(f"{'feature':28s} nan nuniq const std")
for c in FEM_FEATURES:
    x=lab[c]; print(f"{c:28s} {x.isna().sum():3d} {x.nunique():4d} {x.nunique()<=1} {x.std():.4g} min {x.min():.4g} max {x.max():.4g}")
nn=lab[lab[list(FEM_FEATURES)].isna().any(axis=1)]; print('rows with any NaN',len(nn)); print(nn[['site','tile']+[c for c in FEM_FEATURES if nn[c].isna().any()]].head(20))
s=raw[raw.orientation=='sym']; print('sym converged False rows', (~s.converged.astype(bool)).sum(), 'failed_at_s<1 sites', s[s.failed_at_s<1].site.unique())
for o in ['bottom','top']:
    r=raw[raw.orientation==o]; print(o,'failed<1 sites',r[r.failed_at_s<1].site.unique().tolist())
k=kpi[~kpi.heldout]; j=k.merge(fem[TILE_ID+['tile_x0_um','tile_x1_um']],on=TILE_ID,how='outer',suffixes=('','_f'),indicator=True)
print(j._merge.value_counts().to_dict())
print('max |x0 diff|',(j.tile_x0_um-j.tile_x0_um_f).abs().max(),'max |x1 diff|',(j.tile_x1_um-j.tile_x1_um_f).abs().max())
print(k.groupby('site').size().value_counts().to_dict(), 'kpi dup',k.duplicated(TILE_ID).sum())
from pmdb.io import list_sites
m=list_sites(); print(m.columns.tolist())
wcol=[c for c in m.columns if 'width' in c.lower() or c in('W',)]; print(wcol)
if wcol:
    rows=[]
    for _,r in m.iterrows():
        if r['site'] in set(lab.site):
            W=int(r[wcol[0]]); sl=fem_grid_slices(W)
            x0=[a.start*0.05 for a in sl]; x1=[a.stop*0.05 for a in sl]
            g=lab[lab.site==r['site']].sort_values('tile')
            rows.append((np.abs(g.tile_x0_um.values-np.array(x0)).max(),np.abs(g.tile_x1_um.values-np.array(x1)).max(),W))
    a=np.array(rows); print('slice-vs-um maxdiff x0,x1',a[:,0].max(),a[:,1].max(),'n',len(a),'widths',sorted(set(a[:,2])))
print(lab[lab.site==lab.site.iloc[0]][['tile','tile_x0_um','tile_x1_um']].to_string())
b=raw[raw.orientation=='bottom'].set_index(['site','tile','frame']).swelling; t=raw[raw.orientation=='top'].set_index(['site','tile','frame']).swelling; sy=raw[raw.orientation=='sym'].set_index(['site','tile','frame']).swelling
print('max|sym-(b+t)/2| swelling',(sy-(b+t)/2).abs().max()); print('corr bottom/top swell s=1',np.corrcoef(b.xs(10,level='frame'),t.xs(10,level='frame'))[0,1])
print('max|b-t| swell s=1',(b-t).xs(10,level='frame').abs().max())
```

### t1b.py
```python
exec(open('<local scratch file>').read())
from pmdb.io import list_sites
m=list_sites(); lab=fem[~fem.heldout]
for div in (1,2):
    rows=[]
    for _,r in m.iterrows():
        if r['site'] in set(lab.site):
            W=int(r['width'])//div; sl=fem_grid_slices(W)
            x0=np.array([a.start*0.05 for a in sl]); x1=np.array([a.stop*0.05 for a in sl])
            g=lab[lab.site==r['site']].sort_values('tile')
            rows.append((np.abs(g.tile_x0_um.values-x0).max(),np.abs(g.tile_x1_um.values-x1).max(),W))
    a=np.array(rows); print('width div',div,'maxdiff x0,x1 um',a[:,0].max(),a[:,1].max(),'n',len(a))
print(m[m.site.isin(lab.site)].groupby('batch').width.agg(['min','max']))
```

### t2.py
```python
exec(open('<local scratch file>').read())
from scipy.stats import kruskal, spearmanr, rankdata

class _R: pass
class sm:
    @staticmethod
    def add_constant(x): return np.column_stack([np.ones(len(x)),x])
    class OLS:
        def __init__(s,y,X):
            s.y=y;s.X=X
        def fit(s):
            b=np.linalg.lstsq(s.X,s.y,rcond=None)[0]; r=_R(); r.params=b; r.resid=s.y-s.X@b; r.rsquared=1-r.resid.var()/s.y.var(); return r
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
sc=pd.read_csv('outputs/fem/site_curves.csv'); sc=sc[(sc.orientation=='sym')&(~sc.heldout.astype(str).eq('True'))]
sk=pd.read_csv('outputs/kpis/site_kpis.csv'); 
lab=fem[~fem.heldout]; bat=lab.groupby('site').batch.first()
cur_site=lab.groupby('site')[list(FEM_FEATURES)].mean()
metrics=[c for c in sc.columns[9:]]; print(len(metrics),'raw metrics')
raws={}
for s in (5,10):
    g=sc[sc.frame==s].set_index('site')[metrics]; g.columns=[f'{c}@s{s/10}' for c in g.columns]; raws[s]=g
rawsite=pd.concat(raws.values(),axis=1)
kn=[c for c in sk.columns if c not in('batch','site','se_detector','segmenter_version') and pd.api.types.is_numeric_dtype(sk[c])]
ksite=sk[sk.site.isin(bat.index)].set_index('site')[kn]
print('KPI site cols',len(kn),'sites',len(ksite))
def delta(a,b):
    a=np.asarray(a);b=np.asarray(b); 
    return (np.sign(a[:,None]-b[None,:])).mean()
def stats(df,src):
    out=[]
    for c in df.columns:
        x=df[c].reindex(bat.index)
        if x.isna().any() or x.nunique()<2: out.append(dict(feat=c,src=src,p=np.nan,note=f'nan={x.isna().sum()} nuniq={x.nunique()}'));continue
        gs=[x[bat==b].values for b in BATCHES]
        p=kruskal(*gs).pvalue
        out.append(dict(feat=c,src=src,p=p,d13=delta(gs[0],gs[2]),d23=delta(gs[1],gs[2]),d12=delta(gs[0],gs[1])))
    return pd.DataFrame(out)
R=pd.concat([stats(cur_site,'FEM curated(site mean of tiles)'),stats(rawsite,'FEM raw'),stats(ksite,'KPI')],ignore_index=True)
R['maxabs_d']=R[['d13','d23','d12']].abs().max(axis=1)
R.to_csv('<local scratch file>',index=False)
print(R.groupby('src').p.apply(lambda s:(s.notna().sum(),(s<0.05).sum(),(s<0.01).sum())))
print('NaN/constant rows:'); print(R[R.p.isna()])
Rr=R.dropna(subset=['p']).sort_values('p')
print('TOP 15 by KW p'); print(Rr.head(15).to_string())
print('TOP 15 by max|delta|'); print(Rr.sort_values('maxabs_d',ascending=False).head(15).to_string())
for s in ['FEM curated(site mean of tiles)','KPI']:
    print(s); print(Rr[Rr.src==s].head(10 if s.startswith('KPI') else 16).to_string())
print('FEM raw top 10'); print(Rr[Rr.src=='FEM raw'].head(10).to_string())
# 3 redundancy
K=ksite.copy()
rows=[]
for c in cur_site.columns:
    x=cur_site[c].reindex(K.index)
    if x.nunique()<2: rows.append((c,np.nan,'const',np.nan));continue
    best=(0,None)
    for k in K.columns:
        if K[k].nunique()<2 or K[k].isna().any(): continue
        r=spearmanr(x,K[k])[0]
        if abs(r)>abs(best[0]): best=(r,k)
    rows.append((c,best[0],best[1],None))
print('max|rho| per curated FEM feature vs any KPI (site level, n=31, nKPI cols tested)'); print(pd.DataFrame(rows,columns=['feat','rho','kpi','_']).to_string())
print('rho vs K01 specifically:')
for c in cur_site.columns:
    if cur_site[c].nunique()>1: print(f'{c:28s} {spearmanr(cur_site[c].reindex(K.index),K["K01_si_frac_adm"])[0]:.3f}')
# partial: swelling s=1 (site-level raw swelling@s1.0) ~ K01, residual by batch
for name,y in [('site swelling@s1.0',rawsite['swelling@s1.0']),('curated fem_swell_100 site mean',cur_site['fem_swell_100'])]:
    y=y.reindex(bat.index); x=ksite['K01_si_frac_adm'].reindex(bat.index)
    print(name,'spearman vs K01',spearmanr(x,y)[0], 'KW raw p',kruskal(*[y[bat==b] for b in BATCHES]).pvalue)
    X=sm.add_constant(x.values); res=sm.OLS(y.values,X).fit(); r=pd.Series(res.resid,index=bat.index)
    print(' OLS R2',res.rsquared,'slope',res.params[1],'resid KW p',kruskal(*[r[bat==b] for b in BATCHES]).pvalue, 'resid means',{b:round(r[bat==b].mean(),5) for b in BATCHES}, 'resid d13,d23,d12',[round(delta(r[bat==a].values,r[bat==b].values),3) for a,b in [('Batch_1','Batch_3'),('Batch_2','Batch_3'),('Batch_1','Batch_2')]])
    # rank-based
    rx=pd.Series(rankdata(x),index=x.index); ry=pd.Series(rankdata(y),index=y.index); res=sm.OLS(ry.values,sm.add_constant(rx.values)).fit(); r=pd.Series(res.resid,index=bat.index)
    print(' rank-OLS resid KW p',kruskal(*[r[bat==b] for b in BATCHES]).pvalue)
    print(' K01 KW p',kruskal(*[x[bat==b] for b in BATCHES]).pvalue)
# 4 saturation
st=['fem_vm_si_p95_100','fem_si_yield_frac_100','fem_p_si_mean_100','fem_vm_binder_p95_100','fem_vm_gr_p95_100','fem_sxx_mean_100','fem_band_vm_maxdev_100','fem_J_si_mean_100']
print('SITE-level (mean of tiles) distributions'); print(cur_site[st].describe().T[['min','25%','50%','75%','max','std']].assign(cv=lambda d:d['std']/d['50%'].abs()))
print('per batch median/IQR of site means');print(cur_site[st].groupby(bat).agg(['median']).T)
print('tile-level'); print(lab[st].describe().T[['min','25%','50%','75%','max','std']].assign(cv=lambda d:d['std']/d['50%'].abs()))
sc2=sc[sc.frame==10].set_index('site')
print('site-level raw s=1 stress: vm_si_p95',sc2.vm_si_p95_MPa.describe()[['min','50%','max','std']].to_dict(),'si_yield_frac',sc2.si_yield_frac.unique(),'q50_vm_si',sc2.q50_vm_si.describe()[['min','max']].to_dict(), 'q99_vm_si',sc2.q99_vm_si.describe()[['min','max']].to_dict())
print('yield-frac by frame (site mean):',sc.groupby('frame').si_yield_frac.agg(['min','max']).T.round(3).to_dict())
print('within-site tile sd vs between-site sd (fem_vm_si_p95_100):', lab.groupby('site').fem_vm_si_p95_100.std().mean(), lab.groupby('site').fem_vm_si_p95_100.mean().std())
print('Si vm vs Si-yield limit sigmaY(u=0.8): ',3000-3150*3.0/(1+3.0))
```

### t5.py
```python
exec(open('<local scratch file>').read())
import pmdb.classify.model as M
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
table,feats=build_arm_table('FEM',None,fem)
STRESS=['fem_vm_si_p95_100','fem_si_yield_frac_100','fem_p_si_mean_100','fem_vm_binder_p95_100','fem_vm_gr_p95_100','fem_sxx_mean_100','fem_band_vm_maxdev_100']
nonstress=[f for f in feats if f not in STRESS]
print('non-stress',len(nonstress),nonstress)
def lr_model():
    return Pipeline([('imp',SimpleImputer(strategy='median',keep_empty_features=True)),('sc',StandardScaler()),
                     ('rf',LogisticRegression(C=1.0,class_weight='balanced',max_iter=2000))])
rf_model=M.make_stage_model
def site_table(t):
    lab=t[~t.heldout]; g=lab.groupby(['batch','site'],as_index=False)[feats+['tile_x0_um']].mean(); g['tile']=0; g['heldout']=False; return g
def run(name,tbl,fs,mk):
    M.make_stage_model=mk
    sp,tp=M.loso(tbl,fs); m=M.level_metrics(sp).set_index('level').balanced_acc
    print(f'{name:55s} n_feat={len(fs):2d} stage1={m.stage1:.3f} stage2={m.stage2:.3f} e2e={m.end_to_end:.3f}  site_acc_e2e={sp.correct.mean():.3f}')
    M.make_stage_model=rf_model; return sp
st=site_table(table)
run('baseline: tile RF, 16 feats (repo)',table,feats,rf_model)
run('(a) tile RF, non-stress',table,nonstress,rf_model)
run('(b) tile LR, 16 feats',table,feats,lr_model)
run('(c) site RF, 16 feats',st,feats,rf_model)
run('(a)+(b) tile LR, non-stress',table,nonstress,lr_model)
run('(a)+(c) site RF, non-stress',st,nonstress,rf_model)
run('(b)+(c) site LR, 16 feats',st,feats,lr_model)
run('(a)+(b)+(c) site LR, non-stress',st,nonstress,lr_model)
# KPI arm baseline reference & KPI site-level LR for context
kt,kf=build_arm_table('KPI',kpi,None)
run('ref: KPI tile RF (repo)',kt,kf,rf_model)
```
