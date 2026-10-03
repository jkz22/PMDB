# Spec 002 KPI status

## Result

All 25 v1 KPI IDs (43 site-level output columns) are finite for all 31 sites. All requested per-tile
columns are present for 124/124 tiles, with no tile NaNs. The coverage script exits 0.

| Group | IDs | Status |
|---|---|---|
| Si objects and clustering | K01-K09 | 31/31, complete |
| Localisation | K10-K14 | 31/31, complete |
| Cross-phase | K15-K16 | 31/31, complete |
| Diagnostics | D01-D06 | 31/31, complete |
| Artefacts | A01-A03 | 31/31, complete |

## Validation

- Synthetic/non-data suite: 20 passed.
- Real-data suite: 1 passed.
- `scripts/check_kpi_coverage.py`: 25/25 v1 KPIs covered, exit 0.
- Production run: 31/31 sites, 56.3 seconds with `--jobs 8`.
- Segmentation review: 31/31 overlays inspected; no gross site-specific failures.
- K01: 0.0437-0.1645 across all sites, within the 0.001-0.50 sanity interval.

## Output inventory

- `outputs/kpis/site_kpis.csv`: 31 site rows.
- `outputs/kpis/tile_kpis.csv`: 124 tile rows.
- `outputs/kpis/curves.csv`: K08, K10, K12, K13, D01 and D04 curves.
- `outputs/kpis/sensitivity.csv`: all nine K04 `(d, d*)` combinations per site.
- `outputs/kpis/run_log.json`: commit, parameters, versions, timings and per-site status.
- `outputs/kpis/figures/`: verdict KPI strip plots and K10 curves.
- `outputs/overlays/`: 31 review PNGs.

## Approved deviations from revision 2

These departures were approved after the locked definition caused K01 >50% at 29/31 sites and made
K10/K11 site values non-finite. The output column names remain unchanged for catalogue/schema stability.

1. K01 and K02 use non-artefact slice area (`~artefact`, graphite included) rather than admissible area.
2. K10-K13 use non-artefact pixels as the local fraction denominator; their window eligibility test uses
   non-artefact fraction >=0.5. Point-pattern admissible space is unchanged.
3. D-013 and the stripe known-answer test remain unchanged: K05-K09 and K14 use
   `admissible = ~graphite & ~artefact`, so random points cannot land inside graphite.
4. D-011's one allowed global revision was used: the Si threshold is
   `p99 - 0.5 * (p99 - p50)` instead of `median_solid + 4 * MAD_solid`.
5. K08 uses Stoyan's intensity-dependent bandwidth, `h = 0.15 / sqrt(lambda_adm)`, because D-016 locks
   radii/sectors/envelope semantics but does not specify a kernel bandwidth.

See [segmentation-v0-notes.md](segmentation-v0-notes.md) for overlay evidence and per-batch K01 ranges.
