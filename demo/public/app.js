import * as fp from '/lib/fingerprint.js';
import * as cf from '/lib/confound.js';
import { el, lineChart, stripPlot, histogram, scatter, BATCH_COLOR, BATCH_LABEL } from './charts.js';
import { findCell, headerRow, clampTip, storyIsStale, evidenceIsStale, topPcs, reopenSite } from './embed_logic.js';

const BATCHES = ['Batch_1', 'Batch_2', 'Batch_3'];
const short = (b) => b.replace('Batch_', 'Batch ').replace('heldout', 'held-out');
const f2 = (v) => (v == null ? '—' : Number(v).toFixed(2));
const f3 = (v) => (v == null ? '—' : Number(v).toFixed(3));
const pct = (v) => `${Math.round(v * 100)}%`;
const median = (a) => { const s = [...a].sort((x, y) => x - y), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };

const state = {
  view: 'confound', bundle: null, rendered: false,
  confound: { site: '4ih2ggld', delta: 0 },
  arrangement: { show: { Batch_1: true, Batch_2: true, Batch_3: true, Batch_heldout: true } },
  reject: { ti: 0 },
  explorer: { key: 'Batch_3/cfe5vt7s' },
};

const VIEWS = [
  { id: 'confound', title: 'The trap' },
  { id: 'arrangement', title: 'Amount vs arrangement' },
  { id: 'proof', title: 'The proof' },
  { id: 'calls', title: 'Round 1 calls' },
  { id: 'reject', title: 'Reject a shipment' },
  { id: 'explorer', title: 'Site explorer' },
  { id: 'embeddings', title: 'Test calls' },
];

// ---------------------------------------------------------------- html helper
function h(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else if (k === 'html') n.innerHTML = v;
    else n.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) if (c != null && c !== false) n.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return n;
}
const card = (title, body, src, attrs = {}) => h('div', { class: 'card', ...attrs }, title ? h('h3', {}, title) : null, body, src ? h('div', { class: 'src' }, src) : null);
const chip = (batch) => h('span', { class: `chip b-${batch}` }, short(batch));

// ---------------------------------------------------------------- derived model state
let M = null;
function derive(bundle) {
  const D = bundle.data;
  const featRows = D.features;
  const labels = featRows.map((r) => r.batch);
  const pick = (r) => Object.fromEntries(fp.FEATURES.map((f) => [f, r[f]]));
  const X = featRows.map(pick);
  const model = fp.fit(X, labels, fp.FEATURES);
  const inSample = fp.predict(model, X);
  const b3 = labels.map((l, i) => [l, i]).filter(([l]) => l === 'Batch_3').map(([, i]) => i);
  const baseIdx = b3.reduce((best, i) => (inSample[i].p.Batch_3 > inSample[best].p.Batch_3 ? i : best), b3[0]);
  const tGrid = Array.from({ length: 41 }, (_, i) => +(i * 0.05).toFixed(2));
  const trajectory = tGrid.map((t) => { const row = fp.drift(X[baseIdx], t); return { t, row, pred: fp.predict(model, [row])[0] }; });
  const heldX = D.heldoutFeatures.map(pick);
  const heldLive = fp.predict(model, heldX).map((p, i) => ({ ...p, site: D.heldoutFeatures[i].site }));

  const bse = D.rawStats.filter((r) => r.detector === 'BSE');
  const bseRows = bse.map((r) => cf.FEATURES.map((f) => r[f]));
  const bseLabels = bse.map((r) => r.batch);
  const intensityLoo = cf.looAccuracy(bseRows, bseLabels);

  const kpi = new Map([...D.siteKpis, ...D.heldoutSiteKpis].map((r) => [`${r.batch}/${r.site}`, r]));
  const fem = new Map(D.femValidation.map((r) => [`${r.batch}/${r.site}`, r]));
  const loo = new Map(D.looPredictions.map((r) => [`${r.batch}/${r.site}`, r]));
  const bseByKey = new Map([...bse, ...D.heldoutRawStats.filter((r) => r.detector === 'BSE')].map((r) => [`${r.batch}/${r.site}`, r]));
  M = { D, featRows, labels, X, model, baseIdx, base: featRows[baseIdx], trajectory, heldLive, bse, bseRows, bseLabels, intensityLoo, kpi, fem, loo, bseByKey };
}

// ---------------------------------------------------------------- views
const R = {};

R.confound = (root) => {
  const s = state.confound;
  const sites = M.bse.filter((r) => r.batch !== 'Batch_3');
  const idx = M.bse.findIndex((r) => r.site === s.site);
  const row = M.bse[idx];
  const x0 = M.bseRows[idx];
  const xd = cf.offsetSite(x0, s.delta);
  const pred = cf.nearestCentroid(M.bseRows, M.bseLabels, xd);
  const wrong = pred !== row.batch;
  const nonB3 = M.bse.map((r, i) => [r, i]).filter(([r]) => r.batch !== 'Batch_3');
  const flipped = nonB3.filter(([, i]) => cf.nearestCentroid(M.bseRows, M.bseLabels, cf.offsetSite(M.bseRows[i], s.delta)) === 'Batch_3').length;
  const flipped0 = nonB3.filter(([, i]) => cf.nearestCentroid(M.bseRows, M.bseLabels, M.bseRows[i]) === 'Batch_3').length;
  const looRow = M.loo.get(`${row.batch}/${row.site}`);

  const canvas = h('canvas', { class: 'bse', width: 1400, height: 464 });
  drawOffsetImage(canvas, `/outputs/gallery/img/${row.batch}__${row.site}_0.jpg`, s.delta);

  const groups = BATCHES.map((b) => ({ key: b, label: short(b), color: BATCH_COLOR[b],
    values: M.bse.filter((r) => r.batch === b && r.site !== row.site).map((r) => ({ v: r.p1, label: r.site })) }));
  const gi = BATCHES.indexOf(row.batch);
  groups[gi].values.push({ v: xd[0], label: `${row.site} +${s.delta}`, star: true });

  const slider = h('input', { type: 'range', min: 0, max: 25, step: 1, value: s.delta, id: 'delta',
    oninput: (e) => { s.delta = +e.target.value; render(); } });
  const sel = h('select', { onchange: (e) => { s.site = e.target.value; render(); } },
    sites.map((r) => h('option', { value: r.site, selected: r.site === s.site }, `${r.site} (${short(r.batch)})`)));

  root.append(
    head('1 · The trap', 'The obvious model learns the microscope, not the material.'),
    h('p', { class: 'lede' }, `An intensity-only classifier (BSE black level p1, median p50, fraction of zero pixels — no geometry) scores `,
      h('b', {}, f2(M.intensityLoo)), ` leave-one-site-out, better than our model. Now add a flat grey-level offset — a microscope knob, the electrode untouched.`),
    h('div', { class: 'row grow' },
      h('div', { class: 'col', style: 'flex:1.35' },
        card(h('span', {}, 'BSE cross-section · ', sel), h('div', {}, canvas,
          h('div', { style: 'margin-top:10px' }, h('div', { class: 'slider-label' }, h('span', {}, 'grey-level offset added to every pixel'), h('b', { class: 'mono' }, `+${s.delta} DN`)), slider)),
          `image: outputs/gallery/img/${row.batch}__${row.site}_0.jpg (display offset is illustrative; model inputs are outputs/raw_intensity_stats.csv + offset)`),
        h('div', { class: 'row' },
          card('Intensity model says', h('div', {}, h('div', { class: `mid ${wrong ? 'bad' : 'good'}` }, `${short(pred)} ${wrong ? '✗' : '✓'}`),
            h('div', { class: 'note' }, `truth: ${short(row.batch)} · p1 ${f2(x0[0])} → ${f2(xd[0])}`)), null, { style: 'flex:1' }),
          card('Fingerprint model says', h('div', {}, h('div', { class: `mid ${looRow.assigned === row.batch ? 'good' : 'warn'}` }, `${short(looRow.assigned)} ${looRow.assigned === row.batch ? '✓' : '✗'}`),
            h('div', { class: 'note' }, 'leave-one-out call · built from segmentation geometry only, it has no grey-level input to shift')), 'outputs/fingerprint/loo_predictions.csv', { style: 'flex:1' }),
          card('Non-Batch-3 sites now called Batch 3', h('div', {}, h('div', { class: `mid ${flipped > flipped0 ? 'bad' : ''}` }, `${flipped} / ${nonB3.length}`),
            h('div', { class: 'note' }, `${flipped0} at +0 · same offset applied to every Batch 1/2 site`)), null, { style: 'flex:1' }),
        )),
      h('div', { class: 'col', style: 'flex:1' },
        card('BSE black level (p1) by batch — the star is this site', h('div', { style: 'height:360px' }, stripPlot({ groups, yLabel: 'BSE p1 (DN)', yFmt: (v) => v.toFixed(0) })),
          'outputs/raw_intensity_stats.csv'),
        card(null, h('p', { class: 'note' }, 'Batch 3 was largely imaged with an elevated BSE black level (the 4 strongest sites: black level ≈ +19–23, gain ≈ 0.69×). That is an acquisition setting correlated with batch, not a property of the material. Any pixel- or embedding-trained model inhales it.'), 'AGENTS.md · docs/harmonisation.md'),
      )));
};

function drawOffsetImage(canvas, src, delta) {
  const img = new Image();
  img.onload = () => {
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    if (!delta) return;
    const d = ctx.getImageData(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < d.data.length; i += 4) { d.data[i] = Math.min(255, d.data[i] + delta); d.data[i + 1] = Math.min(255, d.data[i + 1] + delta); d.data[i + 2] = Math.min(255, d.data[i + 2] + delta); }
    ctx.putImageData(d, 0, 0);
  };
  img.src = src;
}

R.arrangement = (root) => {
  const show = state.arrangement.show;
  const kRows = [...M.kpi.values()];
  const groups = [...BATCHES, 'Batch_heldout'].map((b) => ({ key: b, label: short(b), color: BATCH_COLOR[b],
    values: kRows.filter((r) => r.batch === b).map((r) => ({ v: r.K01_si_frac_adm, label: r.site, star: b === 'Batch_heldout' })) }));

  const pts = [];
  for (const [key, fr] of M.fem) { const k = M.kpi.get(key); if (k) pts.push({ x: k.K01_si_frac_adm, y: fr.swelling_sym, color: BATCH_COLOR[fr.batch], label: `${fr.site} (${short(fr.batch)})`, star: fr.batch === 'Batch_heldout', lab: fr.batch !== 'Batch_heldout' }); }
  const lab = pts.filter((p) => p.lab);
  const mx = lab.reduce((a, p) => a + p.x, 0) / lab.length, my = lab.reduce((a, p) => a + p.y, 0) / lab.length;
  const sxy = lab.reduce((a, p) => a + (p.x - mx) * (p.y - my), 0), sxx = lab.reduce((a, p) => a + (p.x - mx) ** 2, 0), syy = lab.reduce((a, p) => a + (p.y - my) ** 2, 0);
  const b = sxy / sxx, a = my - b * mx, r2 = (sxy * sxy) / (sxx * syy);
  const medSwell = Object.fromEntries(BATCHES.map((bb) => [bb, median(M.D.femValidation.filter((r) => r.batch === bb).map((r) => r.swelling_sym))]));

  const bands = [0, 1, 2, 3, 4];
  const series = [];
  const all = [...M.featRows.map((r) => ({ ...r })), ...M.D.heldoutFeatures.map((r) => ({ ...r, batch: 'Batch_heldout' }))];
  for (const r of all) {
    if (!show[r.batch] || r.batch === 'Batch_heldout') continue;
    series.push({ points: bands.map((k) => [k, r[`si_depth_rel_band${k}`]]), color: BATCH_COLOR[r.batch], width: 1, opacity: 0.25, label: r.site });
  }
  for (const bb of BATCHES) if (show[bb]) {
    const rows = M.featRows.filter((r) => r.batch === bb);
    series.push({ points: bands.map((k) => [k, median(rows.map((r) => r[`si_depth_rel_band${k}`]))]), color: BATCH_COLOR[bb], width: 5, marker: 5, label: `${short(bb)} median` });
  }
  if (show.Batch_heldout) for (const r of M.D.heldoutFeatures) series.push({ points: bands.map((k) => [k, r[`si_depth_rel_band${k}`]]), color: '#111', width: 2.2, dash: '6 4', label: `held-out ${r.site}` });
  const toggles = h('div', { class: 'tabs-mini' }, [...BATCHES, 'Batch_heldout'].map((bb) => h('button', { class: show[bb] ? 'active' : '', onclick: () => { show[bb] = !show[bb]; render(); } }, short(bb))));

  root.append(
    head('2 · The finding', 'Same amount of silicon. Different arrangement.'),
    h('p', { class: 'lede' }, 'We simulated lithiation inside every real microstructure (68 FEM cases on Modal). Swelling is set by how much Si there is — and the batches hold the same amount. The signal is where the Si sits.'),
    h('div', { class: 'row' },
      card('Si area fraction (K01) by batch', h('div', { style: 'height:300px' }, stripPlot({ groups, w: 480, h: 300, yLabel: 'Si area fraction' })),
        'outputs/kpis/site_kpis.csv · Kruskal–Wallis across batches p = 0.377 (.claude/checkpoint/reports/fem-why.explorer-diag.md)', { style: 'flex:1' }),
      card('Simulated swelling at full charge vs Si fraction', h('div', {}, h('div', { style: 'height:270px' }, scatter({ points: pts, w: 520, h: 290, fit: { a, b }, xLabel: 'Si area fraction (K01)', yLabel: 'electrode swelling', yFmt: (v) => v.toFixed(2) })),
          h('div', { class: 'note' }, h('b', {}, `R² = ${r2.toFixed(3)}`), ` (OLS, ${lab.length} labelled sites, computed live) · median swelling B1 ${f3(medSwell.Batch_1)} · B2 ${f3(medSwell.Batch_2)} · B3 ${f3(medSwell.Batch_3)}`)),
        'outputs/fem/validation.csv (swelling_sym) × outputs/kpis/site_kpis.csv', { style: 'flex:1.1' }),
    ),
    card(h('span', { style: 'display:flex;justify-content:space-between;align-items:center' }, 'Si depth profile, relative to each site’s mean — the fingerprint', toggles),
      h('div', { class: 'row', style: 'align-items:center' },
        h('div', { style: 'height:300px;flex:1' }, series.length ? lineChart({ series, w: 900, h: 300, xTicks: bands, xFmt: (k) => ['top', 'band 1', 'mid-depth', 'band 3', 'bottom'][k], yLabel: 'Si / site mean' }) : h('div', { class: 'note' }, 'No batch selected — toggle a batch above to draw its depth profile.')),
        h('div', { class: 'note', style: 'width:260px' }, h('p', {}, h('b', { class: 'b-Batch_3' }, 'Batch 3 (baseline)'), ': flat — uniform through the coating.'), h('br'),
          h('p', {}, h('b', { class: 'b-Batch_2' }, 'Batch 2'), ': top-heavy, depleted mid-depth (drying migration).'), h('br'),
          h('p', {}, h('b', { class: 'b-Batch_1' }, 'Batch 1'), ': bottom-heavy, variable (sedimentation).'), h('br'),
          h('p', {}, 'Dashed: the 3 held-out sites. Mid-depth band KW p ≈ 0.001 (docs/fingerprint.md).'))),
      'outputs/fingerprint/features.csv · outputs/fingerprint/heldout_features.csv (si_depth_rel_band0–4)'),
  );
};

R.proof = (root) => {
  const D = M.D, ev = D.evaluation, p10 = D.permutation10k, ext = D.extendedEval;
  const nulls = D.nullAccuracies.map((r) => r.accuracy);
  const n = ev.n_sites, correct = Math.round(ev.loo.accuracy * n);
  const conf = ev.loo.confusion;
  const cm = h('table', { class: 'confusion' }, h('tr', {}, h('th', {}, 'true ↓ / called →'), BATCHES.map((b) => h('th', { class: `b-${b}` }, short(b)))),
    BATCHES.map((t) => h('tr', {}, h('th', { class: `b-${t}` }, short(t)), BATCHES.map((p) => {
      const v = conf[t][p], tot = BATCHES.reduce((a, q) => a + conf[t][q], 0);
      return h('td', { style: `background:${t === p ? 'rgba(10,122,42,' : 'rgba(201,52,52,'}${(0.08 + 0.6 * v / tot).toFixed(2)})` }, v);
    }))));
  const d21 = [...D.d21Main.map((r) => ({ ...r, tag: 'main' })), ...D.d21Edge5.filter((r) => r.arm !== 'A0').map((r) => ({ ...r, tag: 'edge5' }))];
  const armName = { A0: 'A0 · fingerprint (16 features)', A1: 'A1 · + 2 physics FEM features', A2: 'A2 · + 2 of 160 raw FEM metrics' };
  const tests = h('table', {}, h('tr', {}, h('th', {}, 'pre-registered test'), h('th', { class: 'num' }, 'LOO correct'), h('th', { class: 'num' }, 'perm p'), h('th', {}, 'rule'), h('th', {}, 'verdict')),
    h('tr', {}, h('td', {}, `Extended fingerprint (${ext.n_features} features)`), h('td', { class: 'num' }, `${Math.round(ext.loo.accuracy * n)}/${n}`), h('td', { class: 'num' }, ext.permutation.p_value.toFixed(4)),
      h('td', {}, `≥ ${f3(ext.rule.loo_min)} and p ≤ ${ext.rule.p_max}`), h('td', { class: ext.adopt ? 'good' : 'bad' }, ext.adopt ? 'adopted' : 'rejected')),
    d21.map((r) => h('tr', {}, h('td', {}, `${armName[r.arm] || r.arm}${r.tag === 'edge5' ? ' (edge5 sensitivity)' : ''}`), h('td', { class: 'num' }, `${r.n_correct}/${r.n}`), h('td', { class: 'num' }, r.perm_p.toFixed(4)),
      h('td', {}, r.arm === 'A0' ? 'reference' : '≥ 23/31 and p ≤ 0.05'), h('td', { class: r.passes_rule === 'reference' ? '' : (r.passes_rule === 'True' ? 'good' : 'bad') }, r.passes_rule === 'reference' ? 'classifier of record' : (r.passes_rule === 'True' ? 'passes' : 'rejected')))));

  root.append(
    head('3 · The proof', 'Small model, adversarially validated.'),
    h('p', { class: 'lede' }, '31 sites forbid fitted weights: robust per-batch naive Bayes on 16 curve-shape features, Mondrian conformal p-values for confidence. Every claim is stress-tested instead of tuned.'),
    h('div', { class: 'row' },
      card('Leave-one-site-out', h('div', {}, h('div', { class: 'big' }, `${correct}/${n}`), h('div', { class: 'note' }, `accuracy ${f3(ev.loo.accuracy)} vs majority baseline ${f3(ev.majority_baseline)}`)), 'outputs/fingerprint/evaluation.json', { style: 'flex:1' }),
      card(`Label permutations (${p10.n_perm_done.toLocaleString()})`, h('div', {}, h('div', { class: 'big' }, `p = ${p10.p_value.toFixed(4)}`), h('div', { class: 'note' }, `${p10.n_null_ge_observed} of ${p10.n_perm_done.toLocaleString()} shuffles reach ${f3(p10.observed_accuracy)} · null mean ${f3(p10.null_mean)}`)), 'outputs/overnight/permutation/permutation_10k.json', { style: 'flex:1' }),
      card('Recall per batch', h('div', { class: 'kv', style: 'font-size:17px' }, BATCHES.flatMap((b) => [h('dt', { class: `b-${b}` }, short(b)), h('dd', {}, `${conf[b][b]}/${BATCHES.reduce((a, q) => a + conf[b][q], 0)}`)])), 'outputs/fingerprint/evaluation.json', { style: 'flex:0.8' }),
    ),
    h('div', { class: 'row grow' },
      card('Null distribution: accuracy with shuffled batch labels', h('div', { style: 'height:290px' }, histogram({ values: nulls, lo: -0.5 / 31, hi: 26.5 / 31, width: 1 / 31, w: 700, h: 290, xLabel: 'LOO accuracy',
        marks: [{ x: ev.majority_baseline, label: `majority baseline ${f3(ev.majority_baseline)}`, color: '#8a8882', dash: '5 4', anchor: 'end' }, { x: p10.observed_accuracy, label: `ours ${f3(p10.observed_accuracy)}`, color: '#0a7a2a' }] })),
        'outputs/overnight/permutation/null_accuracies.csv', { style: 'flex:1.2' }),
      h('div', { class: 'col', style: 'flex:1' },
        card('Confusion matrix (LOO)', cm, 'outputs/fingerprint/evaluation.json'),
        card('Rules written before the run — nothing beat the fingerprint', tests, 'D21 arms use 1,000 permutations (hence p = 0.0060 for A0 vs 0.0025 at 10,000) · docs/eval-plan-oct4.md · outputs/overnight/eval_extended/evaluation.json · outputs/fem_fingerprint/{main,edge5}/metrics.csv'),
      )),
  );
};

R.calls = (root) => {
  const D = M.D;
  const stored = new Map(D.heldoutPredictions.map((r) => [r.site, r]));
  let maxDiff = 0, sameCalls = true;
  for (const p of M.heldLive) {
    const s = stored.get(p.site);
    if (!s || s.assigned !== p.assigned || String(s.ood) !== (p.ood ? 'True' : 'False')) sameCalls = false;
    if (s) maxDiff = Math.max(maxDiff, Math.abs(s.confidence - p.confidence), Math.abs(s.credibility - p.credibility));
    if (s) for (const b of BATCHES) maxDiff = Math.max(maxDiff, Math.abs(s[`p_${b}`] - p.p[b]), Math.abs(s[`score_${b}`] - p.score[b]));
  }
  const ok = sameCalls && maxDiff < 1e-9;
  const jk = D.jackknife.filter((r) => r.drop_k === 3);
  const hp = D.hyperparams;
  const explain = D.heldoutExplain.map((r) => ({ ...r, site: (String(r.site_index).match(/'([^']+)'\)$/) || [])[1] }));
  const narrative = {
    xrv9xvzb: 'Mid-depth Si dip below every Batch 3 site and inside the Batch 2 range — the Batch 2 pattern. It carries a Batch-3-style BSE black level (p1 = 6): we bet on the material, not the microscope.',
    fn0mhxef: 'Mid-depth Si in a range only Batch 3 occupies; Batch 1 effectively excluded. Not a clean baseline match: Batch 2 stays typical overall.',
    '3e122cbj': 'Arrangement typical of every batch, so confidence is 0 — the model refuses to manufacture certainty. It differs from Batch 3 in amount: Si fraction z ≈ +8 vs Batch 3, a loading only Batch 1 reaches.',
  };
  const TRUTH = { fn0mhxef: 'Batch_1', '3e122cbj': 'Batch_2', xrv9xvzb: 'Batch_3' };
  const cards = M.heldLive.map((p) => {
    const j = jk.find((r) => r.site === p.site);
    const hpRows = hp.filter((r) => r.site === p.site);
    const hpAgree = hpRows.filter((r) => r.assigned === p.assigned).length;
    const top = explain.filter((r) => r.site === p.site).sort((a, b) => b.dev_Batch_3 - a.dev_Batch_3).slice(0, 3);
    return h('div', { class: 'card call' },
      h('div', { class: 'site' }, p.site),
      h('div', { class: `batch b-${p.assigned}` }, short(p.assigned)),
      h('div', { class: 'note', style: 'margin:4px 0 8px;font-weight:600' },
        'revealed truth: ',
        h('span', { class: `b-${TRUTH[p.site]}`, style: 'font-weight:700' }, short(TRUTH[p.site])),
        p.assigned === TRUTH[p.site] ? ' ✓' : ' ✗ (our call was wrong)'),
      h('div', { class: 'kv' }, h('dt', {}, 'credibility'), h('dd', {}, f3(p.credibility)), h('dt', {}, 'confidence'), h('dd', {}, f3(p.confidence)), h('dt', {}, 'out of distribution'), h('dd', {}, p.ood ? 'yes' : 'no')),
      h('div', { style: 'margin:10px 0 4px', class: 'note' }, 'conformal p-value per batch'),
      pbars(p.p),
      h('div', { class: 'kv', style: 'margin-top:10px' },
        h('dt', {}, 'jackknife (drop 3 sites)'), h('dd', {}, j ? `${pct(j[`share_${p.assigned}`])} of ${j.n_resamples} refits` : '—'),
        h('dt', {}, 'hyperparameter grid'), h('dd', {}, `${hpAgree}/${hpRows.length} configs`)),
      h('div', { style: 'margin-top:10px', class: 'note' }, h('b', {}, 'Largest deviations from Batch 3: '), top.map((r) => `${r.feature} (${f2(r.dev_Batch_3)})`).join(', ')),
      h('p', { class: 'note', style: 'margin-top:8px' }, narrative[p.site] || ''),
    );
  });
  root.append(
    head('4 · Round 1 calls — truth revealed', 'Three held-out sites: our calls scored 0/3, a perfect rotation of the truth.'),
    h('p', { class: 'lede' }, 'The organisers revealed the labels: every model in the field missed (best team 2/3). The stitching analysis explains why — the batch label follows the crop, not the material. These cards are kept as submitted, with the truth badged. Round 2 (6 new images, the probe model) is in the Test calls view.'),
    h('div', { class: 'calls' }, cards),
    card(null, h('div', { class: `check ${ok ? 'good' : 'bad'}` }, ok
      ? `✓ recomputed live in this browser from outputs/fingerprint/features.csv: identical to outputs/fingerprint/heldout_predictions.csv (max |Δ| = ${maxDiff.toExponential(1)})`
      : `✗ live recomputation differs from outputs/fingerprint/heldout_predictions.csv (calls ${sameCalls ? 'same' : 'DIFFER'}, max |Δ| = ${maxDiff.toExponential(2)}) — results on disk changed; re-run scripts/run_fingerprint.py`),
      'outputs/overnight/stability/jackknife_summary.csv · hyperparam_grid.csv · outputs/fingerprint/heldout_explain.csv'),
  );
};

function pbars(p, { alpha = 0.1, floors } = {}) {
  return h('div', { class: 'pbars' }, BATCHES.map((b) => h('div', { class: 'pbar' },
    h('span', { class: `b-${b}`, style: 'font-weight:600' }, short(b)),
    h('div', { class: 'track' }, h('div', { class: 'fill', style: `width:${(p[b] * 100).toFixed(1)}%;background:${BATCH_COLOR[b]}` }),
      floors ? h('div', { class: 'thr', style: `left:${(Math.max(alpha, floors[b]) * 100).toFixed(1)}%`, title: 'reject threshold' }) : null),
    h('span', { class: 'val' }, f3(p[b])))));
}

R.reject = (root) => {
  const s = state.reject;
  const step = M.trajectory[s.ti];
  const pr = step.pred;
  const n_b = Object.fromEntries(BATCHES.map((b) => [b, M.labels.filter((l) => l === b).length]));
  const floors = Object.fromEntries(BATCHES.map((b) => [b, 1 / (n_b[b] + 1) + 1e-12]));
  const kind = pr.ood ? 'reject' : (pr.assigned === 'Batch_3' ? 'accept' : 'reassign');
  const msg = pr.ood ? ['REJECT — out of distribution', 'less typical than every labelled site of every batch'] :
    pr.assigned === 'Batch_3' ? ['ACCEPT — matches the Batch 3 baseline', `credibility ${f3(pr.credibility)}`] :
      [`FLAG — looks like ${short(pr.assigned)}, not the baseline`, `credibility ${f3(pr.credibility)} · it tells you what kind of wrong`];
  const traj = BATCHES.map((b) => ({ points: M.trajectory.map((st) => [st.t, st.pred.p[b]]), color: BATCH_COLOR[b], width: 2.5, label: short(b) }));
  const bands = [0, 1, 2, 3, 4];
  const prof = [...BATCHES.map((b) => ({ points: bands.map((k) => [k, median(M.featRows.filter((r) => r.batch === b).map((r) => r[`si_depth_rel_band${k}`]))]), color: BATCH_COLOR[b], width: 2, opacity: 0.6 })),
    { points: bands.map((k) => [k, step.row[`si_depth_rel_band${k}`]]), color: '#111', width: 4, marker: 5 }];
  const slider = h('input', { type: 'range', min: 0, max: M.trajectory.length - 1, step: 1, value: s.ti, id: 'drift',
    oninput: (e) => { s.ti = +e.target.value; render(); } });
  root.append(
    head('5 · Reject the unknown shipment', 'A shipment drifts from the baseline toward heavy sedimentation.'),
    h('p', { class: 'lede' }, `Start from the most typical real Batch 3 site (${M.base.site}), then sink Si toward the bottom, coarsen its clustering and grow within-site heterogeneity. The model is the one that made the held-out calls — running live in this browser, no retraining, no thresholds tuned.`),
    h('div', { class: 'row grow' },
      h('div', { class: 'col', style: 'flex:1' },
        h('div', { class: `verdict ${kind}` }, msg[0], h('small', {}, msg[1])),
        card(null, h('div', {}, h('div', { class: 'slider-label' }, h('span', {}, 'drift toward sedimentation'), h('b', { class: 'mono' }, `t = ${step.t.toFixed(2)}`)), slider,
          h('div', { style: 'margin-top:12px' }, pbars(pr.p, { floors })),
          h('div', { class: 'note', style: 'margin-top:8px' }, 'red tick = rejection threshold (p < 0.1 or at the conformal floor 1/(nᵦ+1)); all three rejected ⇒ out of distribution'))),
        card('Depth profile of the drifting shipment (black) vs batch medians', h('div', { style: 'height:230px' }, lineChart({ series: prof, w: 560, h: 230, xTicks: bands, xFmt: (k) => ['top', '', 'mid', '', 'bottom'][k], yLabel: 'Si / site mean' }))),
      ),
      h('div', { class: 'col', style: 'flex:1' },
        card('Conformal p-value per batch along the drift', h('div', { style: 'height:330px' }, lineChart({ series: traj, w: 640, h: 330, yDomain: [0, 1], xLabel: 'drift t', yLabel: 'p-value',
          extra: (svg, x, y, { m, h: hh }) => {
            svg.append(el('rect', { x: m.l, y: y(0.125), width: 1000, height: y(0) - y(0.125) + 6, fill: 'rgba(201,52,52,.08)' }));
            const ood = M.trajectory.find((st) => st.pred.ood);
            if (ood) svg.append(el('text', { x: x(ood.t) + 4, y: y(0.97), fill: '#c93434', 'font-size': 12, 'font-weight': 600 }, `OOD from t = ${ood.t.toFixed(2)}`),
              el('line', { x1: x(ood.t), x2: x(ood.t), y1: m.t, y2: hh - m.b, stroke: '#c93434', 'stroke-dasharray': '3 3' }));
            svg.append(el('line', { x1: x(step.t), x2: x(step.t), y1: m.t, y2: hh - m.b, stroke: '#111', 'stroke-width': 2 }));
          } })), null),
        h('div', { class: 'legend' }, BATCHES.map((b) => h('span', { style: `color:${BATCH_COLOR[b]}` }, short(b)))),
        card(null, h('p', { class: 'note' }, 'The drift is a synthetic perturbation of real features (scripts/demo_reject.py), used to show the accept / re-assign / reject behaviour a manufacturer needs when an unknown batch N arrives. Parity with the Python model is checked by demo/test/parity.test.mjs.'), 'pmdb/fingerprint.py → demo/lib/fingerprint.js'),
      )),
  );
};

R.explorer = (root) => {
  const D = M.D;
  const sites = [...M.featRows.map((r) => ({ batch: r.batch, site: r.site })), ...D.heldoutFeatures.map((r) => ({ batch: 'Batch_heldout', site: r.site }))];
  const key = state.explorer.key;
  const [batch, site] = key.split('/');
  const ho = batch === 'Batch_heldout';
  const thumb = (b, s) => (b === 'Batch_heldout' ? `/outputs/heldout/overlays/${b}__${s}.png` : `/outputs/gallery/img/${b}__${s}_rgb.jpg`);
  const grid = h('div', { class: 'grid' }, sites.map(({ batch: b, site: s }) => h('div', { class: `thumb ${`${b}/${s}` === key ? 'sel' : ''}`, onclick: () => { state.explorer.key = `${b}/${s}`; render(); } },
    h('img', { src: thumb(b, s), loading: 'lazy', alt: s }), h('div', { class: 'lbl' }, h('span', {}, s), h('b', { class: `b-${b}` }, b === 'Batch_heldout' ? 'HO' : short(b).replace('Batch ', 'B'))))));
  const k = M.kpi.get(key), fr = M.fem.get(key), lo = M.loo.get(key), bs = M.bseByKey.get(key);
  const hl = ho ? M.heldLive.find((p) => p.site === site) : null;
  const call = ho ? hl : lo;
  root.append(
    head('6 · Site explorer', `${site} · ${short(batch)}`),
    h('div', { class: 'row grow' },
      h('div', { class: 'col', style: 'flex:1' },
        card(`All ${sites.length} sites — click one`, grid),
        card('Numbers', h('dl', { class: 'kv' },
          h('dt', {}, ho ? 'fingerprint call' : 'fingerprint LOO call'), h('dd', {}, call ? h('span', { class: `b-${call.assigned}` }, `${short(call.assigned)}${ho ? '' : (call.assigned === batch ? ' ✓' : ' ✗')}`) : '—'),
          h('dt', {}, 'credibility / confidence'), h('dd', {}, call ? `${f3(call.credibility)} / ${f3(call.confidence)}` : '—'),
          h('dt', {}, 'Si area fraction (K01)'), h('dd', {}, k ? f3(k.K01_si_frac_adm) : '—'),
          h('dt', {}, 'Si objects / 1000 µm² (K02)'), h('dd', {}, k ? k.K02_si_density_per_1000um2.toFixed(1) : '—'),
          h('dt', {}, 'simulated swelling (full charge)'), h('dd', {}, fr ? f3(fr.swelling_sym) : '—'),
          h('dt', {}, 'BSE black level p1'), h('dd', {}, bs ? bs.p1.toFixed(0) : '—')),
          'outputs/fingerprint · outputs/kpis · outputs/fem/validation.csv · outputs/raw_intensity_stats.csv')),
      h('div', { class: 'col', style: 'flex:1.25' },
        card('Simulated lithiation 0 → 100% charge (von Mises on warped BSE)', h('div', { class: 'media' }, h('img', { src: `/outputs/fem/gifs/${batch}__${site}.gif`, alt: 'FEM' })), 'outputs/fem/gifs/'),
        card('Segmentation overlay', h('div', { class: 'media' }, h('img', { src: ho ? `/outputs/heldout/overlays/${batch}__${site}.png` : `/outputs/overlays/${batch}__${site}.png`, alt: 'overlay' })), ho ? 'outputs/heldout/overlays/' : 'outputs/overlays/'),
      )),
  );
};

function head(kicker, title) {
  return h('div', { class: 'head' }, h('span', { class: 'kicker' }, kicker), h('h1', {}, title));
}

// ---------------------------------------------------------------- shell
const viewEls = {};
function buildShell() {
  const nav = document.getElementById('nav'), main = document.getElementById('views');
  VIEWS.forEach((v, i) => {
    nav.append(h('button', { 'data-view': v.id, onclick: () => go(v.id) }, h('kbd', {}, i + 1), v.title));
    viewEls[v.id] = h('section', { class: 'view', id: `view-${v.id}` });
    main.append(viewEls[v.id]);
  });
}

function render() {
  if (!M) return false;
  const v = state.view;
  document.querySelectorAll('nav button').forEach((b) => b.classList.toggle('active', b.dataset.view === v));
  Object.entries(viewEls).forEach(([id, n]) => n.classList.toggle('active', id === v));
  const root = viewEls[v];
  const focused = document.activeElement && document.activeElement.id;
  const scroll = root.scrollTop;
  const reopen = openSite;
  closeCard(false);
  root.replaceChildren();
  let ok = true;
  try { R[v](root); } catch (e) { ok = false; root.append(card('Could not render this view', h('pre', { class: 'mono' }, String(e.stack || e)))); console.error(e); }
  root.scrollTop = scroll;
  const again = reopenSite(reopen, v, [...root.querySelectorAll('.emb-stories .call')].map((n) => n.dataset.site));
  if (again) openCard(root.querySelector(`.emb-stories .call[data-site="${again}"]`));
  if (focused) document.getElementById(focused)?.focus();
  return ok;
}

function go(id) { state.view = id; history.replaceState(null, '', `#${id}`); render(); }

function setLive(kind, text) {
  const live = document.getElementById('live');
  live.className = `live ${kind}`;
  document.getElementById('live-text').textContent = text;
}

function describeLive() {
  const b = state.bundle, g = b.git || {};
  const t = new Date(b.builtAt).toLocaleTimeString();
  let txt = `live · results v${b.version} @ ${t} · ${g.head || ''}`;
  if (g.upstreamAhead) txt += ` · ${g.upstreamAhead} new on origin/main`;
  const errs = Object.keys(b.errors || {});
  if (errs.length) txt += ` · ${errs.length} file(s) missing`;
  setLive(errs.length ? 'err' : 'ok', txt);
}

function toast(text) {
  const t = document.getElementById('toast');
  t.textContent = text; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => { t.hidden = true; }, 6000);
}

async function load() {
  const res = await fetch('/api/data', { cache: 'no-store' });
  const bundle = await res.json();
  state.bundle = bundle;
  derive(bundle);
  describeLive();
  state.rendered = render();
  if (!state.rendered) throw new Error(`view ${state.view} failed to render`);
  boot();
}

function connect() {
  const es = new EventSource('/api/events');
  es.addEventListener('update', async (e) => {
    const { changed } = JSON.parse(e.data);
    try { await load(); } catch (err) { setLive('err', 'failed to load /api/data'); console.error(err); return; }
    toast(`New results loaded: ${changed.join(', ')}`);
  });
  // retry a failed startup load on the next server event
  for (const ev of ['hello', 'ping']) es.addEventListener(ev, () => { if (!state.rendered) load().catch((err) => console.error(err)); });
  es.addEventListener('git', (e) => { if (!state.rendered) return; state.bundle.git = JSON.parse(e.data); describeLive(); if (state.bundle.git.upstreamAhead) toast(`${state.bundle.git.upstreamAhead} new commit(s) on origin/main — git pull to update`); });
  es.onerror = () => setLive('err', 'server unreachable — showing last loaded data');
  es.onopen = () => state.rendered && describeLive();
}

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT') return;
  const n = Number(e.key);
  if (n >= 1 && n <= VIEWS.length) { go(VIEWS[n - 1].id); return; }
  const dir = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
  if (!dir || e.target.tagName === 'INPUT') return;
  if (state.view === 'confound') { state.confound.delta = Math.max(0, Math.min(25, state.confound.delta + dir)); render(); }
  else if (state.view === 'reject') { state.reject.ti = Math.max(0, Math.min(M.trajectory.length - 1, state.reject.ti + dir)); render(); }
});

// ?autoplay=1 walks the demo script on its own (used to record the backup video).
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function autoplay() {
  const speed = Number(new URLSearchParams(location.search).get('speed') || 1);
  const w = (ms) => sleep(ms / speed);
  go('confound'); state.confound.delta = 0; render(); await w(3500);
  for (let d = 1; d <= 7; d++) { state.confound.delta = d; render(); await w(450); }
  await w(4000);
  go('arrangement'); await w(9000);
  go('proof'); await w(8000);
  go('calls'); await w(9000);
  go('reject'); state.reject.ti = 0; render(); await w(3000);
  for (let i = 1; i < M.trajectory.length; i++) { state.reject.ti = i; render(); await w(i % 5 === 0 ? 1300 : 260); }
  await w(4000);
  go('explorer'); await w(5000);
  document.body.dataset.autoplayDone = '1';
}

// ---------------------------------------------------------------- test calls: probe embeddings
// Each story was written for the recorded call; if refreshed results change the call, a note is shown.
const SITE_STORY = {
  '0eryguqq': { call: 'Batch_3', text: 'Reads as Batch 3 across the whole image: a denser Si particle population with more Si area and porosity than a Batch 2 crop, consistent through the coating depth.' },
  fhwrjtet: { call: 'Batch_3', text: 'Same picture as 0eryguqq: Batch-3-like Si area fraction, particle density and porosity, holding through the depth.' },
  fspqbkxl: { call: 'Batch_2', text: 'Leans Batch 2, but mostly on fine texture our microstructure measurements do not capture; the measurable differences (Si fraction, Si–graphite contact) are small.' },
  '4hq27w4c': { call: 'Batch_2', text: 'Near tie between Batch 2 and Batch 1. Slightly more porosity and Si area, and fewer Si particles, than a Batch 1 crop.' },
  y59rxmxl: { call: 'Batch_1', text: 'Leans Batch 1: lower Si area fraction and fewer Si particles than a Batch 2 crop, with less Si–graphite contact.' },
  soo2ax3r: { call: 'Batch_1', text: 'Leans Batch 1: fewer but larger Si particles (lower density, higher Si area) than a Batch 2 crop, most visible deeper in the coating.' },
};
// Call each evidence PNG in /evidence was rendered for; a different loaded call means the map is stale.
const EVIDENCE_CALL = { '0eryguqq': 'Batch_3', fhwrjtet: 'Batch_3', fspqbkxl: 'Batch_2', '4hq27w4c': 'Batch_2', y59rxmxl: 'Batch_1', soo2ax3r: 'Batch_1' };
const KPI_NAME = { si_frac: 'Si area fraction', si_density_per_1000um2: 'Si particle density', si_mean_area_um2: 'Si particle size',
  si_graphite_contact_frac: 'Si–graphite contact', porosity: 'porosity', depth_frac: 'depth position' };
function pcStory(label, r2, explained) {
  if (!explained) return `Fine texture not captured by our microstructure measurements (KPIs explain ${Math.round(r2 * 100)}% of it).`;
  const parts = label.split(',').map((s) => s.trim()).map((s) => (s[0] === '+' ? 'more ' : 'less ') + s.slice(1).toLowerCase());
  return `Patches high on this dimension show ${parts.join(' and ')} (KPIs explain ${Math.round(r2 * 100)}% of it).`;
}

let openSite = null;
function closeCard(restoreFocus) {
  document.querySelector('.card-overlay')?.remove();
  const site = openSite; openSite = null;
  if (restoreFocus && site) document.querySelector(`.emb-stories .call[data-site="${site}"]`)?.focus();
}
function openCard(card) {
  document.querySelector('.card-overlay')?.remove();
  openSite = card.dataset.site || null;
  const big = card.cloneNode(true); big.classList.add('open'); big.removeAttribute('title');
  const ov = h('div', { class: 'card-overlay', onclick: (e) => { if (e.target === ov) closeCard(true); } }, big);
  document.body.append(ov); big.focus();
}
window.addEventListener('keydown', (e) => { if (e.key === 'Escape' && openSite) closeCard(true); });

R.embeddings = (root) => {
  const P = M.D.probeTest || [], C = M.D.probeContrib || [];
  const test = P.filter((r) => SITE_STORY[r.site]).sort((a, b) => +b[`p_${b.call}`] - +a[`p_${a.call}`]);
  root.append(head('Final model · supervised probe on MicroNet patch embeddings', 'Test calls and why'));
  // 1. call cards
  root.append(h('div', { class: 'emb-stories' }, ...test.map((s) => {
    const p = +s[`p_${s.call}`], ex = +s.explained_share;
    const bars = h('div', { class: 'pbar' }, ...['Batch_1', 'Batch_2', 'Batch_3'].map((b) =>
      h('span', { style: `width:${(+s[`p_${b}`] * 100).toFixed(1)}%;background:${BATCH_COLOR[b]}`, title: `${short(b)} ${(+s[`p_${b}`]).toFixed(2)}` })));
    const nets = Object.keys(KPI_NAME).map((k) => [k, +s[`net_${k}`]]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 3);
    return h('div', { class: 'card call', style: `border-top:4px solid ${BATCH_COLOR[s.call]}`, tabindex: '0', role: 'button', title: 'Click to expand', 'data-site': s.site,
      onclick: (e) => { if (e.target.closest('a')) return; openCard(e.currentTarget); },
      onkeydown: (e) => { if (e.target !== e.currentTarget) return; if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openCard(e.currentTarget); } } },
      h('div', { class: 'call-head' }, h('b', { class: 'mono' }, s.site),
        h('span', { class: 'call-batch', style: `color:${BATCH_COLOR[s.call]}` }, short(s.call)),
        h('span', { class: 'call-p' }, p.toFixed(2),
          h('span', { class: `conf ${p >= 0.8 ? 'hi' : p >= 0.67 ? 'md' : 'lo'}` }, p >= 0.8 ? 'HIGH' : p >= 0.67 ? 'MEDIUM' : 'LOW'))),
      bars,
      h('div', {},
        h('a', { href: `/evidence/${s.site}.png`, target: '_blank', title: 'Open full size' },
          h('img', { src: `/evidence/${s.site}.png`, alt: `Evidence map for ${s.site}`, class: 'evidence' }))),
      ...(evidenceIsStale(EVIDENCE_CALL, s.site, s.call) ? [h('p', { class: 'src' }, `Evidence map was rendered for the call ${EVIDENCE_CALL[s.site] ? short(EVIDENCE_CALL[s.site]) : 'unknown'}; the loaded call is ${short(s.call)}, so the map may not match.`)] : []),
      h('div', { class: 'src' }, 'red zones support the call · blue argue against · yellow = 3 strongest'),
      h('p', {}, SITE_STORY[s.site].text),
      ...(storyIsStale(SITE_STORY[s.site], s.call) ? [h('p', { class: 'src' }, `Explanation written when this site was called ${short(SITE_STORY[s.site].call)}; the loaded call is ${short(s.call)}, so the text may be out of date. Chips and bars are from the loaded data.`)] : []),
      h('div', { class: 'chips' }, ...nets.map(([k, v]) => h('span', { class: 'chip' }, `${v >= 0 ? '▲' : '▼'} ${KPI_NAME[k]}`))),
      h('div', { class: 'split' }, h('span', { style: `width:${ex * 100}%`, class: 'meas' }), h('span', { style: `width:${(1 - ex) * 100}%`, class: 'tex' })),
      h('div', { class: 'src' }, `${Math.round(ex * 100)}% measured microstructure · ${Math.round((1 - ex) * 100)}% fine texture`));
  })));
  // 2. compact heatmap: top 12 dims + rest
  const top = topPcs(C, test.map((x) => x.site), 12);
  const cell = (s, parent, pc) => findCell(C, s, parent, pc);
  const max = Math.max(...C.filter((r) => top.includes(r.pc)).map((r) => Math.abs(+r.contribution)));
  const tip = h('div', { class: 'emb-tip', hidden: true });
  const grid = h('div', { class: 'emb-grid', style: `grid-template-columns: 110px repeat(${top.length + 1}, minmax(54px, 1fr))` });
  grid.append(h('div', {}));
  top.forEach((pc) => { const r = headerRow(C, pc); grid.append(h('div', { class: 'emb-pc', title: pcStory(r.label, +r.r2, r.explained === 'True') },
    pc, h('br'), r.explained === 'True' ? r.label.split(',')[0].replace('+', '↑ ').replace('-', '↓ ') : 'texture')); });
  grid.append(h('div', { class: 'emb-pc' }, 'other 52', h('br'), 'dims'));
  for (const s of test) {
    grid.append(h('div', { class: 'emb-row' }, h('b', { class: 'mono' }, s.site), h('br'), h('span', { style: `color:${BATCH_COLOR[s.call]}` }, short(s.call))));
    const addCell = (c, story, label) => {
      const a = Math.min(1, Math.abs(c) / max) ** 0.6;
      const n = h('div', { class: 'emb-cell', style: `background:${c >= 0 ? `rgba(201,52,52,${a})` : `rgba(42,120,214,${a})`}` }, c.toFixed(1));
      n.addEventListener('mouseenter', (e) => { tip.hidden = false; tip.replaceChildren(h('b', {}, `${label} · ${s.site}`), h('div', {}, story),
        h('div', { class: 'mono' }, `${c >= 0 ? 'pushes toward' : 'pushes away from'} ${short(s.call)} vs ${short(s.runner_up)}: ${c.toFixed(2)}`));
        const pos = clampTip(e.clientX, e.clientY, tip.offsetWidth || 340, tip.offsetHeight || 0, window.innerWidth, window.innerHeight);
        tip.style.left = `${pos.left}px`; tip.style.top = `${pos.top}px`; });
      n.addEventListener('mouseleave', () => { tip.hidden = true; });
      grid.append(n);
    };
    top.forEach((pc) => { const r = cell(s.site, s.parent, pc); addCell(+r.contribution, pcStory(r.label, +r.r2, r.explained === 'True'), pc); });
    const rest = C.filter((x) => x.site === s.site && !top.includes(x.pc)).reduce((t, x) => t + +x.contribution, 0);
    addCell(rest, 'Sum of the remaining 52 embedding dimensions, mostly fine texture.', 'other dims');
  }
  root.append(card('Which embedding dimensions drove each call (red = toward the call, blue = against; hover for meaning)', h('div', { class: 'emb-wrap' }, grid)), tip);
};

buildShell();
window.addEventListener('hashchange', () => { const v = location.hash.slice(1); if (VIEWS.some((x) => x.id === v) && v !== state.view) { state.view = v; render(); } });
const initial = location.hash.slice(1);
if (VIEWS.some((v) => v.id === initial)) state.view = initial;
connect();
let booted = false;
const boot = () => { if (booted) return; booted = true; if (new URLSearchParams(location.search).get('autoplay')) autoplay(); };
load().catch((e) => { if (!state.rendered) setLive('err', 'failed to load /api/data'); console.error(e); });
