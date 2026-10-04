// Tiny dependency-free SVG chart helpers.
const NS = 'http://www.w3.org/2000/svg';
export const BATCH_COLOR = { Batch_1: '#d9772b', Batch_2: '#7a52c7', Batch_3: '#2a78d6', Batch_heldout: '#111111' };
export const BATCH_LABEL = { Batch_1: 'Batch 1', Batch_2: 'Batch 2', Batch_3: 'Batch 3 (baseline)', Batch_heldout: 'held-out' };

export function el(tag, attrs = {}, children = []) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) n.setAttribute(k, v);
  for (const c of [].concat(children)) if (c !== null && c !== undefined) n.append(typeof c === 'string' ? document.createTextNode(c) : c);
  return n;
}

export function scale(d0, d1, r0, r1) {
  const f = (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
  f.domain = [d0, d1];
  return f;
}

export function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1;
  const step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) || mag * 10;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

function frame(w, h) {
  return el('svg', { viewBox: `0 0 ${w} ${h}`, width: '100%', height: '100%', preserveAspectRatio: 'xMidYMid meet' });
}

function axes(svg, x, y, { w, h, m, xTicks, yTicks, xFmt = String, yFmt = String, xLabel, yLabel }) {
  const g = el('g', { 'font-size': 12, fill: '#52514e' });
  for (const t of yTicks) {
    g.append(el('line', { x1: m.l, x2: w - m.r, y1: y(t), y2: y(t), stroke: '#e6e5df' }));
    g.append(el('text', { x: m.l - 6, y: y(t) + 4, 'text-anchor': 'end' }, yFmt(t)));
  }
  for (const t of xTicks) {
    g.append(el('line', { x1: x(t), x2: x(t), y1: h - m.b, y2: h - m.b + 4, stroke: '#8a8882' }));
    g.append(el('text', { x: x(t), y: h - m.b + 17, 'text-anchor': 'middle' }, xFmt(t)));
  }
  g.append(el('line', { x1: m.l, x2: w - m.r, y1: h - m.b, y2: h - m.b, stroke: '#8a8882' }));
  if (xLabel) g.append(el('text', { x: (m.l + w - m.r) / 2, y: h - 4, 'text-anchor': 'middle', 'font-size': 13 }, xLabel));
  if (yLabel) g.append(el('text', { x: 14, y: (m.t + h - m.b) / 2, 'text-anchor': 'middle', 'font-size': 13,
    transform: `rotate(-90 14 ${(m.t + h - m.b) / 2})` }, yLabel));
  svg.append(g);
}

// series: [{points:[[x,y]], color, width, dash, opacity, label}]
export function lineChart({ series, w = 640, h = 360, xTicks, xFmt, yLabel, xLabel, yDomain, extra }) {
  const m = { l: 56, r: 16, t: 14, b: 44 };
  const xs = series.flatMap((s) => s.points.map((p) => p[0]));
  const ys = series.flatMap((s) => s.points.map((p) => p[1]));
  const svg = frame(w, h);
  if (!xs.length) {
    svg.append(el('text', { x: w / 2, y: h / 2, 'text-anchor': 'middle', 'dominant-baseline': 'middle', fill: '#8a8882' }, 'nothing selected'));
    return svg;
  }
  const [ylo, yhi] = yDomain || [Math.min(...ys), Math.max(...ys)];
  const pad = (yhi - ylo) * 0.06;
  const x = scale(Math.min(...xs), Math.max(...xs), m.l + 10, w - m.r - 10);
  const y = scale(ylo - pad, yhi + pad, h - m.b, m.t);
  axes(svg, x, y, { w, h, m, xTicks: xTicks || niceTicks(...x.domain), yTicks: niceTicks(ylo, yhi, 5),
    xFmt, yFmt: (v) => v.toFixed(2).replace(/\.?0+$/, '') || '0', xLabel, yLabel });
  if (extra) extra(svg, x, y, { w, h, m });
  for (const s of series) {
    svg.append(el('polyline', { points: s.points.map(([a, b]) => `${x(a)},${y(b)}`).join(' '), fill: 'none',
      stroke: s.color, 'stroke-width': s.width || 1.5, 'stroke-dasharray': s.dash, opacity: s.opacity ?? 1,
      'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, s.label ? el('title', {}, s.label) : null));
    if (s.marker) for (const [a, b] of s.points) svg.append(el('circle', { cx: x(a), cy: y(b), r: s.marker, fill: s.color }));
  }
  return svg;
}

// groups: [{key,label,color,values:[{v,label,star}]}]
export function stripPlot({ groups, w = 520, h = 340, yLabel, yFmt = (v) => v.toFixed(2) }) {
  const m = { l: 60, r: 12, t: 14, b: 40 };
  const vals = groups.flatMap((g) => g.values.map((d) => d.v));
  const lo = Math.min(...vals), hi = Math.max(...vals), pad = (hi - lo) * 0.08;
  const y = scale(lo - pad, hi + pad, h - m.b, m.t);
  const bw = (w - m.l - m.r) / groups.length;
  const svg = frame(w, h);
  const g = el('g', { 'font-size': 12, fill: '#52514e' });
  for (const t of niceTicks(lo, hi, 5)) {
    g.append(el('line', { x1: m.l, x2: w - m.r, y1: y(t), y2: y(t), stroke: '#e6e5df' }));
    g.append(el('text', { x: m.l - 6, y: y(t) + 4, 'text-anchor': 'end' }, yFmt(t)));
  }
  if (yLabel) g.append(el('text', { x: 14, y: (m.t + h - m.b) / 2, 'text-anchor': 'middle', 'font-size': 13,
    transform: `rotate(-90 14 ${(m.t + h - m.b) / 2})` }, yLabel));
  svg.append(g);
  groups.forEach((grp, i) => {
    const cx = m.l + bw * (i + 0.5);
    svg.append(el('text', { x: cx, y: h - m.b + 20, 'text-anchor': 'middle', 'font-size': 13, fill: grp.color, 'font-weight': 600 }, grp.label));
    const sorted = grp.values.map((d) => d.v).sort((a, b) => a - b);
    if (sorted.length > 1) {
      const med = sorted.length % 2 ? sorted[(sorted.length - 1) / 2] : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
      svg.append(el('line', { x1: cx - bw * 0.3, x2: cx + bw * 0.3, y1: y(med), y2: y(med), stroke: grp.color, 'stroke-width': 3 }));
    }
    grp.values.forEach((d, j) => {
      const jitter = ((j * 0.618) % 1 - 0.5) * bw * 0.4;
      if (d.star) svg.append(star(cx + jitter, y(d.v), 8, grp.color, d.label));
      else svg.append(el('circle', { cx: cx + jitter, cy: y(d.v), r: 5, fill: grp.color, 'fill-opacity': 0.55, stroke: grp.color },
        el('title', {}, `${d.label}: ${yFmt(d.v)}`)));
    });
  });
  return svg;
}

export function star(cx, cy, r, color, title) {
  const pts = [];
  for (let k = 0; k < 10; k++) {
    const a = -Math.PI / 2 + (k * Math.PI) / 5, rr = k % 2 ? r * 0.45 : r;
    pts.push(`${cx + rr * Math.cos(a)},${cy + rr * Math.sin(a)}`);
  }
  return el('polygon', { points: pts.join(' '), fill: color, stroke: '#fff', 'stroke-width': 1 }, title ? el('title', {}, title) : null);
}

// bins over [lo,hi] with `width`; marks: [{x,label,color,dash}]
export function histogram({ values, lo, hi, width, w = 640, h = 300, marks = [], xLabel, xFmt = (v) => v.toFixed(2) }) {
  const m = { l: 52, r: 16, t: 26, b: 44 };
  const nb = Math.round((hi - lo) / width);
  const counts = new Array(nb).fill(0);
  for (const v of values) { const i = Math.min(nb - 1, Math.max(0, Math.floor((v - lo) / width + 1e-9))); counts[i]++; }
  const cmax = Math.max(...counts);
  const x = scale(lo, hi, m.l, w - m.r), y = scale(0, cmax * 1.08, h - m.b, m.t);
  const svg = frame(w, h);
  axes(svg, x, y, { w, h, m, xTicks: niceTicks(lo, hi, 6), yTicks: niceTicks(0, cmax, 4), xFmt, yFmt: String, xLabel, yLabel: 'count' });
  counts.forEach((c, i) => svg.append(el('rect', { x: x(lo + i * width) + 0.5, y: y(c), width: Math.max(1, x(lo + width) - x(lo) - 1),
    height: y(0) - y(c), fill: '#b9b7af' })));
  marks.forEach((mk, i) => {
    svg.append(el('line', { x1: x(mk.x), x2: x(mk.x), y1: m.t - 6, y2: h - m.b, stroke: mk.color, 'stroke-width': 2.5, 'stroke-dasharray': mk.dash }));
    svg.append(el('text', { x: x(mk.x) + (mk.anchor === 'end' ? -6 : 6), y: m.t + 6 + i * 16, fill: mk.color, 'font-size': 13,
      'font-weight': 600, 'text-anchor': mk.anchor || 'start' }, mk.label));
  });
  return svg;
}

// points: [{x,y,color,label,star}]
export function scatter({ points, w = 560, h = 360, xLabel, yLabel, fit, xFmt = (v) => v.toFixed(2), yFmt = (v) => v.toFixed(2) }) {
  const m = { l: 60, r: 16, t: 14, b: 44 };
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  const xl = Math.min(...xs), xh = Math.max(...xs), yl = Math.min(...ys), yh = Math.max(...ys);
  const x = scale(xl - (xh - xl) * 0.05, xh + (xh - xl) * 0.05, m.l, w - m.r);
  const y = scale(yl - (yh - yl) * 0.08, yh + (yh - yl) * 0.08, h - m.b, m.t);
  const svg = frame(w, h);
  axes(svg, x, y, { w, h, m, xTicks: niceTicks(xl, xh, 5), yTicks: niceTicks(yl, yh, 5), xFmt, yFmt, xLabel, yLabel });
  if (fit) svg.append(el('line', { x1: x(xl), x2: x(xh), y1: y(fit.a + fit.b * xl), y2: y(fit.a + fit.b * xh), stroke: '#8a8882', 'stroke-dasharray': '5 4', 'stroke-width': 1.5 }));
  for (const p of points) {
    if (p.star) svg.append(star(x(p.x), y(p.y), 9, p.color, p.label));
    else svg.append(el('circle', { cx: x(p.x), cy: y(p.y), r: 5.5, fill: p.color, 'fill-opacity': 0.7, stroke: p.color }, el('title', {}, p.label)));
  }
  return svg;
}
