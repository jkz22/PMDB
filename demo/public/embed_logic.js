// Pure helpers for the embeddings view (no DOM; unit-tested in test/embed_logic.test.mjs).

// Contribution row for one site, one parent fit and one PC. Rows are parent-specific
// (explained/label/r2 can differ between parents), so never match on the PC alone.
export function findCell(rows, site, parent, pc) {
  return rows.find((x) => x.site === site && (parent == null || x.parent === parent) && x.pc === pc);
}

// Header row for a PC: prefer a row where the PC is explained, so one parent's
// "unexplained" flag does not hide an explained dimension.
export function headerRow(rows, pc) {
  const all = rows.filter((x) => x.pc === pc);
  return all.find((x) => x.explained === 'True') || all[0];
}

// Keep a box of (w, h) at the pointer inside the viewport; flip to the left/above when it overflows.
export function clampTip(x, y, w, h, vw, vh, gap = 14, pad = 8) {
  let left = x + gap, top = y + gap;
  if (left + w > vw - pad) left = x - gap - w;
  if (top + h > vh - pad) top = y - gap - h;
  return { left: Math.max(pad, left), top: Math.max(pad, top) };
}

// A story is only trusted when the loaded call equals the call it was written for.
export function storyIsStale(story, call) {
  return !story || story.call !== call;
}

// An evidence PNG is only trusted when the loaded call equals the call it was rendered for.
export function evidenceIsStale(evidenceCalls, site, call) {
  return !evidenceCalls || evidenceCalls[site] !== call;
}

// Top-n PCs by summed |contribution| over the given (displayed) sites only.
export function topPcs(rows, sites, n) {
  const keep = new Set(sites), tot = {};
  for (const r of rows) if (keep.has(r.site)) tot[r.pc] = (tot[r.pc] || 0) + Math.abs(+r.contribution);
  return Object.keys(tot).sort((a, b) => tot[b] - tot[a]).slice(0, n);
}
