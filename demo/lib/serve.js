// Node-only helpers for the demo server (kept out of server.mjs so they are unit-testable).
import fs from 'node:fs';
import path from 'node:path';

// Parse a single "bytes=a-b" header. Returns {start,end}, 'invalid' (answer 416) or null (ignore header).
export function parseRange(header, size) {
  const m = /^bytes=(\d*)-(\d*)$/.exec(header || '');
  if (!m || (m[1] === '' && m[2] === '')) return null;
  let start, end;
  if (m[1] === '') { // suffix range: last N bytes
    const n = Number(m[2]);
    if (n === 0 || size === 0) return 'invalid';
    start = Math.max(0, size - n); end = size - 1;
  } else {
    start = Number(m[1]); end = m[2] === '' ? size - 1 : Number(m[2]);
  }
  if (end >= size) end = size - 1;
  if (!(start < size && start <= end)) return 'invalid';
  return { start, end };
}

// Real path of abs if it lies inside dir after resolving symlinks, else null.
export async function realInside(dir, abs) {
  try {
    const [rd, ra] = await Promise.all([fs.promises.realpath(dir), fs.promises.realpath(abs)]);
    return ra === rd || ra.startsWith(rd + path.sep) ? ra : null;
  } catch { return undefined; } // undefined = does not exist
}

export async function serveFile(req, res, dir, abs, mime) {
  let fh;
  try {
    fh = await fs.promises.open(abs, 'r');
    const real = await realInside(dir, abs);
    if (real === null) { await fh.close(); res.writeHead(403).end(); return; }
    // Verify the opened handle is the file at the resolved path (guards against a swap after open).
    const [hs, rs] = [await fh.stat(), real ? await fs.promises.stat(real) : null];
    if (!rs || !hs.isFile() || hs.ino !== rs.ino || hs.dev !== rs.dev) { await fh.close(); res.writeHead(real === undefined ? 404 : 403).end(); return; }
    const size = hs.size;
    const type = mime[path.extname(real).toLowerCase()] || 'application/octet-stream';
    const range = parseRange(req.headers.range, size);
    if (range === 'invalid') { await fh.close(); res.writeHead(416, { 'Content-Range': `bytes */${size}` }).end(); return; }
    if (range) res.writeHead(206, { 'Content-Type': type, 'Content-Range': `bytes ${range.start}-${range.end}/${size}`,
      'Accept-Ranges': 'bytes', 'Content-Length': range.end - range.start + 1, 'Cache-Control': 'no-cache' });
    else res.writeHead(200, { 'Content-Type': type, 'Content-Length': size, 'Cache-Control': 'no-cache' });
    const stream = fh.createReadStream(range ? { start: range.start, end: range.end } : {});
    stream.on('error', () => { if (!res.headersSent) res.writeHead(404).end(); else res.destroy(); });
    stream.pipe(res);
  } catch {
    if (fh) await fh.close().catch(() => {});
    if (!res.headersSent) res.writeHead(404).end('not found'); else res.destroy();
  }
}

// Upstream state from `git rev-list --count` (true count) and the truncated `git log` (display only).
export function upstreamState(countOut, logOut, limit = 20) {
  const log = logOut ? logOut.split('\n').filter(Boolean).slice(0, limit) : [];
  const n = countOut === null || countOut === undefined ? NaN : parseInt(countOut, 10);
  return { upstreamLog: log, upstreamAhead: Number.isNaN(n) ? null : n };
}
