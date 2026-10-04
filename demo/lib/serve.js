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
  if (!(start <= end && end < size)) return 'invalid';
  return { start, end };
}

// Real path of abs if it lies inside dir after resolving symlinks, else null.
export async function realInside(dir, abs) {
  try {
    const [rd, ra] = await Promise.all([fs.promises.realpath(dir), fs.promises.realpath(abs)]);
    return ra === rd || ra.startsWith(rd + path.sep) ? ra : null;
  } catch { return undefined; } // undefined = does not exist
}

export function serveFile(req, res, dir, abs, mime) {
  realInside(dir, abs).then((real) => {
    if (real === null) { res.writeHead(403).end(); return; }
    if (real === undefined) { res.writeHead(404).end('not found'); return; }
    fs.stat(real, (err, st) => {
      if (err || !st.isFile()) { res.writeHead(404).end('not found'); return; }
      const type = mime[path.extname(real).toLowerCase()] || 'application/octet-stream';
      const range = parseRange(req.headers.range, st.size);
      if (range === 'invalid') { res.writeHead(416, { 'Content-Range': `bytes */${st.size}` }).end(); return; }
      const opts = range || {};
      if (range) res.writeHead(206, { 'Content-Type': type, 'Content-Range': `bytes ${range.start}-${range.end}/${st.size}`,
        'Accept-Ranges': 'bytes', 'Content-Length': range.end - range.start + 1, 'Cache-Control': 'no-cache' });
      else res.writeHead(200, { 'Content-Type': type, 'Content-Length': st.size, 'Cache-Control': 'no-cache' });
      const stream = fs.createReadStream(real, opts);
      stream.on('error', () => { if (!res.headersSent) res.writeHead(404).end(); else res.destroy(); });
      stream.pipe(res);
    });
  });
}

// Upstream state from `git rev-list --count` (true count) and the truncated `git log` (display only).
export function upstreamState(countOut, logOut, limit = 20) {
  const log = logOut ? logOut.split('\n').filter(Boolean).slice(0, limit) : [];
  const n = countOut === null || countOut === undefined ? NaN : parseInt(countOut, 10);
  return { upstreamLog: log, upstreamAhead: Number.isNaN(n) ? null : n };
}
