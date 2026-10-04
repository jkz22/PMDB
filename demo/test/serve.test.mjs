import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { parseRange, serveFile, upstreamState } from '../lib/serve.js';

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'demo-'));
const root = path.join(tmp, 'root'); fs.mkdirSync(root);
fs.writeFileSync(path.join(root, 'a.txt'), '0123456789');
fs.writeFileSync(path.join(tmp, 'secret.txt'), 'secret');
fs.symlinkSync(path.join(tmp, 'secret.txt'), path.join(root, 'link.txt'));

const server = http.createServer((req, res) => serveFile(req, res, root, path.join(root, new URL(req.url, 'http://x').pathname), {}));
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const get = (p, headers) => fetch(`http://127.0.0.1:${server.address().port}${p}`, { headers });
test.after(() => { server.close(); fs.rmSync(tmp, { recursive: true, force: true }); });

test('parseRange bounds', () => {
  assert.deepEqual(parseRange('bytes=2-4', 10), { start: 2, end: 4 });
  assert.deepEqual(parseRange('bytes=5-', 10), { start: 5, end: 9 });
  assert.deepEqual(parseRange('bytes=-3', 10), { start: 7, end: 9 });
  for (const h of ['bytes=5-2', 'bytes=10-', 'bytes=20-30']) assert.equal(parseRange(h, 10), 'invalid', h);
  assert.deepEqual(parseRange('bytes=0-10', 10), { start: 0, end: 9 });
  assert.equal(parseRange('junk', 10), null);
});

test('out-of-range request gets 416 with Content-Range', async () => {
  const r = await get('/a.txt', { Range: 'bytes=20-30' });
  assert.equal(r.status, 416);
  assert.equal(r.headers.get('content-range'), 'bytes */10');
  const ok = await get('/a.txt', { Range: 'bytes=2-4' });
  assert.equal(ok.status, 206);
  assert.equal(await ok.text(), '234');
  const clamped = await get('/a.txt', { Range: 'bytes=7-999' });
  assert.equal(clamped.status, 206);
  assert.equal(clamped.headers.get('content-range'), 'bytes 7-9/10');
  assert.equal(await clamped.text(), '789');
  assert.equal(await (await get('/a.txt')).text(), '0123456789');
});

test('symlink escaping the root is rejected with 403', async () => {
  assert.equal((await get('/link.txt')).status, 403);
  assert.equal((await get('/a.txt')).status, 200);
  assert.equal((await get('/missing.txt')).status, 404);
});

test('stream error does not crash the process', async () => {
  const orig = fs.createReadStream;
  fs.createReadStream = (...a) => { const s = orig(path.join(root, 'gone')); return s; };
  try {
    const r = await get('/a.txt').catch((e) => ({ status: 'aborted', e }));
    assert.ok(r.status === 404 || r.status === 'aborted' || r.status === 200);
  } finally { fs.createReadStream = orig; }
  assert.equal((await get('/a.txt')).status, 200);
});

test('upstream count is not capped by the truncated log', () => {
  const log = Array.from({ length: 45 }, (_, i) => `abc${i} msg`).join('\n');
  const s = upstreamState('45\n', log);
  assert.equal(s.upstreamAhead, 45);
  assert.equal(s.upstreamLog.length, 20);
  assert.deepEqual(upstreamState(null, null), { upstreamLog: [], upstreamAhead: null });
});
