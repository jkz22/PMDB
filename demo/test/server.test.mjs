import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { statSync } from 'node:fs';
import { createServer } from 'node:net';
import path from 'node:path';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const clipSize = statSync(path.join(root, 'outputs/clips/01_confound_flip.mp4')).size;
let child;
let baseUrl;
let childOutput = '';

async function freePort() {
  const probe = createServer();
  await new Promise((resolve, reject) => {
    probe.once('error', reject);
    probe.listen(0, '127.0.0.1', resolve);
  });
  const { port } = probe.address();
  await new Promise((resolve, reject) => probe.close((error) => error ? reject(error) : resolve()));
  return port;
}

before(async () => {
  const port = await freePort();
  baseUrl = `http://127.0.0.1:${port}`;
  child = spawn(process.execPath, [
    path.join(root, 'demo/server.mjs'),
    '--port', String(port),
    '--host', '127.0.0.1',
  ], { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] });
  child.stdout.setEncoding('utf8');
  child.stderr.setEncoding('utf8');
  child.stdout.on('data', (chunk) => { childOutput += chunk; });
  child.stderr.on('data', (chunk) => { childOutput += chunk; });

  const deadline = Date.now() + 15000;
  let lastError;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) throw new Error(`server exited early (${child.exitCode}): ${childOutput}`);
    try {
      const response = await fetch(`${baseUrl}/api/health`);
      if (response.ok) return;
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`server did not become healthy: ${lastError || ''}\n${childOutput}`);
});

after(async () => {
  if (!child || child.exitCode !== null) return;
  const exited = new Promise((resolve) => child.once('exit', resolve));
  child.kill('SIGTERM');
  const didExit = await Promise.race([
    exited.then(() => true),
    new Promise((resolve) => setTimeout(() => resolve(false), 2500)),
  ]);
  if (!didExit) {
    child.kill('SIGKILL');
    await exited;
  }
});

test('/api/health reports a healthy server', async () => {
  const response = await fetch(`${baseUrl}/api/health`);
  assert.equal(response.status, 200);
  assert.equal((await response.json()).ok, true);
});

test('full clip GET returns the complete file', async () => {
  const response = await fetch(`${baseUrl}/outputs/clips/01_confound_flip.mp4`);
  assert.equal(response.status, 200);
  assert.equal(Number(response.headers.get('content-length')), clipSize);
  assert.equal((await response.arrayBuffer()).byteLength, clipSize);
});

test('byte range returns the requested first 100 bytes', async () => {
  const response = await fetch(`${baseUrl}/outputs/clips/01_confound_flip.mp4`, {
    headers: { Range: 'bytes=0-99' },
  });
  assert.equal(response.status, 206);
  assert.equal(Number(response.headers.get('content-length')), 100);
  assert.equal(response.headers.get('content-range'), `bytes 0-99/${clipSize}`);
  assert.equal((await response.arrayBuffer()).byteLength, 100);
});

test('suffix byte range returns the final 100 bytes', async () => {
  const response = await fetch(`${baseUrl}/outputs/clips/01_confound_flip.mp4`, {
    headers: { Range: 'bytes=-100' },
  });
  assert.equal(response.status, 206);
  assert.equal(response.headers.get('content-range'), `bytes ${clipSize - 100}-${clipSize - 1}/${clipSize}`);
  assert.equal((await response.arrayBuffer()).byteLength, 100);
});

test('reversed byte range returns 416', async () => {
  const response = await fetch(`${baseUrl}/outputs/clips/01_confound_flip.mp4`, {
    headers: { Range: 'bytes=500-100' },
  });
  assert.equal(response.status, 416);
  assert.equal(response.headers.get('content-range'), `bytes */${clipSize}`);
});

test('range starting at file size returns 416', async () => {
  const response = await fetch(`${baseUrl}/outputs/clips/01_confound_flip.mp4`, {
    headers: { Range: `bytes=${clipSize}-` },
  });
  assert.equal(response.status, 416);
  assert.equal(response.headers.get('content-range'), `bytes */${clipSize}`);
});

test('range with neither bound returns 416', async () => {
  const response = await fetch(`${baseUrl}/outputs/clips/01_confound_flip.mp4`, {
    headers: { Range: 'bytes=-' },
  });
  assert.equal(response.status, 416);
  assert.equal(response.headers.get('content-range'), `bytes */${clipSize}`);
});

test('encoded traversal cannot escape the static root', async () => {
  const response = await fetch(`${baseUrl}/outputs/..%2fAGENTS.md`);
  assert.ok(response.status === 403 || response.status === 404);
});
