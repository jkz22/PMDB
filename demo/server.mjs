// PMDB hackathon demo server: zero dependencies, Node >= 18.
//   node demo/server.mjs [--port 8080]
// Serves the dashboard, the read-only repo outputs/ tree, a JSON bundle of the
// committed results (/api/data) and a server-sent event stream (/api/events)
// that fires whenever a watched result file changes, so the dashboard always
// shows what is on disk right now.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { parseCSV } from './lib/csv.js';
import { serveFile, upstreamState } from './lib/serve.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const argPort = process.argv.indexOf('--port');
const PORT = Number(argPort > 0 ? process.argv[argPort + 1] : process.env.PORT || 8080);
const HOST = process.env.HOST || '127.0.0.1'; // loopback by default; HOST=0.0.0.0 to expose on the LAN
const POLL_MS = 2000;
const GIT_POLL_S = Number(process.env.DEMO_GIT_POLL || 0);

// key -> repo-relative path. Everything the dashboard shows comes from here.
const SOURCES = {
  features: 'outputs/fingerprint/features.csv',
  heldoutFeatures: 'outputs/fingerprint/heldout_features.csv',
  heldoutPredictions: 'outputs/fingerprint/heldout_predictions.csv',
  heldoutExplain: 'outputs/fingerprint/heldout_explain.csv',
  looPredictions: 'outputs/fingerprint/loo_predictions.csv',
  evaluation: 'outputs/fingerprint/evaluation.json',
  permutation10k: 'outputs/overnight/permutation/permutation_10k.json',
  nullAccuracies: 'outputs/overnight/permutation/null_accuracies.csv',
  jackknife: 'outputs/overnight/stability/jackknife_summary.csv',
  hyperparams: 'outputs/overnight/stability/hyperparam_grid.csv',
  extendedEval: 'outputs/overnight/eval_extended/evaluation.json',
  d21Main: 'outputs/fem_fingerprint/main/metrics.csv',
  d21Edge5: 'outputs/fem_fingerprint/edge5/metrics.csv',
  rawStats: 'outputs/raw_intensity_stats.csv',
  heldoutRawStats: 'outputs/heldout/raw_intensity_stats.csv',
  siteKpis: 'outputs/kpis/site_kpis.csv',
  heldoutSiteKpis: 'outputs/heldout/kpis/site_kpis.csv',
  femValidation: 'outputs/fem/validation.csv',
  femRunLog: 'outputs/fem/run_log.json',
  probeTest: 'outputs/probe_explain/test_final_predictions.csv',
  probeContrib: 'outputs/probe_explain/test_contributions.csv',
};

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json', '.csv': 'text/csv; charset=utf-8', '.md': 'text/markdown; charset=utf-8',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.gif': 'image/gif',
  '.svg': 'image/svg+xml', '.mp4': 'video/mp4', '.webm': 'video/webm',
};

const STATIC_ROOTS = {
  '/lib/': path.join(HERE, 'lib'),
  '/outputs/': path.join(ROOT, 'outputs'),
  '/docs/': path.join(ROOT, 'docs'),
  '/': path.join(HERE, 'public'),
};

function readSource(rel) {
  const abs = path.join(ROOT, rel);
  const text = fs.readFileSync(abs, 'utf8');
  return rel.endsWith('.json') ? JSON.parse(text.replace(/(?<=[:\[,]\s*)(-?Infinity|NaN)(?=\s*[,\]}])/g, 'null')) : parseCSV(text);
}

let bundle = null;
let version = 0;
const mtimes = {};
const gitState = { head: null, branch: null, upstreamAhead: null, upstreamLog: [], checked: null };

function mtimeOf(rel) {
  try { return fs.statSync(path.join(ROOT, rel)).mtimeMs; } catch { return null; }
}

function buildBundle() {
  const data = {}, errors = {}, files = {};
  for (const [key, rel] of Object.entries(SOURCES)) {
    files[key] = { path: rel, mtime: mtimeOf(rel) };
    try { data[key] = readSource(rel); } catch (e) { errors[key] = String(e.message || e); }
  }
  version += 1;
  bundle = { version, builtAt: new Date().toISOString(), files, errors, data, git: gitState };
  return bundle;
}

function changedSources() {
  const changed = [];
  for (const rel of Object.values(SOURCES)) {
    const m = mtimeOf(rel);
    if (mtimes[rel] !== undefined && mtimes[rel] !== m) changed.push(rel);
    mtimes[rel] = m;
  }
  return changed;
}

const clients = new Set();
function broadcast(event, payload) {
  const msg = `event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`;
  for (const res of clients) res.write(msg);
}

function git(args) {
  return new Promise((resolve) => execFile('git', args, { cwd: ROOT, timeout: 30000 },
    (err, stdout) => resolve(err ? null : stdout.trim())));
}

async function refreshGit(fetchUpstream) {
  if (fetchUpstream) await git(['fetch', '--quiet', 'origin']);
  gitState.head = await git(['rev-parse', '--short', 'HEAD']);
  gitState.branch = await git(['rev-parse', '--abbrev-ref', 'HEAD']);
  const log = await git(['log', '--oneline', 'HEAD..origin/main']);
  const count = await git(['rev-list', '--count', 'HEAD..origin/main']);
  Object.assign(gitState, upstreamState(log === null ? null : count, log));
  gitState.checked = new Date().toISOString();
}

function serveStatic(req, res, urlPath) {
  for (const [prefix, dir] of Object.entries(STATIC_ROOTS)) {
    if (!urlPath.startsWith(prefix)) continue;
    let rel = decodeURIComponent(urlPath.slice(prefix.length)) || 'index.html';
    const abs = path.resolve(dir, rel);
    if (!abs.startsWith(dir + path.sep) && abs !== dir) { res.writeHead(403).end(); return; }
    serveFile(req, res, dir, abs, MIME);
    return;
  }
  res.writeHead(404).end();
}

const server = http.createServer((req, res) => {
  const urlPath = new URL(req.url, 'http://x').pathname;
  if (urlPath === '/api/data') {
    res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
    res.end(JSON.stringify(bundle));
  } else if (urlPath === '/api/events') {
    res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-store', Connection: 'keep-alive' });
    res.write(`event: hello\ndata: ${JSON.stringify({ version })}\n\n`);
    clients.add(res);
    req.on('close', () => clients.delete(res));
  } else if (urlPath === '/api/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ ok: true, version, errors: Object.keys(bundle.errors) }));
  } else {
    serveStatic(req, res, urlPath);
  }
});

changedSources();
await refreshGit(false);
buildBundle();
setInterval(() => {
  const changed = changedSources();
  if (changed.length) {
    buildBundle();
    console.log(`[demo] results changed (v${version}):`, changed.join(', '));
    broadcast('update', { version, changed });
  }
}, POLL_MS);
setInterval(() => broadcast('ping', { t: Date.now() }), 25000);
if (GIT_POLL_S > 0) {
  setInterval(async () => {
    const before = gitState.upstreamAhead;
    await refreshGit(true);
    if (gitState.upstreamAhead !== before) broadcast('git', gitState);
  }, GIT_POLL_S * 1000);
}

server.listen(PORT, HOST, () => {
  const errs = Object.keys(bundle.errors);
  console.log(`[demo] PMDB dashboard on http://${HOST}:${PORT}  (git ${gitState.head}, ${Object.keys(SOURCES).length - errs.length}/${Object.keys(SOURCES).length} result files loaded)`);
  if (errs.length) console.log('[demo] missing/unreadable:', bundle.errors);
});
