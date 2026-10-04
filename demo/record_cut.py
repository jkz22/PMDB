#!/usr/bin/env python3
"""Record the ~75 s "director's cut" video: the dashboard with a camera that zooms onto the panel being talked about.

Same facts pipeline and invariants as record_submission.py; fewer words per caption, larger type, chapter strip.
Usage: node demo/server.mjs &  python demo/record_cut.py [--url http://localhost:8080] [--out demo/backup/pmdb-cut.mp4]
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent / "submission"))
from build import check_invariants, collect, display, render  # noqa: E402

CHAPTERS = ["The trap", "The finding", "The proof", "The calls", "Batch QC"]

OVERLAY_CSS = """
header.top { position: relative; z-index: 3; }
main { z-index: 1; }
main#views { transform-origin: 0 0; transition: transform .9s cubic-bezier(.65,0,.35,1); will-change: transform; }
.cut-ring { box-shadow: 0 0 0 4px #ffcf5c, 0 6px 24px rgba(255,207,92,.35) !important; transition: box-shadow .4s; }
#cut-cap { position: fixed; left: 0; right: 0; bottom: 34px; margin: 0 auto; width: fit-content; max-width: 1300px; z-index: 9999;
  background: rgba(20,20,19,.92); color: #fff; font: 700 36px/1.3 system-ui, sans-serif; padding: 16px 32px;
  border-radius: 14px; text-align: center; box-shadow: 0 8px 30px rgba(0,0,0,.3); transition: opacity .25s; }
#cut-cap:empty { opacity: 0; }
#cut-cap b { color: #ffcf5c; }
#cut-chapters { position: fixed; top: 0; left: 0; right: 0; height: 56px; z-index: 9998; display: flex; gap: 0;
  background: #141413; transition: opacity .4s; }
#cut-chapters.off { opacity: 0; }
#cut-chapters div { flex: 1; display: flex; align-items: center; justify-content: center; color: #6f6d66;
  font: 600 20px system-ui, sans-serif; letter-spacing: .5px; text-transform: uppercase; border-bottom: 4px solid #2a2a28;
  transition: color .3s, border-color .3s; }
#cut-chapters div.done { color: #a9a69c; border-color: #6f6d66; }
#cut-chapters div.on { color: #ffcf5c; border-color: #ffcf5c; }
#cut-card { position: fixed; inset: 0; z-index: 10000; background: #141413; color: #faf9f5; display: flex;
  flex-direction: column; justify-content: center; align-items: center; text-align: center; transition: opacity .5s; }
#cut-card.off { opacity: 0; pointer-events: none; }
#cut-card h1 { font: 800 64px/1.12 system-ui, sans-serif; margin: 0 0 28px; letter-spacing: -1.5px; max-width: 1400px; }
#cut-card p { font: 400 34px/1.45 system-ui, sans-serif; margin: 0 0 10px; color: #c9c7bf; max-width: 1300px; }
#cut-card .k { color: #ffcf5c; }
#cut-card .s { font-size: 26px; color: #8d8a80; margin-top: 18px; }
"""

ZOOM_JS = """([needles, pad, maxScale, ring]) => {
  const main = document.getElementById('views');
  const view = main.querySelector('.view.active');
  const m = new DOMMatrix(getComputedStyle(main).transform === 'none' ? undefined : getComputedStyle(main).transform);
  const base = { left: 0, top: document.querySelector('header.top').getBoundingClientRect().bottom };
  const natural = r => ({ left: (r.left - base.left - m.e) / m.a + base.left, top: (r.top - base.top - m.f) / m.d + base.top,
    right: (r.right - base.left - m.e) / m.a + base.left, bottom: (r.bottom - base.top - m.f) / m.d + base.top });
  const cards = [...view.querySelectorAll('.card')];
  const hits = needles.map(n => {
    const hit = n.startsWith('css:') ? view.querySelector(n.slice(4))
      : cards.find(c => c.innerText.toLowerCase().includes(n.toLowerCase()));
    if (!hit) throw new Error('zoom: nothing matches ' + n);
    return hit;
  });
  document.querySelectorAll('.cut-ring').forEach(e => e.classList.remove('cut-ring'));
  if (ring) hits.forEach(e => e.classList.add('cut-ring'));
  const picked = hits.map(e => natural(e.getBoundingClientRect()));
  const x0 = Math.min(...picked.map(r => r.left)) - pad, y0 = Math.min(...picked.map(r => r.top)) - pad;
  const x1 = Math.max(...picked.map(r => r.right)) + pad, y1 = Math.max(...picked.map(r => r.bottom)) + pad;
  const W = window.innerWidth, H = window.innerHeight - base.top - 120;
  const s = Math.min(maxScale, W / (x1 - x0), H / (y1 - y0));
  const tx = (W - (x1 - x0) * s) / 2 - (x0 - base.left) * s;
  const ty = (H - (y1 - y0) * s) / 2 - (y0 - base.top) * s;
  main.style.transform = `translate(${tx}px, ${ty}px) scale(${s})`;
}"""


def arguments():
    parser = argparse.ArgumentParser(description="Record the zoomed, captioned PMDB director's-cut video.")
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--out", default="demo/backup/pmdb-cut.mp4")
    parser.add_argument("--allow-unreviewed", default="", metavar="DIR[,DIR]",
                        help="outputs/ dirs you have checked do not change the story (skipped by the invariant check)")
    return parser.parse_args()


class Camera:
    def __init__(self, page, values, origin):
        self.page = page
        self.values = values
        self.origin = origin
        self.marks = {}

    def wait(self, seconds):
        self.page.wait_for_timeout(int(seconds * 1000))

    def mark(self, name):
        self.marks[name] = round(time.monotonic() - self.origin, 2)

    def card(self, template):
        html = render(template, self.values)
        self.page.evaluate("h => { const c = document.getElementById('cut-card'); c.innerHTML = h;"
                           " c.classList.remove('off'); }", html)

    def uncard(self):
        self.page.evaluate("() => document.getElementById('cut-card').classList.add('off')")

    def cap(self, template, extra=None):
        html = render(template, {**self.values, **(extra or {})})
        self.page.evaluate("h => { document.getElementById('cut-cap').innerHTML = h; }", html)

    def chapter(self, index):
        self.page.evaluate("i => [...document.querySelectorAll('#cut-chapters div')].forEach((d, k) =>"
                           " { d.className = k < i ? 'done' : k === i ? 'on' : ''; })", index)

    def zoom(self, *needles, pad=14, max_scale=1.9, ring=False):
        self.page.evaluate(ZOOM_JS, [list(needles), pad, max_scale, ring])
        self.wait(1.0)

    def wide(self):
        self.page.evaluate("() => { document.getElementById('views').style.transform = 'none'; }")
        self.wait(0.9)

    def view(self, n):
        self.page.evaluate("() => { const m = document.getElementById('views'); m.style.transition = 'none';"
                           " m.style.transform = 'none'; m.getBoundingClientRect(); m.style.transition = ''; }")
        self.page.keyboard.press(str(n))
        self.wait(0.5)

    def step(self, times, pause):
        for _ in range(times):
            self.page.keyboard.press("ArrowRight")
            self.wait(pause)

    def headline(self):
        for line in self.page.locator("#views").inner_text().splitlines():
            if line.strip().startswith(("ACCEPT", "FLAG", "REJECT")):
                return line.strip()
        raise RuntimeError("no ACCEPT/FLAG/REJECT headline found in #views")

    def step_until(self, prefix, pause=0.25):
        limit = int(self.page.eval_on_selector("#drift", "e => e.max")) + 2
        for _ in range(limit):
            headline = self.headline()
            if headline.startswith(prefix):
                return headline
            if headline.startswith("REJECT") and prefix == "FLAG":
                raise RuntimeError("drift reached REJECT before FLAG")
            self.page.keyboard.press("ArrowRight")
            self.wait(pause)
        raise RuntimeError(f"drift never reached {prefix}")


def script(c):
    banner = "css:.col > :first-child"
    c.card("<h1>{{batch_count_word}} batches.<br>The same battery anode.</h1>"
           "<p>Which batch did this sample come from — and is a new shipment out of spec?</p>"
           "<p class='s'>PMDB · {{fp_n}} SEM cross-sections · {{heldout_n}} unlabelled held-out sites</p>")
    c.wait(4.5)

    c.view(1)
    c.chapter(0)
    c.zoom("BSE cross-section")
    c.uncard()
    c.cap("The obvious model reads pixel brightness. It scores <b>{{intensity_acc}}</b>.")
    c.wait(3.5)
    c.cap("Now brighten the image by <b>one grey level</b>. The electrode is unchanged.")
    c.wait(2.2)
    c.step(1, 0.3)
    c.zoom("intensity model says", "non-batch-3 sites", ring=False)
    c.mark("flip")
    c.cap("<b>Every</b> {{batch_1}} and {{batch_2}} site is now called {{batch_3}}: <b>{{flip}}</b>.")
    c.step(6, 0.3)
    c.wait(1.6)
    c.zoom("intensity model says", "non-batch-3 sites")
    c.page.evaluate("() => [...document.querySelectorAll('#view-confound .card')].find(e =>"
                    " e.innerText.toLowerCase().includes('fingerprint model says')).classList.add('cut-ring')")
    c.cap("It learned the microscope. Ours reads <b>geometry only</b>, so it doesn't move.")
    c.wait(3.6)

    c.view(2)
    c.chapter(1)
    c.zoom("Si area fraction")
    c.cap("Every batch holds the <b>same amount</b> of silicon (p = {{si_kw_p}}).")
    c.wait(3.3)
    c.zoom("simulated swelling")
    c.cap("{{fem_runs}} FEM charging runs on Modal ({{fem_cost}}): swelling tracks that amount, R² {{fem_r2}}.")
    c.wait(4)
    c.zoom("Si depth profile")
    c.cap("What differs is <b>where it sits</b>: top-heavy, bottom-heavy, uniform.")
    c.wait(4)

    c.view(3)
    c.chapter(2)
    c.zoom("leave-one-site-out", "label permutations")
    c.cap("{{fp_features}} depth-shape features: <b>{{fp_score}}</b> leave-one-out, <b>p = {{perm_p}}</b>.")
    c.wait(3.8)
    c.zoom("rules written before the run")
    c.cap("Every pre-registered challenger failed its own rule. We kept the honest model.")
    c.wait(4)

    c.view(4)
    c.chapter(3)
    c.zoom("3e122cbj", "xrv9xvzb", max_scale=1.6)
    c.cap("Held-out calls, recomputed live in the browser — identical to Python.")
    c.wait(3.2)
    c.zoom("3e122cbj", ring=True)
    c.cap("{{call_3e122cbj_site}}: {{call_3e122cbj_batch}}, <b>confidence {{call_3e122cbj_conf}}</b>."
          " It says when it doesn't know.")
    c.wait(4)

    c.view(5)
    c.chapter(4)
    c.zoom(banner, "drift toward sedimentation", "depth profile of the drifting")
    c.cap("A new shipment drifts toward sedimentation. First: <b>accepted</b>.")
    c.mark("accept")
    c.wait(2.6)
    flag = c.step_until("FLAG", pause=0.35)
    batch = re.search(r"looks like (Batch \d+)", flag)
    if not batch:
        raise RuntimeError(f"could not read assigned batch from verdict: {flag}")
    c.mark("flag")
    c.cap("<b>Flagged</b>: it looks like {{flag_batch}}. The model names the kind of wrong.",
          extra={"flag_batch": batch.group(1)})
    c.wait(3.2)
    c.cap("Drift further…")
    c.step_until("REJECT", pause=0.2)
    c.mark("reject")
    c.cap("…and every batch rejects it: <b>out of distribution</b>.")
    c.wait(2.8)
    c.cap("Accept, re-assign or reject: the QC call a manufacturer needs.")
    c.wait(3)

    c.cap("")
    c.card("<h1>Can't be fooled by a brightness knob.<br>Knows when it doesn't know.</h1>"
           "<p>Geometry-only fingerprint · physics-checked with FEM · calibrated · live</p>"
           "<p class='k'>github.com/jkz22/PMDB</p>")
    c.mark("close")
    c.wait(4.5)


def main():
    options = arguments()
    facts = collect()
    allowed = {d for d in options.allow_unreviewed.split(",") if d}
    skipped = sorted(allowed & set(facts["unreviewed_output_dirs"]))
    if skipped:
        print(f"warning: recording with unreviewed outputs/ dirs: {', '.join(skipped)}", file=sys.stderr)
    check_invariants({**facts, "unreviewed_output_dirs": [d for d in facts["unreviewed_output_dirs"] if d not in allowed]})
    values = display(facts)
    output = Path(options.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    with tempfile.TemporaryDirectory(prefix="pmdb-cut-") as video_dir:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1600, "height": 900}, device_scale_factor=1,
                                          record_video_dir=video_dir, record_video_size={"width": 1600, "height": 900})
            origin = time.monotonic()
            page = context.new_page()
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: m.type == "error" and errors.append(m.text))
            page.goto(f"{options.url.rstrip('/')}/#confound")
            page.add_style_tag(content=OVERLAY_CSS)
            page.evaluate("chapters => { for (const id of ['cut-cap', 'cut-card', 'cut-chapters']) {"
                          " const e = document.createElement('div'); e.id = id; document.body.append(e); }"
                          " document.getElementById('cut-chapters').innerHTML ="
                          " chapters.map(t => `<div>${t}</div>`).join(''); }", CHAPTERS)
            page.wait_for_selector("#delta")
            camera = Camera(page, values, origin)
            script(camera)
            webm = page.video.path()
            context.close()
            browser.close()
        if errors:
            print("\n".join(errors), file=sys.stderr)
            sys.exit(1)
        subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-ss", "0.5", "-i", webm,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "slow",
                        "-movflags", "+faststart", "-an", str(output)], check=True)
    print(output)
    print(json.dumps({name: round(max(0, at - 0.5), 2) for name, at in camera.marks.items()}, indent=1))


if __name__ == "__main__":
    main()
