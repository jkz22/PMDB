#!/usr/bin/env python3
"""Record the <=2 min captioned submission video: dashboard walkthrough with title cards and captions.

Usage: node demo/server.mjs &  python demo/record_submission.py [--url http://localhost:8080] [--out demo/backup/pmdb-submission.mp4]
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

OVERLAY_CSS = """
#sub-cap { position: fixed; left: 0; right: 0; bottom: 28px; margin: 0 auto; width: fit-content; max-width: 1380px; z-index: 9999;
  background: rgba(20,20,19,.88); color: #fff; font: 600 27px/1.35 system-ui, sans-serif; padding: 14px 26px;
  border-radius: 12px; text-align: center; box-shadow: 0 6px 24px rgba(0,0,0,.25); transition: opacity .25s; }
#sub-cap:empty { opacity: 0; }
#sub-cap.top { bottom: auto; top: 64px; }
#sub-cap b { color: #ffcf5c; }
#sub-card { position: fixed; inset: 0; z-index: 10000; background: #141413; color: #faf9f5; display: flex;
  flex-direction: column; justify-content: center; align-items: center; text-align: center; transition: opacity .4s; }
#sub-card.off { opacity: 0; pointer-events: none; }
#sub-card h1 { font: 800 76px/1.1 system-ui, sans-serif; margin: 0 0 24px; letter-spacing: -1px; max-width: 1350px; }
#sub-card p { font: 400 32px/1.45 system-ui, sans-serif; margin: 0 0 12px; color: #c9c7bf; max-width: 1250px; }
#sub-card .k { color: #ffcf5c; }
"""


def arguments():
    parser = argparse.ArgumentParser(description="Record the captioned PMDB submission video.")
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--out", default="demo/backup/pmdb-submission.mp4")
    return parser.parse_args()


class Director:
    def __init__(self, page, values, origin):
        self.page = page
        self.values = values
        self.origin = origin
        self.marks = {}

    def wait(self, seconds):
        self.page.wait_for_timeout(int(seconds * 1000))

    def mark(self, name):
        self.marks[name] = time.monotonic() - self.origin

    def card(self, template):
        html = render(template, self.values)
        self.page.evaluate("h => { const c = document.getElementById('sub-card'); c.innerHTML = h; c.classList.remove('off'); }", html)

    def uncard(self):
        self.page.evaluate("() => document.getElementById('sub-card').classList.add('off')")

    def cap(self, template, top=False, extra=None):
        html = render(template, {**self.values, **(extra or {})})
        self.page.evaluate("([h, t]) => { const c = document.getElementById('sub-cap'); c.innerHTML = h;"
                           " c.classList.toggle('top', t); }", [html, top])

    def view(self, n):
        self.page.keyboard.press(str(n))
        self.wait(0.6)

    def step(self, times, pause, key="ArrowRight"):
        for _ in range(times):
            self.page.keyboard.press(key)
            self.wait(pause)

    def verdict_headline(self):
        for line in self.page.locator("#views").inner_text().splitlines():
            line = line.strip()
            if line.startswith(("ACCEPT", "FLAG", "REJECT")):
                return line
        raise RuntimeError("no ACCEPT/FLAG/REJECT headline found in #views")

    def step_until(self, prefix, pause=0.3):
        limit = int(self.page.eval_on_selector("#drift", "e => e.max")) + 2
        for _ in range(limit):
            headline = self.verdict_headline()
            if headline.startswith(prefix):
                return headline
            if headline.startswith("REJECT") and prefix == "FLAG":
                raise RuntimeError("drift reached REJECT before FLAG")
            self.page.keyboard.press("ArrowRight")
            self.wait(pause)
        raise RuntimeError(f"drift never reached {prefix}")


def script(d):
    page = d.page
    d.card("<h1>PMDB</h1><p>Which batch did this battery electrode come from?</p>"
           "<p>{{fp_n}} SEM sites · {{batch_count}} batches · {{heldout_n}} unlabelled held-out sites</p>"
           "<p class='k'>Answered from where the silicon sits, not from how the microscope was set.</p>")
    d.wait(6)
    d.view(1)
    d.uncard()
    d.cap("An intensity model scores <b>{{intensity_acc}}</b> on this data. Looks like a winner.")
    d.wait(5)
    d.cap("Brighten the image by <b>{{offset_step_word}} grey level</b> "
          "(a microscope setting; the electrode is unchanged)…")
    d.step(1, 2.5)
    d.mark("confound_flip")
    d.cap("…and it calls <b>every</b> {{batch_1}} and {{batch_2}} site {{batch_3}} ({{flip}}). "
          "It learned the microscope, not the material.")
    d.step(6, 0.5)
    d.wait(4.5)
    d.cap("Our model only uses geometry, so a brightness offset cannot move a single feature.")
    d.wait(4)

    d.view(2)
    d.cap("All composition KPIs are batch-blind: the {{batch_count_word_lower}} batches contain the same material.", top=True)
    d.mark("view2_top")
    d.wait(5)
    d.cap("{{fem_runs}} FEM charging simulations ({{fem_cost}} on Modal): swelling follows the Si amount (R² {{fem_r2}}), "
          "and the batches hold the same amount (p {{si_kw_p}}).", top=True)
    d.wait(7)
    d.cap("What differs is <b>where the silicon sits</b> through the depth: top-heavy, bottom-heavy, uniform.", top=True)
    d.wait(6)

    d.view(3)
    d.cap("Fingerprint from {{fp_features}} curve-shape features: <b>{{fp_score}}</b> leave-one-out, "
          "p = {{perm_p}} over {{perm_n}} label shuffles.")
    d.wait(7)
    d.cap("Pre-registered challengers (FEM features, XGBoost on screened KPIs at {{xgb_kpi_only}}) "
          "all failed their own rules. We kept the honest model.")
    d.wait(7)

    d.view(4)
    d.cap("Held-out calls with calibrated confidence, recomputed live in the browser and identical to the Python output.", top=True)
    d.mark("view4_top")
    d.wait(6.5)
    d.cap("{{call_3e122cbj_site}}: {{call_3e122cbj_batch}} with "
          "<b>confidence {{call_3e122cbj_conf}}</b>. It looks typical of every batch, and the model says so.", top=True)
    d.wait(6.5)

    d.view(5)
    d.cap("A new shipment drifts away from the baseline toward sedimentation…")
    d.wait(3)
    flag_headline = d.step_until("FLAG")
    batch = re.search(r"looks like (Batch \d+)", flag_headline)
    if not batch:
        raise RuntimeError(f"could not read assigned batch from verdict: {flag_headline}")
    d.cap("…it is flagged: <b>looks like {{flag_batch}}</b>. It tells you what kind of wrong it is.",
          extra={"flag_batch": batch.group(1)})
    d.mark("reject_flag")
    d.wait(3)
    d.step_until("REJECT")
    d.cap("Drift further and every batch rejects it: <b>out of distribution</b>.")
    d.mark("reject_reject")
    d.wait(3.5)
    d.cap("This is the QC answer a manufacturer needs: accept, re-assign, or reject the shipment.")
    d.wait(4.5)

    d.view(6)
    d.cap("Every site can be explored: overlays, depth profiles and FEM swelling animations.")
    d.wait(5.5)
    d.cap("")
    d.card("<h1>Can't be fooled by a brightness knob.<br>Knows when it doesn't know.</h1>"
           "<p>Geometry-only fingerprint · physics-checked with FEM · calibrated, pre-registered, live</p>"
           "<p class='k'>github.com/jkz22/PMDB</p>")
    d.mark("closing_card")
    d.wait(6)


def main():
    options = arguments()
    facts = collect()
    check_invariants(facts)
    values = display(facts)
    output = Path(options.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    with tempfile.TemporaryDirectory(prefix="pmdb-submission-") as video_dir:
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
            page.evaluate("() => { for (const id of ['sub-cap', 'sub-card']) { const e = document.createElement('div');"
                          " e.id = id; document.body.append(e); } }")
            page.wait_for_selector("#delta")
            director = Director(page, values, origin)
            script(director)
            markers = director.marks
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
    print(json.dumps({name: round(max(0, at - 0.5), 2) for name, at in markers.items()}, indent=1))


if __name__ == "__main__":
    main()
