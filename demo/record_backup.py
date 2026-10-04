#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright


def arguments():
    parser = argparse.ArgumentParser(description="Record the scripted PMDB dashboard walkthrough.")
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--out", default="demo/backup/pmdb-dashboard-demo.mp4")
    parser.add_argument("--speed", type=float, default=1)
    options = parser.parse_args()
    if options.speed <= 0:
        parser.error("--speed must be greater than zero")
    return options


def main():
    options = arguments()
    output = Path(options.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    target_url = f"{options.url.rstrip('/')}/?autoplay=1&speed={options.speed:g}#confound"
    console_errors = []
    page_errors = []

    with tempfile.TemporaryDirectory(prefix="pmdb-dashboard-recording-") as video_dir:
        browser = None
        context = None
        webm_path = None
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                context = browser.new_context(
                    viewport={"width": 1600, "height": 900},
                    record_video_dir=video_dir,
                    record_video_size={"width": 1600, "height": 900},
                    device_scale_factor=1,
                )
                page = context.new_page()
                page.on(
                    "console",
                    lambda message: console_errors.append(message.text)
                    if message.type == "error"
                    else None,
                )
                page.on("pageerror", lambda error: page_errors.append(str(error)))
                page.goto(target_url, wait_until="domcontentloaded", timeout=60_000)
                page.wait_for_function(
                    "document.body && document.body.dataset.autoplayDone === '1'",
                    timeout=240_000,
                )
                page.wait_for_timeout(1_000)
                video = page.video
                context.close()
                context = None
                webm_path = Path(video.path())
                browser.close()
                browser = None
        except Exception as error:
            print(f"Recording failed: {error}", file=sys.stderr)
            return 1
        finally:
            if context is not None:
                context.close()
            if browser is not None:
                browser.close()

        if not webm_path or not webm_path.is_file():
            print("Recording failed: Playwright did not produce a WebM file.", file=sys.stderr)
            return 1
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            print("Recording failed: ffmpeg is not installed or not on PATH.", file=sys.stderr)
            return 1
        command = [
            ffmpeg,
            "-y",
            "-ss",
            "0.5",
            "-i",
            str(webm_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            "-preset",
            "slow",
            "-movflags",
            "+faststart",
            "-an",
            str(output),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            print(result.stderr, file=sys.stderr)
            return result.returncode

    for error in console_errors:
        print(f"Console error: {error}")
    for error in page_errors:
        print(f"Page error: {error}", file=sys.stderr)
    print(f"Saved {output}")
    return 1 if page_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
