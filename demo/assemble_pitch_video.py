#!/usr/bin/env python3
"""Assemble the submission video from the outputs/clips animations + title cards.

The four clips (scripts/make_clips.py) carry the ideas and their own headlines;
this stitches them with matching dark title cards into one <=2 min MP4.
Edit CARDS below to change the narrative; everything else is mechanical.

    python demo/assemble_pitch_video.py [--out demo/backup/pmdb-submission-v3.mp4]
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "outputs" / "clips"
BG, FG, DIM, ACCENT = "#0d1117", "#f0f3f6", "#8b949e", "#ffcf5c"

# (kind, payload, seconds). Card payload: (headline, [sublines]); clip payload: filename.
SEQUENCE = [
    ("card", ("Three batches. Same recipe.\nWho made this electrode?",
              ["31 labelled SEM cross-sections · 3 unknown sites",
               "First instinct: train a model on the images. Watch what happens."]), 6.0),
    ("clip", "01_confound_flip.mp4", 1.5),
    ("card", ("The obvious model learns the microscope,\nnot the material.",
              ["Every composition statistic is batch-blind — same silicon, same amounts.",
               "The real difference is arrangement."]), 5.5),
    ("clip", "02_depth_sweep.mp4", 1.5),
    ("card", ("Does arrangement matter?\nWe simulated charging to check.",
              ["Finite-element lithiation on the real segmented microstructures."]), 4.5),
    ("clip", "03_fem_swelling.mp4", 1.5),
    ("card", ("What a manufacturer actually needs:\naccept or reject the shipment.",
              ["Calibrated confidence — a model that says “I don’t know” when that is the truth."]), 5.0),
    ("clip", "04_verdict_reject.mp4", 2.0),
    ("card", ("Can’t be fooled by a brightness knob.\nKnows when it doesn’t know.",
              ["github.com/jkz22/PMDB"]), 7.0),
]


def render_card(headline: str, sublines: list[str], path: Path) -> None:
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
    y = 0.60
    fig.text(0.5, y, headline, ha="center", va="center", color=FG,
             fontsize=44, fontweight="bold", family="DejaVu Sans", linespacing=1.4)
    y -= 0.10 + 0.05 * headline.count("\n")
    for i, line in enumerate(sublines):
        colour = ACCENT if line.startswith("github") else DIM
        fig.text(0.5, y - 0.07 * i, line, ha="center", va="center", color=colour,
                 fontsize=23, family="DejaVu Sans")
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="demo/backup/pmdb-submission-v3.mp4")
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    with tempfile.TemporaryDirectory(prefix="pmdb-video-") as td:
        tmp = Path(td)
        segments = []
        for i, (kind, payload, seconds) in enumerate(SEQUENCE):
            seg = tmp / f"seg{i:02d}.mp4"
            if kind == "card":
                png = tmp / f"card{i:02d}.png"
                render_card(*payload, png)
                cmd = [ffmpeg, "-y", "-loglevel", "error", "-loop", "1", "-t", f"{seconds}",
                       "-i", str(png), "-vf", "fade=t=in:d=0.4,scale=1920:1080,fps=30",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", str(seg)]
            else:
                # play the clip once, then freeze its last frame for `seconds`
                cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(CLIPS / payload),
                       "-vf", f"tpad=stop_mode=clone:stop_duration={seconds},fps=30,scale=1920:1080",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", str(seg)]
            subprocess.run(cmd, check=True)
            segments.append(seg)

        listfile = tmp / "list.txt"
        listfile.write_text("".join(f"file '{s.as_posix()}'\n" for s in segments))
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", str(listfile), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-preset", "slow", "-crf", "21", "-movflags", "+faststart", "-an",
                        str(out)], check=True)

    probe = subprocess.run([ffmpeg, "-i", str(out)], capture_output=True, text=True)
    duration = [line for line in probe.stderr.splitlines() if "Duration" in line]
    print(out)
    print(duration[0].strip() if duration else "duration unknown")


if __name__ == "__main__":
    main()
