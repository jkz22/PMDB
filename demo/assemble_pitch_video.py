#!/usr/bin/env python3
"""Assemble the submission video from the outputs/clips animations + title/result cards.

The clips (scripts/make_clips.py) carry the ideas and their own headlines; this
stitches them with matching dark cards into one <=2 min MP4. Edit SEQUENCE and
the two results tables below to change the narrative; the rest is mechanical.

    python demo/assemble_pitch_video.py [--out demo/backup/pmdb-submission-v4.mp4]
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
COL = {"Batch 1": "#3d8bfd", "Batch 2": "#ff7a45", "Batch 3": "#22c38e"}

# One model throughout: linear probe on MicroNet patch embeddings,
# leave-one-parent-out 0.71, permutation p = 0.005 (docs/final_predictions.md).
# Round 1 (3 sites): the model's parent-excluded calls vs the revealed truth.
ROUND1 = [  # site, called, truth
    ("fn0mhxef", "Batch 2", "Batch 1"),
    ("3e122cbj", "Batch 1", "Batch 2"),
    ("xrv9xvzb", "Batch 3", "Batch 3"),
]
# Round 2 (6 sites): the model's calls (no per-site confidences shown).
ROUND2 = [  # site, call
    ("0eryguqq", "Batch 3"),
    ("fhwrjtet", "Batch 3"),
    ("fspqbkxl", "Batch 2"),
    ("y59rxmxl", "Batch 1"),
    ("soo2ax3r", "Batch 1"),
    ("4hq27w4c", "Batch 2"),
]

# (kind, payload, seconds). card: (headline, [sublines]); clip: filename (seconds
# = freeze on last frame); image: a render_* function name.
SEQUENCE = [
    ("card", ("Three batches. Same recipe.\nWho made this electrode?",
              ["31 labelled SEM cross-sections · then two rounds of unknowns: 3 images, then 6 more"]), 6.0),
    ("clip", "01_confound_flip.mp4", 1.5),
    ("card", ("The obvious model learns the microscope,\nnot the material.",
              ["Every composition statistic is batch-blind — same silicon, same amounts.",
               "The real difference is arrangement."]), 5.5),
    ("clip", "02_depth_sweep.mp4", 1.5),
    ("card", ("Does arrangement matter?\nWe simulated charging to check.",
              ["Finite-element lithiation on the real microstructures, free lateral expansion."]), 4.5),
    ("clip", "03_fem_swelling.mp4", 1.5),
    ("image", "render_round1", 9.0),
    ("image", "render_round2", 10.0),
    ("card", ("Reads the electrode, not the microscope.\nKnows when it doesn’t know.",
              ["github.com/jkz22/PMDB"]), 7.0),
]


def _base_fig(kicker: str, headline: str):
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
    fig.text(0.06, 0.90, kicker, fontsize=20, color=ACCENT, fontweight="bold", family="DejaVu Sans")
    fig.text(0.06, 0.82, headline, fontsize=40, color=FG, fontweight="bold", family="DejaVu Sans")
    return fig


def render_round1(path: Path) -> None:
    fig = _base_fig("ROUND 1 · 3 unknown images", "Scored against the revealed truth.")
    for i, (site, called, truth) in enumerate(ROUND1):
        y = 0.62 - i * 0.13
        ok = called == truth
        fig.text(0.08, y, site, fontsize=26, color=DIM, family="DejaVu Sans Mono")
        fig.text(0.36, y, called, fontsize=26, color=COL[called], fontweight="bold", family="DejaVu Sans")
        fig.text(0.56, y, "→", fontsize=26, color=DIM)
        fig.text(0.61, y, f"true {truth}", fontsize=26, color=COL[truth], family="DejaVu Sans")
        fig.text(0.85, y, "✓" if ok else "✗", fontsize=28, fontweight="bold",
                 color="#22c38e" if ok else "#ff4d5e", family="DejaVu Sans")
    fig.text(0.06, 0.18, "Labels no team fully cracked (best in the field: 2/3). Why so hard?",
             fontsize=22, color=FG, family="DejaVu Sans")
    fig.text(0.06, 0.11, "The 34 “sites” are crops of 13 parent electrodes — the batch label follows\n"
             "the crop, not the material.", fontsize=22, color=ACCENT, family="DejaVu Sans", va="top")
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def render_round2(path: Path) -> None:
    fig = _base_fig("ROUND 2 · 6 new images", "The calls.")
    for i, (site, call) in enumerate(ROUND2):
        x = 0.10 + (i // 3) * 0.45
        y = 0.60 - (i % 3) * 0.13
        fig.text(x, y, site, fontsize=26, color=DIM, family="DejaVu Sans Mono")
        fig.text(x + 0.21, y, call, fontsize=26, color=COL[call], fontweight="bold", family="DejaVu Sans")
    fig.text(0.06, 0.15, "Validated leave-one-parent-out, so no sibling-crop leakage:\n"
             "accuracy 0.71, permutation p = 0.005.", fontsize=21, color=FG, family="DejaVu Sans", va="top")
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


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
    parser.add_argument("--out", default="demo/backup/pmdb-submission-v4.mp4")
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    with tempfile.TemporaryDirectory(prefix="pmdb-video-") as td:
        tmp = Path(td)
        segments = []
        for i, (kind, payload, seconds) in enumerate(SEQUENCE):
            seg = tmp / f"seg{i:02d}.mp4"
            if kind in ("card", "image"):
                png = tmp / f"card{i:02d}.png"
                if kind == "card":
                    render_card(*payload, png)
                else:
                    globals()[payload](png)
                cmd = [ffmpeg, "-y", "-loglevel", "error", "-loop", "1", "-t", f"{seconds}",
                       "-i", str(png), "-vf", "fade=t=in:d=0.4,scale=1920:1080,fps=30",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", str(seg)]
            else:
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
