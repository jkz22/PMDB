"""Generate QC overview figures for auditing imaging confounds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pmdb.io import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATA_ROOT,
    get_cache_root,
    get_data_root,
    list_sites,
    load_site,
)

DEFAULT_OUTPUTS_DIR = REPO_ROOT / "outputs"


def generate_contact_sheet(
    df_manifest: pd.DataFrame,
    output_path: Path,
    cache_root: Path | None = None,
    data_root: Path | None = None,
) -> None:
    """Generate outputs/qc_contact_sheet.png with one row per site, grouped by batch."""
    num_sites = len(df_manifest)
    fig, axes = plt.subplots(num_sites, 3, figsize=(12, max(6, num_sites * 1.5)), squeeze=False)

    col_names = ["BSE", "Inlens", "SE_type"]

    for idx, row in df_manifest.iterrows():
        b = row["batch"]
        s = row["site"]
        se_det = row["se_detector"]

        site_obj = load_site(b, s, resolution="half", normalise="percentile", cache_root=cache_root, data_root=data_root)

        for col_idx, col_name in enumerate(col_names):
            ax = axes[idx, col_idx]
            # Downsample thumbnail further for display if needed
            thumb = site_obj.image[::2, ::2, col_idx]
            ax.imshow(thumb, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])

            if idx == 0:
                ax.set_title(col_name, fontsize=11, fontweight="bold")

            if col_idx == 0:
                ax.set_ylabel(f"{b}\n{s}\n({se_det})", fontsize=8, rotation=0, labelpad=40, va="center")

    fig.suptitle("PMDB Site Overview: Multi-Detector Contact Sheet", fontsize=14, y=1.002)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved contact sheet to {output_path}")


def generate_raw_stats_plot(
    stats_csv_path: Path,
    output_path: Path,
) -> None:
    """Generate outputs/raw_stats_by_batch.png showing BSE p1, p50, and p99 per site coloured by batch."""
    if not stats_csv_path.exists():
        raise FileNotFoundError(f"Statistics file not found: {stats_csv_path}. Run build_cache.py first.")

    df = pd.read_csv(stats_csv_path)
    bse_df = df[df["detector"] == "BSE"].copy()
    bse_df.sort_values(by=["batch", "site"], inplace=True)
    bse_df.reset_index(drop=True, inplace=True)

    batches = bse_df["batch"].unique()
    batch_colors = {
        "Batch_1": "#1f77b4",
        "Batch_2": "#2ca02c",
        "Batch_3": "#d62728",
    }

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    x_indices = np.arange(len(bse_df))
    site_labels = [f"{row['batch'].split('_')[-1]}:{row['site']}" for _, row in bse_df.iterrows()]

    metrics = [
        ("p1", "BSE 1st Percentile (p1)", ax1),
        ("p50", "BSE Median (p50)", ax2),
        ("p99", "BSE 99th Percentile (p99)", ax3),
    ]

    for col, title, ax in metrics:
        for batch_name in batches:
            mask = (bse_df["batch"] == batch_name).to_numpy()
            color = batch_colors.get(batch_name, "black")
            ax.scatter(
                x_indices[mask],
                bse_df.loc[mask, col],
                label=batch_name,
                color=color,
                s=40,
                alpha=0.9,
            )
            ax.plot(
                x_indices[mask],
                bse_df.loc[mask, col],
                color=color,
                alpha=0.4,
                linestyle="--",
            )
        ax.set_ylabel(col, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.6)
        if ax == ax1:
            ax.legend(title="Batch", loc="upper left")

    ax3.set_xticks(x_indices)
    ax3.set_xticklabels(site_labels, rotation=90, fontsize=8)
    ax3.set_xlabel("Site (Batch:ID)", fontsize=11)

    fig.suptitle("Raw Intensity Statistics by Batch (BSE Channel)", fontsize=14, y=0.995)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved stats by batch plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QC overview figures.")
    parser.add_argument("--data-root", type=Path, default=None, help="Root directory for raw data.")
    parser.add_argument("--cache-root", type=Path, default=None, help="Root directory for cache.")
    parser.add_argument("--outputs-dir", type=Path, default=None, help="Output directory for figures.")
    args = parser.parse_args()

    d_root = get_data_root(args.data_root)
    c_root = get_cache_root(args.cache_root)
    o_dir = args.outputs_dir if args.outputs_dir is not None else DEFAULT_OUTPUTS_DIR

    df = list_sites(data_root=d_root)

    contact_sheet_path = o_dir / "qc_contact_sheet.png"
    stats_plot_path = o_dir / "raw_stats_by_batch.png"
    stats_csv_path = o_dir / "raw_intensity_stats.csv"

    generate_contact_sheet(df, contact_sheet_path, cache_root=c_root, data_root=d_root)
    generate_raw_stats_plot(stats_csv_path, stats_plot_path)


if __name__ == "__main__":
    main()
