"""Simulation result container and npz I/O (P14). Pure numpy."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FIELD_KEYS = ("J", "sxx", "szz", "sxz", "syy", "vm")


@dataclass
class SimResult:
    labels: np.ndarray
    px_um: float
    s: np.ndarray
    converged: np.ndarray
    u_nodes: np.ndarray  # (n, H+1, W+1, 2) float32, (ux, uz), uz positive toward image row 0
    fields: dict[str, np.ndarray]  # FIELD_KEYS -> (n, H, W) float32, NaN for unconverged frames
    failed_at_s: float  # nan if none
    substeps: list[dict] = field(default_factory=list)
    wall_s: float = 0.0


def save_npz(r: SimResult, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        labels=r.labels.astype(np.uint8),
        s=np.asarray(r.s, dtype=np.float64),
        converged=np.asarray(r.converged, dtype=bool),
        u_nodes=r.u_nodes.astype(np.float32),
        px_um=np.float64(r.px_um),
        failed_at_s=np.float64(r.failed_at_s),
        **{k: r.fields[k].astype(np.float32) for k in FIELD_KEYS},
    )


def load_npz(path: Path) -> SimResult:
    with np.load(Path(path)) as z:
        return SimResult(
            labels=z["labels"], px_um=float(z["px_um"]), s=z["s"], converged=z["converged"],
            u_nodes=z["u_nodes"], fields={k: z[k] for k in FIELD_KEYS},
            failed_at_s=float(z["failed_at_s"]), substeps=[], wall_s=0.0,
        )
