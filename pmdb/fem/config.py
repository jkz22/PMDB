"""Loader for the single FEM parameter set (configs/fem/fem.yaml)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

FEM_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "fem" / "fem.yaml"


def load_params(path: Path = FEM_CONFIG_PATH) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def params_hash(p: dict) -> str:
    return hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest()[:12]


def case_params_hash(p: dict, lateral: str = "both") -> str:
    """Hash of the params plus the lateral boundary condition. run_case pops `solver.lateral` out of the params
    before simulating, so it must be folded back in here; the default "both" leaves the hash unchanged."""
    return params_hash(p if lateral == "both" else {**p, "lateral": lateral})
