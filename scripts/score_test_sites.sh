#!/usr/bin/env bash
# Test day: score new organiser images. All compute runs on Modal. See docs/patch_mil.md (Test day runbook).
set -euo pipefail
cd "$(dirname "$0")/.."
ls data_test/Batch_test/img_*_BSE.tif >/dev/null 2>&1 || { echo "copy organiser TIFFs to data_test/Batch_test/img_<site>_<BSE|Inlens|ETD|SE>.tif"; exit 1; }
test -f outputs/menu/selection.json || { echo "outputs/menu/selection.json missing: the menu must be frozen (modal run modal_patch_mil.py --mode menu) before test images"; exit 1; }
modal run modal_test_prep.py::main
modal run modal_patch_mil.py --mode test
