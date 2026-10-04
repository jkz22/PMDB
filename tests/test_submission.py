from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "demo/submission"
sys.path.insert(0, str(SUBMISSION))
from build import StoryChange, TemplateError, check_invariants, collect, render  # noqa: E402


def test_renderer_rejects_unknown_keys():
    with pytest.raises(TemplateError, match="unknown template key"):
        render("value: {{not_a_fact}}", {})


def test_story_invariant_violation_raises():
    facts = collect()
    facts["flip1"] = facts["flip_total"] - 1
    with pytest.raises(StoryChange, match="STORY CHANGE:"):
        check_invariants(facts)


def test_check_matches_committed_facts():
    result = subprocess.run(
        [sys.executable, str(SUBMISSION / "build.py"), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "facts.json matches outputs/" in result.stdout
