from __future__ import annotations

import re
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


def test_heldout_assignment_invariant_raises():
    facts = collect()
    call = next(c for c in facts["heldout"] if c["site"] == "fn0mhxef")
    call["assigned"] = "Batch_1"
    with pytest.raises(
        StoryChange,
        match="STORY CHANGE: fn0mhxef is now Batch_1, but the card text explains Batch_3",
    ):
        check_invariants(facts)


def test_dashboard_inputs_are_in_provenance():
    dashboard = (ROOT / "demo/server.mjs").read_text()
    dashboard_inputs = {
        path.removeprefix("outputs/") for path in re.findall(r"'(outputs/[^']+)'", dashboard)
    }
    assert dashboard_inputs
    assert dashboard_inputs <= set(collect()["sources"])


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


def test_committed_outputs_match_templates():
    """Generated files must equal their template render; edit demo/submission/templates/, not the output."""
    import json

    import build

    rendered = build.render_outputs(build.display(json.loads(build.FACTS_JSON.read_text())))
    drifted = [str(t) for t, text in rendered.items() if (ROOT / t).read_text() != text]
    assert not drifted, f"edited without updating demo/submission/templates/: {drifted}"
