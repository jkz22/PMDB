#!/usr/bin/env python3
"""Render the submission materials from demo/submission/facts.py.

    python demo/submission/build.py            # write facts.json and every generated file
    python demo/submission/build.py --check    # compare committed facts.json with outputs/, exit 3 if stale
    python demo/submission/build.py --video    # render, then re-record demo/backup/pmdb-submission.mp4
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
from facts import collect  # noqa: E402

TEMPLATES = HERE / "templates"
FACTS_JSON = HERE / "facts.json"
VIDEO = "demo/backup/pmdb-submission.mp4"
HELDOUT_SITES = {"3e122cbj", "fn0mhxef", "xrv9xvzb"}
RENDERED = [("submission.md", "docs/slides/submission.md"), ("qa.md", "docs/slides/qa.md"),
            ("run-of-show.md", "docs/slides/run-of-show.md"), ("pitch.html", "docs/slides/pitch.html")]
README = "README.md"
README_TEMPLATE = "readme_block.md"
START, END = "<!-- submission:start -->", "<!-- submission:end -->"
TOKEN = re.compile(r"\{\{([^{}]*)\}\}")


class StoryChange(RuntimeError):
    """A narrative invariant the written materials depend on no longer holds."""


class TemplateError(RuntimeError):
    """A template used a key the display layer does not provide."""


def percent(value):
    return f"{round(value * 100)}"


def percent_exact(value):
    return f"{value * 100:.1f}".rstrip("0").rstrip(".")


def integer_words(number):
    small = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
             "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
             "eighteen", "nineteen")
    tens = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
    if number < 20:
        return small[number]
    if number < 100:
        return tens[number // 10] + (f"-{small[number % 10]}" if number % 10 else "")
    if number < 1000:
        hundreds, remainder = divmod(number, 100)
        return f"{small[hundreds]} hundred" + (f" {integer_words(remainder)}" if remainder else "")
    if number < 1_000_000:
        thousands, remainder = divmod(number, 1000)
        return f"{integer_words(thousands)}\nthousand" + (f" {integer_words(remainder)}" if remainder else "")
    return f"{number:,}"


def display(facts):
    """Pre-formatted strings for the templates, matching the wording of the written materials."""
    calls = {c["site"]: c for c in facts["heldout"]}
    arms = {(a["family"], a["tag"], a["arm"]): a for a in facts["challengers"]}
    jk_low = min(c["jk_share_min"] for c in facts["heldout"])
    jk_high = max(c["jk_share_max"] for c in facts["heldout"])
    values = {
        "fp_n": f"{facts['fp_n']}",
        "fp_correct": f"{facts['fp_correct']}",
        "fp_features": f"{facts['fp_features']}",
        "fp_score": f"{facts['fp_correct']}/{facts['fp_n']}",
        "fp_score_words": f"{facts['fp_correct']} of {facts['fp_n']}",
        "batch_count": f"{len(facts['swell_median'])}",
        "batch_count_word": integer_words(len(facts["swell_median"])).capitalize(),
        "batch_count_word_lower": integer_words(len(facts["swell_median"])),
        "heldout_n": f"{len(facts['heldout'])}",
        "heldout_n_word": integer_words(len(facts["heldout"])).capitalize(),
        "fp_acc": f"{facts['fp_acc']:.3f}",
        "majority_baseline": f"{facts['majority_baseline']:.3f}",
        "perm_n": f"{facts['perm_n']:,}",
        "perm_n_words": integer_words(facts["perm_n"]),
        "perm_p": f"{facts['perm_p']:.4f}",
        "perm_null_p95": f"{facts['perm_null_p95']:.3f}",
        "intensity_acc": f"{facts['intensity_acc']:.2f}",
        "offset_step_word": integer_words(1),
        "flip": f"{facts['flip1']} of {facts['flip_total']}",
        "flip7": f"{facts['flip7']} of {facts['flip_total']}",
        "fem_runs": f"{facts['fem_runs']}",
        "fem_sites": f"{facts['fem_sites']}",
        "fem_missing": f"{facts['fem_missing']}",
        "fem_cost": f"${facts['fem_cost_usd']:.2f}",
        "fem_r2": f"{facts['fem_r2']:.2f}",
        "fem_r2_3": f"{facts['fem_r2']:.3f}",
        "si_kw_p": f"{facts['si_kw_p']:.2f}",
        "si_kw_p_3": f"{facts['si_kw_p']:.3f}",
        "jk_refits": f"{facts['jk_refits']:,}",
        "jk_stable": f"{percent(jk_low)}–{percent(jk_high)}%",
        "hp_configs": f"{facts['hp_configs']}",
        "xgb_kpi_only": f"{facts['xgb_kpi_only_correct']}/{facts['fp_n']}",
        "xgb_kpi_only_words": f"{facts['xgb_kpi_only_correct']} of {facts['fp_n']}",
        "xgb_kpi_only_p": f"{arms[('xgb_kpi', 'main', 'X1')]['perm_p']:.2f}",
        "xgb_best": f"{facts['xgb_best_correct']}/{facts['fp_n']}",
        "fem_fp_a1": f"{arms[('fem_fingerprint', 'main', 'A1')]['correct']}/{facts['fp_n']}",
        "fem_fp_a2": f"{arms[('fem_fingerprint', 'main', 'A2')]['correct']}/{facts['fp_n']}",
    }
    for batch, median in facts["swell_median"].items():
        batch_number = batch.split("_")[1]
        values[f"batch_{batch_number}"] = batch.replace("_", " ")
        values[f"swell_b{batch_number}"] = f"{median:.3f}"
    for site, call in calls.items():
        values.update({
            f"call_{site}_batch": call["assigned"].replace("_", " "),
            f"call_{site}_cred": f"{call['credibility']:.3f}",
            f"call_{site}_cred2": f"{call['credibility']:.2f}",
            f"call_{site}_conf": f"{call['confidence']:.3f}",
            f"call_{site}_conf2": f"{call['confidence']:.2f}",
            f"call_{site}_jk": f"{percent_exact(call['jk_share_min'])}–{percent_exact(call['jk_share_max'])}%",
            f"call_{site}_hp": f"{call['hp_agree']}/{call['hp_configs']}",
            f"call_{site}_site": site,
        })
        if site == "3e122cbj":
            confidence = float(call["confidence"])
            values["call_3e122cbj_conf_plain"] = f"{confidence:g}"
            values["call_3e122cbj_conf_word"] = (
                integer_words(int(confidence)) if confidence.is_integer() else f"{confidence:g}"
            )
    return values


def check_invariants(facts):
    """The materials assert these; rendering text that has become false is worse than failing."""
    sites = {c["site"]: c for c in facts["heldout"]}
    if facts["any_challenger_passes"]:
        passing = [a["arm"] for a in facts["challengers"] if a["passes"]]
        raise StoryChange(f"STORY CHANGE: a pre-registered challenger now passes its rule: {', '.join(passing)}")
    if facts["flip1"] != facts["flip_total"]:
        raise StoryChange("STORY CHANGE: a +1 grey level no longer flips every Batch 1/2 site "
                          f"({facts['flip1']} of {facts['flip_total']})")
    if facts["fem_missing"] != 0:
        raise StoryChange(f"STORY CHANGE: {facts['fem_missing']} FEM cases are missing, so '0 failures' is false")
    if set(sites) != HELDOUT_SITES:
        raise StoryChange(f"STORY CHANGE: the held-out sites are now {sorted(sites)}, not {sorted(HELDOUT_SITES)}")
    if sites["3e122cbj"]["confidence"] != 0:
        raise StoryChange("STORY CHANGE: 3e122cbj no longer has confidence 0 "
                          f"({sites['3e122cbj']['confidence']})")
    if facts["unreviewed_output_dirs"]:
        raise StoryChange("STORY CHANGE: unreviewed result directories: "
                          f"{', '.join(facts['unreviewed_output_dirs'])}")


def render(template, values):
    """Substitute {{key}}; an unknown key or any leftover brace pair is an error."""
    def one(match):
        key = match.group(1).strip()
        if key not in values:
            raise TemplateError(f"unknown template key: {{{{{key}}}}}")
        return values[key]

    text = TOKEN.sub(one, template)
    if "{{" in text or "}}" in text:
        raise TemplateError(f"unrendered placeholder left in output: {text[max(0, text.find('{{') - 40):][:120]!r}")
    return text


def header(name):
    return f"<!-- generated by demo/submission/build.py — edit demo/submission/templates/{name} -->"


def with_header(name, text):
    if text.startswith("<!DOCTYPE"):
        first, _, rest = text.partition("\n")
        return f"{first}\n{header(name)}\n{rest}"
    return f"{header(name)}\n\n{text}"


def render_readme(values):
    readme = (ROOT / README).read_text()
    start, end = readme.find(START), readme.find(END)
    if start < 0 or end < 0 or start >= end:
        raise TemplateError(f"{README} is missing the {START} / {END} markers")
    block = render((TEMPLATES / README_TEMPLATE).read_text(), values)
    inner = f"\n{header(README_TEMPLATE)}\n\n{block.strip()}\n\n"
    return readme[:start + len(START)] + inner + readme[end:]


def render_outputs(values):
    rendered = {}
    for name, target in RENDERED:
        rendered[target] = with_header(name, render((TEMPLATES / name).read_text(), values))
    rendered[README] = render_readme(values)
    return rendered


def build(values):
    rendered = render_outputs(values)
    for target, text in rendered.items():
        (ROOT / target).write_text(text)
    return list(rendered)


def report_drift(old, new):
    """Print what moved between the committed facts.json and a fresh collect(); True if anything differs."""
    changed = False
    for key in sorted(set(old) | set(new)):
        if key == "sources":
            continue
        if old.get(key) != new.get(key):
            changed = True
            print(f"changed fact: {key}: {old.get(key)!r} -> {new.get(key)!r}")
    old_sources, new_sources = old.get("sources", {}), new.get("sources", {})
    for key in sorted(set(old_sources) | set(new_sources)):
        if old_sources.get(key) != new_sources.get(key):
            changed = True
            print(f"changed source: {key}: {old_sources.get(key)} -> {new_sources.get(key)}")
    fresh = [d for d in new.get("unreviewed_output_dirs", []) if d not in old.get("unreviewed_output_dirs", [])]
    for directory in fresh:
        print(f"new unreviewed output dir: outputs/{directory}")
    return changed


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render the submission materials from outputs/.")
    parser.add_argument("--check", action="store_true", help="compare committed facts.json with outputs/, write nothing")
    parser.add_argument("--video", action="store_true", help="re-record the submission video after rendering")
    options = parser.parse_args(argv)

    facts = collect()
    if options.check:
        if not FACTS_JSON.exists():
            print(f"missing {FACTS_JSON.relative_to(ROOT)}; run python demo/submission/build.py")
            return 3
        changed = report_drift(json.loads(FACTS_JSON.read_text()), facts)
        print("facts.json is stale" if changed else "facts.json matches outputs/")
        return 3 if changed else 0

    check_invariants(facts)
    rendered = render_outputs(display(facts))
    FACTS_JSON.write_text(json.dumps(facts, indent=1, sort_keys=True) + "\n")
    for target, text in rendered.items():
        (ROOT / target).write_text(text)
    for target in [FACTS_JSON.relative_to(ROOT).as_posix(), *rendered]:
        print(target)
    if options.video:
        subprocess.run([sys.executable, str(ROOT / "demo/record_submission.py"), "--out", str(ROOT / VIDEO)],
                       cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
