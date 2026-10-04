"""Offline Stage 1 (src/rule_extraction.py): plans built from the article's
own headings and instruction sentences, never invented."""
import json
import re
from pathlib import Path

import pytest

from rule_extraction import rule_based_extraction
from structure_extraction import _validate_structure

DATA = Path(__file__).resolve().parent.parent / "data"
ARTICLES = {r["siis_response"]["title"]: r["siis_response"]
            for r in json.loads((DATA / "siis_responses.json").read_text(encoding="utf-8"))["responses"]}


def _words(t):
    return set(re.findall(r"[a-z0-9]+", t.lower()))


@pytest.mark.parametrize("title", sorted(ARTICLES))
def test_every_kit_article_gives_a_valid_grounded_plan(title):
    art = ARTICLES[title]
    raw = rule_based_extraction(art)
    assert raw is not None
    plan = _validate_structure(raw, f"{art['title']}\n{art['content']}")
    assert plan.contexts and plan.contexts[0].actions
    source = _words(art["content"])
    for action in plan.contexts[0].actions:
        assert action.description.startswith("It will") and 5 <= len(action.description.split()) <= 7
        for step in action.stepGroups[0].steps:
            # every word of every step comes from the article (no invented steps)
            assert _words(step) <= source, step


def test_offline_plan_is_specific_not_the_old_two_line_echo():
    raw = rule_based_extraction(ARTICLES["Blank or black display on a smartphone or tablet"], "screen_blank_black")
    names = [a["actionName"] for a in raw["contexts"][0]["actions"]]
    assert names == ["Check for Physical Damage and Liquid Exposure", "Force a Restart",
                     "Charge the Device", "Attempt to Power On"]
    assert raw["contexts"][0]["title"] == "Screen blank black"


def test_instruction_chains_are_split_one_interaction_per_step():
    raw = rule_based_extraction({"title": "Touch", "content": "## Touch sensitivity\n"
                                 "To do this, go to Settings, tap Display, and then tap the switch next to Touch sensitivity."})
    assert raw["contexts"][0]["actions"][0]["stepGroups"][0]["steps"] == [
        "Go to Settings.", "Tap Display.", "Tap the switch next to Touch sensitivity."]


def test_article_without_instructions_gives_nothing():
    assert rule_based_extraction({"title": "About", "content": "## Understanding screens\nScreens are made of glass."}) is None
