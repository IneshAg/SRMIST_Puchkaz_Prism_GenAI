"""Real-world phrasings (collected from repair guides and support forums, see
docs/DEV_NOTES.md "Vocabulary sources") must land in the right category, and
the extensions must not move any kit complaint to a different category."""
import pytest

from enrichment import extract_symptom

CASES = [
    ("my phone is typing by itself", "screen_ghost_touch"),
    ("apps open on their own and it scrolls by itself", "screen_ghost_touch"),
    ("top half of the touch screen is not responding", "touch_unresponsive"),
    ("dead zone at the bottom of the screen", "touch_unresponsive"),
    ("half the screen went black", "half_screen_dark"),
    ("half of my screen is dark, other half works", "half_screen_dark"),
    ("black screen of death but it still vibrates", "screen_blank_black"),
    ("screen won't come on but I can hear notifications", "screen_blank_black"),
    ("stuck on a blue screen with tiny text", "screen_blank_black"),
    ("screen just dark, nothing visible", "screen_blank_black"),
    ("green line on my screen", "distorted_display"),
    ("pink lines across the display", "distorted_display"),
    ("screen burn-in, ghost image of the keyboard", "distorted_display"),
    ("screen is smashed", "screen_cracked"),
    ("display cracked and touch doesn't work in places", "screen_cracked"),
    ("touch is laggy, takes a second to respond", "touch_lag"),
    # must NOT be pulled into screen categories
    ("my battery drains fast", "battery_drain"),
    ("wifi keeps dropping", "wifi_connectivity"),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_real_world_phrasing_classifies(text, expected):
    assert extract_symptom(text)[0].category == expected
