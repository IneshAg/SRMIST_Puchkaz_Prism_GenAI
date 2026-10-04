"""
Stage 0 — Query Enrichment (Member A)

    normalize_query(raw_complaint)  -> canonical technical query + extracted
                                        device / symptom metadata
    generate_variations(canonical)  -> 8-10 paraphrases spanning
                                        formal / casual / keyword /
                                        frustrated / typo registers

Design
------
Customer complaints arrive in wildly different registers (a careful
formal sentence, a two-word keyword search, a typo-ridden rant). Stage 1
(LLM structuring) and Stage 2 (deeplink retrieval) both work far better
against a single clean technical query than against raw free text, and
the semantic cache (Stage 3) gets much better hit-rate recall if it has
several paraphrases of the same underlying request to match against
(that's what the roadmap calls "A5 — query variations": without them, a
cache primed on one phrasing of "screen is black" won't hit on "display
won't turn on").

`normalize_query` is regex/keyword based, not an LLM call: it extracts
the device model and a symptom category from a small taxonomy built by
inspecting the 20 sample complaints in data/input.txt (screen blank/black,
flicker, crack, touch unresponsive, touch lag, half-screen dark, inner
screen failure, distorted display, undersized display, floating
assistant icon). This is fast, deterministic, and — critically — cannot
hallucinate a symptom that isn't there, which matters for the "no
hallucinations" requirement.

`generate_variations` is template-based per symptom category by default
(deterministic, always produces valid output even with zero API keys
configured), and will additionally ask the configured LLM for extra
paraphrases when one is available (see llm_client.py) — those are
programmatically validated (non-empty, reasonable length, deduped)
before being mixed in, so a misbehaving LLM call degrades gracefully
instead of corrupting the output.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from llm_client import LLMClient, MockLLMClient, get_llm_client

REGISTERS = ["formal", "casual", "keyword", "frustrated", "typo"]
MIN_VARIATIONS = 8
MAX_VARIATIONS = 10

# ---------------------------------------------------------------------------
# Device extraction
# ---------------------------------------------------------------------------

_DEVICE_PATTERN = re.compile(
    # Rebranded kit data (TechCorp / Nexa). Listed first; the Galaxy/Samsung
    # branches below are kept so older phrasings still resolve.
    r"(Nexa\s+(?:Fold|Flip)\s*X?\d+|(?:Fold|Flip)\s*X\d+"
    r"|Nexa\s+X\d+\s*(?:Ultra|Plus|\+|FE)?|TechCorp\s+X\d+\s*(?:Ultra|Plus|\+|FE)?"
    r"|Nexa\s+A\d+(?:\s*/\s*A\d+)?"
    r"|Nexa\s+Tab\s*[A-Za-z0-9]*"
    r"|TechCorp\s+[A-Z0-9]{3,}G?\s+tablet"
    r"|Galaxy\s+Z\s+Flip\s*\d+|Galaxy\s+Flip\s*\d+|Z\s+Flip\s*\d+"
    r"|Galaxy\s+Z\s+Fold\s*\d+|Galaxy\s+Fold\s*\d+"
    r"|Galaxy\s+S\d+\s*(?:Ultra|Plus|\+|FE)?"
    r"|Galaxy\s+A\d+(?:\s*/\s*A\d+)?"
    r"|Galaxy\s+Note\s*\d+\s*(?:Ultra|\+)?"
    r"|Galaxy\s+Tab\s*[A-Za-z0-9]*\s*(?:FE|Ultra|\+)?|Galaxy\s+tablet"
    r"|Galaxy\s+Watch\s*\d*\s*(?:Classic|Pro|Ultra)?"
    r"|Galaxy\s+Buds\s*[A-Za-z0-9]*\s*(?:Pro)?"
    r"|SM-[A-Z]\d{3,4}[A-Z0-9]*"
    r"|Samsung\s+[A-Z0-9]{3,}G?\s+tablet"
    r"|Samsung\s+Galaxy\s+[A-Za-z0-9]+(?:\s*/\s*[A-Za-z0-9]+)?(?:\s+Ultra)?"
    r"|Samsung\s+S\*+\s*Ultra)",
    re.IGNORECASE,
)


UNKNOWN_DEVICE_LABEL = "TechCorp device"  # fallback when no specific model is recognized


def extract_device(raw: str) -> str:
    # Drop a "Samsung" brand prefix in front of "Nexa" first: regex search is
    # leftmost-first, so "Samsung Galaxy Z Flip 6" used to be captured by the
    # generic "Samsung Galaxy <word>" branch as just "Samsung Galaxy Z".
    raw = re.sub(r"(?i)\btechcorp\s+(?=nexa\b)", "", raw)
    match = _DEVICE_PATTERN.search(raw)
    if not match:
        return UNKNOWN_DEVICE_LABEL
    device = re.sub(r"\s+", " ", match.group(0)).strip()
    if re.match(r"(?i)^sm-", device):
        # SM-A536E-style model codes are conventionally all-caps; a keyword-style
        # complaint typed in lowercase ("my sm-a536e...") shouldn't come out as
        # "Sm-a536e" from the generic title-casing below.
        return device.upper()
    # Normalize casing: "galaxy s22" -> "Nexa S22"
    return " ".join(w[0].upper() + w[1:] if w and w[0].isalpha() else w for w in device.split())


# ---------------------------------------------------------------------------
# Symptom taxonomy — (category, matcher keywords, formal/casual/keyword text)
# ---------------------------------------------------------------------------

@dataclass
class Symptom:
    category: str
    label: str
    # Matching is (component word present) AND (problem word present), not one long
    # exact phrase — an earlier version required e.g. "camera app crashes" verbatim
    # and silently missed "the camera app on my S24 keeps crashing" (different word
    # order/tense). component_terms=[] means "no specific part" (whole-device
    # symptoms like overheating/slow performance), so only problem_terms need match.
    component_terms: List[str]
    problem_terms: List[str]
    subject: str  # the device component this is about — "screen", "battery", "Wi-Fi
                  # connection", etc. — inserted by the template, never assumed to be
                  # "screen" (see the canonical-query bug this replaced: a battery
                  # complaint was coming out as "screen is exhibiting a display issue").
    formal: str
    casual: str
    keyword: str


SYMPTOM_TAXONOMY: List[Symptom] = [
    # -- display/screen (data/input.txt's 20 samples are all in this group) --
    # More specific screen categories are listed first: extract_symptom() keeps the
    # first symptom seen on a tied score, so e.g. flicker-then-blank (2 hits: "screen"
    # + "flicker") outranks the generic blank/black category on a complaint that
    # mentions both, matching how these 20 samples actually read.
    # Physical damage first: when a complaint mentions a crack alongside a
    # symptom it causes ("cracked ... touch doesn't work in places"), the crack
    # is the actionable root cause (repair), so it wins the tie.
    Symptom(
        "screen_cracked",
        "Screen physically cracked/damaged",
        ["screen", "display"],
        ["crack", "cracked", "shattered", "shatter", "smashed", "spider web", "spiderweb",
         "broken glass", "screen broke"],
        "screen",
        "has a physical crack across it",
        "is cracked and busted up",
        "cracked physical damage",
    ),
    Symptom(
        "screen_flicker_then_blank",
        "Screen flickers then goes blank",
        ["screen", "display"],
        ["flash", "flicker"],
        "screen",
        "flickers briefly and then goes completely blank",
        "keeps flickering and then just goes blank",
        "flicker then blank",
    ),
    Symptom(
        "screen_ghost_touch",
        "Screen registers touches on its own (ghost touches)",
        ["screen", "touch", "display", "phone", "typing", "apps"],
        # "on its own"/"by itself" alone used to also fire on any spontaneous-failure
        # complaint (a screen that "goes black on its own" is not a ghost-touch report) --
        # kept touch-specific instead so it can only fire alongside real touch behavior.
        ["ghost touch", "random touches", "phantom touch", "touch input on its own",
         "touching on its own", "taps on its own", "touches on its own",
         # (web-researched phrasing, see docs/DEV_NOTES.md "Vocabulary sources")
         "typing by itself", "types by itself", "typing on its own", "types on its own",
         "scrolls by itself", "scrolling by itself", "scrolls on its own", "opens apps by itself",
         "opening apps by itself", "apps open on their own", "does things on its own",
         "doing things on its own", "phantom tap", "ghost tap", "false touch"],
        "screen",
        "registers touch input on its own without anyone touching it",
        "keeps tapping stuff by itself, ghost touches everywhere",
        "ghost touch screen registers on its own",
    ),
    Symptom(
        "inner_screen_failure",
        "Inner/foldable screen failed, outer screen fine",
        ["inner screen", "cover screen", "fold"],
        ["stopped working", "no image", "dead", "doesn't respond", "doesnt respond",
         "not responding", "stopped", "died"],
        "inner foldable display",
        "has stopped producing an image or responding to touch, while the cover screen remains functional",
        "just died but the cover screen still works",
        "dead, cover screen works, foldable",
    ),
    Symptom(
        "touch_unresponsive",
        "Touchscreen unresponsive / can't interact",
        ["screen", "touch", "touchscreen", "display"],
        ["unresponsive", "doesn't respond", "doesnt respond", "does not respond", "won't respond",
         "wont respond", "not responding", "can't interact", "cant interact", "unable to interact",
         # (web-researched phrasing, see docs/DEV_NOTES.md "Vocabulary sources")
         "stops responding", "stopped responding", "refuses to respond", "not responding to touch",
         "dead zone", "dead spot", "not registering", "won't register", "wont register",
         "doesn't register", "doesnt register", "touch not working", "touch stopped working",
         "touch doesn't work", "touch isn't working", "touch is not working"],
        "touchscreen",
        "does not respond to touch input at all",
        "won't respond no matter how I tap it",
        "unresponsive no input",
    ),
    Symptom(
        "touch_lag",
        "Touch input delayed / laggy",
        ["touch", "tap", "input"],
        ["delayed", "laggy", "lag", "noticeable delay", "slow to respond", "delay", "lagging",
         "sluggish", "takes a second to respond", "slow response"],
        "touch input",
        "exhibits a noticeable delay before responding",
        "feels laggy, there's a delay when I tap stuff",
        "delay laggy",
    ),

    # -- input / hardware buttons -- added after scanning data/deeplinks.json's actual
    # text for clusters the original 26 categories didn't cover ("keyboard", "power
    # button"/"volume button" phrasing show up repeatedly with no matching category).
    Symptom(
        "keyboard_typing_problem",
        "On-screen keyboard not typing correctly / unresponsive",
        # "when i type"/"while typing"/etc. cover a common real phrasing that never
        # says the word "keyboard" at all ("something weird when I type, letters
        # come out wrong") -- understandable to a person, but the plain "keyboard"
        # component word alone used to miss it entirely, sending it to the guarded
        # LLM fallback (still correct, just slower/less certain than it needs to be
        # for a phrasing this common). These are multi-word phrases, not the bare
        # word "type", specifically so this doesn't start firing on unrelated
        # complaints that merely mention a device "type" or model.
        ["keyboard", "when i type", "while typing", "when typing", "as i type"],
        ["not typing", "wont type", "won't type", "stuck", "lagging", "not working",
         "keeps freezing", "autocorrect is broken", "keeps messing up", "mistypes",
         "letters come out", "come out wrong", "come out jumbled", "jumbled", "gibberish",
         "fails to register"],
        "keyboard",
        "fails to register keystrokes correctly or becomes unresponsive during use",
        "keeps messing up when I type, keys stick or don't register",
        "keyboard not typing stuck",
    ),
    Symptom(
        "hardware_button_unresponsive",
        "Power or volume button physically stuck/unresponsive",
        ["power button", "volume button", "volume key", "power key", "button"],
        ["stuck", "not working", "doesn't work", "doesnt work", "stopped working",
         "unresponsive", "wont press", "won't press", "hard to press"],
        "button",
        "has become physically stuck or unresponsive to presses",
        "just doesn't click anymore, feels stuck",
        "button stuck not working",
    ),

    Symptom(
        "half_screen_dark",
        "Half of the display is dark/unresponsive",
        ["screen", "display", "half", "side"],
        ["half black", "one side", "half dark", "half is completely dark",
         "half completely dark", "half is dead",
         # (web-researched phrasing, see docs/DEV_NOTES.md "Vocabulary sources")
         "half the screen", "half of the screen", "half of my screen", "half the display",
         "half of the display", "other half", "half went black", "half is black"],
        "screen",
        "shows one half completely dark while the other half functions normally",
        "half is dead, other half's fine",
        "half dark half working",
    ),
    Symptom(
        "screen_partial_lit",
        "Only part of the display lights up",
        ["screen", "display", "icons"],
        ["only three app icons", "partially lit", "some icons light up", "stays dark and only",
         "illuminates a small portion", "icons light up"],
        "screen",
        "only illuminates a small portion while the remainder stays dark",
        "only a few icons light up, rest is dark",
        "partially lit rest dark",
    ),
    Symptom(
        "screen_blank_black",
        "Screen completely blank/black, no display output",
        ["screen", "display"],
        ["completely blank", "goes blank", "blank", "black screen", "completely black",
         "totally black", "went black", "stays dark", "won't turn on", "wont turn on",
         "no image", "doesn't display anything", "doesnt display anything", "dark screen",
         "black", "white and no text", "is dead", "screen dead", "dead screen",
         # added after the rebranded paraphrase eval (scripts/eval_paraphrase_hits.py)
         # missed "stuck on a blue screen" and "screen just dark, nothing visible"
         "blue screen", "just dark", "is dark", "went dark", "screen dark",
         "nothing visible", "nothing is visible", "won't boot", "wont boot",
         # (web-researched phrasing, see docs/DEV_NOTES.md "Vocabulary sources")
         "black screen of death", "stays black", "won't come on", "wont come on", "no display",
         "still vibrates", "still rings", "can hear notifications", "won't wake", "wont wake",
         "screen is off and"],
        "screen",
        "displays no image at all and will not turn on",
        "is just totally black, nothing shows up",
        "black no display",
    ),
    Symptom(
        "distorted_display",
        "Display output visually distorted / discolored",
        ["screen", "display"],
        ["distorted", "warped", "glitch", "green tint", "color tint", "discolor", "weird tint",
         "weird color", "tint across",
         # (web-researched phrasing, see docs/DEV_NOTES.md "Vocabulary sources")
         "green line", "pink line", "purple line", "white line", "lines on", "lines across",
         "vertical line", "horizontal line", "stripes", "color banding", "colors are off",
         "colours are off", "green screen", "pink screen", "burn-in", "burn in", "ghost image",
         "dead pixel", "pixelated"],
        "screen",
        "renders visual output that appears distorted or shows an abnormal color tint",
        "looks all warped and messed up, weird colors and everything",
        "distorted visual glitch tint",
    ),
    Symptom(
        "screen_undersized",
        "Display doesn't fill full screen size",
        ["screen", "display"],
        ["doesn't fill", "stays small", "expand to full size", "not full size", "shrunk down",
         "does not scale", "doesn't scale"],
        "screen",
        "does not scale to fill the full display area",
        "looks all shrunk down, not using the whole screen",
        "small not full size scaling",
    ),
    Symptom(
        "floating_assistant_icon",
        "Unwanted floating assistant/shortcut icon on screen",
        ["screen", "icon", "overlay", "circle", "bubble"],
        ["floating", "hovers", "shortcuts"],
        "screen",
        "displays a persistent floating icon overlay that I would like removed",
        "has this annoying floating bubble thing I want gone",
        "remove floating assistant icon overlay",
    ),

    # -- battery / power --
    Symptom(
        "battery_drain",
        "Battery drains unusually fast",
        ["battery"],
        ["drain", "dies so fast", "dead by", "doesn't last", "doesnt last", "runs out",
         "dying fast", "life is bad", "fast", "drains"],
        "battery",
        "drains unusually quickly even under light everyday use",
        "is dying crazy fast, barely lasts half a day",
        "drains fast battery life",
    ),
    Symptom(
        "charging_fails_or_slow",
        "Device won't charge or charges very slowly",
        ["charg"],  # stem: matches charge/charging/charger/charges
        ["won't", "wont", "not charging", "slowly", "slow", "barely", "stopped", "doesn't", "doesnt", "no matter what cable"],
        "charging",
        "is extremely slow or does not charge at all when plugged in",
        "is barely charging, if at all, no matter how long it's plugged in",
        "wont charge charging slow",
    ),
    Symptom(
        "overheating",
        "Device overheats during normal use",
        [],
        ["overheat", "gets very hot", "getting hot", "too hot", "extremely hot", "gets hot",
         "gets extremely hot", "heats up", "excessively hot", "super hot"],
        "device",
        "becomes excessively hot during normal use",
        "gets super hot for no reason",
        "overheating hot device",
    ),
    Symptom(
        "random_restarts",
        "Device randomly restarts/reboots",
        [],
        ["randomly restart", "reboots on its own", "restarts by itself", "random reboot",
         "keeps restarting", "restarts on its own", "reboots itself"],
        "device",
        "randomly restarts on its own with no warning or error message",
        "just reboots itself out of nowhere, no warning",
        "random restart reboot itself",
    ),
    Symptom(
        "sluggish_performance",
        "Device runs slow/laggy overall",
        [],
        ["running slow", "gotten slow", "is slow", "everything is laggy", "takes forever",
         "very sluggish", "freezes a lot", "hangs a lot", "noticeably slower",
         "taking longer to respond"],
        "device",
        "runs noticeably slower than usual, with everything taking longer to respond",
        "is laggy and slow all of a sudden, everything takes forever",
        "running slow laggy performance",
    ),

    # -- connectivity --
    Symptom(
        "wifi_connectivity",
        "Wi-Fi keeps disconnecting or won't connect",
        ["wifi", "wi-fi", "wireless network"],
        ["disconnect", "won't connect", "wont connect", "can't connect", "cant connect",
         "drop", "keeps dropping"],
        "Wi-Fi connection",
        "repeatedly disconnects from Wi-Fi networks or fails to connect at all",
        "keeps dropping out on me",
        "wifi keeps disconnecting",
    ),
    Symptom(
        "bluetooth_pairing",
        "Bluetooth won't pair or keeps disconnecting",
        ["bluetooth"],
        ["won't pair", "wont pair", "keeps disconnecting", "not connecting", "won't connect",
         "wont connect", "pairing"],
        "Bluetooth connection",
        "fails to pair with, or keeps disconnecting from, other accessories",
        "won't connect to my earbuds no matter what",
        "bluetooth wont pair disconnects",
    ),
    Symptom(
        "no_signal_network",
        "No mobile signal / can't call or text",
        ["signal", "network", "service", "bars", "carrier"],
        ["no signal", "no service", "no network", "can't call", "cant call", "cannot receive",
         "zero bars", "no bars", "fail to go through"],
        "mobile network signal",
        "shows no signal, so calls and texts fail to go through",
        "has zero bars, can't call or text anyone",
        "no signal no network",
    ),
    Symptom(
        "mobile_data_connectivity",
        "Mobile data won't connect or keeps dropping",
        ["mobile data", "cellular data", "data connection"],
        ["won't connect", "wont connect", "not working", "keeps dropping", "no internet",
         "not connecting", "wont turn on", "won't turn on", "keeps cutting out",
         "fails to connect", "drops the connection"],
        "mobile data connection",
        "fails to connect to mobile data or repeatedly drops the connection",
        "keeps cutting out, no internet unless I'm on Wi-Fi",
        "mobile data not working no internet",
    ),
    Symptom(
        "gps_location_inaccurate",
        "GPS/location not working or reporting the wrong place",
        ["gps", "location"],
        ["not working", "inaccurate", "wrong location", "cant find", "can't find",
         "not accurate", "off by", "not updating", "wont find me", "won't find me"],
        "GPS/location",
        "reports an inaccurate location or fails to acquire a signal at all",
        "keeps showing me in the wrong spot or won't find me at all",
        "gps location not working inaccurate",
    ),
    Symptom(
        "vibration_not_working",
        "Device no longer vibrates for calls/notifications",
        ["vibrate", "vibration", "vibrating"],
        ["not working", "stopped", "wont vibrate", "won't vibrate", "doesnt vibrate",
         "doesn't vibrate", "no longer", "isn't vibrating", "isnt vibrating"],
        "vibration",
        "no longer vibrates for incoming calls or notifications",
        "just stopped buzzing for calls and texts",
        "vibration not working stopped",
    ),

    # -- audio --
    Symptom(
        "distorted_sound",
        "Audio is distorted or crackling",
        ["sound", "audio", "speaker", "earbuds", "buds", "call quality"],
        ["distort", "crackl", "static", "garbled", "choppy"],  # "crackl" stems crackly/crackling
        "audio output",
        "is distorted or crackling during calls and playback",
        "sounds all crackly and messed up",
        "distorted crackling audio",
    ),
    Symptom(
        "no_sound",
        "No sound from speaker",
        ["sound", "speaker", "audio", "volume"],
        ["no sound", "silent", "can't hear", "cant hear", "dead silent", "not producing sound"],
        "speaker",
        "produces no sound at all during calls or media playback",
        "is dead silent, no sound comes out",
        "no sound speaker silent",
    ),

    # -- camera --
    Symptom(
        "camera_crashes",
        "Camera app won't open or crashes",
        ["camera"],
        ["crash", "force clos", "force-clos", "won't open", "wont open", "freeze", "closes"],
        "camera app",
        "fails to open or force-closes immediately when launched",
        "just crashes the second I try to open it",
        "camera crashes wont open",
    ),
    Symptom(
        "camera_blurry",
        "Photos come out blurry",
        ["camera", "photo", "picture", "pic"],
        ["blurry", "out of focus", "fuzzy", "not clear", "grainy"],
        "camera",
        "produces photos that are consistently blurry or out of focus",
        "takes photos that come out all blurry",
        "camera blurry photos out of focus",
    ),

    # -- storage / apps --
    Symptom(
        "storage_full",
        "Storage full, can't install apps/updates",
        ["storage", "memory", "space"],
        ["full", "out of", "not enough", "no space", "almost full"],
        "internal storage",
        "is full, preventing new apps or updates from installing",
        "is totally full, can't download or update anything",
        "storage full cant install",
    ),
    Symptom(
        "app_crashing",
        "A specific app keeps crashing",
        ["app"],
        ["crash", "force clos", "force-clos", "keeps closing", "stops working", "closing"],
        "an app",
        "repeatedly crashes or force-closes during use",
        "just keeps crashing on me over and over",
        "app keeps crashing force close",
    ),

    # -- security / biometrics / notifications / updates --
    Symptom(
        "fingerprint_face_unlock_fail",
        "Fingerprint/face unlock not working",
        ["fingerprint", "face recognition", "face unlock", "biometric"],
        ["won't", "wont", "not working", "stopped working", "fail", "doesn't recognize",
         "doesnt recognize", "won't recognize", "wont recognize", "not recognizing", "recognize me"],
        "fingerprint/face unlock",
        "repeatedly fails to recognize and unlock the device",
        "scanner just won't recognize me anymore",
        "fingerprint face unlock not working",
    ),
    Symptom(
        "no_notifications",
        "Not receiving notifications",
        ["notification"],
        ["not getting", "not receiving", "not showing", "missing", "stopped getting",
         "no notifications", "aren't showing", "arent showing", "not being delivered"],
        "notifications",
        "are not being delivered for messages or app alerts",
        "just aren't showing up at all anymore",
        "notifications not showing up missing",
    ),
    Symptom(
        "software_update_fails",
        "Software update fails to install",
        ["update"],
        ["fail", "won't install", "wont install", "error", "keeps failing", "stuck",
         "doesn't install", "doesnt install"],
        "software update",
        "repeatedly fails to download or install",
        "keeps failing every single time I try",
        "software update fails wont install",
    ),

    # -- accessibility --
    Symptom(
        "screen_reader_accessibility_fail",
        "TalkBack / accessibility screen reader not working",
        ["talkback", "screen reader"],
        ["not working", "stopped", "not reading", "wont read", "won't read",
         "stopped talking", "not speaking", "stopped reading"],
        "TalkBack",
        "has stopped reading screen content aloud",
        "just stopped talking, not reading anything on screen anymore",
        "talkback not reading stopped",
    ),
]

_DEFAULT_SYMPTOM = Symptom(
    "unclassified_issue",
    "Unclassified device issue (needs manual review)",
    [],
    [],
    "device",
    "is experiencing an issue that could not be automatically classified from the description given",
    "is just not working right and I'm not sure why",
    "device issue unclassified",
)


def _symptom_confidence_from_hits(hits: int) -> float:
    """Map a keyword-match score to a confidence in [0.0, 1.0]. hits is always
    0, 1, or 2 now (extract_symptom caps each side to boolean presence): 0 is
    the unclassified fallback, 1 is a single weak signal (only possible for a
    whole-device category matched on a problem term alone, no component gate)
    -- real evidence but not strong evidence, so 0.5 not 1.0 -- and 2 means both
    the component and the problem were found, which is full confidence.
    """
    if hits <= 0:
        return 0.0
    if hits == 1:
        return 0.5
    return 1.0


def extract_symptom(raw: str) -> Tuple[Symptom, float]:
    """Score each taxonomy entry by (component word present) + (problem word
    present) rather than requiring one long exact phrase. A whole-device
    symptom (component_terms=[]) is scored on problem terms alone; a
    part-specific symptom (e.g. camera, battery, Wi-Fi) needs at least one hit
    from EACH list to be eligible at all — this is what stops a stray
    substring collision (e.g. "crackly" containing "crack") from misfiring a
    screen_cracked match on an audio complaint that never mentions a screen.

    Returns (symptom, confidence). confidence is 0.0 for the unclassified
    fallback — callers (pipeline.py, the API response, Stage 1/2) can use
    this to know Stage 0 has no real signal here, rather than silently
    treating "unclassified_issue" as just another category with no asterisk
    on it.
    """
    lowered = raw.lower()
    best: Optional[Symptom] = None
    best_hits = 0
    for symptom in SYMPTOM_TAXONOMY:
        # Capped to 0/1 per side, not summed across every matching keyword-list
        # entry: two entries that both match (e.g. "blank" and "goes blank") are
        # the same underlying evidence, not two independent signals, and a
        # category with a longer list of near-duplicate phrasings must not
        # outscore a more specific category on that account alone.
        component_hits = 1 if any(kw in lowered for kw in symptom.component_terms) else 0
        problem_hits = 1 if any(kw in lowered for kw in symptom.problem_terms) else 0
        if symptom.component_terms and component_hits == 0:
            continue  # part-specific symptom, but that part was never mentioned
        if problem_hits == 0:
            continue  # no evidence of the actual problem, regardless of component
        hits = component_hits + problem_hits
        if hits > best_hits:
            best, best_hits = symptom, hits
    if best is None:
        return _DEFAULT_SYMPTOM, 0.0
    return best, _symptom_confidence_from_hits(best_hits)


# ---------------------------------------------------------------------------
# Guarded LLM fallback classifier — ONLY for complaints the deterministic
# matcher above found nothing for. Never runs when a keyword match already
# succeeded, so it can never override real evidence; never returns a category
# outside the known taxonomy, so it can never hallucinate a new one; and its
# result is always scored below the is_low_confidence threshold (see
# LLM_FALLBACK_CONFIDENCE), so pipeline.py's existing cache gate keeps it out
# of the cache automatically — no changes needed there. It exists purely to
# give Stage 1/2 a real, specific symptom_category to work with instead of
# the generic "could not be automatically classified" text, for the genuinely
# unseen-taxonomy scenarios the keyword matcher structurally can't reach.
# ---------------------------------------------------------------------------

LLM_FALLBACK_CONFIDENCE = 0.4  # deliberately < 0.5: always is_low_confidence, never cached

_VALID_SYMPTOM_CATEGORIES = {s.category for s in SYMPTOM_TAXONOMY} | {_DEFAULT_SYMPTOM.category}


def _strip_code_fence(text: str) -> str:
    """LLMs (Gemini especially) commonly wrap a JSON response in a markdown
    code fence (```json ... ```) even when explicitly instructed to "respond
    with strict JSON and nothing else" — found by testing a real Gemini call,
    not by inspection: it correctly classified a keyboard complaint but
    returned the literal string '```json\\n{"category": "keyboard_typing_problem"}\\n```',
    which json.loads() rejects outright (it starts with a backtick, not `{`
    or `[`) and enrichment.py's bare `except Exception: pass` was silently
    swallowing, making a correct LLM answer look like a failed one. Stripping
    a leading/trailing fence (with an optional "json" language tag) before
    parsing is a no-op on a response that's already plain JSON, so this is
    always safe to apply, not just for Gemini.
    """
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


LLM_EXTRA_VARIATIONS = 4  # paraphrases requested from the LLM on the non-keyword path


def _validated_category(category) -> Optional[str]:
    """Whitelist check shared by both LLM classification paths: only an
    exact, known, non-default category code survives."""
    if (
        isinstance(category, str)
        and category in _VALID_SYMPTOM_CATEGORIES
        and category != _DEFAULT_SYMPTOM.category
    ):
        return category
    return None


def _llm_classify_and_paraphrase(raw: str, client: LLMClient, n: int = LLM_EXTRA_VARIATIONS
                                 ) -> Tuple[Optional[str], List[str]]:
    """ONE round-trip that does both jobs the unclassified path needs:
    pick a category from the whitelist AND paraphrase the complaint.

    Before this, an unclassified complaint paid for two sequential LLM
    calls — _llm_classify_symptom() in normalize_query(), then
    _llm_variations() in generate_variations() — each a full network
    round-trip on the cold path. The paraphrases don't depend on the
    category (they reword the customer's own complaint, not the canonical
    template text), so nothing is lost by asking for both at once, and the
    cold-path LLM latency/cost for llm_fallback/unclassified cases halves.

    Same guardrails as before, applied independently to each half: the
    category must pass the whitelist (else None -> caller keeps
    unclassified_issue), and the variations are returned raw here and
    validated in generate_variations() exactly as _llm_variations() output
    always was. A response that only has "category" (or only "variations")
    still yields the half it has. Never raises.
    """
    category_list = "\n".join(f"- {s.category}: {s.label}" for s in SYMPTOM_TAXONOMY)
    system_prompt = (
        "You do two things for a TechCorp device support complaint.\n"
        "1. Classify it into EXACTLY ONE of the category codes listed below, or "
        "\"unclassified_issue\" if none of them genuinely describe the complaint. "
        "Never invent a new category code that isn't in the list.\n"
        f"2. Write {n} short alternate phrasings of the same complaint for "
        "search/cache matching. Only reword it — never add new symptoms, "
        "devices, or facts that are not already present.\n"
        "Respond with strict JSON and nothing else: "
        '{"category": "<one exact code from the list>", "variations": ["...", "..."]}\n\n'
        + category_list
    )
    user_prompt = f"Complaint: {raw}"
    try:
        parsed = json.loads(_strip_code_fence(client.complete(system_prompt, user_prompt)))
    except Exception:
        return None, []
    if not isinstance(parsed, dict):
        return None, []
    category = _validated_category(parsed.get("category"))
    raw_variations = parsed.get("variations")
    variations = (
        [v.strip() for v in raw_variations if isinstance(v, str) and v.strip()]
        if isinstance(raw_variations, list) else []
    )
    return category, variations


def _llm_classify_symptom(raw: str, client: LLMClient) -> Optional[str]:
    """Ask the LLM to pick ONE of the known category codes, or admit it
    doesn't know. The whitelist check against _VALID_SYMPTOM_CATEGORIES is
    the real guardrail — even if the LLM ignores the instructions and
    returns free text, an unrecognized string, or garbage, this returns
    None and the caller keeps the safe unclassified_issue default. Never
    raises: any parse/network failure is treated the same as "no answer."
    """
    category_list = "\n".join(f"- {s.category}: {s.label}" for s in SYMPTOM_TAXONOMY)
    system_prompt = (
        "You classify a TechCorp device support complaint into EXACTLY ONE of the "
        "category codes listed below, or \"unclassified_issue\" if none of them "
        "genuinely describe the complaint. Never invent a new category code that "
        "isn't in the list. Never add a symptom, device, or detail that isn't "
        "already in the complaint. Respond with strict JSON and nothing else: "
        '{"category": "<one exact code from the list>"}\n\n' + category_list
    )
    user_prompt = f"Complaint: {raw}"
    try:
        raw_resp = client.complete(system_prompt, user_prompt)
        parsed = json.loads(_strip_code_fence(raw_resp))
        category = parsed.get("category") if isinstance(parsed, dict) else None
        return _validated_category(category)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Canonical query + variations
# ---------------------------------------------------------------------------

@dataclass
class EnrichmentResult:
    raw_query: str
    canonical_query: str
    device: str
    symptom_category: str
    symptom_label: str
    device_confidence: float = 0.0    # 1.0 if a specific model was recognized, 0.0 if the
                                       # generic "TechCorp device" fallback was used
    symptom_confidence: float = 0.0   # 0.0 for "unclassified_issue"; see _symptom_confidence_from_hits
    classification_source: str = "keyword_match"  # "keyword_match" | "llm_fallback" | "unclassified"
    query_variations: List[str] = field(default_factory=list)
    # Paraphrases already obtained by the combined classify+paraphrase call in
    # normalize_query(). None = no LLM call was made (generate_variations may
    # make its own); a list (possibly empty) = already paid for, don't call
    # again. Internal plumbing only — deliberately not part of to_dict().
    llm_variations_prefetched: Optional[List[str]] = field(default=None, repr=False)

    @property
    def overall_confidence(self) -> float:
        """The weaker of the two signals — a confidently-identified device
        with an unclassified symptom (or vice versa) is still a low-confidence
        enrichment overall; don't let one strong signal mask the other being
        absent.
        """
        return min(self.device_confidence, self.symptom_confidence)

    @property
    def is_low_confidence(self) -> bool:
        """True when Stage 0 genuinely doesn't know what it's looking at.
        Downstream stages / the API response should treat this as a signal
        to be conservative — e.g. lower a Goal's score, or prefer a null
        deeplink over a guessed one — not as license to answer anyway."""
        return self.overall_confidence < 0.5

    def to_dict(self) -> dict:
        return {
            "raw_query": self.raw_query,
            "canonical_query": self.canonical_query,
            "device": self.device,
            "symptom_category": self.symptom_category,
            "symptom_label": self.symptom_label,
            "device_confidence": self.device_confidence,
            "symptom_confidence": self.symptom_confidence,
            "overall_confidence": self.overall_confidence,
            "is_low_confidence": self.is_low_confidence,
            "classification_source": self.classification_source,
            "query_variations": self.query_variations,
        }


def _clean_raw(raw: str) -> str:
    text = raw.strip()
    # strip leading numbering like "1." or list markers, and wrapping quotes
    text = re.sub(r'^\s*\d+[\.\)]\s*', "", text)
    text = text.strip('"“”\' ')
    return text


def normalize_query(raw_complaint: str, llm_client: Optional[LLMClient] = None) -> EnrichmentResult:
    """Turn a raw customer complaint into a canonical technical query.

    Primary path is deterministic (regex + keyword taxonomy) — no LLM call,
    so it cannot invent a device or symptom that isn't actually present in
    the text. If (and only if) that deterministic match finds nothing at
    all, and a real LLM client is configured, a tightly-guarded fallback
    (_llm_classify_symptom) gets one attempt to pick a category from the
    exact known list — never to override a keyword match that already
    succeeded, never to introduce a category outside the taxonomy, and
    always scored at LLM_FALLBACK_CONFIDENCE (< 0.5), so it's structurally
    unable to enter the semantic cache regardless of how confident the
    device extraction is (see pipeline.py's confidence gate). With no LLM
    configured (the default), behavior is unchanged: 100% deterministic.
    """
    cleaned = _clean_raw(raw_complaint)
    device = extract_device(cleaned)
    symptom, symptom_confidence = extract_symptom(cleaned)
    classification_source = "keyword_match" if symptom.category != _DEFAULT_SYMPTOM.category else "unclassified"
    prefetched: Optional[List[str]] = None

    if symptom.category == _DEFAULT_SYMPTOM.category:
        client = llm_client or get_llm_client()
        if True:
            # One call for both classification and paraphrases (see
            # _llm_classify_and_paraphrase) — generate_variations() reuses
            # the paraphrases instead of making a second round-trip.
            llm_category, prefetched = _llm_classify_and_paraphrase(cleaned, client)
            if isinstance(client, MockLLMClient):
                # The mock classifies demo categories but has no real
                # paraphrases; don't let its empty list block a real client
                # passed to generate_variations() later.
                prefetched = None
            if llm_category is not None:
                matched = next((s for s in SYMPTOM_TAXONOMY if s.category == llm_category), None)
                if matched is not None:
                    symptom = matched
                    symptom_confidence = LLM_FALLBACK_CONFIDENCE
                    classification_source = "llm_fallback"

    device_confidence = 0.0 if device == UNKNOWN_DEVICE_LABEL else 1.0
    canonical = f"{device} {symptom.subject} {symptom.formal}."
    return EnrichmentResult(
        raw_query=raw_complaint,
        canonical_query=canonical,
        device=device,
        symptom_category=symptom.category,
        symptom_label=symptom.label,
        device_confidence=device_confidence,
        symptom_confidence=symptom_confidence,
        classification_source=classification_source,
        llm_variations_prefetched=prefetched,
    )


def _inject_typos(text: str, seed: int) -> str:
    """Deterministic, mild typo injection: adjacent-letter swaps and a
    dropped vowel here and there — enough to exercise fuzzy matching
    without mangling the query beyond recognition.
    """
    rng = random.Random(seed)
    chars = list(text)
    n_edits = max(1, len(chars) // 25)
    for _ in range(n_edits):
        if len(chars) < 4:
            break
        i = rng.randrange(1, len(chars) - 1)
        op = rng.choice(["swap", "drop", "double"])
        if op == "swap" and chars[i].isalpha() and chars[i + 1].isalpha():
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif op == "drop" and chars[i].lower() in "aeiou":
            chars.pop(i)
        elif op == "double" and chars[i].isalpha():
            chars.insert(i, chars[i])
    return "".join(chars)


def _template_variations(result: EnrichmentResult) -> List[str]:
    device, symptom = result.device, next(
        (s for s in SYMPTOM_TAXONOMY if s.category == result.symptom_category), _DEFAULT_SYMPTOM
    )
    variations: List[str] = [
        f"I am experiencing an issue where the {device} {symptom.subject} {symptom.formal}. "
        f"Could you please advise on the appropriate troubleshooting steps?",
        f"The {symptom.subject} on my {device} {symptom.formal}; I would appreciate guidance on how to resolve this.",
        f"hey so my {device} {symptom.subject} {symptom.casual}, kinda annoying, help?",
        f"my {device}'s {symptom.subject} {symptom.casual} idk whats going on",
        f"{device} {symptom.keyword}",
        f"{device} {symptom.keyword} fix",
        f"This is so frustrating!! My {device} {symptom.subject} {symptom.casual} and nothing I do works!",
        f"I'm really annoyed, my {device} {symptom.subject} {symptom.formal} and I've tried everything already!",
    ]
    # two typo-register variants, seeded off the canonical text for determinism
    variations.append(_inject_typos(f"{device} {symptom.subject} {symptom.casual}", seed=1))
    variations.append(_inject_typos(f"{device} {symptom.subject} {symptom.formal}", seed=2))
    return variations


def _llm_variations(result: EnrichmentResult, client: LLMClient, n: int) -> List[str]:
    system_prompt = (
        "You paraphrase a customer's device-support complaint into short alternate "
        "phrasings for search/cache matching. Only reword the given complaint — "
        "never add new symptoms, devices, or facts that are not already present. "
        "Return a JSON array of strings and nothing else."
    )
    user_prompt = (
        f"Canonical query: {result.canonical_query}\n"
        f"Original complaint: {result.raw_query}\n"
        f"Give {n} short alternate phrasings of this exact same complaint."
    )
    try:
        raw = client.complete(system_prompt, user_prompt)
        candidates = json.loads(_strip_code_fence(raw))
        if not isinstance(candidates, list):
            return []
        return [c.strip() for c in candidates if isinstance(c, str) and c.strip()]
    except Exception:
        return []


def generate_variations(
    result: EnrichmentResult,
    llm_client: Optional[LLMClient] = None,
) -> List[str]:
    """Produce 8-10 query variations spanning formal / casual / keyword /
    frustrated / typo registers.

    Always returns a valid, deduped, length-bounded list even if no LLM
    is configured — the template generator alone satisfies the 8-10
    requirement across all five registers. If a real LLM client is
    available AND Stage 0 isn't already confident (classification_source
    != "keyword_match"), it contributes extra paraphrases, which are
    validated the same way before being mixed in (never trusted to
    self-constrain).

    The keyword_match skip is a real, measured latency fix, not a
    hypothetical one: live-tested against Gemini, a confidently
    keyword-matched query (full deterministic template coverage already)
    still paid a real ~1s network round-trip here for "extra paraphrase
    variety" before the cache was ever even checked — directly working
    against the ≤300ms fast-path budget. llm_fallback / unclassified
    results skip this gate and still get the LLM call: they're already
    paying an LLM cost for classification (or have no keyword signal at
    all), so the extra diversity is worth it there.
    """
    base = _template_variations(result)

    client = llm_client or get_llm_client()
    if result.classification_source != "keyword_match":
        if result.llm_variations_prefetched is not None:
            extra = result.llm_variations_prefetched  # already fetched in normalize_query's single call
        else:
            extra = _llm_variations(result, client, n=LLM_EXTRA_VARIATIONS)
        # validate: non-empty, plausible length, must still mention the device
        # or symptom keyword so a hallucinated unrelated paraphrase is dropped
        _matched_symptom = next(
            (s for s in SYMPTOM_TAXONOMY if s.category == result.symptom_category), _DEFAULT_SYMPTOM
        )
        _relevance_terms = _matched_symptom.component_terms + _matched_symptom.problem_terms
        valid_extra = [
            e for e in extra
            if 3 <= len(e.split()) <= 40
            and (result.device.split()[0].lower() in e.lower()
                 or any(kw in e.lower() for kw in _relevance_terms))
        ]
        # base[0:6] = formal/casual/keyword pairs, base[6:8] = frustrated pair,
        # base[8:10] = the two typo variants. Swap out only the frustrated slots
        # for LLM diversity — formal/casual/keyword/typo registers must survive
        # even when the LLM contributes nothing usable (extra == [], e.g. a bad
        # or filtered-out response), otherwise a real-LLM run silently loses the
        # typo register the docstring promises is always present.
        base = base[:6] + base[8:10] + valid_extra

    seen = set()
    deduped: List[str] = []
    for v in base:
        key = v.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(v.strip())

    # Guarantee the [8, 10] contract regardless of how many survived.
    while len(deduped) < MIN_VARIATIONS:
        deduped.append(_inject_typos(result.canonical_query, seed=len(deduped) + 10))
    return deduped[:MAX_VARIATIONS]


def enrich(raw_complaint: str, llm_client: Optional[LLMClient] = None) -> EnrichmentResult:
    """Full Stage 0 entry point: normalize (+ guarded LLM classification
    fallback) then generate variations. Resolves the LLM client once and
    threads the same instance through both steps, rather than letting each
    resolve it independently, so a real (non-mock) provider only gets
    initialized a single time per call.
    """
    client = llm_client or get_llm_client()
    result = normalize_query(raw_complaint, llm_client=client)
    result.query_variations = generate_variations(result, llm_client=client)
    return result
