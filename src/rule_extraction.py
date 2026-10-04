"""
Offline Stage 1: deterministic extraction from the SIIS article.

Used when no LLM is reachable (no key, and the hosted service is asleep or
gone). Instead of the old mock -- one action made of the article's first two
lines -- this reads the article's own structure:

  * the markdown headings ("### Step 2: Force a Restart") become actions,
  * the imperative sentences under each heading become its steps
    ("Press and hold both the Power button ... for 20 seconds."),
  * comma chains like "go to Settings, tap Display, and then tap the switch
    next to Touch sensitivity" are split into one step per interaction,
  * informational sections ("Understanding Video Flickering") with no
    instruction in them are skipped.

Every step is copied from the source text, so it cannot invent a step; the
result then goes through the same validation/repair as the LLM output, and
Stage 2 assigns categories, deeplinks and critical-last ordering.
"""
from __future__ import annotations

import re
from typing import List, Optional

_IMPERATIVE = {
    "open", "go", "tap", "press", "hold", "select", "turn", "remove", "insert", "charge",
    "connect", "disconnect", "check", "inspect", "restart", "reset", "back", "wipe", "clean",
    "disable", "enable", "adjust", "increase", "decrease", "visit", "schedule", "swipe",
    "navigate", "use", "try", "update", "install", "uninstall", "contact", "plug", "unplug",
    "wait", "make", "ensure", "shine", "examine", "let", "leave", "keep", "set", "change",
    "clear", "delete", "drag", "launch", "close", "find", "touch", "reinsert", "allow",
    "avoid", "bring", "take", "switch", "toggle", "follow", "power", "eject", "attach",
}
_LEADING_FILLER = re.compile(
    r"^(?:first|next|now|then|finally|also|alternatively|additionally|afterwards?)[,:]?\s+|"
    r"^(?:please|kindly)\s+|^(?:let's|lets)\s+(?=try\b)|^(?:you can|you may|you should|you'll need to)\s+|"
    r"^(?:to do this|to fix this|to resolve this(?: issue)?)[,:]?\s+",
    re.IGNORECASE,
)
_INFO_HEADING = re.compile(
    r"\b(understanding|factors|overview|about|why|what (?:is|are|causes)|common ways|"
    r"introduction|summary|normal ldis?|cracked screen|ink blots|bleeding pixels)\b",
    re.IGNORECASE,
)
_HEADING_PREFIX = re.compile(r"^(?:step\s*\d+\s*[:.\-]\s*|\d+\s*[.)]\s*)", re.IGNORECASE)
MAX_ACTIONS = 6
MAX_STEPS = 6


def _sections(content: str) -> List[tuple]:
    """[(heading, body)] in article order. Text before the first heading
    (the "Smartphone,Tablet ... (...):" preamble) is dropped."""
    parts = re.split(r"(?m)^\s*#{1,6}\s+(.+?)\s*$|(?<=:)\s*#{1,6}\s+(.+?)\s*$", content)
    out, i = [], 1
    while i < len(parts):
        heading = parts[i] or parts[i + 1] or ""
        body = parts[i + 2] if i + 2 < len(parts) else ""
        out.append((heading.strip(), (body or "").strip()))
        i += 3
    return out


def _clean_sentence(s: str) -> str:
    s = s.strip().strip('"').strip()
    for _ in range(3):
        s = _LEADING_FILLER.sub("", s).strip()
    if not s:
        return s
    s = s[0].upper() + s[1:]
    return s if s.endswith((".", "!", "?")) else s + "."


def _is_instruction(s: str) -> bool:
    first = re.findall(r"[A-Za-z']+", s.lower())[:1]
    return bool(first) and first[0] in _IMPERATIVE and len(s.split()) <= 40


def _split_chain(sentence: str) -> List[str]:
    """"Go to Settings, tap Display, and then tap the switch next to X." ->
    three steps. Only splits where every piece is itself an instruction."""
    pieces = [p.strip(" ,.") for p in re.split(r",\s*(?:and\s+)?(?:then\s+)?|\s+and then\s+", sentence)]
    pieces = [p for p in pieces if p]
    if len(pieces) > 1 and all(_is_instruction(p) for p in pieces):
        return [_clean_sentence(p) for p in pieces]
    return [sentence]


def _steps_from(body: str) -> List[str]:
    steps: List[str] = []
    for raw in re.split(r"(?<=[.!?])\s+|\n+", body):
        s = _clean_sentence(raw)
        # "To do this, go to Settings, ..." -- the instruction starts mid-sentence
        m = re.search(r"\b(?:go to|open|tap|press and hold)\b.*", s, re.IGNORECASE)
        if not _is_instruction(s) and m and m.start() > 0 and ", " in s[: m.start() + 2]:
            s = _clean_sentence(m.group(0))
        if not s or not _is_instruction(s):
            continue
        for step in _split_chain(s):
            if step not in steps:
                steps.append(step)
    return steps[:MAX_STEPS]


def _action_name(heading: str) -> str:
    name = _HEADING_PREFIX.sub("", heading).strip(" :.-")
    return name[:60]


def _description(name: str) -> str:
    words = re.findall(r"[A-Za-z0-9'-]+", name.lower())
    words = [w for w in words if w not in {"the", "a", "an", "your", "device's"}][:4]
    desc = ["It", "will", "help"] + words
    while len(desc) < 5:
        desc.append("now")
    return " ".join(desc[:7]) + "."


def _topic(title: str) -> str:
    t = re.sub(r"\s+on\s+(?:a|your)\s+.*$", "", title.strip(), flags=re.IGNORECASE)
    t = re.sub(r"[^A-Za-z0-9 ]", " ", t)
    return " ".join(t.split()[:4]) or "Device"


def rule_based_extraction(siis_response: dict) -> Optional[dict]:
    """Raw dict in the Stage 1 LLM output shape, or None if the article has
    no extractable instructions."""
    title = str(siis_response.get("title", "")).strip()
    content = str(siis_response.get("content", "")).strip()
    actions = []
    for heading, body in _sections(content):
        if _INFO_HEADING.search(heading):
            continue
        steps = _steps_from(body)
        if not steps:
            continue
        name = _action_name(heading) or steps[0].rstrip(".")
        actions.append({
            "actionName": name,
            "description": _description(name),
            "stepGroups": [{"steps": steps}],
            "category": "auto" if any(re.search(r"\bsettings\b", s, re.I) for s in steps) else "manual",
        })
        if len(actions) >= MAX_ACTIONS:
            break
    if not actions:  # article without headings: one action from its instructions
        steps = _steps_from(content)
        if not steps:
            return None
        actions = [{"actionName": _topic(title), "description": _description(_topic(title)),
                    "stepGroups": [{"steps": steps}], "category": "manual"}]
    topic = _topic(title)
    return {"contexts": [{
        "goal": f"Follow these steps to perform this {topic.title()} Troubleshooting",
        "title": topic,
        "score": 0.6,  # rule-based: lower confidence than an LLM extraction
        "actions": actions,
    }]}
