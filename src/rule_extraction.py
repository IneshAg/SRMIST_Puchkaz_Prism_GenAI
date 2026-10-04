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
    "force", "reconnect", "replace", "repeat", "restore", "free", "move", "back up",
}
# verbs that read naturally as "It will <verb> ..." in a description
_DESC_VERBS = {
    "check", "force", "charge", "attempt", "restart", "reset", "clean", "remove", "update",
    "schedule", "visit", "enable", "disable", "adjust", "turn", "inspect", "try", "clear",
    "connect", "replace", "contact", "install", "uninstall", "boot", "use",
    "verify", "review", "open", "perform", "customize", "create", "exit", "access", "test",
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


_LEAD_CLAUSE = re.compile(
    r"^(?:(?:to|if|when|once|after|before|while)\b[^,]{0,80},|for\b[^:]{0,60}:)\s*(?:please\s+)?",
    re.IGNORECASE,
)


def _steps_from(body: str) -> List[str]:
    steps: List[str] = []
    for raw in re.split(r"(?<=[.!?])\s+|\n+", body):
        s = _clean_sentence(raw)
        # "To restart, swipe down ..." / "If damaged, please visit ..." -- drop
        # the lead-in clause so the instruction itself is the step
        if s and not _is_instruction(s):
            s = _clean_sentence(_LEAD_CLAUSE.sub("", s))
        if s and not _is_instruction(s):
            # "For fast repairs you can trust, visit a local service center."
            m = re.search(r",\s*(?:please\s+)?([A-Za-z]+)\b", s)
            while m and m.group(1).lower() not in _IMPERATIVE:
                m = re.compile(r",\s*(?:please\s+)?([A-Za-z]+)\b").search(s, m.end())
            if m:
                s = _clean_sentence(s[m.start(1):])
        if not s or not _is_instruction(s):
            continue
        if s.rstrip(".").endswith(":") or re.search(r"\bfollowing steps\b", s, re.IGNORECASE):
            continue  # "Try the following steps:" introduces steps, it isn't one
        for step in _split_chain(s):
            if step not in steps:
                steps.append(step)
    return steps[:MAX_STEPS]


def _action_name(heading: str) -> str:
    name = _HEADING_PREFIX.sub("", heading).strip(" :.-")
    return name[:60]


_SMALL = {"the", "a", "an", "your", "and", "or", "of", "to", "for", "on", "in", "with"}


def _description(name: str) -> str:
    """5-7 words starting "It will": "Force a Restart" -> "It will force a
    restart."; noun headings -> "It will help with safe mode."."""
    words = re.findall(r"[A-Za-z0-9'/-]+", name)
    lw = [w.lower() for w in words]
    if lw and lw[0] in _DESC_VERBS:
        body = lw[:5]
    else:
        body = ["help", "with"] + [w for w in lw if w not in _SMALL][:3]
    while body and body[-1] in _SMALL:
        body.pop()
    desc = ["It", "will"] + body
    while len(desc) < 5:
        desc.append("properly" if len(desc) == 4 else "now")
    return " ".join(desc[:7]) + "."


def _topic(title: str) -> str:
    # "Screen flickers when using the Camera on a smartphone" -> "Screen flickers"
    t = re.sub(r"\s+(?:on|when|while|with|after|if|for)\s+.*$", "", title.strip(), flags=re.IGNORECASE)
    t = re.sub(r"[^A-Za-z0-9 ]", " ", t)
    return " ".join(t.split()[:4]) or "Device"


def _title(category: Optional[str], topic: str) -> str:
    """2-3 word sentence-case title. Prefer the Stage 0 symptom category
    ("inner_screen_failure" -> "Inner screen failure"), else the article topic."""
    words = [w for w in (category or "").split("_") if w and w not in {"then", "issue", "unclassified"}]
    if category and category != "unclassified_issue" and len(words) >= 2:
        return " ".join(words[:3]).capitalize()
    tw = [w for w in topic.split() if w.lower() not in _SMALL][:3]
    return " ".join(tw) if len(tw) >= 2 else topic


def rule_based_extraction(siis_response: dict, category: Optional[str] = None) -> Optional[dict]:
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
            # on-screen navigation = a settings screen (Stage 2 re-labels
            # restart / reset / safe mode as critical and maps the deeplink)
            "category": "auto" if any(re.match(r"(?:tap|go to|navigate|open settings|select)\b", s, re.I)
                                      for s in steps) else "manual",
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
    topic_tc = " ".join(w if (i and w.lower() in _SMALL) else w[:1].upper() + w[1:].lower()
                        for i, w in enumerate(topic.split()))
    return {"contexts": [{
        "goal": f"Follow these steps to perform this {topic_tc} Troubleshooting",
        "title": _title(category, topic),
        "score": 0.6,  # rule-based: lower confidence than an LLM extraction
        "actions": actions,
    }]}
