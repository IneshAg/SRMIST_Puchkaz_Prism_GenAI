"""
Response Scrubber (Zero-URL-Leak Guard)

Recursively sanitizes troubleshooting engine responses to enforce the
zero-URL-leak rule from the theme brief:
- Strips http://, https://, www., bare domains, and email-like strings
  from all user-visible text fields (goal, titles, descriptions, messages, steps).
- Allows deeplink schemes (e.g. bixby://, samsungapps://, intent://) ONLY in
  deeplink fields (e.g. actionableDeeplink.deeplink, validationDeeplink.deeplink).
- Strips any deeplink schemes that leak into user-visible text fields.
- Validates text fields against schema rules:
  * goal: matches regex r"^Follow these steps to perform this .+ (Troubleshooting|Configuration)$"
  * title: 2-3 words
  * description: starts with "It will" and has 5-7 words
  Works sentence by sentence: a sentence that was only a link instruction
  ("Visit samsung.com for details") is dropped whole instead of leaving
  fragments like "Visit for details"; markdown links keep only their label.
  A step that scrubs to nothing is dropped; a step group / action with no
  real steps left is dropped; if nothing remains, contexts [] so the pipeline
  adds fallback "no_match". Never invents steps.
- A web URL in an auto action's deeplink is replaced by the catalog's
  generic placeholder (voiceassist://dummy_positive); manual/critical
  actions never carry an actionable deeplink.
- Idempotent and non-mutating.
"""
from __future__ import annotations

import copy
import re
from typing import Any

# Known top-level domains for bare-domain detection
_TLDS = (
    r"(?:com|org|net|edu|gov|mil|int|io|co|in|ai|me|info|biz|tv|app|dev|"
    r"xyz|tech|online|store|site|us|uk|ca|de|jp|fr|au|ru|ch|it|nl|se|no|es|ly)"
)

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Matches any URI scheme (e.g. http://, https://, bixby://, intent://)
# Preserves trailing sentence punctuation (. , ! ? ; :)
_SCHEME_RE = re.compile(
    r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s<>\"]+?(?=[.,!?;:)\]]*(?:\s|$))",
    re.IGNORECASE,
)

# Matches www. URLs
_WWW_RE = re.compile(
    r"\bwww\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s<>\"]+?)?(?=[.,!?;:)\]]*(?:\s|$))",
    re.IGNORECASE,
)

# Matches bare domains such as samsung.com or support.samsung.com/path
_DOMAIN_RE = re.compile(
    r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+"
    + _TLDS
    + r"(?::\d+)?(?:/[^\s<>\"]+?)?(?=[.,!?;:)\]]*(?:\s|$))\b",
    re.IGNORECASE,
)

_MD_LINK_RE = re.compile(r"!?\[([^\]]*)\]\(([^)]*)\)")
_LINK_FILLER = {
    "visit", "see", "go", "to", "at", "or", "and", "the", "our", "website", "site", "link",
    "here", "for", "more", "info", "information", "details", "please", "check", "out",
    "open", "click", "this", "page", "on", "a", "an", "you", "can", "also", "guide", "url",
    "email", "contact", "us", "help", "support", "call", "write", "online", "web",
    "if", "needed", "need", "in", "browser", "via", "with", "from", "by", "your", "any",
    "questions", "is", "are", "available", "official", "visiting",
}

_DISALLOWED_DEEPLINK_PREFIXES = (
    "http://",
    "https://",
    "ftp://",
    "file://",
    "ws://",
    "wss://",
)

GOAL_PATTERN = r"^Follow these steps to perform this .+ (Troubleshooting|Configuration)$"


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _is_valid_goal(goal: str) -> bool:
    return bool(re.match(GOAL_PATTERN, (goal or "").strip()))


def _is_valid_title(title: str) -> bool:
    return 2 <= _word_count(title or "") <= 3


def _is_valid_description(desc: str) -> bool:
    desc = (desc or "").strip()
    return desc.startswith("It will") and (5 <= _word_count(desc) <= 7)


def _placeholder_deeplink() -> str:
    """The catalog's own generic placeholder (voiceassist://dummy_positive after
    the rebrand). Hard-coding the old bixby:// value produced a URI that is not
    in deeplinks.json."""
    try:
        import deeplink_mapping
        deeplink_mapping._load_catalog()
        return deeplink_mapping._placeholder
    except Exception:
        return "voiceassist://dummy_positive"


def _scrub_text(text: str) -> tuple[str, bool]:
    """Scrub URLs, emails, bare domains, and deeplink schemes from text fields.

    Returns (cleaned_text, was_modified).
    """
    if not isinstance(text, str):
        return text, False

    orig = text
    # Markdown links/images: keep the visible label, drop the target.
    text = _MD_LINK_RE.sub(lambda m: "\x00" + m.group(1) + "\x00", text)
    # Work sentence by sentence: "Visit https://x or www.y." minus its links is
    # "Visit or." -- noise, not a step. Drop a sentence whose remainder has no
    # real content left; otherwise keep the remainder.
    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        cleaned_s = _EMAIL_RE.sub("", sentence)
        cleaned_s = _SCHEME_RE.sub("", cleaned_s)
        cleaned_s = _WWW_RE.sub("", cleaned_s)
        cleaned_s = _DOMAIN_RE.sub("", cleaned_s)
        cleaned_s = cleaned_s.replace("\x00", "")
        if cleaned_s != sentence:
            cleaned_s = re.sub(r"\(\s*\)|\[\s*\]|<\s*>", "", cleaned_s)
            content = [w for w in re.findall(r"[A-Za-z0-9']+", cleaned_s.lower()) if w not in _LINK_FILLER]
            if len(content) < 2:
                continue
        kept.append(cleaned_s)
    text = " ".join(kept)
    text = re.sub(
        r"\s*\(([^()]*)\)",
        lambda m: "" if not [w for w in re.findall(r"[A-Za-z0-9']+", m.group(1).lower()) if w not in _LINK_FILLER] else m.group(0),
        text,
    )
    # Any leftover link syntax fragments
    text = re.sub(r"\]\(|\(\s*\)|\[\s*\]", "", text)

    # Clean up whitespace and punctuation spacing
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    cleaned = text.strip()
    was_modified = cleaned != orig.strip()
    return cleaned, was_modified


def _scrub_deeplink(deeplink: str) -> tuple[str, bool]:
    """Validate deeplink field: allow non-web deeplink schemes (e.g. bixby://),
    reject web URLs, emails, and bare domains.
    Returns (cleaned_deeplink, was_cleared).
    """
    if not isinstance(deeplink, str):
        return deeplink, False

    cleaned = deeplink.strip()
    lowered = cleaned.lower()

    # Reject web protocols, www, or emails
    if lowered.startswith(_DISALLOWED_DEEPLINK_PREFIXES) or lowered.startswith("www.") or _EMAIL_RE.search(cleaned):
        return "", True

    # Must possess a valid non-web URI scheme (e.g. bixby://, intent://)
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", cleaned):
        return "", bool(cleaned)

    return cleaned, False


def _scrub_raw_values(key: str | None, value: Any) -> Any:
    """Recursively scrub string values within dictionaries and lists."""
    if isinstance(value, dict):
        return {k: _scrub_raw_values(k, v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_scrub_raw_values(key, item) for item in value]
    elif isinstance(value, str):
        if key == "deeplink":
            cleaned, _ = _scrub_deeplink(value)
            return cleaned
        cleaned, _ = _scrub_text(value)
        return cleaned
    return value


def _sanitize_contexts(contexts_list: list) -> list:
    """Validate and apply fallback rules to scrubbed contexts list."""
    valid_contexts = []
    for ctx in contexts_list:
        if not isinstance(ctx, dict):
            continue

        # Rule 1: Goal validation
        if "goal" in ctx:
            goal_raw = ctx["goal"]
            if not isinstance(goal_raw, str):
                continue
            goal_clean, goal_mod = _scrub_text(goal_raw)
            ctx["goal"] = goal_clean
            if not goal_clean or (goal_mod and not _is_valid_goal(goal_clean)):
                continue

        # Rule 1: Title validation (2-3 words)
        if "title" in ctx:
            title_raw = ctx["title"]
            if not isinstance(title_raw, str):
                continue
            title_clean, title_mod = _scrub_text(title_raw)
            ctx["title"] = title_clean
            if not title_clean or (title_mod and not _is_valid_title(title_clean)):
                continue

        if "actions" in ctx and isinstance(ctx["actions"], list):
            had_actions = len(ctx["actions"]) > 0
            valid_actions = []
            for act in ctx["actions"]:
                if not isinstance(act, dict):
                    continue

                act_name = act.get("actionName")
                if not isinstance(act_name, str) or not act_name.strip():
                    continue

                # Rule 1: Description validation ("It will", 5-7 words)
                act_desc = act.get("description")
                if not isinstance(act_desc, str):
                    continue
                desc_clean, desc_mod = _scrub_text(act_desc)
                act["description"] = desc_clean
                if not desc_clean or (desc_mod and not _is_valid_description(desc_clean)):
                    continue

                step_groups = act.get("stepGroups", [])
                if not isinstance(step_groups, list) or not step_groups:
                    continue  # no steps -> drop the action; never invent placeholder steps

                valid_groups = []
                for grp in step_groups:
                    if not isinstance(grp, dict):
                        continue

                    is_affected = False

                    # Check steps
                    raw_steps = grp.get("steps")
                    if not isinstance(raw_steps, list) or not raw_steps:
                        is_affected = True
                    else:
                        clean_steps = []
                        for s in raw_steps:
                            if isinstance(s, str):
                                s_clean, s_mod = _scrub_text(s)
                                if s_clean:
                                    clean_steps.append(s_clean)
                                # a step that was only a link ("Visit x.com") is
                                # dropped; the rest of the group is kept
                            else:
                                is_affected = True
                        if not clean_steps:
                            is_affected = True
                        else:
                            grp["steps"] = clean_steps

                    # Check actionableDeeplink
                    act_dl = grp.get("actionableDeeplink")
                    dl_cleared = False
                    if isinstance(act_dl, dict):
                        raw_dl = act_dl.get("deeplink")
                        if isinstance(raw_dl, str):
                            dl_clean, was_cleared = _scrub_deeplink(raw_dl)
                            act_dl["deeplink"] = dl_clean
                            if was_cleared or not dl_clean:
                                dl_cleared = True
                        elif raw_dl is None or raw_dl == "":
                            dl_cleared = True

                        raw_dl_desc = act_dl.get("description")
                        if isinstance(raw_dl_desc, str):
                            dl_desc_clean, _ = _scrub_text(raw_dl_desc)
                            act_dl["description"] = dl_desc_clean
                            if not dl_desc_clean:
                                act_dl["description"] = "Open the relevant Settings screen"
                        if "message" in act_dl and isinstance(act_dl["message"], str):
                            act_dl["message"], _ = _scrub_text(act_dl["message"])

                    val_dl = grp.get("validationDeeplink")
                    if isinstance(val_dl, dict):
                        raw_vdl = val_dl.get("deeplink")
                        if isinstance(raw_vdl, str):
                            vdl_clean, was_vcleared = _scrub_deeplink(raw_vdl)
                            val_dl["deeplink"] = vdl_clean
                            if was_vcleared or not vdl_clean:
                                dl_cleared = True
                                grp["validationDeeplink"] = None  # no placeholder exists for validation links
                        if not val_dl.get("key") or not isinstance(val_dl.get("key"), str):
                            grp["validationDeeplink"] = None

                    cat = act.get("category") or act.get("actionCategory")
                    # A web URL in place of a catalog deeplink: an auto action keeps
                    # its category and gets the catalog's generic placeholder (the
                    # brief's "valid Settings screen not in the catalog").
                    # manual/critical actions may not carry a deeplink at all.
                    if cat == "auto":
                        if dl_cleared and isinstance(act_dl, dict):
                            act_dl["deeplink"] = _placeholder_deeplink()
                    elif act_dl is not None:
                        grp["actionableDeeplink"] = None

                    if is_affected:
                        continue  # nothing real left in this group

                    valid_groups.append(grp)

                if not valid_groups:
                    continue

                act["stepGroups"] = valid_groups
                valid_actions.append(act)

            if had_actions and not valid_actions:
                continue

            ctx["actions"] = valid_actions

        else:
            # Handle flat context dicts
            cat = ctx.get("category") or ctx.get("actionCategory")
            raw_dl = ctx.get("deeplink")
            dl_cleared = False
            if isinstance(raw_dl, str):
                dl_clean, was_cleared = _scrub_deeplink(raw_dl)
                ctx["deeplink"] = dl_clean
                if was_cleared or not dl_clean:
                    dl_cleared = True

            if dl_cleared and cat == "auto":
                ctx["deeplink"] = _placeholder_deeplink()

            if "description" in ctx:
                desc_clean, desc_mod = _scrub_text(ctx["description"])
                ctx["description"] = desc_clean
                if not desc_clean or (desc_mod and not _is_valid_description(desc_clean)):
                    continue

        valid_contexts.append(ctx)

    return valid_contexts


def scrub_response(response: dict) -> dict:
    """Recursively remove or reject http://, https://, www., bare domains and
    email-like strings from all user-visible text fields (goal, titles,
    descriptions, messages, steps). Allow deeplink schemes (e.g. bixby://) only
    in deeplink fields. Never mutates the input dictionary.

    If a text field is empty or breaks its rules after scrubbing (title 2-3 words,
    description starts with "It will" and has 5-7 words, goal matches the goal regex
    in src/schema.py or structure_extraction.py), do not emit it: empty steps,
    groups and actions are dropped, and if nothing valid remains contexts is []
    so the pipeline adds fallback "no_match". Never invent new content or URLs.

    A web URL in an auto action's deeplink becomes the catalog placeholder;
    manual/critical actions never carry an actionable deeplink.
    """
    if not isinstance(response, dict):
        return response

    res = copy.deepcopy(response)

    # Process and sanitize contexts (both top-level and nested response.contexts)
    if "contexts" in res and isinstance(res["contexts"], list):
        res["contexts"] = _sanitize_contexts(res["contexts"])
    elif "response" in res and isinstance(res["response"], dict) and "contexts" in res["response"]:
        res["response"]["contexts"] = _sanitize_contexts(res["response"]["contexts"])
    else:
        # Check flat response dict for deeplink clearing and auto category
        cat = res.get("category") or res.get("actionCategory")
        raw_dl = res.get("deeplink")
        dl_cleared = False
        if isinstance(raw_dl, str):
            dl_clean, was_cleared = _scrub_deeplink(raw_dl)
            res["deeplink"] = dl_clean
            if was_cleared or not dl_clean:
                dl_cleared = True
        if dl_cleared and cat == "auto":
            res["deeplink"] = _placeholder_deeplink()
            if "category" in res:
                res["category"] = "manual"
            if "actionCategory" in res:
                res["actionCategory"] = "manual"

    # Validate and clean top-level fields if present
    if "goal" in res:
        goal_str, goal_mod = _scrub_text(res["goal"])
        res["goal"] = goal_str
        if not goal_str or (goal_mod and not _is_valid_goal(goal_str)):
            del res["goal"]

    if "title" in res:
        title_str, title_mod = _scrub_text(res["title"])
        res["title"] = title_str
        if not title_str or (title_mod and not _is_valid_title(title_str)):
            del res["title"]

    if "description" in res:
        desc_str, desc_mod = _scrub_text(res["description"])
        res["description"] = desc_str
        if not desc_str or (desc_mod and not _is_valid_description(desc_str)):
            del res["description"]

    if "message" in res and isinstance(res["message"], str):
        msg_str, _ = _scrub_text(res["message"])
        res["message"] = msg_str

    if "steps" in res and isinstance(res["steps"], list):
        clean_steps = []
        for s in res["steps"]:
            if isinstance(s, str):
                s_clean, _ = _scrub_text(s)
                clean_steps.append(s_clean)
            else:
                clean_steps.append(s)
        res["steps"] = clean_steps

    # Clean residual nested structures (like actionables/validation at top-level)
    for k, v in list(res.items()):
        if k not in ("contexts", "response", "goal", "title", "description", "message", "steps"):
            res[k] = _scrub_raw_values(k, v)

    return res

