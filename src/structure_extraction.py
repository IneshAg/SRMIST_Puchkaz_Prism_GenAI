from __future__ import annotations

import json
import re
from typing import Any

from enrichment import EnrichmentResult
from llm_client import LLMClient, get_llm_client
from relevance_gate import assess_reference_relevance
from schema import (
    Action,
    ContextDeeplinkResponse,
    Goal,
    StepGroup,
)



SYSTEM_PROMPT = """
You are the Structure Extraction stage of a Samsung Smart Guided
Troubleshooting Engine.

Your job is to faithfully convert the supplied Samsung troubleshooting
reference text into a structured troubleshooting response.

The reference text has already passed a deterministic relevance
check upstream.

Do not perform another broad relevance search.

Assume the reference contains potentially relevant material and
focus on faithfully extracting the supported troubleshooting
procedures.

Extract the useful troubleshooting procedures that are explicitly present
in the reference.

IMPORTANT:
- Use ONLY information supported by the supplied reference text.
- NEVER invent troubleshooting actions.
- NEVER invent troubleshooting steps.
- NEVER invent Settings deeplinks.
- Stage 2 will handle deeplink mapping, so actionableDeeplink must be null.
- Extract the troubleshooting flow already present in the source.
- Group steps that belong to the same physical Settings screen or feature
  under one Action.
- Preserve the logical order of the troubleshooting procedure.
- Place all "critical" actions after "auto" and "manual" actions.
- If multiple critical actions exist, preserve their source order.
- Critical actions should appear after normal actions.
- Manual/service actions should remain manual.

Action ordering rules:
1. Preserve the troubleshooting sequence as much as possible.
2. However, all "critical" actions must appear after all "auto"
   and "manual" actions.
3. Do not move steps between actions merely to satisfy ordering.

- Extract ALL relevant troubleshooting procedures supported by the reference.
- Do not stop after finding the first applicable procedure.
- Include each distinct troubleshooting operation as a separate Action
  when it belongs to a different screen, feature, or operation.
- Preserve the source order.
- Do not add procedures that are not supported by the reference.

OUTPUT ONLY VALID JSON.
Do not use markdown.
Do not use ```json fences.

Required shape:

{
  "contexts": [
    {
      "goal": "...",
      "title": "...",
      "score": 0.0,
      "actions": [
        {
          "actionName": "...",
          "description": "...",
          "stepGroups": [
            {
              "steps": ["...", "..."],
              "actionableDeeplink": null,
              "validationDeeplink": null
            }
          ],
          "category": "auto"
        }
      ]
    }
  ]
}

Formatting rules:

- goal must follow the pattern:

  "Follow these steps to perform this <Topic> Troubleshooting"
  OR
  "Follow these steps to perform this <Topic> Configuration"

- <Topic> must be a natural, human-readable topic derived from
  the reference title or troubleshooting content.

- Do NOT use internal machine-readable labels such as:
  "touch_lag",
  "screen_flicker_then_blank",
  "battery_issue", etc.

- title must contain exactly 2 to 3 words.

- title should be a concise human-readable version of the
  troubleshooting topic from the reference title.

- score must be between 0 and 1.

- description must contain exactly 5 to 7 words.

- description must begin exactly with:
  "It will"

- Count words separated by spaces.

- The description must explain what the action accomplishes.

- Keep the description concise. Do not add unnecessary words.

Examples of VALID descriptions:
"It will check camera settings."
"It will restore normal screen function."
"It will verify the internet connection."
"It will identify third-party app issues."

Example of INVALID description:
"It will adjust camera settings to fix flickering."
Reason: 8 words.

Before returning the JSON, verify every description yourself.

For every description:
1. Confirm it starts with "It will".
2. Count the words.
3. Ensure the count is between 5 and 7.
4. Rewrite it if necessary before returning the JSON.

- steps must contain only information explicitly supported
  by the reference.

- Actions must correspond to a single physical Settings screen,
  device feature, or clearly defined troubleshooting operation.

- Group steps together only when they belong to the same screen
  or feature.

- If two operations lead to different Settings screens or
  different features, create separate Actions.

- For Settings-based actions:
  * actionName should clearly identify the screen or feature.
  * Preserve the navigation hierarchy from the reference.
  * Do not remove meaningful navigation steps.

- category must be exactly one of:
  "auto"
  "manual"
  "critical"

- actionableDeeplink must always be null.

- validationDeeplink must always be null.

- The supplied reference has already passed the deterministic
  relevance gate upstream.

- The deterministic relevance gate is authoritative.
  Do NOT reject the reference because the exact symptom wording
  is not repeated word-for-word in the reference.

- When the relevance gate passes, extract the troubleshooting
  procedures that address the same device feature, symptom area,
  or troubleshooting topic.

- Use ONLY procedures and steps explicitly supported by the
  supplied reference.

- Do NOT invent a procedure to make the reference fit the query.

- Do NOT perform another broad relevance search.

- Do NOT return an empty result merely because the source uses
  different wording for the same troubleshooting issue.

- Return an empty result only when the supplied reference contains
  no actual troubleshooting or configuration procedure that can
  be extracted.

- If the reference contains applicable procedures, extract them
  faithfully into the required Goal → Action → StepGroup structure.
"""
def _strip_code_fence(text: str) -> str:
    """
    Remove markdown code fences if the LLM accidentally returns them.
    """
    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):]

    elif text.startswith("```"):
        text = text[len("```"):]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()

def _extract_json(text: str) -> dict[str, Any]:
    """
    Safely extract a JSON object from the LLM response.
    """

    if not isinstance(text, str):
        text = str(text)

    cleaned = text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()

    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            f"No JSON object found in LLM response: "
            f"{repr(cleaned[:500])}"
        )

    candidate = cleaned[start:end + 1]

    try:
        result = json.loads(candidate)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON returned by LLM: {exc}"
        ) from exc

    if not isinstance(result, dict):
        raise ValueError(
            "LLM JSON response must be an object"
        )

    return result

def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _validate_title(title: str) -> bool:
    """
    Title must contain 2-3 words.
    """
    return 2 <= _word_count(title) <= 3

def _normalize_description(description: str) -> str:
    """
    Normalize an action description to the required format.

    Required:
    - starts with "It will"
    - 5 to 7 words
    """

    description = description.strip()

    # Remove trailing punctuation while processing.
    description = description.rstrip(".!?")

    words = description.split()

    # Ensure the required prefix.
    if not description.startswith("It will"):
        description = f"It will {description}"

    words = description.split()

    # Already valid.
    if 5 <= len(words) <= 7:
        return description + "."

    # Remove common unnecessary trailing phrases.
    removable_phrases = [
        ["to", "fix", "flickering"],
        ["to", "resolve", "the", "issue"],
        ["to", "fix", "the", "issue"],
        ["to", "solve", "the", "problem"],
        ["for", "better", "performance"],
    ]

    for phrase in removable_phrases:

        phrase_len = len(phrase)

        if len(words) > 7 and [
            word.lower()
            for word in words[-phrase_len:]
        ] == phrase:

            words = words[:-phrase_len]

            if 5 <= len(words) <= 7:
                return " ".join(words) + "."

    # Generic deterministic fallback.
    if len(words) > 7:
        words = words[:7]

    # Ensure at least 5 words.
    while len(words) < 5:
        words.append("effectively")

    return " ".join(words) + "."

def _validate_description(description: str) -> bool:
    """
    Description:
    - must start with "It will"
    - must contain 5-7 words
    """
    if not description.startswith("It will"):
        return False

    return 5 <= _word_count(description) <= 7

def _validate_goal(goal: str) -> bool:
    """
    Goal must use the required troubleshooting/configuration wording.
    """

    pattern = (
        r"^Follow these steps to perform this .+ "
        r"(Troubleshooting|Configuration)$"
    )

    return re.match(pattern, goal) is not None

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "your",
    "you",
    "is",
    "are",
    "this",
    "that",
    "it",
    "be",
    "as",
    "at",
    "by",
}


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    return {
        word
        for word in words
        if word not in _STOPWORDS and len(word) > 2
    }

def _step_is_supported(
    step: str,
    source_text: str,
) -> bool:
    """
    Check whether a generated step is grounded in the source.

    Allows:
    - exact source phrases
    - paraphrased steps with at least two meaningful overlaps
    - short navigation steps such as "Go to Settings"
      when the referenced screen/feature appears in the source
    """

    step_clean = re.sub(r"\s+", " ", step.lower()).strip()
    source_clean = re.sub(r"\s+", " ", source_text.lower()).strip()

    if not step_clean or not source_clean:
        return False

    # 1. Exact phrase match.
    if step_clean in source_clean:
        return True

    step_tokens = _tokens(step)
    source_tokens = _tokens(source_text)

    if not step_tokens:
        return False

    overlap = step_tokens.intersection(source_tokens)
    # 2. Normal paraphrased step.
    if len(overlap) >= 2:
        return True

    # 3. Short navigation step.
    navigation_words = {
        "settings",
        "display",
        "reset",
        "restart",
        "battery",
        "camera",
        "network",
        "bluetooth",
        "wifi",
        "keyboard",
        "sound",
        "accessibility",
        "connections",
    }

    if len(step_tokens) <= 2 and overlap.intersection(
        navigation_words
    ):
        return True

    return False

_KEEP_CASE = {"samsung", "galaxy", "bixby", "wi-fi", "bluetooth", "android", "one", "ui"}


def _sentence_case(text: str) -> str:
    """'Screen Display Damage' -> 'Screen display damage'. Acronyms (USB,
    SIM, 5G) and brand names keep their casing."""
    words = text.split()
    out = []
    for i, w in enumerate(words):
        if i == 0:
            out.append(w[:1].upper() + w[1:])
        elif w.isupper() and len(w) > 1 or any(ch.isdigit() for ch in w) or w.lower() in _KEEP_CASE:
            out.append(w)
        else:
            out.append(w.lower())
    return " ".join(out)


def _repair_title(title: str) -> str:
    words = re.findall(r"[\w'-]+", title)
    # "Display Troubleshooting" carries no information beyond the goal line
    while len(words) > 2 and words[-1].lower() in {"troubleshooting", "configuration", "issue", "issues"}:
        words.pop()
    words = words[:3]
    if len(words) < 2:
        words = (words or ["Device"]) + ["issue"]
    return _sentence_case(" ".join(words))


def _repair_goal(goal: str, title: str) -> str:
    """Programmatic fix instead of rejecting the whole plan: collapse
    'X Troubleshooting Troubleshooting', and rebuild an off-format goal from
    the title."""
    goal = re.sub(r"\b(Troubleshooting|Configuration)(\s+\1\b)+", r"\1", goal.strip().rstrip("."))
    if _validate_goal(goal):
        return goal
    topic = re.sub(r"\s+(troubleshooting|configuration)$", "", title.strip(), flags=re.IGNORECASE)
    topic = " ".join(w[:1].upper() + w[1:] for w in topic.split()) or "Device"
    return f"Follow these steps to perform this {topic} Troubleshooting"


_SMALL_WORDS = {"a", "an", "and", "or", "the", "of", "to", "in", "on", "for", "with", "via", "at", "by"}


def _title_case(name: str) -> str:
    words = name.split()
    return " ".join(
        w if (w.isupper() and len(w) > 1) else
        (w.lower() if (0 < i < len(words) - 1 and w.lower() in _SMALL_WORDS) else w[:1].upper() + w[1:])
        for i, w in enumerate(words)
    )


def _validate_structure(
    raw: dict[str, Any],
    source_text: str,
) -> ContextDeeplinkResponse:
    """
    Validate the complete Stage 1 response.
    """

    # Pydantic schema validation.
    response = ContextDeeplinkResponse.model_validate(raw)

    for context in response.contexts:

        # Repair first (brief: "Enforce programmatic validation, trimming,
        # and correction loops"), then validate what's left.
        context.title = _repair_title(context.title)
        context.goal = _repair_goal(context.goal, context.title)

        # ---------------------------
        # Goal
        # ---------------------------
        if not _validate_goal(context.goal):
            raise ValueError(
                f"Invalid goal format: {context.goal}"
            )

        # ---------------------------
        # Title
        # ---------------------------
        if not _validate_title(context.title):
            raise ValueError(
                f"Title must contain 2-3 words: "
                f"{context.title}"
            )

        # ---------------------------
        # Score
        # ---------------------------
        if not 0 <= context.score <= 1:
            raise ValueError(
                f"Score must be between 0 and 1: "
                f"{context.score}"
            )

        # ---------------------------
        # Actions
        # ---------------------------
        # Critical actions must appear after non-critical actions.
        categories = [
            action.category.value
            for action in context.actions
        ]

        critical_positions = [
            i
            for i, category in enumerate(categories)
            if category == "critical"
        ]

        if critical_positions:
            # Stable re-order instead of discarding a whole valid plan.
            context.actions.sort(
                key=lambda a: 1 if a.category.value == "critical" else 0
            )
        for action in context.actions:
            action.actionName = _title_case(action.actionName)
            action.description = _normalize_description(
                action.description
            )

            if not _validate_description(
                action.description
            ):
                raise ValueError(
                    f"Invalid description: "
                    f"{action.description}"
                )

            # ---------------------------
            # Step Groups
            # ---------------------------
            for group in action.stepGroups:

                if not group.steps:
                    raise ValueError(
                        f"Action '{action.actionName}' "
                        f"contains no steps"
                    )

                # Stage 1 MUST NOT create deeplinks.
                if group.actionableDeeplink is not None:
                    raise ValueError(
                        "Stage 1 must not create "
                        "actionable deeplinks"
                    )

                if group.validationDeeplink is not None:
                    raise ValueError(
                        "Stage 1 must not create "
                        "validation deeplinks"
                    )

                # ---------------------------
                # Grounding
                # ---------------------------
                # Drop ungrounded steps instead of discarding the entire
                # plan because of one paraphrased line.
                dropped = [s for s in group.steps if not _step_is_supported(s, source_text)]
                if dropped:
                    print(f"[Stage 1] dropped {len(dropped)} ungrounded step(s): {dropped}")
                group.steps = [s for s in group.steps if _step_is_supported(s, source_text)]

            action.stepGroups = [g for g in action.stepGroups if g.steps]

        context.actions = [a for a in context.actions if a.stepGroups]

    response.contexts = [c for c in response.contexts if c.actions]
    return response

def structure_extraction(
    siis_response: dict,
    enrichment: EnrichmentResult,
    llm_client: LLMClient | None = None,
) -> ContextDeeplinkResponse:
    """
    Stage 1:

        SIIS reference text
                ↓
        Goal → Action → StepGroup
    """

    # -----------------------------------------
    # 1. Make sure we actually have input
    # -----------------------------------------
    if not siis_response:
        return ContextDeeplinkResponse(contexts=[])

    title = str(
        siis_response.get("title", "")
    ).strip()

    content = str(
        siis_response.get("content", "")
    ).strip()

    if not content:
        return ContextDeeplinkResponse(contexts=[])

    # The complete source used for grounding.
    source_text = f"{title}\n{content}"
    

# -----------------------------------------
# Deterministic relevance gate
# -----------------------------------------
    is_relevant, relevance_score, matched_terms = (
        assess_reference_relevance(
            siis_response,
            enrichment,
        )
    )

    if not is_relevant:

        print(
            "[Stage 1] relevance gate rejected "
            f"reference. matched={sorted(matched_terms)}"
        )

        return ContextDeeplinkResponse(
            contexts=[]
      )

# -----------------------------------------
# Continue to LLM only when relevant
# -----------------------------------------
    client = llm_client or get_llm_client()

    # Offline (no key, hosted service unreachable): build the plan from the
    # article's own headings and instruction sentences instead of asking the
    # mock "LLM", which could only echo the first two lines of the article.
    from llm_client import MockLLMClient
    if isinstance(client, MockLLMClient):
        from rule_extraction import rule_based_extraction
        raw = rule_based_extraction(siis_response, category=enrichment.symptom_category)
        if raw is None:
            return ContextDeeplinkResponse(contexts=[])
        try:
            return _validate_structure(raw, source_text)
        except Exception as exc:
            print(f"[Stage 1] offline extraction failed validation: {type(exc).__name__}: {exc}")
            return ContextDeeplinkResponse(contexts=[])
    # -----------------------------------------
    # 3. Build the user prompt
    # -----------------------------------------
    user_prompt = f"""
Customer information:

Device:
{enrichment.device}

Symptom category:
{enrichment.symptom_category}

Symptom:
{enrichment.symptom_label}

Canonical query:
{enrichment.canonical_query}


Samsung troubleshooting reference:

Title:
{title}

Content:
{content}

The reference has already passed the relevance gate.
Do not reconsider whether the reference is relevant.

Extract ALL applicable troubleshooting or configuration
procedures supported by the supplied reference.

Do not assume the issue is touchscreen-related.
Use the customer's enriched symptom and the supplied
reference to determine which supported procedures apply.

Extract the troubleshooting structure.

Return JSON only.
"""

    # -----------------------------------------
    # 4. Call the LLM
    # -----------------------------------------
    try:
        raw_response = client.complete(
            SYSTEM_PROMPT,
            user_prompt,
        )
              
        parsed = _extract_json(raw_response)

        validated = _validate_structure(
            parsed,
            source_text,
        )

        return validated

    except Exception as exc:
        print(
            f"\n[Stage 1] extraction failed: "
            f"{type(exc).__name__}: {exc}"
        )
        # §4.2.3: on any failure, return an empty list — not a crash.
        # The pipeline will attach "fallback": "no_match" automatically.
        return ContextDeeplinkResponse(contexts=[])
