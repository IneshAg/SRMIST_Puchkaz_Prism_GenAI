"""
Tests for src/scrubber.py (Zero-URL-Leak Guard).

Verifies:
- Nested structures are recursively traversed and scrubbed.
- A URL in each user-visible text field (goal, title, description, message, steps)
  is removed.
- Legitimate deeplink schemes (e.g. bixby://) survive in deeplink fields.
- Deeplink schemes leaking into text fields are removed.
- URL-only title: drops context, returns contexts [], passes schema validation.
- Description dropping below 5 words: drops action/context, passes schema validation.
- Goal becoming invalid: drops context, returns contexts [], passes schema validation.
- Auto action with web-URL deeplink: stays auto, gets the catalog placeholder
  (voiceassist://dummy_positive), passes schema validation.
- A step group whose only step was a URL is dropped (no invented placeholder
  steps); a manual action never carries an actionable deeplink.
- Idempotency: scrub_response(scrub_response(x)) == scrub_response(x).
- Immutability: input object is never mutated.
"""
import copy

from schema import ContextDeeplinkResponse
from scrubber import scrub_response


def test_nested_structures():
    """Verify recursive scrubbing across nested dictionaries and lists."""
    nested = {
        "contexts": [
            {
                "goal": "Follow these steps to perform this https://samsung.com/troubleshoot Display Troubleshooting",
                "title": "Display Troubleshooting",
                "score": 0.95,
                "actions": [
                    {
                        "actionName": "Display Settings",
                        "description": "It will visit https://samsung.com/support display settings today.",
                        "stepGroups": [
                            {
                                "steps": [
                                    "Navigate to Settings.",
                                    "Check help at www.samsung.com/help for details.",
                                    "Contact support@samsung.com if needed.",
                                    "Open samsung.com in browser.",
                                ],
                                "actionableDeeplink": {
                                    "deeplink": "bixby://dummy_positive",
                                    "description": "Valid deeplink",
                                    "message": "Visit http://badlink.com for info.",
                                },
                                "validationDeeplink": {
                                    "deeplink": "bixby://masked/val/266037d0c5",
                                    "key": "Valid Key",
                                },
                            }
                        ],
                        "category": "manual",
                    }
                ],
            }
        ]
    }

    scrubbed = scrub_response(nested)

    context = scrubbed["contexts"][0]
    assert "https://" not in context["goal"]
    assert "samsung.com" not in context["goal"]

    action = context["actions"][0]
    assert "https://" not in action["description"]
    assert "samsung.com" not in action["description"]

    group = action["stepGroups"][0]
    # Steps 2-4 were nothing but "go to this link / email us" instructions:
    # dropped whole instead of leaving fragments like "Check help at for details."
    assert group["steps"] == ["Navigate to Settings."]
    joined = " ".join(group["steps"])
    assert "samsung.com" not in joined and "@" not in joined
    # manual action: no actionable deeplink (brief: manual cannot carry one)
    assert group["actionableDeeplink"] is None
    # Legitimate validation deeplink survives
    assert group["validationDeeplink"]["deeplink"] == "bixby://masked/val/266037d0c5"


def test_url_in_each_text_field():
    """Verify that a URL in each user-visible text field is removed."""
    # 1. URL in goal
    res_goal = scrub_response({"goal": "Follow these steps to perform this https://samsung.com Display Troubleshooting"})
    assert "https://" not in res_goal["goal"]
    assert res_goal["goal"] == "Follow these steps to perform this Display Troubleshooting"

    # 2. URL in title
    res_title = scrub_response({"title": "Display http://support.samsung.com Troubleshooting"})
    assert "http://" not in res_title["title"]
    assert res_title["title"] == "Display Troubleshooting"

    # 3. URL in description
    res_desc = scrub_response({"description": "It will check https://samsung.com/page display settings now."})
    assert "https://" not in res_desc["description"]
    assert res_desc["description"] == "It will check display settings now."

    # 4. URL in message
    res_msg = scrub_response({"message": "Please see www.samsung.com for troubleshooting."})
    assert "www." not in res_msg["message"]
    assert res_msg["message"] == ""  # was only a "see this site" pointer: nothing left

    # 5. URL in steps
    res_steps = scrub_response({
        "steps": [
            "Open Settings.",
            "Go to https://samsung.com/setup for guide.",
            "Browse samsung.com/help today.",
            "Email kidshome.pin@samsung.com for code.",
        ]
    })
    for step in res_steps["steps"]:
        assert "https://" not in step
        assert "samsung.com" not in step
        assert "@" not in step


def test_legitimate_deeplink_must_survive():
    """Verify legitimate deeplink schemes survive while invalid web URLs in deeplink fields are removed."""
    response = {
        "actionableDeeplink": {
            "deeplink": "bixby://dummy_positive",
            "description": "Legit deeplink",
        },
        "validationDeeplink": {
            "deeplink": "bixby://masked/val/266037d0c5",
            "key": "test_key",
        },
        "webDeeplink": {
            "deeplink": "https://external.samsung.com/intent",
        },
        "bareDomainDeeplink": {
            "deeplink": "samsung.com/open",
        },
    }

    scrubbed = scrub_response(response)
    # Legitimate deeplinks must survive
    assert scrubbed["actionableDeeplink"]["deeplink"] == "bixby://dummy_positive"
    assert scrubbed["validationDeeplink"]["deeplink"] == "bixby://masked/val/266037d0c5"
    # Web URLs in deeplink fields must be rejected/removed
    assert scrubbed["webDeeplink"]["deeplink"] == ""
    assert scrubbed["bareDomainDeeplink"]["deeplink"] == ""


def test_deeplink_scheme_in_text_field_is_removed():
    """Verify deeplink schemes (e.g. bixby://) are allowed ONLY in deeplink fields,

    and are removed if they appear in text fields.
    """
    response = {
        "description": "It will execute bixby://dummy_positive actions automatically today.",
        "steps": ["Run bixby://settings/display to open menu."],
        "actionableDeeplink": {
            "deeplink": "bixby://dummy_positive",
        },
    }

    scrubbed = scrub_response(response)
    # Deeplink scheme removed from user-visible description and steps
    assert "bixby://" not in scrubbed["description"]
    assert "bixby://" not in scrubbed["steps"][0]
    # Allowed in deeplink field
    assert scrubbed["actionableDeeplink"]["deeplink"] == "bixby://dummy_positive"


def test_url_only_title():
    """URL-only title scrubbed to empty; context pruned; passes schema validation."""
    response = {
        "contexts": [
            {
                "goal": "Follow these steps to perform this Display Troubleshooting",
                "title": "https://samsung.com/troubleshoot",
                "score": 0.95,
                "actions": [
                    {
                        "actionName": "Display Settings",
                        "description": "It will configure display settings properly.",
                        "stepGroups": [
                            {
                                "steps": ["Open Settings."],
                                "actionableDeeplink": {
                                    "deeplink": "bixby://dummy_positive",
                                    "description": "Automated troubleshooting execution",
                                },
                            }
                        ],
                        "category": "manual",
                    }
                ],
            }
        ]
    }
    out = scrub_response(response)
    # Title had only a URL, becomes empty string -> breaks 2-3 words rule -> pruned
    assert out["contexts"] == []
    # Assert output still passes schema validation
    validated = ContextDeeplinkResponse.model_validate(out)
    assert validated.contexts == []


def test_description_drops_below_5_words():
    """Description dropping below 5 words after URL removal is pruned; passes schema validation."""
    response = {
        "contexts": [
            {
                "goal": "Follow these steps to perform this Display Troubleshooting",
                "title": "Display Troubleshooting",
                "score": 0.95,
                "actions": [
                    {
                        "actionName": "Display Settings",
                        # "It will open https://samsung.com/page now." -> "It will open now." (4 words, < 5)
                        "description": "It will open https://samsung.com/page now.",
                        "stepGroups": [
                            {
                                "steps": ["Open Settings."],
                                "actionableDeeplink": {
                                    "deeplink": "bixby://dummy_positive",
                                    "description": "Automated troubleshooting execution",
                                },
                            }
                        ],
                        "category": "manual",
                    }
                ],
            }
        ]
    }
    out = scrub_response(response)
    # Action description dropped below 5 words -> pruned -> nothing valid remains -> contexts is []
    assert out["contexts"] == []
    validated = ContextDeeplinkResponse.model_validate(out)
    assert validated.contexts == []


def test_goal_becomes_invalid():
    """Goal breaking pattern after URL removal is pruned; passes schema validation."""
    response = {
        "contexts": [
            {
                # Stripping URL leaves "Follow these steps on for Troubleshooting", which fails goal regex
                "goal": "Follow these steps on https://samsung.com/page for Troubleshooting",
                "title": "Display Troubleshooting",
                "score": 0.95,
                "actions": [
                    {
                        "actionName": "Display Settings",
                        "description": "It will configure display settings properly.",
                        "stepGroups": [
                            {
                                "steps": ["Open Settings."],
                                "actionableDeeplink": {
                                    "deeplink": "bixby://dummy_positive",
                                    "description": "Automated troubleshooting execution",
                                },
                            }
                        ],
                        "category": "manual",
                    }
                ],
            }
        ]
    }
    out = scrub_response(response)
    assert out["contexts"] == []
    validated = ContextDeeplinkResponse.model_validate(out)
    assert validated.contexts == []


def test_auto_action_with_web_url_deeplink():
    """Auto action with a web-URL deeplink keeps category auto and gets the catalog placeholder; passes schema validation."""
    response = {
        "contexts": [
            {
                "goal": "Follow these steps to perform this Display Troubleshooting",
                "title": "Display Troubleshooting",
                "score": 0.95,
                "actions": [
                    {
                        "actionName": "Display Settings",
                        "description": "It will configure display settings properly.",
                        "stepGroups": [
                            {
                                "steps": ["Open Settings."],
                                "actionableDeeplink": {
                                    "deeplink": "https://samsung.com/display/settings",
                                    "description": "Automated troubleshooting execution",
                                },
                            }
                        ],
                        "category": "auto",
                        "actionCategory": "auto",
                    }
                ],
            }
        ]
    }
    out = scrub_response(response)
    # Output must still pass schema validation
    validated = ContextDeeplinkResponse.model_validate(out)
    assert len(validated.contexts) == 1
    ctx = validated.contexts[0]
    act = ctx.actions[0]
    # Stays auto (it is a settings screen), placeholder from the catalog, not the web URL
    assert act.category == "auto"
    assert act.stepGroups[0].actionableDeeplink.deeplink == "voiceassist://dummy_positive"


def test_url_only_step_group_is_dropped_not_replaced_with_invented_steps():
    """A step group whose only step is a URL has nothing real left: it is dropped
    (never replaced with a made-up "Check device settings." step)."""
    response = {
        "contexts": [
            {
                "goal": "Follow these steps to perform this Display Troubleshooting",
                "title": "Display Troubleshooting",
                "score": 0.95,
                "actions": [
                    {
                        "actionName": "Display Settings",
                        "description": "It will configure display settings properly.",
                        "stepGroups": [
                            {
                                "steps": ["https://samsung.com/step"],
                                "actionableDeeplink": {
                                    "deeplink": "bixby://dummy_positive",
                                    "description": "Automated troubleshooting execution",
                                },
                            }
                        ],
                        "category": "manual",
                    }
                ],
            }
        ]
    }
    out = scrub_response(response)
    validated = ContextDeeplinkResponse.model_validate(out)
    assert validated.contexts == [] or not validated.contexts[0].actions
    assert "Check device settings." not in str(out)


def test_idempotency():
    """Verify that scrub_response is completely idempotent."""
    payloads = [
        {},
        {"query": "my screen is flickering"},
        {
            "contexts": [
                {
                    "goal": "Follow these steps to perform this Display Troubleshooting",
                    "title": "Screen issue",
                    "actions": [
                        {
                            "actionName": "Display Settings",
                            "description": "It will fix error properly now.",
                            "stepGroups": [
                                {
                                    "steps": ["Step with user@example.com email."],
                                    "actionableDeeplink": {
                                        "deeplink": "bixby://dummy_positive",
                                        "description": "Automated troubleshooting execution",
                                    },
                                }
                            ],
                        }
                    ],
                }
            ]
        },
    ]

    for p in payloads:
        once = scrub_response(p)
        twice = scrub_response(once)
        assert once == twice, f"Idempotency failed for: {p}"


def test_never_mutates_input():
    """Verify that scrub_response never modifies the input dictionary or nested objects."""
    original = {
        "goal": "Follow these steps to perform this https://samsung.com Display Troubleshooting",
        "nested": {
            "list": ["Visit www.samsung.com", "Normal text"],
            "deeplink": "bixby://dummy_positive",
            "dict": {"inner": "Contact me at info@test.org."},
        },
    }

    snapshot = copy.deepcopy(original)
    scrubbed = scrub_response(original)

    # Input dictionary remains identical to snapshot
    assert original == snapshot
    # Scrubbed output is modified
    assert scrubbed != original
    assert "https://" not in scrubbed["goal"]
    assert "www.samsung.com" not in scrubbed["nested"]["list"][0]
    assert "info@test.org" not in scrubbed["nested"]["dict"]["inner"]

