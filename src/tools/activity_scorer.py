"""Activity scoring tool — ranks activities by keyword overlap with user interests.

Pure Python, no external API.  Uses simple normalised keyword matching.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _tokenise(text: str) -> set[str]:
    """Lowercase and split text into a set of word tokens."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def score_activities(
    activities: list[str],
    interests: list[str],
) -> list[dict[str, Any]]:
    """Score and rank activities against user interests.

    Each activity gets a score from 0–100 based on how many interest
    keywords appear in the activity description.

    Args:
        activities: List of activity names / descriptions.
        interests: List of user interest keywords or phrases.

    Returns:
        List of dicts sorted by descending score, each containing
        ``activity``, ``score``, and ``matched_interests``.
    """
    if not interests:
        return [
            {"activity": a, "score": 0.0, "matched_interests": []}
            for a in activities
        ]

    interest_tokens = {i: _tokenise(i) for i in interests}
    scored: list[dict[str, Any]] = []

    for activity in activities:
        act_tokens = _tokenise(activity)
        matched: list[str] = []
        for interest, itokens in interest_tokens.items():
            if itokens & act_tokens:
                matched.append(interest)

        score = round((len(matched) / len(interests)) * 100, 1)
        scored.append(
            {
                "activity": activity,
                "score": score,
                "matched_interests": matched,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


# LangChain-compatible tool wrapper
def activity_scorer_tool(activities: list[str], interests: list[str]) -> str:
    """Rank activities by how well they match user interests.

    Args:
        activities: List of activity names or descriptions.
        interests: List of user interest keywords.

    Returns:
        JSON string of ranked activities with match scores (0-100).
    """
    results = score_activities(activities, interests)
    return json.dumps(results, indent=2)
