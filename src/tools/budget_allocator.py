"""Budget allocation tool — pure Python, no external API.

Splits a total travel budget across days and spending categories
using heuristic percentage breakdowns based on travel style.
"""

from __future__ import annotations

import json
from typing import Any, Literal


# Percentage splits by travel style
_STYLE_SPLITS: dict[str, dict[str, int]] = {
    "budget": {
        "accommodation_pct": 30,
        "food_pct": 25,
        "activities_pct": 25,
        "transport_pct": 20,
    },
    "mid": {
        "accommodation_pct": 40,
        "food_pct": 25,
        "activities_pct": 20,
        "transport_pct": 15,
    },
    "luxury": {
        "accommodation_pct": 45,
        "food_pct": 25,
        "activities_pct": 20,
        "transport_pct": 10,
    },
}


def allocate_budget(
    total_budget: float,
    num_days: int,
    num_travelers: int = 1,
    travel_style: Literal["budget", "mid", "luxury"] = "mid",
) -> dict[str, Any]:
    """Split *total_budget* across days and categories.

    Args:
        total_budget: Total trip budget in the user's currency.
        num_days: Number of travel days.
        num_travelers: Number of people travelling.
        travel_style: One of ``budget``, ``mid``, or ``luxury``.

    Returns:
        Dict with percentage splits, per-day-per-person amount,
        and a breakdown list for each day.
    """
    if num_days <= 0:
        return {"error": "num_days must be positive"}
    if num_travelers <= 0:
        return {"error": "num_travelers must be positive"}
    if total_budget <= 0:
        return {"error": "total_budget must be positive"}

    style = travel_style if travel_style in _STYLE_SPLITS else "mid"
    splits = _STYLE_SPLITS[style]

    per_day = round(total_budget / num_days, 2)
    per_day_per_person = round(per_day / num_travelers, 2)

    daily_breakdown: list[dict[str, Any]] = []
    for day in range(1, num_days + 1):
        daily_breakdown.append(
            {
                "day": day,
                "total": per_day,
                "accommodation": round(per_day * splits["accommodation_pct"] / 100, 2),
                "food": round(per_day * splits["food_pct"] / 100, 2),
                "activities": round(per_day * splits["activities_pct"] / 100, 2),
                "transport": round(per_day * splits["transport_pct"] / 100, 2),
            }
        )

    return {
        "total_budget": total_budget,
        "num_days": num_days,
        "num_travelers": num_travelers,
        "travel_style": style,
        **splits,
        "per_day": per_day,
        "per_day_per_person": per_day_per_person,
        "daily_breakdown": daily_breakdown,
    }


# LangChain-compatible tool wrapper
def budget_tool(total_budget: float, num_days: int, num_travelers: int = 1, travel_style: str = "mid") -> str:
    """Allocate a travel budget across days and spending categories.

    Args:
        total_budget: Total trip budget.
        num_days: Number of travel days.
        num_travelers: Number of travelers.
        travel_style: One of 'budget', 'mid', or 'luxury'.

    Returns:
        JSON string with per-day budget breakdown by category.
    """
    result = allocate_budget(total_budget, num_days, num_travelers, travel_style)  # type: ignore[arg-type]
    return json.dumps(result, indent=2)
