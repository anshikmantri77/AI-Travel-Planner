"""Itinerary Planner Agent — constructs a structured day-by-day trip plan.

Uses two tools:
  1. allocate_budget – splits budget across days/categories
  2. score_activities – ranks activities against user interests

Produces a complete itinerary JSON consumed by the HITL / finalize steps.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.config import get_settings
from src.tools.budget_allocator import budget_tool
from src.tools.activity_scorer import activity_scorer_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangChain @tool wrappers
# ---------------------------------------------------------------------------

@tool
def allocate_budget(total_budget: float, num_days: int, num_travelers: int = 1, travel_style: str = "mid") -> str:
    """Allocate a travel budget across days and spending categories.

    Args:
        total_budget: Total trip budget.
        num_days: Number of travel days.
        num_travelers: Number of travelers.
        travel_style: One of 'budget', 'mid', or 'luxury'.
    """
    return budget_tool(total_budget, num_days, num_travelers, travel_style)


@tool
def score_activities(activities: list[str], interests: list[str]) -> str:
    """Rank candidate activities by how well they match the traveler's interests.

    Args:
        activities: List of activity names or descriptions.
        interests: List of user interest keywords.
    """
    return activity_scorer_tool(activities, interests)


PLANNER_TOOLS = [allocate_budget, score_activities]


def _get_llm():
    settings = get_settings()
    if settings.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=settings.LLM_MODEL, temperature=0.4, api_key=settings.GROQ_API_KEY)
    elif settings.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0.4, api_key=settings.OPENAI_API_KEY)
    else:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=settings.LLM_MODEL, temperature=0.4, api_key=settings.ANTHROPIC_API_KEY)
    return llm.bind_tools(PLANNER_TOOLS)


PLANNER_SYSTEM_PROMPT = """You are an expert travel itinerary planner. Given research data and the user's \
travel request, produce a detailed day-by-day itinerary.

You MUST use the allocate_budget tool once to get per-day budget guidance, and SHOULD use score_activities \
to rank potential activities against the user's interests.

After gathering tool outputs, produce a JSON object with EXACTLY these keys:
- trip_summary: string (2-3 sentence overview of the trip)
- duration_days: integer
- total_budget_used: number
- days: array of day objects, each with:
    - day: integer
    - date: string (YYYY-MM-DD)
    - theme: string (short theme for the day)
    - morning: object with activity, duration, cost
    - afternoon: object with activity, duration, cost
    - evening: object with activity, duration, cost
    - accommodation: object with name, type, cost_per_night
    - daily_budget: number
- packing_suggestions: list of strings
- important_notes: string

Return ONLY the JSON object, no markdown fences or extra text."""


async def run_planner_agent(
    travel_request: dict[str, Any],
    research_output: dict[str, Any],
    hitl_feedback: str | None = None,
    hitl_modifications: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the planner agent.

    Parameters
    ----------
    travel_request : dict
        Original user request.
    research_output : dict
        Output from the Research Agent.
    hitl_feedback : str, optional
        User feedback from a previous rejection/modification.
    hitl_modifications : dict, optional
        Specific modifications requested by the user.
    """
    destination = travel_request.get("destination", "unknown")
    budget_min = travel_request.get("budget_min", 500)
    budget_max = travel_request.get("budget_max", 2000)
    avg_budget = (budget_min + budget_max) / 2
    interests = travel_request.get("interests", [])
    start_date = travel_request.get("start_date", "")
    end_date = travel_request.get("end_date", "")
    num_travelers = travel_request.get("num_travelers", 1)

    # Calculate trip duration
    from datetime import datetime
    try:
        sd = datetime.strptime(start_date, "%Y-%m-%d")
        ed = datetime.strptime(end_date, "%Y-%m-%d")
        num_days = max((ed - sd).days, 1)
    except (ValueError, TypeError):
        num_days = 5

    attractions = research_output.get("top_attractions", [])
    tips = research_output.get("local_tips", [])
    weather = research_output.get("weather_summary", "")
    areas = research_output.get("best_areas_to_stay", [])
    cuisine = research_output.get("cuisine_highlights", [])

    user_prompt = (
        f"Plan a {num_days}-day trip to {destination} for {num_travelers} traveler(s).\n"
        f"Dates: {start_date} to {end_date}\n"
        f"Budget range: {budget_min}-{budget_max} (target ~{avg_budget})\n"
        f"Interests: {', '.join(interests)}\n\n"
        f"Research data:\n"
        f"- Top attractions: {json.dumps(attractions)}\n"
        f"- Local tips: {json.dumps(tips)}\n"
        f"- Weather: {weather}\n"
        f"- Best areas to stay: {json.dumps(areas)}\n"
        f"- Cuisine highlights: {json.dumps(cuisine)}\n"
    )

    if hitl_feedback:
        user_prompt += f"\nUser feedback from previous draft: {hitl_feedback}\n"
    if hitl_modifications:
        user_prompt += f"\nSpecific modifications requested: {json.dumps(hitl_modifications)}\n"

    user_prompt += "\nUse your tools, then produce the JSON itinerary."

    llm = _get_llm()
    tool_map = {t.name: t for t in PLANNER_TOOLS}

    messages: list[Any] = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    for _ in range(10):
        response = await llm.ainvoke(messages)
        messages.append(response)

        if response.tool_calls:
            from langchain_core.messages import ToolMessage
            for tc in response.tool_calls:
                tool_fn = tool_map.get(tc["name"])
                if tool_fn:
                    try:
                        result = tool_fn.invoke(tc["args"])
                    except Exception as exc:
                        result = json.dumps({"error": str(exc)})
                else:
                    result = json.dumps({"error": f"Unknown tool: {tc['name']}"})
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
            continue

        text = response.content if isinstance(response.content, str) else str(response.content)
        return _parse_itinerary_output(text)

    return _default_itinerary(destination, num_days)


def _parse_itinerary_output(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Could not parse planner output as JSON, wrapping raw text.")
        return {
            "trip_summary": cleaned[:500],
            "duration_days": 0,
            "total_budget_used": 0,
            "days": [],
            "packing_suggestions": [],
            "important_notes": "Output could not be fully parsed.",
        }


def _default_itinerary(destination: str, num_days: int) -> dict[str, Any]:
    return {
        "trip_summary": f"A {num_days}-day trip to {destination}.",
        "duration_days": num_days,
        "total_budget_used": 0,
        "days": [],
        "packing_suggestions": [],
        "important_notes": "Itinerary generation reached max iterations. Please try again.",
    }
