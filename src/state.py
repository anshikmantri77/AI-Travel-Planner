"""LangGraph state schema for the AI Travel Planner workflow."""

from __future__ import annotations

from typing import Any, Literal

from typing_extensions import TypedDict


class TravelRequest(TypedDict, total=False):
    """User-submitted travel request fields."""

    destination: str
    start_date: str
    end_date: str
    budget_min: float
    budget_max: float
    interests: list[str]
    num_travelers: int


class PlannerState(TypedDict, total=False):
    """Top-level state flowing through the LangGraph workflow.

    Every node reads from / writes to this shared dict.
    Fields use ``total=False`` so nodes only need to set the keys they own.
    """

    # identifiers
    session_id: str

    # user input
    travel_request: TravelRequest

    # agent outputs
    research_output: dict[str, Any]
    draft_itinerary: dict[str, Any]

    # HITL control
    hitl_status: Literal["pending", "approved", "rejected", "modified"]
    hitl_feedback: str
    hitl_modifications: dict[str, Any]

    # final artefact
    final_plan: dict[str, Any]

    # workflow meta
    workflow_stage: str
    error: str | None
    revision_count: int
