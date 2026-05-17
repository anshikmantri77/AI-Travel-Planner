"""Tests for the LangGraph orchestrator — state transitions and HITL flow."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from langgraph.types import Command

from src.orchestrator import compile_graph

MOCK_RESEARCH = {
    "destination_overview": "Test destination.",
    "top_attractions": ["Attraction 1"],
    "local_tips": ["Tip 1"],
    "safety_notes": "Safe.",
    "weather_summary": "Sunny.",
    "best_areas_to_stay": ["Area 1"],
    "cuisine_highlights": ["Dish 1"],
}

MOCK_ITINERARY = {
    "trip_summary": "Test trip.",
    "duration_days": 3,
    "total_budget_used": 900,
    "days": [],
    "packing_suggestions": [],
    "important_notes": "",
}

START = (date.today() + timedelta(days=30)).isoformat()
END = (date.today() + timedelta(days=33)).isoformat()

VALID_STATE = {
    "session_id": "test-001",
    "travel_request": {
        "destination": "Tokyo",
        "start_date": START,
        "end_date": END,
        "budget_min": 500,
        "budget_max": 2000,
        "interests": ["food", "culture"],
        "num_travelers": 2,
    },
    "hitl_status": "pending",
    "hitl_feedback": "",
    "hitl_modifications": {},
    "revision_count": 0,
    "error": None,
    "workflow_stage": "started",
}


@pytest.mark.asyncio
@patch("src.agents.research_agent.run_research_agent", new_callable=AsyncMock, return_value=MOCK_RESEARCH)
@patch("src.agents.planner_agent.run_planner_agent", new_callable=AsyncMock, return_value=MOCK_ITINERARY)
async def test_happy_path(mock_planner, mock_research):
    """Full happy path: create → interrupt → approve → finalize."""
    graph = compile_graph()
    config = {"configurable": {"thread_id": "happy-path"}}

    # Run until HITL interrupt
    await graph.ainvoke(VALID_STATE, config=config)

    snapshot = await graph.aget_state(config)
    assert snapshot.next  # should be paused
    state = dict(snapshot.values)
    assert state["workflow_stage"] == "awaiting_review"
    assert state["draft_itinerary"] == MOCK_ITINERARY

    # Resume with approve
    await graph.ainvoke(
        Command(resume={"action": "approved", "feedback": "", "modifications": {}}),
        config=config,
    )

    snapshot = await graph.aget_state(config)
    state = dict(snapshot.values)
    assert state["workflow_stage"] == "completed"
    assert state["final_plan"] is not None


@pytest.mark.asyncio
@patch("src.agents.research_agent.run_research_agent", new_callable=AsyncMock, return_value=MOCK_RESEARCH)
@patch("src.agents.planner_agent.run_planner_agent", new_callable=AsyncMock, return_value=MOCK_ITINERARY)
async def test_reject_routes_back_to_research(mock_planner, mock_research):
    """Rejecting should loop back through research → plan → HITL."""
    graph = compile_graph()
    config = {"configurable": {"thread_id": "reject-test"}}

    await graph.ainvoke(VALID_STATE, config=config)

    # Reject
    await graph.ainvoke(
        Command(resume={"action": "rejected", "feedback": "More outdoor activities", "modifications": {}}),
        config=config,
    )

    # Should be paused at HITL again after re-researching + re-planning
    snapshot = await graph.aget_state(config)
    assert snapshot.next  # paused again
    state = dict(snapshot.values)
    assert state["revision_count"] == 1

    # Now approve
    await graph.ainvoke(
        Command(resume={"action": "approved", "feedback": "", "modifications": {}}),
        config=config,
    )
    snapshot = await graph.aget_state(config)
    state = dict(snapshot.values)
    assert state["workflow_stage"] == "completed"


@pytest.mark.asyncio
@patch("src.agents.research_agent.run_research_agent", new_callable=AsyncMock, return_value=MOCK_RESEARCH)
@patch("src.agents.planner_agent.run_planner_agent", new_callable=AsyncMock, return_value=MOCK_ITINERARY)
async def test_max_revisions_auto_finalizes(mock_planner, mock_research):
    """After MAX_REVISIONS rejections, the graph should auto-finalize."""
    graph = compile_graph()
    config = {"configurable": {"thread_id": "max-rev-test"}}

    with patch("src.orchestrator.get_settings") as mock_settings:
        mock_settings.return_value.MAX_REVISIONS = 2

        await graph.ainvoke(VALID_STATE, config=config)

        # Reject twice
        for _ in range(2):
            await graph.ainvoke(
                Command(resume={"action": "rejected", "feedback": "Not good enough", "modifications": {}}),
                config=config,
            )

        # Third reject should auto-finalize
        await graph.ainvoke(
            Command(resume={"action": "rejected", "feedback": "Still not happy", "modifications": {}}),
            config=config,
        )

    snapshot = await graph.aget_state(config)
    state = dict(snapshot.values)
    assert state["workflow_stage"] == "completed"
    assert state["final_plan"] is not None


@pytest.mark.asyncio
async def test_validation_failure_stops_graph():
    """Invalid request should stop at validation."""
    graph = compile_graph()
    config = {"configurable": {"thread_id": "invalid-test"}}

    bad_state = {
        **VALID_STATE,
        "session_id": "bad-001",
        "travel_request": {
            "destination": "",
            "start_date": "invalid",
            "end_date": "invalid",
            "budget_min": 0,
            "budget_max": 0,
            "interests": [],
            "num_travelers": 1,
        },
    }

    await graph.ainvoke(bad_state, config=config)

    snapshot = await graph.aget_state(config)
    state = dict(snapshot.values)
    assert state.get("error") is not None
    assert "validation_failed" in state.get("workflow_stage", "")
