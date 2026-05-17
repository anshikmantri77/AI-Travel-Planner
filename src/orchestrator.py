"""LangGraph orchestrator — the StateGraph that drives the travel planning workflow.

Nodes:
  validate_request → research → plan_itinerary → hitl_checkpoint → process_feedback → finalize

The hitl_checkpoint node uses LangGraph's ``interrupt()`` to pause the
graph and surface the draft plan for human review.  The graph is compiled
with ``MemorySaver`` so state persists across HTTP requests.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt, Command

from src.config import get_settings
from src.state import PlannerState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

async def validate_request(state: PlannerState) -> dict[str, Any]:
    """Validate the incoming travel request."""
    req = state.get("travel_request", {})
    errors: list[str] = []

    if not req.get("destination"):
        errors.append("destination is required")

    start = req.get("start_date", "")
    end = req.get("end_date", "")
    try:
        sd = datetime.strptime(start, "%Y-%m-%d")
        ed = datetime.strptime(end, "%Y-%m-%d")
        if ed <= sd:
            errors.append("end_date must be after start_date")
    except (ValueError, TypeError):
        errors.append("start_date and end_date must be valid YYYY-MM-DD strings")

    budget_min = req.get("budget_min", 0)
    budget_max = req.get("budget_max", 0)
    if budget_min < 0 or budget_max < 0:
        errors.append("budget values must be non-negative")
    if budget_min >= budget_max:
        errors.append("budget_min must be less than budget_max")

    num_travelers = req.get("num_travelers", 1)
    if not (1 <= num_travelers <= 20):
        errors.append("num_travelers must be between 1 and 20")

    interests = req.get("interests", [])
    if not (1 <= len(interests) <= 10):
        errors.append("interests must have 1–10 items")

    if errors:
        return {
            "error": "; ".join(errors),
            "workflow_stage": "validation_failed",
        }

    return {
        "workflow_stage": "validated",
        "error": None,
    }


async def research(state: PlannerState) -> dict[str, Any]:
    """Run the Research Agent."""
    from src.agents.research_agent import run_research_agent

    logger.info("Running research agent for session %s", state.get("session_id"))
    try:
        output = await run_research_agent(state["travel_request"])
    except Exception as exc:
        logger.exception("Research agent failed")
        return {
            "error": f"Research agent error: {exc}",
            "workflow_stage": "research_failed",
        }

    return {
        "research_output": output,
        "workflow_stage": "research_complete",
        "error": None,
    }


async def plan_itinerary(state: PlannerState) -> dict[str, Any]:
    """Run the Planner Agent."""
    from src.agents.planner_agent import run_planner_agent

    logger.info("Running planner agent for session %s", state.get("session_id"))
    try:
        output = await run_planner_agent(
            travel_request=state["travel_request"],
            research_output=state.get("research_output", {}),
            hitl_feedback=state.get("hitl_feedback"),
            hitl_modifications=state.get("hitl_modifications"),
        )
    except Exception as exc:
        logger.exception("Planner agent failed")
        return {
            "error": f"Planner agent error: {exc}",
            "workflow_stage": "planning_failed",
        }

    return {
        "draft_itinerary": output,
        "workflow_stage": "awaiting_review",
        "hitl_status": "pending",
        "error": None,
    }


async def hitl_checkpoint(state: PlannerState) -> dict[str, Any]:
    """Pause the graph and wait for human feedback.

    Uses ``interrupt()`` which raises a special exception caught by
    the LangGraph runtime, persisting state and halting execution.
    The graph resumes when the caller invokes ``graph.ainvoke(…)``
    with the ``Command(resume=...)`` value.
    """
    logger.info("HITL checkpoint reached for session %s", state.get("session_id"))
    feedback = interrupt(
        {
            "message": "Draft itinerary ready for review",
            "draft_itinerary": state.get("draft_itinerary", {}),
        }
    )
    # When resumed, `feedback` contains the user's review payload
    return {
        "hitl_status": feedback.get("action", "approved"),
        "hitl_feedback": feedback.get("feedback", ""),
        "hitl_modifications": feedback.get("modifications", {}),
        "workflow_stage": "feedback_received",
    }


async def process_feedback(state: PlannerState) -> Command:
    """Route based on HITL feedback: approve → finalize, reject → research, modify → plan."""
    settings = get_settings()
    action = state.get("hitl_status", "approved")
    revision_count = state.get("revision_count", 0)

    if action in ("approved", "approve"):
        return Command(goto="finalize", update={"workflow_stage": "finalizing"})

    # Check revision limit
    if revision_count >= settings.MAX_REVISIONS:
        logger.warning("Max revisions (%d) reached for session %s — auto-finalizing",
                        settings.MAX_REVISIONS, state.get("session_id"))
        return Command(
            goto="finalize",
            update={
                "workflow_stage": "finalizing",
                "hitl_feedback": (state.get("hitl_feedback", "") +
                                  " [Auto-finalized: max revision limit reached]"),
            },
        )

    new_revision = revision_count + 1

    if action in ("rejected", "reject"):
        return Command(goto="research", update={"revision_count": new_revision, "workflow_stage": "re_researching"})

    # "modified" — re-plan with modifications
    return Command(goto="plan_itinerary", update={"revision_count": new_revision, "workflow_stage": "re_planning"})


async def finalize(state: PlannerState) -> dict[str, Any]:
    """Assemble the final approved plan."""
    final: dict[str, Any] = {
        "session_id": state.get("session_id", ""),
        "destination": state.get("travel_request", {}).get("destination", ""),
        "travel_request": state.get("travel_request", {}),
        "research_summary": state.get("research_output", {}),
        "itinerary": state.get("draft_itinerary", {}),
        "status": "approved",
        "revision_count": state.get("revision_count", 0),
    }

    return {
        "final_plan": final,
        "workflow_stage": "completed",
        "hitl_status": "approved",
    }


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _after_validation(state: PlannerState) -> str:
    if state.get("error"):
        return END
    return "research"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and return the compiled LangGraph StateGraph."""
    graph = StateGraph(PlannerState)

    graph.add_node("validate_request", validate_request)
    graph.add_node("research", research)
    graph.add_node("plan_itinerary", plan_itinerary)
    graph.add_node("hitl_checkpoint", hitl_checkpoint)
    graph.add_node("process_feedback", process_feedback)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("validate_request")

    graph.add_conditional_edges("validate_request", _after_validation, {"research": "research", END: END})
    graph.add_edge("research", "plan_itinerary")
    graph.add_edge("plan_itinerary", "hitl_checkpoint")
    graph.add_edge("hitl_checkpoint", "process_feedback")
    # process_feedback uses Command(goto=...) so no static edge needed
    graph.add_edge("finalize", END)

    return graph


def compile_graph():
    """Compile the graph with MemorySaver checkpointer for state persistence."""
    memory = MemorySaver()
    graph = build_graph()
    return graph.compile(checkpointer=memory)
