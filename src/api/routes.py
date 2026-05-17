"""FastAPI route handlers for the Travel Planner API."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from langgraph.types import Command

from src.api.models import (
    ErrorResponse,
    FinalPlanResponse,
    HealthResponse,
    PlanCreatedResponse,
    PlanStatusResponse,
    ReviewRequestBody,
    ReviewResponse,
    TravelRequestBody,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _graph(request: Request):
    """Retrieve the compiled graph from app state."""
    return request.app.state.graph


def _sessions(request: Request):
    """Retrieve the session store from app state."""
    return request.app.state.sessions


# ---------------------------------------------------------------------------
# Helper to extract state snapshot from LangGraph
# ---------------------------------------------------------------------------

async def _get_graph_state(graph, thread_id: str) -> dict[str, Any]:
    """Get the current state snapshot from the LangGraph checkpointer."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await graph.aget_state(config)
    return dict(snapshot.values) if snapshot and snapshot.values else {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Simple health check."""
    return HealthResponse()


@router.post(
    "/plan",
    response_model=PlanCreatedResponse,
    responses={422: {"model": ErrorResponse}},
    tags=["planning"],
)
async def create_plan(body: TravelRequestBody, request: Request):
    """Submit a new travel planning request.

    Starts the LangGraph workflow which runs until the HITL interrupt,
    then returns the session ID and draft itinerary for review.
    """
    graph = _graph(request)
    sessions = _sessions(request)

    session_id = await sessions.create()

    # Build initial state
    initial_state = {
        "session_id": session_id,
        "travel_request": {
            "destination": body.destination,
            "start_date": body.start_date.isoformat(),
            "end_date": body.end_date.isoformat(),
            "budget_min": body.budget_min,
            "budget_max": body.budget_max,
            "interests": body.interests,
            "num_travelers": body.num_travelers,
        },
        "hitl_status": "pending",
        "hitl_feedback": "",
        "hitl_modifications": {},
        "revision_count": 0,
        "error": None,
        "workflow_stage": "started",
    }

    config = {"configurable": {"thread_id": session_id}}

    try:
        # This runs until the interrupt() in hitl_checkpoint
        await graph.ainvoke(initial_state, config=config)
    except Exception as exc:
        logger.exception("Graph execution failed for session %s", session_id)
        await sessions.update(session_id, status="error", workflow_stage="error")
        raise HTTPException(status_code=500, detail=f"Planning workflow error: {exc}")

    # Read state after interrupt
    state = await _get_graph_state(graph, session_id)
    error = state.get("error")
    if error:
        await sessions.update(session_id, status="error", workflow_stage="error")
        raise HTTPException(status_code=400, detail=error)

    stage = state.get("workflow_stage", "awaiting_review")
    draft = state.get("draft_itinerary")
    await sessions.update(session_id, status="awaiting_review", workflow_stage=stage)

    return PlanCreatedResponse(
        session_id=session_id,
        status="awaiting_review",
        draft_itinerary=draft,
        message="Draft itinerary ready for review. Use POST /plan/{id}/review to approve, reject, or modify.",
    )


@router.get(
    "/plan/{session_id}",
    response_model=PlanStatusResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["planning"],
)
async def get_plan_status(session_id: str, request: Request):
    """Get the current plan status and draft itinerary."""
    sessions = _sessions(request)
    session = await sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    graph = _graph(request)
    state = await _get_graph_state(graph, session_id)

    return PlanStatusResponse(
        session_id=session_id,
        status=session.get("status", "unknown"),
        workflow_stage=state.get("workflow_stage", session.get("workflow_stage", "unknown")),
        hitl_status=state.get("hitl_status"),
        draft_itinerary=state.get("draft_itinerary"),
        error=state.get("error"),
        revision_count=state.get("revision_count", 0),
    )


@router.post(
    "/plan/{session_id}/review",
    response_model=ReviewResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    tags=["planning"],
)
async def submit_review(session_id: str, body: ReviewRequestBody, request: Request):
    """Submit HITL feedback to resume the workflow."""
    sessions = _sessions(request)
    session = await sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    graph = _graph(request)
    config = {"configurable": {"thread_id": session_id}}

    # Verify the graph is actually waiting at the interrupt
    snapshot = await graph.aget_state(config)
    if not snapshot or not snapshot.next:
        raise HTTPException(
            status_code=409,
            detail="Plan is not currently awaiting review. Check status with GET /plan/{id}.",
        )

    # Resume the graph with the user's feedback via Command(resume=...)
    resume_payload = {
        "action": body.action,
        "feedback": body.feedback or "",
        "modifications": body.modifications or {},
    }

    try:
        await graph.ainvoke(Command(resume=resume_payload), config=config)
    except Exception as exc:
        logger.exception("Graph resume failed for session %s", session_id)
        raise HTTPException(status_code=500, detail=f"Resume failed: {exc}")

    # Read updated state
    state = await _get_graph_state(graph, session_id)
    stage = state.get("workflow_stage", "unknown")
    status = "completed" if stage == "completed" else "in_progress"

    await sessions.update(session_id, status=status, workflow_stage=stage)

    # If the workflow is still running (rejected/modified → loops back to HITL),
    # we need to check if it paused at interrupt again
    snapshot = await graph.aget_state(config)
    if snapshot and snapshot.next:
        status = "awaiting_review"
        stage = state.get("workflow_stage", "awaiting_review")
        await sessions.update(session_id, status=status, workflow_stage=stage)

    return ReviewResponse(
        session_id=session_id,
        status=status,
        workflow_stage=stage,
        hitl_status=state.get("hitl_status"),
        draft_itinerary=state.get("draft_itinerary"),
        final_plan=state.get("final_plan") if stage == "completed" else None,
        message="Plan approved and finalized." if stage == "completed" else "Feedback processed, draft updated.",
    )


@router.get(
    "/plan/{session_id}/final",
    response_model=FinalPlanResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    tags=["planning"],
)
async def get_final_plan(session_id: str, request: Request):
    """Retrieve the finalized plan (only available after approval)."""
    sessions = _sessions(request)
    session = await sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    graph = _graph(request)
    state = await _get_graph_state(graph, session_id)
    final = state.get("final_plan")

    if not final:
        current_stage = state.get("workflow_stage", "unknown")
        raise HTTPException(
            status_code=409,
            detail=f"Plan not yet finalized. Current stage: {current_stage}",
        )

    return FinalPlanResponse(
        session_id=session_id,
        status="completed",
        final_plan=final,
    )
