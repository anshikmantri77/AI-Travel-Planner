"""Tests for the FastAPI endpoints.

Uses pytest + httpx AsyncClient. Agent calls are mocked so tests
run without API keys or network access.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.main import app


@pytest_asyncio.fixture
async def client():
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


VALID_PAYLOAD = {
    "destination": "Barcelona",
    "start_date": (date.today() + timedelta(days=30)).isoformat(),
    "end_date": (date.today() + timedelta(days=35)).isoformat(),
    "budget_min": 500,
    "budget_max": 2000,
    "interests": ["history", "food", "architecture"],
    "num_travelers": 2,
}

MOCK_RESEARCH = {
    "destination_overview": "Barcelona is a beautiful Mediterranean city.",
    "top_attractions": ["Sagrada Familia", "Park Güell", "Gothic Quarter"],
    "local_tips": ["Get a T-Casual metro pass"],
    "safety_notes": "Generally safe for tourists.",
    "weather_summary": "Warm and sunny, 25-30°C.",
    "best_areas_to_stay": ["Eixample", "Gothic Quarter"],
    "cuisine_highlights": ["Paella", "Tapas"],
}

MOCK_ITINERARY = {
    "trip_summary": "A 5-day trip to Barcelona.",
    "duration_days": 5,
    "total_budget_used": 1500,
    "days": [
        {
            "day": 1,
            "date": "2025-09-01",
            "theme": "Arrival & Gothic Quarter",
            "morning": {"activity": "Arrive, check in", "duration": "2h", "cost": 0},
            "afternoon": {"activity": "Gothic Quarter walk", "duration": "3h", "cost": 0},
            "evening": {"activity": "Tapas tour", "duration": "2h", "cost": 40},
            "accommodation": {"name": "Hotel Example", "type": "hotel", "cost_per_night": 120},
            "daily_budget": 200,
        }
    ],
    "packing_suggestions": ["Comfortable walking shoes", "Sunscreen"],
    "important_notes": "Book Sagrada Familia tickets in advance.",
}


@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_create_plan_invalid_payload(client: AsyncClient):
    resp = await client.post("/plan", json={"destination": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_plan_invalid_dates(client: AsyncClient):
    payload = {**VALID_PAYLOAD, "start_date": "2020-01-01"}
    resp = await client.post("/plan", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_plan_budget_validation(client: AsyncClient):
    payload = {**VALID_PAYLOAD, "budget_min": 3000, "budget_max": 1000}
    resp = await client.post("/plan", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
@patch("src.agents.research_agent.run_research_agent", new_callable=AsyncMock, return_value=MOCK_RESEARCH)
@patch("src.agents.planner_agent.run_planner_agent", new_callable=AsyncMock, return_value=MOCK_ITINERARY)
async def test_full_happy_path(mock_planner, mock_research, client: AsyncClient):
    """Create plan → get status → approve → get final plan."""
    # 1. Create plan
    resp = await client.post("/plan", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert data["status"] == "awaiting_review"
    session_id = data["session_id"]

    # 2. Check status
    resp = await client.get(f"/plan/{session_id}")
    assert resp.status_code == 200
    assert resp.json()["workflow_stage"] in ("awaiting_review", "feedback_received")

    # 3. GET /final before approval → 409
    resp = await client.get(f"/plan/{session_id}/final")
    assert resp.status_code == 409

    # 4. Approve
    resp = await client.post(f"/plan/{session_id}/review", json={"action": "approve"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"

    # 5. GET /final after approval → 200
    resp = await client.get(f"/plan/{session_id}/final")
    assert resp.status_code == 200
    final = resp.json()
    assert "final_plan" in final


@pytest.mark.asyncio
async def test_get_plan_not_found(client: AsyncClient):
    resp = await client.get("/plan/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_review_not_found(client: AsyncClient):
    resp = await client.post("/plan/nonexistent/review", json={"action": "approve"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_final_not_found(client: AsyncClient):
    resp = await client.get("/plan/nonexistent/final")
    assert resp.status_code == 404


@pytest.mark.asyncio
@patch("src.agents.research_agent.run_research_agent", new_callable=AsyncMock, return_value=MOCK_RESEARCH)
@patch("src.agents.planner_agent.run_planner_agent", new_callable=AsyncMock, return_value=MOCK_ITINERARY)
async def test_reject_then_approve(mock_planner, mock_research, client: AsyncClient):
    """Reject once, then approve on the second attempt."""
    resp = await client.post("/plan", json=VALID_PAYLOAD)
    session_id = resp.json()["session_id"]

    # Reject
    resp = await client.post(
        f"/plan/{session_id}/review",
        json={"action": "reject", "feedback": "Need more museums"},
    )
    assert resp.status_code == 200

    # Approve
    resp = await client.post(f"/plan/{session_id}/review", json={"action": "approve"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"


@pytest.mark.asyncio
@patch("src.agents.research_agent.run_research_agent", new_callable=AsyncMock, return_value=MOCK_RESEARCH)
@patch("src.agents.planner_agent.run_planner_agent", new_callable=AsyncMock, return_value=MOCK_ITINERARY)
async def test_modify_then_approve(mock_planner, mock_research, client: AsyncClient):
    """Modify once, then approve."""
    resp = await client.post("/plan", json=VALID_PAYLOAD)
    session_id = resp.json()["session_id"]

    # Modify
    resp = await client.post(
        f"/plan/{session_id}/review",
        json={"action": "modify", "modifications": {"day_1_morning": "Visit museum instead"}},
    )
    assert resp.status_code == 200

    # Approve
    resp = await client.post(f"/plan/{session_id}/review", json={"action": "approve"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"
