"""Unit tests for the travel planner tools."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.budget_allocator import allocate_budget
from src.tools.activity_scorer import score_activities


# ---------------------------------------------------------------------------
# Budget allocator tests
# ---------------------------------------------------------------------------

class TestBudgetAllocator:
    def test_basic_split(self):
        result = allocate_budget(1000, 5, 2, "mid")
        assert result["total_budget"] == 1000
        assert result["num_days"] == 5
        assert result["per_day"] == 200.0
        assert result["per_day_per_person"] == 100.0
        assert len(result["daily_breakdown"]) == 5

    def test_percentages_sum_to_100(self):
        result = allocate_budget(1000, 5, 1, "mid")
        pcts = (
            result["accommodation_pct"]
            + result["food_pct"]
            + result["activities_pct"]
            + result["transport_pct"]
        )
        assert pcts == 100

    def test_budget_style(self):
        budget_result = allocate_budget(1000, 5, 1, "budget")
        luxury_result = allocate_budget(1000, 5, 1, "luxury")
        assert budget_result["accommodation_pct"] < luxury_result["accommodation_pct"]

    def test_daily_breakdown_fields(self):
        result = allocate_budget(500, 3, 1, "mid")
        day = result["daily_breakdown"][0]
        assert "accommodation" in day
        assert "food" in day
        assert "activities" in day
        assert "transport" in day
        assert day["day"] == 1

    def test_invalid_inputs(self):
        assert "error" in allocate_budget(-100, 5, 1, "mid")
        assert "error" in allocate_budget(1000, 0, 1, "mid")
        assert "error" in allocate_budget(1000, 5, 0, "mid")

    def test_unknown_style_falls_back_to_mid(self):
        result = allocate_budget(1000, 5, 1, "ultra")
        assert result["travel_style"] == "mid"


# ---------------------------------------------------------------------------
# Activity scorer tests
# ---------------------------------------------------------------------------

class TestActivityScorer:
    def test_ranks_by_interest_match(self):
        activities = ["Historical museum tour", "Beach surfing", "Art gallery visit"]
        interests = ["history", "art"]
        results = score_activities(activities, interests)
        assert len(results) == 3
        # Activities matching interests should rank higher
        assert results[0]["score"] >= results[-1]["score"]

    def test_perfect_match(self):
        activities = ["History and art workshop"]
        interests = ["history", "art"]
        results = score_activities(activities, interests)
        assert results[0]["score"] == 100.0
        assert len(results[0]["matched_interests"]) == 2

    def test_no_match(self):
        activities = ["Skydiving adventure"]
        interests = ["cooking", "museums"]
        results = score_activities(activities, interests)
        assert results[0]["score"] == 0.0
        assert results[0]["matched_interests"] == []

    def test_empty_interests(self):
        results = score_activities(["Visit park"], [])
        assert results[0]["score"] == 0.0

    def test_empty_activities(self):
        results = score_activities([], ["food"])
        assert results == []

    def test_sorted_descending(self):
        activities = ["Cooking class", "Random walk", "Food market tour"]
        interests = ["food", "cooking"]
        results = score_activities(activities, interests)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Weather tool tests (mocked HTTP)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_weather_tool_returns_forecast():
    from src.tools.weather import get_weather_forecast

    mock_geocode = {
        "results": [{"latitude": 41.39, "longitude": 2.17}]
    }
    mock_forecast = {
        "daily": {
            "time": ["2025-08-01", "2025-08-02"],
            "temperature_2m_max": [30, 31],
            "temperature_2m_min": [22, 23],
            "precipitation_sum": [0, 0.5],
            "wind_speed_10m_max": [15, 20],
        }
    }

    mock_response_geo = MagicMock()
    mock_response_geo.json.return_value = mock_geocode
    mock_response_geo.raise_for_status = MagicMock()

    mock_response_fc = MagicMock()
    mock_response_fc.json.return_value = mock_forecast
    mock_response_fc.raise_for_status = MagicMock()

    with patch("src.tools.weather.httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.get = AsyncMock(side_effect=[mock_response_geo, mock_response_fc])
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        result = await get_weather_forecast("Barcelona")

    assert "city" in result
    assert result["city"] == "Barcelona"
    assert "daily" in result
    assert len(result["daily"]["date"]) == 2


@pytest.mark.asyncio
async def test_weather_tool_geocode_failure():
    from src.tools.weather import get_weather_forecast

    mock_response = MagicMock()
    mock_response.json.return_value = {}
    mock_response.raise_for_status = MagicMock()

    with patch("src.tools.weather.httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.get = AsyncMock(return_value=mock_response)
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        result = await get_weather_forecast("NonexistentCity12345")

    assert "error" in result


# ---------------------------------------------------------------------------
# Web search tool tests (mocked HTTP)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_web_search_fallback_when_no_key():
    """When SERPER_API_KEY is empty, mock data should be returned."""
    from src.tools.web_search import web_search

    with patch("src.tools.web_search.get_settings") as mock_settings:
        mock_settings.return_value.SERPER_API_KEY = ""
        results = await web_search("Barcelona attractions")

    assert len(results) > 0
    assert "mock" in results[0]["title"].lower() or "mock" in results[0]["snippet"].lower()


@pytest.mark.asyncio
async def test_web_search_with_key():
    from src.tools.web_search import web_search

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "organic": [
            {"title": "Barcelona Guide", "snippet": "A guide to Barcelona", "link": "https://example.com"}
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("src.tools.web_search.get_settings") as mock_settings, \
         patch("src.tools.web_search.httpx.AsyncClient") as MockClient:
        mock_settings.return_value.SERPER_API_KEY = "test-key"

        instance = AsyncMock()
        instance.post = AsyncMock(return_value=mock_response)
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        results = await web_search("Barcelona attractions")

    assert len(results) == 1
    assert results[0]["title"] == "Barcelona Guide"
