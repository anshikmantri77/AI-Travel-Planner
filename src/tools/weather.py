"""Weather forecast tool using the free Open-Meteo API.

Two-step process:
1. Geocode the city name → lat/lon via Open-Meteo Geocoding API
2. Fetch 7-day forecast from Open-Meteo Forecast API
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


async def _geocode(city: str) -> tuple[float, float] | None:
    """Resolve a city name to (latitude, longitude)."""
    params = {"name": city, "count": 1, "language": "en", "format": "json"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(GEOCODE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        logger.error("Geocoding request failed for '%s': %s", city, exc)
        return None

    results = data.get("results")
    if not results:
        logger.warning("No geocoding results for '%s'", city)
        return None
    return results[0]["latitude"], results[0]["longitude"]


async def get_weather_forecast(city: str) -> dict[str, Any]:
    """Fetch a 7-day weather forecast for *city*.

    Args:
        city: City name (e.g. "Paris", "Tokyo").

    Returns:
        Dict with ``city``, ``latitude``, ``longitude``, and ``daily``
        containing arrays of date, temperature_max, temperature_min,
        precipitation_sum, and wind_speed_max.
    """
    coords = await _geocode(city)
    if coords is None:
        return {"error": f"Could not geocode city: {city}"}

    lat, lon = coords
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
        "timezone": "auto",
        "forecast_days": 7,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(FORECAST_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        logger.error("Open-Meteo forecast request failed: %s", exc)
        return {"error": f"Weather API error: {exc}"}

    daily = data.get("daily", {})
    return {
        "city": city,
        "latitude": lat,
        "longitude": lon,
        "daily": {
            "date": daily.get("time", []),
            "temperature_max_c": daily.get("temperature_2m_max", []),
            "temperature_min_c": daily.get("temperature_2m_min", []),
            "precipitation_mm": daily.get("precipitation_sum", []),
            "wind_speed_max_kmh": daily.get("wind_speed_10m_max", []),
        },
    }


# LangChain-compatible sync wrapper
def weather_tool(city: str) -> str:
    """Get a 7-day weather forecast for a city.

    Args:
        city: City or destination name (e.g. 'Barcelona').

    Returns:
        JSON string of the 7-day forecast with daily high/low temps, precipitation, and wind.
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = pool.submit(asyncio.run, get_weather_forecast(city)).result()
    else:
        result = asyncio.run(get_weather_forecast(city))

    return json.dumps(result, indent=2)
