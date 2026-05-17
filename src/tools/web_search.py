"""Web search tool using the Serper API (https://serper.dev).

Falls back to mock data when SERPER_API_KEY is not configured so the
system stays functional during local development.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from src.config import get_settings

logger = logging.getLogger(__name__)

SERPER_URL = "https://google.serper.dev/search"

# ---------------------------------------------------------------------------
# Mock fallback data
# ---------------------------------------------------------------------------
_MOCK_RESULTS: list[dict[str, str]] = [
    {
        "title": "Top attractions in your destination (mock)",
        "snippet": "This is mock search data. Configure SERPER_API_KEY for real results.",
        "link": "https://example.com/attractions",
    },
    {
        "title": "Travel tips and safety info (mock)",
        "snippet": "Mock travel safety information. Set SERPER_API_KEY in .env for live data.",
        "link": "https://example.com/safety",
    },
    {
        "title": "Local cuisine guide (mock)",
        "snippet": "Mock cuisine highlights. Enable Serper integration for real recommendations.",
        "link": "https://example.com/food",
    },
]


async def web_search(query: str, num_results: int = 5) -> list[dict[str, Any]]:
    """Search the web via Serper API.

    Parameters
    ----------
    query:
        The search query string.
    num_results:
        Max number of organic results to return (capped at 10).

    Returns
    -------
    list of dicts with keys ``title``, ``snippet``, ``link``.
    """
    settings = get_settings()
    if not settings.SERPER_API_KEY:
        logger.warning("SERPER_API_KEY not set – returning mock search results.")
        return _MOCK_RESULTS[:num_results]

    headers = {
        "X-API-KEY": settings.SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": min(num_results, 10)}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(SERPER_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        logger.error("Serper API request failed: %s", exc)
        return _MOCK_RESULTS[:num_results]

    organic: list[dict[str, Any]] = data.get("organic", [])
    return [
        {
            "title": r.get("title", ""),
            "snippet": r.get("snippet", ""),
            "link": r.get("link", ""),
        }
        for r in organic[:num_results]
    ]


# LangChain-compatible tool wrapper (sync, for agent binding)
def web_search_tool(query: str) -> str:
    """Search the web for travel-related information.

    Args:
        query: The search query string.

    Returns:
        JSON string of search results with title, snippet, and link.
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            results = pool.submit(asyncio.run, web_search(query)).result()
    else:
        results = asyncio.run(web_search(query))

    return json.dumps(results, indent=2)
