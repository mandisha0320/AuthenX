"""
authenx/services/search_service.py

Web Search Service for News Headline Verification.

Supports two backends (configured via SEARCH_BACKEND env var):
    - "serpapi"     : Google Search via SerpAPI (requires SERPAPI_KEY)
    - "duckduckgo"  : DuckDuckGo HTML scraping via duckduckgo_search lib (free, no key)

Default backend: duckduckgo (zero-cost, no API key required for demos).

Each result is returned as:
    {"title": str, "snippet": str, "url": str}
"""

import os
import logging
import requests
import html
import re
from typing import Optional

logger = logging.getLogger("authenx.search_service")

SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "duckduckgo").lower()
SERPAPI_KEY     = os.getenv("SERPAPI_KEY", "")
RESULTS_COUNT   = 5  # Top N results to fetch


# ─── DuckDuckGo Backend ────────────────────────────────────────────────────────

def _search_duckduckgo(query: str, num_results: int = RESULTS_COUNT) -> list[dict]:
    """
    Fetch search results via the duckduckgo_search library.
    Requires: pip install duckduckgo-search
    """
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num_results):
                results.append({
                    "title":   r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url":     r.get("href", ""),
                })
        logger.info(f"DuckDuckGo returned {len(results)} results for query: {query!r}")
        return results
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
        return []


# ─── SerpAPI Backend ──────────────────────────────────────────────────────────

def _search_serpapi(query: str, num_results: int = RESULTS_COUNT) -> list[dict]:
    """
    Fetch search results via SerpAPI (Google Search).
    Requires SERPAPI_KEY environment variable.
    """
    if not SERPAPI_KEY:
        logger.error("SERPAPI_KEY is not set. Cannot use SerpAPI backend.")
        return []

    params = {
        "q":       query,
        "api_key": SERPAPI_KEY,
        "num":     num_results,
        "engine":  "google",
    }
    try:
        resp = requests.get("https://serpapi.com/search", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("organic_results", [])[:num_results]:
            results.append({
                "title":   item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url":     item.get("link", ""),
            })
        logger.info(f"SerpAPI returned {len(results)} results for query: {query!r}")
        return results
    except Exception as e:
        logger.error(f"SerpAPI search failed: {e}")
        return []


# ─── Public Interface ──────────────────────────────────────────────────────────

def fetch_search_results(query: str, num_results: int = RESULTS_COUNT) -> list[dict]:
    """
    Route a search query to the configured backend and return results.

    Args:
        query:       The search query string (e.g., the news headline).
        num_results: Number of results to retrieve (default 5).

    Returns:
        List of dicts: [{"title": ..., "snippet": ..., "url": ...}, ...]
        Empty list if search fails or backend is misconfigured.
    """
    if SEARCH_BACKEND == "serpapi":
        return _search_serpapi(query, num_results)
    else:
        # Default: DuckDuckGo (free, no API key)
        return _search_duckduckgo(query, num_results)


def build_corpus_texts(results: list[dict]) -> list[str]:
    """
    Combine title + snippet for each result into a single searchable string.
    Used for embedding comparison.
    """
    texts = []
    for r in results:
        combined = f"{r.get('title', '')}. {r.get('snippet', '')}".strip(". ")
        if combined:
            texts.append(combined)
    return texts
