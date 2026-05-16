#!/usr/bin/env python3
"""Semantic Scholar API client for citation graph traversal."""

import logging
import os
import time

import requests

log = logging.getLogger(__name__)

S2_API = "https://api.semanticscholar.org/graph/v1"

PAPER_FIELDS = "paperId,externalIds,title,authors,year,venue,citationCount"
PAPER_FIELDS_FULL = PAPER_FIELDS + ",references.paperId,references.title,references.authors,references.year,citations.paperId,citations.title,citations.authors,citations.year"
AUTHOR_FIELDS = "authorId,name,affiliations,homepage,paperCount,citationCount,hIndex"


class S2Client:
    """Thin Semantic Scholar client with rate limiting and retries."""

    def __init__(self, api_key: str | None = None, delay: float | None = None):
        self.api_key = api_key or os.environ.get("S2_API_KEY", "")
        self.delay = delay if delay is not None else (0.12 if self.api_key else 3.5)
        self._last_call = 0.0

    def _get(self, path: str, params: dict | None = None) -> dict | None:
        """GET with rate limiting and retry on 429."""
        elapsed = time.time() - self._last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        url = f"{S2_API}/{path.lstrip('/')}"
        for attempt in range(4):
            try:
                self._last_call = time.time()
                resp = requests.get(url, params=params, headers=headers, timeout=30)
                if resp.status_code == 429:
                    wait = min(2 ** attempt * 5, 60)
                    log.warning(f"S2 429 on {path}, sleeping {wait}s")
                    time.sleep(wait)
                    continue
                if resp.status_code == 404:
                    return None
                if resp.status_code != 200:
                    log.warning(f"S2 {resp.status_code} on {path}: {resp.text[:200]}")
                    return None
                return resp.json()
            except Exception as e:
                log.warning(f"S2 request failed (attempt {attempt+1}): {e}")
                if attempt < 3:
                    time.sleep(2 ** attempt)
        return None

    # --- Paper lookups ---

    def paper_by_arxiv_id(self, arxiv_id: str) -> dict | None:
        """Resolve an arXiv ID to an S2 paper."""
        return self._get(f"paper/ArXiv:{arxiv_id}", {"fields": PAPER_FIELDS})

    def paper_by_s2_id(self, paper_id: str) -> dict | None:
        """Look up a paper by S2 paper ID."""
        return self._get(f"paper/{paper_id}", {"fields": PAPER_FIELDS})

    def paper_by_title(self, title: str) -> dict | None:
        """Search for a paper by title, return best match."""
        data = self._get("paper/search", {"query": title, "limit": "5", "fields": PAPER_FIELDS})
        if not data or not data.get("data"):
            return None
        # Return first result (S2 ranks by relevance)
        return data["data"][0]

    # --- Expansion ---

    def get_citations(self, paper_id: str, limit: int = 50) -> list[dict]:
        """Get papers that cite this paper."""
        data = self._get(f"paper/{paper_id}/citations",
                         {"fields": PAPER_FIELDS, "limit": str(limit)})
        if not data or not data.get("data"):
            return []
        return [c["citingPaper"] for c in data["data"] if c.get("citingPaper", {}).get("paperId")]

    def get_references(self, paper_id: str, limit: int = 50) -> list[dict]:
        """Get papers referenced by this paper."""
        data = self._get(f"paper/{paper_id}/references",
                         {"fields": PAPER_FIELDS, "limit": str(limit)})
        if not data or not data.get("data"):
            return []
        return [r["citedPaper"] for r in data["data"] if r.get("citedPaper", {}).get("paperId")]

    def get_paper_authors(self, paper_id: str) -> list[dict]:
        """Get detailed author info for a paper."""
        data = self._get(f"paper/{paper_id}/authors", {"fields": AUTHOR_FIELDS})
        if not data or not data.get("data"):
            return []
        return data["data"]

    def get_author_papers(self, author_id: str, limit: int = 20) -> list[dict]:
        """Get recent papers by an author."""
        data = self._get(f"author/{author_id}/papers",
                         {"fields": PAPER_FIELDS, "limit": str(limit)})
        if not data or not data.get("data"):
            return []
        return data["data"]

    # --- Search ---

    def search_papers(self, query: str, year_range: str = "2024-2026",
                      limit: int = 100) -> list[dict]:
        """Keyword search for papers."""
        data = self._get("paper/search",
                         {"query": query, "year": year_range, "limit": str(limit),
                          "fields": PAPER_FIELDS})
        if not data or not data.get("data"):
            return []
        return data["data"]

    def search_author(self, name: str) -> dict | None:
        """Search for an author by name, return best match."""
        data = self._get("author/search",
                         {"query": name, "limit": "5", "fields": AUTHOR_FIELDS})
        if not data or not data.get("data"):
            return None
        return data["data"][0]
