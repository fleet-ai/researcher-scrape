#!/usr/bin/env python3
"""OpenAlex API client for citation graph traversal.

Mirrors s2_client.S2Client's public surface so expand.py can swap engines
without other changes. All methods return the same shape s2_client returns:
paper dicts with keys {paperId, title, authors, year, venue, citationCount};
author dicts with keys {authorId, name, affiliations, hIndex, citationCount,
paperCount}.

Rate limit: OpenAlex polite pool is 10 req/s when a mailto is set. Default
delay of 0.11s keeps us safely under.
"""

import logging
import os
import time
import urllib.parse

import requests

log = logging.getLogger(__name__)

OA_API = "https://api.openalex.org"
OA_EMAIL = os.environ.get("OPENALEX_EMAIL", "scraper@fleet.ai")

WORK_FIELDS = "id,title,display_name,publication_year,cited_by_count,authorships,primary_location,referenced_works"
AUTHOR_FIELDS = "id,display_name,affiliations,last_known_institutions,summary_stats,works_count,cited_by_count"


def _short_id(oa_id: str) -> str:
    """Strip the https://openalex.org/ prefix from an OpenAlex ID."""
    if not oa_id:
        return ""
    return oa_id.rsplit("/", 1)[-1]


def _work_to_paper(w: dict) -> dict:
    """Convert an OpenAlex work object to the s2-shaped paper dict."""
    if not w:
        return {}
    authorships = w.get("authorships") or []
    authors = []
    for a in authorships:
        auth = a.get("author") or {}
        insts = a.get("institutions") or []
        authors.append({
            "authorId": _short_id(auth.get("id", "")),
            "name": auth.get("display_name", ""),
            "affiliations": [i.get("display_name", "") for i in insts if i.get("display_name")],
        })
    primary = w.get("primary_location") or {}
    source = primary.get("source") or {}
    return {
        "paperId": _short_id(w.get("id", "")),
        "title": w.get("title") or w.get("display_name") or "",
        "authors": authors,
        "year": w.get("publication_year"),
        "venue": source.get("display_name", ""),
        "citationCount": w.get("cited_by_count", 0),
        "referenced_works": [_short_id(r) for r in (w.get("referenced_works") or [])],
    }


def _author_to_dict(a: dict) -> dict:
    """Convert an OpenAlex author object to the s2-shaped author dict."""
    if not a:
        return {}
    stats = a.get("summary_stats") or {}
    insts = a.get("last_known_institutions") or []
    if not insts:
        # Fallback: try current affiliations list
        aff = a.get("affiliations") or []
        insts = [x.get("institution", {}) for x in aff[:2] if x.get("institution")]
    return {
        "authorId": _short_id(a.get("id", "")),
        "name": a.get("display_name", ""),
        "affiliations": [i.get("display_name", "") for i in insts if i.get("display_name")],
        "hIndex": stats.get("h_index", 0),
        "citationCount": a.get("cited_by_count", 0),
        "paperCount": a.get("works_count", 0),
        "homepage": "",
    }


class OAClient:
    """OpenAlex client with polite-pool rate limiting and retries.

    Public surface matches s2_client.S2Client so expand.py is engine-agnostic.
    """

    def __init__(self, mailto: str | None = None, delay: float | None = None):
        self.mailto = mailto or OA_EMAIL
        self.delay = delay if delay is not None else 0.11  # ~9 req/s, under polite pool 10 rps
        self._last_call = 0.0

    def _get(self, path: str, params: dict | None = None) -> dict | None:
        elapsed = time.time() - self._last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

        if params is None:
            params = {}
        params["mailto"] = self.mailto

        url = f"{OA_API}/{path.lstrip('/')}"
        for attempt in range(4):
            try:
                self._last_call = time.time()
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    wait = min(2 ** attempt * 5, 60)
                    log.warning(f"OA 429 on {path}, sleeping {wait}s")
                    time.sleep(wait)
                    continue
                if resp.status_code == 404:
                    return None
                if resp.status_code != 200:
                    log.warning(f"OA {resp.status_code} on {path}: {resp.text[:200]}")
                    return None
                return resp.json()
            except Exception as e:
                log.warning(f"OA request failed (attempt {attempt + 1}): {e}")
                if attempt < 3:
                    time.sleep(2 ** attempt)
        return None

    # --- Paper lookups ---

    def paper_by_arxiv_id(self, arxiv_id: str) -> dict | None:
        """Resolve an arXiv ID (e.g. '2409.17652') to a paper.

        Uses the arXiv-assigned DOI (`10.48550/arXiv.{id}`) which OpenAlex
        indexes for all recent arxiv submissions (~2022 onward). For older
        papers without a DOI, fall back to abs-URL landing-page lookup, then
        title search.
        """
        clean = arxiv_id.strip().split("v")[0]  # strip version suffix if present
        data = self._get(f"works/doi:10.48550/arXiv.{clean}", {"select": WORK_FIELDS})
        if data and data.get("id"):
            return _work_to_paper(data)
        # Older arxiv IDs (no DOI): try landing-page URL match
        data = self._get("works", {
            "filter": f"primary_location.landing_page_url:https://arxiv.org/abs/{clean}",
            "per-page": "1",
            "select": WORK_FIELDS,
        })
        if data and data.get("results"):
            return _work_to_paper(data["results"][0])
        return None

    def paper_by_s2_id(self, paper_id: str) -> dict | None:
        """Look up a paper by OpenAlex work ID (e.g. 'W2741809807').

        Method name kept for interface parity with S2Client.
        """
        if not paper_id:
            return None
        pid = _short_id(paper_id)
        data = self._get(f"works/{pid}", {"select": WORK_FIELDS})
        return _work_to_paper(data) if data else None

    def paper_by_title(self, title: str) -> dict | None:
        """Search for a paper by title, return best match."""
        data = self._get("works", {
            "search": title,
            "per-page": "5",
            "select": WORK_FIELDS,
        })
        if not data or not data.get("results"):
            return None
        return _work_to_paper(data["results"][0])

    # --- Expansion ---

    def get_citations(self, paper_id: str, limit: int = 50) -> list[dict]:
        """Papers that cite this paper."""
        pid = _short_id(paper_id)
        data = self._get("works", {
            "filter": f"cites:{pid}",
            "per-page": str(min(limit, 200)),
            "select": WORK_FIELDS,
            "sort": "publication_year:desc",
        })
        if not data or not data.get("results"):
            return []
        return [_work_to_paper(w) for w in data["results"][:limit] if w.get("id")]

    def get_references(self, paper_id: str, limit: int = 30) -> list[dict]:
        """Papers this paper references.

        OpenAlex bundles referenced_works into the work object itself, so we
        fetch the work once and return shallow dicts (paperId only). expand.py
        looks up full titles later when it visits each ref.
        """
        pid = _short_id(paper_id)
        work = self._get(f"works/{pid}", {"select": "referenced_works"})
        if not work:
            return []
        refs = work.get("referenced_works") or []
        return [{"paperId": _short_id(r), "title": ""} for r in refs[:limit]]

    def get_paper_authors(self, paper_id: str) -> list[dict]:
        """Detailed author info for a paper.

        OpenAlex embeds authorships in the work; we don't pay extra RPS for
        author stats here (h-index/citations get filled in later by Phase 6
        enrichment, only for researchers who survive filtering).
        """
        pid = _short_id(paper_id)
        data = self._get(f"works/{pid}", {"select": "authorships"})
        if not data:
            return []
        authors = []
        for a in data.get("authorships") or []:
            auth = a.get("author") or {}
            insts = a.get("institutions") or []
            name = auth.get("display_name", "")
            if not name:
                continue
            authors.append({
                "authorId": _short_id(auth.get("id", "")),
                "name": name,
                "affiliations": [i.get("display_name", "") for i in insts if i.get("display_name")],
                # Stats intentionally 0 — expand.py enriches from OpenAlex later.
                "hIndex": 0,
                "citationCount": 0,
                "paperCount": 0,
            })
        return authors

    def get_author_papers(self, author_id: str, limit: int = 20) -> list[dict]:
        """Recent papers by an author (most recent first)."""
        aid = _short_id(author_id)
        data = self._get("works", {
            "filter": f"author.id:{aid}",
            "per-page": str(min(limit, 200)),
            "sort": "publication_year:desc",
            "select": WORK_FIELDS,
        })
        if not data or not data.get("results"):
            return []
        return [_work_to_paper(w) for w in data["results"][:limit]]

    # --- Search ---

    def search_papers(self, query: str, year_range: str = "2024-2026",
                      limit: int = 100) -> list[dict]:
        """Keyword search for papers within a year range 'YYYY-YYYY'."""
        try:
            start, end = year_range.split("-")
            filter_str = f"from_publication_date:{start}-01-01,to_publication_date:{end}-12-31"
        except ValueError:
            filter_str = ""
        params = {
            "search": query,
            "per-page": str(min(limit, 200)),
            "select": WORK_FIELDS,
            "sort": "cited_by_count:desc",
        }
        if filter_str:
            params["filter"] = filter_str
        data = self._get("works", params)
        if not data or not data.get("results"):
            return []
        return [_work_to_paper(w) for w in data["results"][:limit]]

    def search_author(self, name: str) -> dict | None:
        """Search for an author by name, return best match."""
        data = self._get("authors", {
            "search": name,
            "per-page": "5",
            "select": AUTHOR_FIELDS,
        })
        if not data or not data.get("results"):
            return None
        return _author_to_dict(data["results"][0])
