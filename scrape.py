#!/usr/bin/env python3
"""
arXiv researcher scraper for RL, post-training, world models, and environment simulation.

Fetches papers from arXiv, enriches author profiles via Semantic Scholar,
and outputs a CSV spreadsheet of researchers.
"""

import argparse
import csv
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import arxiv
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "researchers.csv"

SEARCH_QUERIES = [
    "reinforcement learning",
    "RLHF",
    "reward model",
    "post-training alignment",
    "post-training optimization",
    "world model",
    "world models",
    "environment simulation",
    "policy optimization",
    "GRPO",
    "DPO direct preference optimization",
    "reinforcement learning from human feedback",
    "sim-to-real transfer",
    "model-based reinforcement learning",
]

ARXIV_CATEGORIES = ["cs.LG", "cs.AI", "cs.CL", "cs.RO", "stat.ML"]

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
S2_RATE_LIMIT_DELAY = 1.0  # seconds between S2 calls (free tier)


def build_arxiv_query(query: str, categories: list[str]) -> str:
    cat_filter = " OR ".join(f"cat:{c}" for c in categories)
    return f'all:"{query}" AND ({cat_filter})'


def fetch_papers(query: str, start_date: datetime, end_date: datetime, max_results: int = 500) -> list[arxiv.Result]:
    """Fetch arXiv papers matching query within date range."""
    full_query = build_arxiv_query(query, ARXIV_CATEGORIES)
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    results = []
    for paper in client.results(search):
        pub_date = paper.published.replace(tzinfo=None)
        if pub_date < start_date:
            break
        if pub_date <= end_date:
            results.append(paper)

    return results


def search_semantic_scholar_author(name: str, paper_title: str) -> dict | None:
    """Look up an author on Semantic Scholar by searching for their paper."""
    try:
        resp = requests.get(
            f"{SEMANTIC_SCHOLAR_API}/paper/search",
            params={"query": paper_title, "limit": 1, "fields": "authors"},
            timeout=10,
        )
        if resp.status_code == 429:
            log.warning("S2 rate limited, sleeping 30s")
            time.sleep(30)
            return None
        if resp.status_code != 200:
            return None

        data = resp.json()
        if not data.get("data"):
            return None

        authors = data["data"][0].get("authors", [])
        # Find the matching author
        name_lower = name.lower()
        for author in authors:
            if name_lower in author.get("name", "").lower() or author.get("name", "").lower() in name_lower:
                return get_author_details(author["authorId"])

    except Exception as e:
        log.debug(f"S2 search failed for {name}: {e}")
    return None


def get_author_details(author_id: str) -> dict | None:
    """Get author details from Semantic Scholar."""
    try:
        time.sleep(S2_RATE_LIMIT_DELAY)
        resp = requests.get(
            f"{SEMANTIC_SCHOLAR_API}/author/{author_id}",
            params={"fields": "name,url,homepage,externalIds"},
            timeout=10,
        )
        if resp.status_code == 429:
            log.warning("S2 rate limited on author fetch, sleeping 30s")
            time.sleep(30)
            return None
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception as e:
        log.debug(f"S2 author fetch failed for {author_id}: {e}")
    return None


def extract_linkedin_from_s2(author_data: dict) -> str:
    """Try to extract LinkedIn URL from Semantic Scholar external IDs."""
    if not author_data:
        return ""
    external_ids = author_data.get("externalIds") or {}
    # S2 doesn't directly provide LinkedIn, but sometimes has DBLP/ORCID
    # which can help find profiles. Return empty for now.
    return ""


def extract_homepage_from_s2(author_data: dict) -> str:
    """Extract homepage URL from Semantic Scholar."""
    if not author_data:
        return ""
    return author_data.get("homepage") or ""


def load_existing_data() -> tuple[list[dict], set[str]]:
    """Load existing CSV data and return rows + set of (name, paper_link) for dedup."""
    rows = []
    seen = set()
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                seen.add((row["name"], row["paper_link"]))
    return rows, seen


def save_data(rows: list[dict]):
    """Save rows to CSV."""
    DATA_DIR.mkdir(exist_ok=True)
    fieldnames = ["name", "paper_link", "paper_title", "arxiv_category", "published_date", "linkedin", "personal_website"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Sort by published_date descending, then name
        rows.sort(key=lambda r: (r.get("published_date", ""), r.get("name", "")), reverse=True)
        writer.writerows(rows)


def scrape(start_date: datetime, end_date: datetime, enrich: bool = True):
    """Main scrape pipeline."""
    existing_rows, seen = load_existing_data()
    new_rows = []
    author_cache: dict[str, dict | None] = {}  # name -> S2 data
    papers_found = 0

    for query in SEARCH_QUERIES:
        log.info(f"Searching arXiv for: {query}")
        papers = fetch_papers(query, start_date, end_date, max_results=200)
        log.info(f"  Found {len(papers)} papers")
        papers_found += len(papers)

        for paper in papers:
            paper_link = paper.entry_id
            categories = ",".join(paper.categories[:3])
            pub_date = paper.published.strftime("%Y-%m-%d")

            for author in paper.authors:
                name = str(author)
                key = (name, paper_link)
                if key in seen:
                    continue
                seen.add(key)

                linkedin = ""
                website = ""

                if enrich and name not in author_cache:
                    log.debug(f"  Enriching: {name}")
                    author_cache[name] = search_semantic_scholar_author(name, paper.title)
                    time.sleep(S2_RATE_LIMIT_DELAY)

                if enrich and name in author_cache:
                    s2_data = author_cache[name]
                    linkedin = extract_linkedin_from_s2(s2_data)
                    website = extract_homepage_from_s2(s2_data)

                new_rows.append({
                    "name": name,
                    "paper_link": paper_link,
                    "paper_title": paper.title.replace("\n", " ").strip(),
                    "arxiv_category": categories,
                    "published_date": pub_date,
                    "linkedin": linkedin,
                    "personal_website": website,
                })

    all_rows = existing_rows + new_rows
    save_data(all_rows)
    log.info(f"Done. Papers scanned: {papers_found}. New researcher-paper entries: {len(new_rows)}. Total rows: {len(all_rows)}")


def main():
    parser = argparse.ArgumentParser(description="Scrape arXiv for researcher profiles")
    parser.add_argument("--days", type=int, default=1, help="Number of days to look back (default: 1 for nightly)")
    parser.add_argument("--backfill-months", type=int, default=0, help="Backfill N months of data")
    parser.add_argument("--no-enrich", action="store_true", help="Skip Semantic Scholar enrichment")
    args = parser.parse_args()

    end_date = datetime.now(timezone.utc).replace(tzinfo=None)

    if args.backfill_months > 0:
        start_date = end_date - timedelta(days=args.backfill_months * 30)
        log.info(f"Backfilling {args.backfill_months} months: {start_date.date()} to {end_date.date()}")
    else:
        start_date = end_date - timedelta(days=args.days)
        log.info(f"Scraping last {args.days} day(s): {start_date.date()} to {end_date.date()}")

    scrape(start_date, end_date, enrich=not args.no_enrich)


if __name__ == "__main__":
    main()
