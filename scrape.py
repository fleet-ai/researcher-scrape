#!/usr/bin/env python3
"""
arXiv + OpenAlex researcher scraper for RL, post-training, world models,
and environment simulation.

Two-phase approach:
1. arXiv API — discover papers by keyword + category, get all authors
2. OpenAlex API — search same keywords with country filter to get structured
   institution/country data for US-based (and other) researchers

Outputs both a full CSV and a US-only filtered CSV for hiring pipelines.
"""

import argparse
import csv
import logging
import re
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import arxiv
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
CSV_ALL = DATA_DIR / "researchers.csv"
CSV_US = DATA_DIR / "researchers_us.csv"

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

# OpenAlex search queries (broader phrasing works better for their search)
OPENALEX_QUERIES = [
    "reinforcement learning RLHF language model",
    "reinforcement learning human feedback LLM",
    "GRPO group relative policy optimization",
    "reward model language model alignment",
    "post-training alignment large language model",
    "world model deep reinforcement learning",
    "sim-to-real transfer robot reinforcement learning",
    "model-based reinforcement learning deep learning",
    "DPO direct preference optimization language model",
    "policy optimization reinforcement learning neural network",
]

ARXIV_CATEGORIES = ["cs.LG", "cs.AI", "cs.CL", "cs.RO", "stat.ML"]

OPENALEX_API = "https://api.openalex.org"
OPENALEX_EMAIL = "scraper@fleet.ai"

FIELDNAMES = [
    "name",
    "paper_link",
    "paper_title",
    "arxiv_category",
    "published_date",
    "institution",
    "institution_type",
    "country",
    "city",
    "linkedin_search_url",
    "google_scholar_url",
    "personal_website",
]


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------

def build_arxiv_query(query: str, categories: list[str]) -> str:
    cat_filter = " OR ".join(f"cat:{c}" for c in categories)
    return f'all:"{query}" AND ({cat_filter})'


def extract_arxiv_id(entry_id: str) -> str:
    m = re.search(r"(\d{4}\.\d{4,5})", entry_id)
    return m.group(1) if m else ""


def fetch_papers(query: str, start_date: datetime, end_date: datetime, max_results: int = 500) -> list[arxiv.Result]:
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


# ---------------------------------------------------------------------------
# OpenAlex enrichment (direct keyword search with institution data)
# ---------------------------------------------------------------------------

def openalex_search(query: str, start_date: str, end_date: str, country_code: str | None = None, per_page: int = 200) -> list[dict]:
    """Search OpenAlex for works matching query with optional country filter. Paginates through all results."""
    all_results = []
    cursor = "*"
    filters = [f"from_publication_date:{start_date}", f"to_publication_date:{end_date}"]
    if country_code:
        filters.append(f"institutions.country_code:{country_code}")

    while cursor:
        try:
            params = {
                "search": query,
                "filter": ",".join(filters),
                "select": "doi,title,authorships,publication_date",
                "per_page": min(per_page, 200),
                "cursor": cursor,
                "mailto": OPENALEX_EMAIL,
            }
            resp = requests.get(f"{OPENALEX_API}/works", params=params, timeout=30)
            if resp.status_code == 429:
                log.warning("OpenAlex rate limited, sleeping 5s")
                time.sleep(5)
                continue
            if resp.status_code != 200:
                log.warning(f"OpenAlex returned {resp.status_code}")
                break
            data = resp.json()
            results = data.get("results", [])
            all_results.extend(results)
            cursor = data.get("meta", {}).get("next_cursor")
            if not results:
                break
            time.sleep(0.2)
        except Exception as e:
            log.warning(f"OpenAlex search failed: {e}")
            break

    return all_results


def extract_us_authors_from_openalex(works: list[dict]) -> list[dict]:
    """Extract US-affiliated author rows from OpenAlex works."""
    rows = []
    seen = set()
    for work in works:
        title = (work.get("title") or "").replace("\n", " ").strip()
        doi = work.get("doi") or ""
        pub_date = work.get("publication_date") or ""
        paper_link = doi if doi else ""

        for auth in work.get("authorships", []):
            countries = auth.get("countries", [])
            if "US" not in countries:
                continue

            name = auth.get("author", {}).get("display_name") or auth.get("raw_author_name", "")
            if not name:
                continue

            key = (name, title)
            if key in seen:
                continue
            seen.add(key)

            # Get US institution info
            institution = ""
            institution_type = ""
            city = ""
            for inst in auth.get("institutions", []):
                if inst.get("country_code") == "US":
                    institution = inst.get("display_name", "")
                    institution_type = inst.get("type", "")
                    break

            # City from raw affiliation
            raw_strings = auth.get("raw_affiliation_strings", [])
            if raw_strings:
                parts = [p.strip() for p in raw_strings[0].split(",")]
                if len(parts) >= 3:
                    city = parts[-2]

            rows.append({
                "name": name,
                "paper_link": paper_link,
                "paper_title": title,
                "arxiv_category": "",
                "published_date": pub_date,
                "institution": institution,
                "institution_type": institution_type,
                "country": "US",
                "city": city,
                "linkedin_search_url": build_linkedin_search_url(name),
                "google_scholar_url": build_google_scholar_url(name),
                "personal_website": "",
            })
    return rows


# ---------------------------------------------------------------------------
# Search URL builders
# ---------------------------------------------------------------------------

def build_linkedin_search_url(name: str) -> str:
    q = urllib.parse.quote_plus(f"{name} reinforcement learning OR machine learning")
    return f"https://www.google.com/search?q=site%3Alinkedin.com%2Fin+{q}"


def build_google_scholar_url(name: str) -> str:
    q = urllib.parse.quote_plus(f'author:"{name}"')
    return f"https://scholar.google.com/scholar?q={q}"


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def load_existing_data(csv_path: Path) -> tuple[list[dict], set[str]]:
    rows = []
    seen = set()
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                seen.add((row["name"], row["paper_link"]))
    return rows, seen


def save_data(rows: list[dict], csv_path: Path):
    DATA_DIR.mkdir(exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        rows.sort(key=lambda r: (r.get("published_date", ""), r.get("name", "")), reverse=True)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def scrape(start_date: datetime, end_date: datetime, enrich: bool = True):
    existing_all, seen_all = load_existing_data(CSV_ALL)
    existing_us, seen_us = load_existing_data(CSV_US)
    papers_found = 0

    # ---- Phase 1: arXiv papers (all countries, no institution data) ----
    all_papers: list[arxiv.Result] = []
    paper_set: set[str] = set()

    for query in SEARCH_QUERIES:
        log.info(f"[arXiv] Searching: {query}")
        papers = fetch_papers(query, start_date, end_date, max_results=200)
        log.info(f"  Found {len(papers)} papers")
        papers_found += len(papers)
        for p in papers:
            aid = extract_arxiv_id(p.entry_id)
            if aid and aid not in paper_set:
                paper_set.add(aid)
                all_papers.append(p)

    log.info(f"[arXiv] Total unique papers: {len(all_papers)}")

    # Build rows from arXiv (no institution data)
    new_all_rows = []
    for paper in all_papers:
        paper_link = paper.entry_id
        categories = ",".join(paper.categories[:3])
        pub_date = paper.published.strftime("%Y-%m-%d")
        for author in paper.authors:
            name = str(author)
            key = (name, paper_link)
            if key in seen_all:
                continue
            seen_all.add(key)
            new_all_rows.append({
                "name": name,
                "paper_link": paper_link,
                "paper_title": paper.title.replace("\n", " ").strip(),
                "arxiv_category": categories,
                "published_date": pub_date,
                "institution": "",
                "institution_type": "",
                "country": "",
                "city": "",
                "linkedin_search_url": build_linkedin_search_url(name),
                "google_scholar_url": build_google_scholar_url(name),
                "personal_website": "",
            })

    # ---- Phase 2: OpenAlex US researchers (rich institution data) ----
    new_us_rows = []
    if enrich:
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        for query in OPENALEX_QUERIES:
            log.info(f"[OpenAlex] Searching US: {query}")
            works = openalex_search(query, start_str, end_str, country_code="US")
            us_rows = extract_us_authors_from_openalex(works)
            # Dedup against existing US data
            for row in us_rows:
                key = (row["name"], row["paper_title"])
                if key not in seen_us:
                    seen_us.add(key)
                    new_us_rows.append(row)
            log.info(f"  {len(works)} works -> {len(us_rows)} US author entries (new: {len(new_us_rows)})")

    # Save all data
    all_rows = existing_all + new_all_rows
    save_data(all_rows, CSV_ALL)

    us_rows_total = existing_us + new_us_rows
    save_data(us_rows_total, CSV_US)

    log.info(
        f"Done. arXiv papers: {papers_found} (unique: {len(all_papers)}). "
        f"New all entries: {len(new_all_rows)}. Total all: {len(all_rows)}. "
        f"New US entries: {len(new_us_rows)}. Total US: {len(us_rows_total)}"
    )


def main():
    parser = argparse.ArgumentParser(description="Scrape arXiv + OpenAlex for researcher profiles")
    parser.add_argument("--days", type=int, default=1, help="Number of days to look back (default: 1 for nightly)")
    parser.add_argument("--backfill-months", type=int, default=0, help="Backfill N months of data")
    parser.add_argument("--no-enrich", action="store_true", help="Skip OpenAlex US enrichment (faster)")
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
