#!/usr/bin/env python3
"""
US researcher scraper for RL, post-training, world models, and environment simulation.

Two strategies, both via OpenAlex:
1. Conference authors — pull all authors from top ML/RL venues (NeurIPS, ICML, ICLR, etc.)
2. Lab authors — pull all authors affiliated with known AI research labs

Filters to US-based, non-university researchers. No LLM needed — the venue IS the
relevance signal.
"""

import argparse
import csv
import logging
import re
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "researchers.csv"

OPENALEX_API = "https://api.openalex.org"
OPENALEX_EMAIL = "scraper@fleet.ai"

# --- Top ML/RL conferences (OpenAlex source IDs) ---
CONFERENCES = {
    "S4306420609": "NeurIPS",
    "S4306419644": "ICML",
    "S4306419637": "ICLR",
    "S4210191458": "AAAI",
    "S4306419146": "AISTATS",
    "S4306506823": "CoRL",
    "S4306420803": "RSS",
    "S4306420508": "ACL",
    "S4306418267": "EMNLP",
}

# --- Known AI research labs in the US (OpenAlex institution IDs) ---
AI_LABS = {
    "I1291425158": "Google",
    "I4210114444": "Meta",
    "I4210161460": "OpenAI",
    "I4210127875": "NVIDIA",
    "I4210153776": "Apple",
    "I1311688040": "Amazon",
    "I4210156221": "Allen AI (AI2)",
    "I4391768151": "Toyota Research Institute",
    "I4210114115": "IBM Research (Watson)",
    "I4210085935": "IBM Research (Almaden)",
    "I4387154989": "Hugging Face",
    # DeepMind US papers often list Google as institution
}

# OpenAlex topic subfields — restrict lab papers to AI/ML/CV
# https://docs.openalex.org/api-entities/topics
AI_CV_SUBFIELDS = "subfields/1702|subfields/1707"  # Artificial Intelligence | Computer Vision

# Venue keywords to exclude (repositories, medical, non-CS journals)
EXCLUDED_VENUE_KEYWORDS = {
    "zenodo", "cancer", "medicine", "medical", "genetics", "genomics",
    "quantum information", "solid-state circuits", "social science",
    "diabetes", "clinical", "biomedical", "health informatics",
    "pharmaceutical", "epidemiol",
}

# Institution types to EXCLUDE (we want industry/lab, not tenured faculty)
EXCLUDED_INSTITUTION_TYPES = {"education", "healthcare"}

# Regex to detect non-human "author" names (AI models listed as authors on Zenodo etc.)
_BAD_AUTHOR_RE = re.compile(r"\(.*\)|^\d|^[A-Z]{2,}\d|GPT|Gemini|Claude|LLaMA|Llama|Mistral|Copilot", re.IGNORECASE)

FIELDNAMES = [
    "name",
    "paper_link",
    "paper_title",
    "published_date",
    "venue",
    "institution",
    "institution_type",
    "city",
    "linkedin_search_url",
    "google_scholar_url",
]


# ---------------------------------------------------------------------------
# OpenAlex queries
# ---------------------------------------------------------------------------

def openalex_paginate(filter_str: str, select: str = "doi,title,authorships,publication_date,primary_location") -> list[dict]:
    """Paginate through OpenAlex works matching a filter."""
    all_results = []
    cursor = "*"

    while cursor:
        try:
            resp = requests.get(
                f"{OPENALEX_API}/works",
                params={
                    "filter": filter_str,
                    "select": select,
                    "per_page": 200,
                    "cursor": cursor,
                    "mailto": OPENALEX_EMAIL,
                },
                timeout=30,
            )
            if resp.status_code == 429:
                log.warning("OpenAlex rate limited, sleeping 5s")
                time.sleep(5)
                continue
            if resp.status_code != 200:
                log.warning(f"OpenAlex returned {resp.status_code}: {resp.text[:200]}")
                break
            data = resp.json()
            results = data.get("results", [])
            all_results.extend(results)
            cursor = data.get("meta", {}).get("next_cursor")
            if not results:
                break
            time.sleep(0.15)
        except Exception as e:
            log.warning(f"OpenAlex query failed: {e}")
            break

    return all_results


def fetch_conference_papers(source_id: str, start_date: str, end_date: str) -> list[dict]:
    """Fetch all papers from a conference venue within date range."""
    filt = f"primary_location.source.id:{source_id},type:article,from_publication_date:{start_date},to_publication_date:{end_date}"
    return openalex_paginate(filt)


def fetch_lab_papers(institution_id: str, start_date: str, end_date: str) -> list[dict]:
    """Fetch AI/ML/CV papers from a specific institution within date range."""
    filt = (
        f"authorships.institutions.id:{institution_id},"
        f"topics.subfield.id:{AI_CV_SUBFIELDS},"
        f"type:article,"
        f"from_publication_date:{start_date},to_publication_date:{end_date}"
    )
    return openalex_paginate(filt)


# ---------------------------------------------------------------------------
# Author extraction
# ---------------------------------------------------------------------------

def extract_us_nonacademic_authors(works: list[dict], seen: set[str], venue: str = "") -> list[dict]:
    """Extract US-based non-university authors from OpenAlex works."""
    rows = []
    for work in works:
        title = (work.get("title") or "").replace("\n", " ").strip()
        if not title:
            continue
        doi = work.get("doi") or ""
        pub_date = work.get("publication_date") or ""

        # Get venue from primary_location if not provided
        work_venue = venue
        if not work_venue:
            loc = work.get("primary_location") or {}
            src = loc.get("source") or {}
            work_venue = src.get("display_name", "")

        # Skip papers from noisy repositories
        if any(kw in work_venue.lower() for kw in EXCLUDED_VENUE_KEYWORDS):
            continue

        for auth in work.get("authorships", []):
            if "US" not in auth.get("countries", []):
                continue

            name = auth.get("author", {}).get("display_name") or auth.get("raw_author_name", "")
            if not name or _BAD_AUTHOR_RE.search(name):
                continue

            # Find the US institution
            institution = ""
            institution_type = ""
            city = ""
            for inst in auth.get("institutions", []):
                if inst.get("country_code") == "US":
                    institution = inst.get("display_name", "")
                    institution_type = inst.get("type", "")
                    break

            # Skip universities and hospitals
            if institution_type in EXCLUDED_INSTITUTION_TYPES:
                continue

            key = (name, title)
            if key in seen:
                continue
            seen.add(key)

            raw_strings = auth.get("raw_affiliation_strings", [])
            if raw_strings:
                parts = [p.strip() for p in raw_strings[0].split(",")]
                if len(parts) >= 3:
                    city = parts[-2]

            rows.append({
                "name": name,
                "paper_link": doi,
                "paper_title": title,
                "published_date": pub_date,
                "venue": work_venue,
                "institution": institution,
                "institution_type": institution_type,
                "city": city,
                "linkedin_search_url": build_linkedin_search_url(name),
                "google_scholar_url": build_google_scholar_url(name),
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

def load_existing_data() -> tuple[list[dict], set[str]]:
    rows = []
    seen = set()
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                seen.add((row["name"], row["paper_title"]))
    return rows, seen


def save_data(rows: list[dict]):
    DATA_DIR.mkdir(exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        rows.sort(key=lambda r: (r.get("published_date", ""), r.get("name", "")), reverse=True)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def scrape(start_date: datetime, end_date: datetime):
    existing_rows, seen = load_existing_data()
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    new_rows = []

    # Phase 1: Conference papers
    for source_id, conf_name in CONFERENCES.items():
        log.info(f"[Conference] {conf_name} ({source_id})")
        works = fetch_conference_papers(source_id, start_str, end_str)
        rows = extract_us_nonacademic_authors(works, seen, venue=conf_name)
        new_rows.extend(rows)
        log.info(f"  {len(works)} papers -> {len(rows)} new US industry entries")

    # Phase 2: Known AI lab papers
    for inst_id, lab_name in AI_LABS.items():
        log.info(f"[Lab] {lab_name} ({inst_id})")
        works = fetch_lab_papers(inst_id, start_str, end_str)
        rows = extract_us_nonacademic_authors(works, seen)
        new_rows.extend(rows)
        log.info(f"  {len(works)} papers -> {len(rows)} new entries")

    all_rows = existing_rows + new_rows
    save_data(all_rows)

    unique_names = len(set(r["name"] for r in all_rows))
    log.info(f"Done. New: {len(new_rows)}. Total: {len(all_rows)}. Unique researchers: {unique_names}")


def main():
    parser = argparse.ArgumentParser(description="Scrape OpenAlex for US-based industry ML researchers from top venues & labs")
    parser.add_argument("--days", type=int, default=2, help="Days to look back (default: 2)")
    parser.add_argument("--backfill-months", type=int, default=0, help="Backfill N months")
    args = parser.parse_args()

    end_date = datetime.now(timezone.utc).replace(tzinfo=None)

    if args.backfill_months > 0:
        start_date = end_date - timedelta(days=args.backfill_months * 30)
        log.info(f"Backfilling {args.backfill_months} months: {start_date.date()} to {end_date.date()}")
    else:
        start_date = end_date - timedelta(days=args.days)
        log.info(f"Scraping last {args.days} day(s): {start_date.date()} to {end_date.date()}")

    scrape(start_date, end_date)


if __name__ == "__main__":
    main()
