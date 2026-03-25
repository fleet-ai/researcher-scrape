#!/usr/bin/env python3
"""
US industry ML researcher discovery pipeline.

Three data sources:
1. Conference websites — scrape NeurIPS/ICML/ICLR accepted paper JSONs directly
2. OpenAlex conferences — other venues (AAAI, AISTATS, CoRL, RSS, ACL, EMNLP)
3. OpenAlex lab papers — AI/ML papers from known research labs

Then:
- Filter to US-based, non-academic researchers
- Fetch OpenAlex author profiles for h-index and citation metrics
- Filter to h-index 5-80 (recruitable range: not too junior, not whales)
- LLM relevance filter via Sonnet: keeps only RL, post-training, world models, env sim researchers
- Output is researcher-centric (one row per person), ranked by priority score
"""

import argparse
import csv
import json
import logging
import math
import os
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

# --- Conference website JSON endpoints (NeurIPS/ICML/ICLR) ---
# These conferences moved to OpenReview and are no longer indexed by OpenAlex.
# We scrape accepted paper data directly from the conference virtual site JSONs.
CONFERENCE_JSONS = {
    "NeurIPS 2025": "https://neurips.cc/static/virtual/data/neurips-2025-orals-posters.json",
    "ICML 2025": "https://icml.cc/static/virtual/data/icml-2025-orals-posters.json",
    "ICLR 2025": "https://iclr.cc/static/virtual/data/iclr-2025-orals-posters.json",
}

# --- Other conferences via OpenAlex (source IDs) ---
OPENALEX_CONFERENCES = {
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
    "I4210161460": "OpenAI",
    "I4210127875": "NVIDIA",
    "I4210153776": "Apple",
    "I4210156221": "Allen AI (AI2)",
    "I4391768151": "Toyota Research Institute",
    "I4210114115": "IBM Research (Watson)",
    "I4210085935": "IBM Research (Almaden)",
    "I4387154989": "Hugging Face",
}

# US industry institution patterns for conference website filtering.
# Matched case-insensitively against institution names from conference data.
US_INDUSTRY_PATTERNS = [
    "google", "deepmind", "meta ai", "meta fair", "facebook", "openai", "nvidia",
    "apple", "amazon", "aws ", "microsoft", "allen ai", "ai2 ", "hugging face",
    "huggingface", "anthropic", "cohere", "databricks", "mosaic", "together ai", "allen institute",
    "anyscale", "salesforce", "adobe", "intel labs", "intel corporation",
    "qualcomm", "ibm research", "toyota research", "samsung research",
    "sony ai", "sonyai", "waymo", "cruise", "aurora innovation", "zoox",
    "nuro", "argo ai", "scale ai", "character ai", "character.ai",
    "inflection", "xai", "x.ai", "stability ai", "midjourney", "runway",
    "palantir", "tesla", "uber", "lyft", "snap inc", "spotify", "netflix",
    "oracle", "bloomberg", "two sigma", "citadel", "jane street", "de shaw",
    "renaissance tech", "jump trading", "hudson river",
    "jpmorgan", "jp morgan", "goldman sachs", "morgan stanley",
    "boeing", "lockheed", "raytheon", "northrop", "general dynamics",
    "johns hopkins apl", "jhuapl", "mitre", "lincoln lab", "lincoln laboratory",
    "sandia", "los alamos", "llnl", "argonne", "oak ridge", "brookhaven",
    "pacific northwest", "national renewable",
]

# Patterns that must match as whole words (not substrings)
_US_INDUSTRY_EXACT = {"meta", "aws", "ai2"}
_US_INDUSTRY_WORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _US_INDUSTRY_EXACT) + r")\b", re.IGNORECASE
)

# Academic institution keywords to exclude (for conference website data)
ACADEMIC_KEYWORDS = [
    "university", "université", "universität", "universidad", "universidade",
    "universita", "college", "school of", "institute of technology",
    "polytechnic", "école", "eth zurich", "epfl", "kaist", "postech",
    "tsinghua", "peking", "fudan", "zhejiang", "nanjing", "shanghai jiao",
    "seoul national", "tokyo", "kyoto", "osaka", "oxford", "cambridge",
    "imperial college", "ucl", "edinburgh", "inria", "cnrs", "max planck",
    "chinese academy", "cas ", "academia sinica",
]

# OpenAlex topic subfields — restrict lab papers to AI/ML/CV
AI_CV_SUBFIELDS = "subfields/1702|subfields/1707"

# Venue keywords to exclude (repositories, medical, non-CS journals)
EXCLUDED_VENUE_KEYWORDS = {
    "zenodo", "cancer", "medicine", "medical", "genetics", "genomics",
    "quantum information", "solid-state circuits", "social science",
    "diabetes", "clinical", "biomedical", "health informatics",
    "pharmaceutical", "epidemiol",
}

# Institution types to EXCLUDE
EXCLUDED_INSTITUTION_TYPES = {"education", "healthcare"}

# Regex to detect non-human "author" names
_BAD_AUTHOR_RE = re.compile(r"\(.*\)|^\d|^[A-Z]{2,}\d|GPT|Gemini|Claude|LLaMA|Llama|Mistral|Copilot", re.IGNORECASE)

# h-index range for recruitable researchers
H_INDEX_MIN = 5
H_INDEX_MAX = 80

# LLM relevance filter config
OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "anthropic/claude-sonnet-4.5"

RELEVANCE_PROMPT = """You are filtering researchers for a hiring pipeline. We want people who work on:

1. **Reinforcement Learning** (RL, RLHF, GRPO, DPO, PPO, policy optimization, reward modeling, multi-agent RL, offline RL)
2. **Post-training** (alignment, preference optimization, instruction tuning, constitutional AI, red-teaming)
3. **World models** (model-based RL, learned simulators, predictive models of environments)
4. **Environment simulation** (sim-to-real, robotics environments, physics simulation for RL, procedural generation)
5. **LLM training & optimization** (pretraining at scale, efficient training, distributed training, training infrastructure)
6. **Agentic AI** (tool use, code generation agents, autonomous agents, planning, reasoning chains)

We do NOT want: security researchers, hardware/chip designers, quantum computing, HCI/UX, bioinformatics, networking, databases (unless ML-for-DB), pure NLP linguistics, pure computer vision with no RL/agent component.

For each researcher below, respond with ONLY a JSON array of objects with "id" (the researcher number) and "relevant" (true/false). No explanation needed.

Researchers:
"""

FIELDNAMES = [
    "priority_score",
    "name",
    "openalex_id",
    "h_index",
    "i10_index",
    "cited_by_count",
    "works_count",
    "2yr_mean_citedness",
    "institution",
    "institution_type",
    "city",
    "paper_count_in_window",
    "top_paper_title",
    "top_paper_link",
    "venues",
    "linkedin_search_url",
    "google_scholar_url",
]


# ---------------------------------------------------------------------------
# Conference website scraping (NeurIPS, ICML, ICLR)
# ---------------------------------------------------------------------------

def _is_us_industry(institution: str) -> bool:
    """Check if an institution name looks like a US industry/lab (not academic)."""
    inst_lower = institution.lower().strip()
    if not inst_lower or inst_lower == "none":
        return False
    # Exclude academic institutions
    if any(kw in inst_lower for kw in ACADEMIC_KEYWORDS):
        return False
    # Match known US industry labs (substring patterns)
    if any(pat in inst_lower for pat in US_INDUSTRY_PATTERNS):
        return True
    # Match whole-word patterns (e.g. "Meta" but not "metadata")
    if _US_INDUSTRY_WORD_RE.search(institution):
        return True
    return False


def fetch_conference_website_papers(conf_name: str, url: str) -> list[dict]:
    """Fetch accepted papers from a conference virtual site JSON endpoint."""
    log.info(f"[Conference Website] Fetching {conf_name} from {url}")
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        log.info(f"  {len(results)} papers fetched")
        return results
    except Exception as e:
        log.warning(f"  Failed to fetch {conf_name}: {e}")
        return []


def collect_authors_from_conference_json(papers: list[dict], authors: dict, venue: str):
    """Extract US industry authors from conference website JSON into the authors dict.

    These authors don't have OpenAlex IDs yet — we use their name as a temporary key
    prefixed with 'conf:' to distinguish from OpenAlex-sourced authors.
    """
    for paper in papers:
        title = (paper.get("name") or "").replace("\n", " ").strip()
        if not title:
            continue

        for auth in paper.get("authors", []):
            name = (auth.get("fullname") or "").strip()
            institution = (auth.get("institution") or "").strip()
            if not name or _BAD_AUTHOR_RE.search(name):
                continue
            if not _is_us_industry(institution):
                continue

            paper_info = {"title": title, "doi": "", "date": "2025", "venue": venue}
            # Use name+institution as a composite key for dedup
            key = f"conf:{name.lower()}:{institution.lower()}"

            if key not in authors:
                authors[key] = {
                    "name": name,
                    "institution": institution,
                    "institution_type": "company",
                    "city": "",
                    "papers": [paper_info],
                    "_from_conference_site": True,
                }
            else:
                authors[key]["papers"].append(paper_info)


def merge_conference_authors_with_openalex(authors: dict) -> dict:
    """Merge conference website authors with OpenAlex authors where they overlap.

    If an OpenAlex-keyed author has the same name+institution as a conf:-keyed author,
    merge their papers. Conference-only authors are kept as-is (no OpenAlex resolution
    needed — acceptance at NeurIPS/ICML/ICLR is a strong quality signal).
    """
    conf_authors = {k: v for k, v in authors.items() if k.startswith("conf:")}
    oa_authors = {k: v for k, v in authors.items() if not k.startswith("conf:")}

    if not conf_authors:
        return authors

    # Build lookup: lowercase name -> list of OpenAlex author IDs
    name_to_oa = {}
    for aid, info in oa_authors.items():
        key = info["name"].lower()
        name_to_oa.setdefault(key, []).append(aid)

    merged = 0
    for conf_key, conf_info in conf_authors.items():
        name_lower = conf_info["name"].lower()
        if name_lower in name_to_oa:
            # Merge papers into existing OpenAlex author
            target_aid = name_to_oa[name_lower][0]
            oa_authors[target_aid]["papers"].extend(conf_info["papers"])
            merged += 1
        else:
            # Keep as conference-only author
            oa_authors[conf_key] = conf_info

    log.info(f"  Merged {merged} conference authors with existing OpenAlex entries; {len(conf_authors) - merged} conference-only")
    return oa_authors


# ---------------------------------------------------------------------------
# OpenAlex helpers
# ---------------------------------------------------------------------------

def openalex_get(endpoint: str, params: dict) -> dict | None:
    """Single OpenAlex API call with exponential backoff on rate limit."""
    params["mailto"] = OPENALEX_EMAIL
    for attempt in range(5):
        try:
            resp = requests.get(f"{OPENALEX_API}/{endpoint}", params=params, timeout=30)
            if resp.status_code == 429:
                wait = min(5 * (2 ** attempt), 60)
                log.warning(f"Rate limited, sleeping {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                log.warning(f"OpenAlex {resp.status_code}: {resp.text[:200]}")
                return None
            return resp.json()
        except Exception as e:
            log.warning(f"OpenAlex request failed: {e}")
            return None
    return None


def openalex_paginate(filter_str: str, select: str = "doi,title,authorships,publication_date,primary_location") -> list[dict]:
    """Paginate through OpenAlex works matching a filter."""
    all_results = []
    cursor = "*"

    while cursor:
        data = openalex_get("works", {
            "filter": filter_str,
            "select": select,
            "per_page": 200,
            "cursor": cursor,
        })
        if not data:
            break
        results = data.get("results", [])
        all_results.extend(results)
        cursor = data.get("meta", {}).get("next_cursor")
        if not results:
            break
        time.sleep(0.15)

    return all_results


def fetch_conference_papers(source_id: str, start_date: str, end_date: str) -> list[dict]:
    filt = f"primary_location.source.id:{source_id},type:article,from_publication_date:{start_date},to_publication_date:{end_date}"
    return openalex_paginate(filt)


def fetch_lab_papers(institution_id: str, start_date: str, end_date: str) -> list[dict]:
    filt = (
        f"authorships.institutions.id:{institution_id},"
        f"topics.subfield.id:{AI_CV_SUBFIELDS},"
        f"type:article,"
        f"from_publication_date:{start_date},to_publication_date:{end_date}"
    )
    return openalex_paginate(filt)


def fetch_author_profiles(author_ids: list[str]) -> dict[str, dict]:
    """Batch-fetch author profiles from OpenAlex. Returns {author_id: profile}."""
    profiles = {}
    batch_size = 50
    id_list = list(author_ids)

    for i in range(0, len(id_list), batch_size):
        batch = id_list[i:i + batch_size]
        filter_str = "id:" + "|".join(batch)
        data = openalex_get("authors", {
            "filter": filter_str,
            "select": "id,display_name,summary_stats,cited_by_count,works_count,last_known_institutions",
            "per_page": 50,
        })
        if data:
            for author in data.get("results", []):
                aid = author["id"].split("/")[-1]
                profiles[aid] = author
        time.sleep(0.15)
        if (i // batch_size) % 20 == 0 and i > 0:
            log.info(f"  Fetched {i}/{len(id_list)} author profiles")

    return profiles


# ---------------------------------------------------------------------------
# Author extraction
# ---------------------------------------------------------------------------

def collect_authors_from_works(works: list[dict], authors: dict, venue: str = ""):
    """Extract US non-academic authors from works into the authors dict."""
    for work in works:
        title = (work.get("title") or "").replace("\n", " ").strip()
        if not title:
            continue
        doi = work.get("doi") or ""
        pub_date = work.get("publication_date") or ""

        work_venue = venue
        if not work_venue:
            loc = work.get("primary_location") or {}
            src = loc.get("source") or {}
            work_venue = src.get("display_name", "")

        if any(kw in work_venue.lower() for kw in EXCLUDED_VENUE_KEYWORDS):
            continue

        for auth in work.get("authorships", []):
            if "US" not in auth.get("countries", []):
                continue

            author_obj = auth.get("author") or {}
            author_id = (author_obj.get("id") or "").split("/")[-1]
            name = author_obj.get("display_name") or auth.get("raw_author_name", "")
            if not name or not author_id or _BAD_AUTHOR_RE.search(name):
                continue

            institution = ""
            institution_type = ""
            city = ""
            for inst in auth.get("institutions", []):
                if inst.get("country_code") == "US":
                    institution = inst.get("display_name", "")
                    institution_type = inst.get("type", "")
                    break

            if institution_type in EXCLUDED_INSTITUTION_TYPES:
                continue

            raw_strings = auth.get("raw_affiliation_strings", [])
            if raw_strings:
                parts = [p.strip() for p in raw_strings[0].split(",")]
                if len(parts) >= 3:
                    city = parts[-2]

            paper = {"title": title, "doi": doi, "date": pub_date, "venue": work_venue}

            if author_id not in authors:
                authors[author_id] = {
                    "name": name,
                    "institution": institution,
                    "institution_type": institution_type,
                    "city": city,
                    "papers": [paper],
                }
            else:
                existing = authors[author_id]
                existing["papers"].append(paper)
                if pub_date > (existing["papers"][0].get("date") or ""):
                    existing["institution"] = institution or existing["institution"]
                    existing["institution_type"] = institution_type or existing["institution_type"]
                    existing["city"] = city or existing["city"]


# ---------------------------------------------------------------------------
# LLM relevance filter
# ---------------------------------------------------------------------------

def filter_relevant_researchers(researchers: list[dict], api_key: str) -> list[dict]:
    """Use Sonnet via OpenRouter to filter researchers by topic relevance.

    Each researcher dict must have 'name', 'papers' (list of paper dicts with 'title').
    Returns only researchers deemed relevant.
    """
    batch_size = 25
    relevant_ids = set()
    total = len(researchers)

    for i in range(0, total, batch_size):
        batch = researchers[i:i + batch_size]

        # Build the researcher list for the prompt
        lines = []
        for j, r in enumerate(batch):
            titles = [p["title"] for p in r["papers"][:5]]  # Up to 5 paper titles
            titles_str = " | ".join(titles)
            lines.append(f"{j+1}. {r['name']} @ {r['institution']} — Papers: {titles_str}")

        prompt = RELEVANCE_PROMPT + "\n".join(lines)

        for attempt in range(3):
            try:
                resp = requests.post(
                    OPENROUTER_API,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": OPENROUTER_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                        "max_tokens": 2000,
                    },
                    timeout=60,
                )
                if resp.status_code == 429:
                    log.warning("OpenRouter rate limited, sleeping 10s")
                    time.sleep(10)
                    continue
                if resp.status_code != 200:
                    log.warning(f"OpenRouter {resp.status_code}: {resp.text[:200]}")
                    break

                content = resp.json()["choices"][0]["message"]["content"]
                # Extract JSON array from response (handle markdown code blocks)
                content = content.strip()
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\s*", "", content)
                    content = re.sub(r"\s*```$", "", content)

                results = json.loads(content)
                for item in results:
                    if item.get("relevant"):
                        idx = item["id"] - 1  # 1-indexed to 0-indexed
                        if 0 <= idx < len(batch):
                            relevant_ids.add(id(batch[idx]))
                break
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                log.warning(f"Failed to parse LLM response (attempt {attempt+1}): {e}")
                if attempt < 2:
                    time.sleep(2)
                    continue
                # On final failure, include all in batch (fail open)
                log.warning("  Including entire batch as fallback")
                for r in batch:
                    relevant_ids.add(id(r))
            except Exception as e:
                log.warning(f"OpenRouter request failed: {e}")
                break

        kept = sum(1 for r in batch if id(r) in relevant_ids)
        log.info(f"  LLM filter batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}: {kept}/{len(batch)} relevant")
        time.sleep(0.5)

    return [r for r in researchers if id(r) in relevant_ids]


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

def save_data(rows: list[dict]):
    DATA_DIR.mkdir(exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def scrape(start_date: datetime, end_date: datetime, skip_llm: bool = False, conferences_only: bool = False):
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # {openalex_author_id or conf:key: {name, institution, ..., papers: [...]}}
    authors: dict[str, dict] = {}

    # Phase 1: Conference website papers (NeurIPS, ICML, ICLR)
    for conf_name, json_url in CONFERENCE_JSONS.items():
        papers = fetch_conference_website_papers(conf_name, json_url)
        before = len(authors)
        collect_authors_from_conference_json(papers, authors, venue=conf_name)
        log.info(f"  -> {len(authors) - before} US industry researchers extracted")

    if not conferences_only:
        # Phase 2: OpenAlex conference papers (AAAI, AISTATS, CoRL, etc.)
        for source_id, conf_name in OPENALEX_CONFERENCES.items():
            log.info(f"[OpenAlex Conference] {conf_name} ({source_id})")
            works = fetch_conference_papers(source_id, start_str, end_str)
            before = len(authors)
            collect_authors_from_works(works, authors, venue=conf_name)
            log.info(f"  {len(works)} papers -> {len(authors) - before} new researchers")

        # Phase 3: Known AI lab papers via OpenAlex
        for inst_id, lab_name in AI_LABS.items():
            log.info(f"[Lab] {lab_name} ({inst_id})")
            works = fetch_lab_papers(inst_id, start_str, end_str)
            before = len(authors)
            collect_authors_from_works(works, authors)
            log.info(f"  {len(works)} papers -> {len(authors) - before} new researchers")

    log.info(f"Total unique researchers found: {len(authors)}")

    # Phase 4: Merge conference website authors with OpenAlex authors
    authors = merge_conference_authors_with_openalex(authors)

    # Phase 5: Fetch author profiles for OpenAlex-keyed authors
    profiles = {}
    if not conferences_only:
        openalex_ids = [k for k in authors if not k.startswith("conf:")]
        ids_needing_profiles = [k for k in openalex_ids if "_profile" not in authors[k]]
        log.info(f"Fetching author profiles for {len(ids_needing_profiles)} researchers...")
        profiles = fetch_author_profiles(ids_needing_profiles)
        log.info(f"  Got {len(profiles)} profiles")

    # Phase 6: h-index filter for OpenAlex authors; conference-only authors pass through
    filtered_authors = {}
    conf_only_count = 0
    for author_id, info in authors.items():
        if author_id.startswith("conf:"):
            # Conference-only authors: NeurIPS/ICML/ICLR acceptance is a quality signal
            info["_profile"] = {}
            info["_h_index"] = 0
            filtered_authors[author_id] = info
            conf_only_count += 1
            continue
        profile = info.get("_profile") or profiles.get(author_id, {})
        h = (profile.get("summary_stats") or {}).get("h_index") or 0
        if H_INDEX_MIN <= h <= H_INDEX_MAX:
            info["_profile"] = profile
            info["_h_index"] = h
            filtered_authors[author_id] = info

    log.info(f"After h-index filter ({H_INDEX_MIN}-{H_INDEX_MAX}): {len(filtered_authors)} researchers ({conf_only_count} conference-only)")

    # Phase 5: LLM relevance filter
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not skip_llm and openrouter_key:
        researcher_list = [
            {"id": aid, **info}
            for aid, info in filtered_authors.items()
        ]
        log.info(f"Running LLM relevance filter on {len(researcher_list)} researchers...")
        relevant = filter_relevant_researchers(researcher_list, openrouter_key)
        relevant_ids = {r["id"] for r in relevant}
        filtered_authors = {aid: info for aid, info in filtered_authors.items() if aid in relevant_ids}
        log.info(f"After LLM filter: {len(filtered_authors)} researchers")
    elif not skip_llm:
        log.warning("OPENROUTER_API_KEY not set — skipping LLM relevance filter")

    # Build final rows
    rows = []
    for author_id, info in filtered_authors.items():
        profile = info["_profile"]
        stats = profile.get("summary_stats", {})

        papers = sorted(info["papers"], key=lambda p: p.get("date", ""), reverse=True)
        top_paper = papers[0] if papers else {}
        venues = sorted(set(p["venue"] for p in papers if p["venue"]))

        h = info["_h_index"]
        citedness_2yr = stats.get("2yr_mean_citedness") or 0
        n_papers = len(papers)

        score = round(h + n_papers * 10 + math.log1p(citedness_2yr) * 15, 1)

        rows.append({
            "priority_score": score,
            "name": info["name"],
            "openalex_id": author_id,
            "h_index": h,
            "i10_index": stats.get("i10_index", ""),
            "cited_by_count": profile.get("cited_by_count", ""),
            "works_count": profile.get("works_count", ""),
            "2yr_mean_citedness": round(citedness_2yr, 2) if citedness_2yr else "",
            "institution": info["institution"],
            "institution_type": info["institution_type"],
            "city": info["city"],
            "paper_count_in_window": n_papers,
            "top_paper_title": top_paper.get("title", ""),
            "top_paper_link": top_paper.get("doi", ""),
            "venues": "; ".join(venues),
            "linkedin_search_url": build_linkedin_search_url(info["name"]),
            "google_scholar_url": build_google_scholar_url(info["name"]),
        })

    rows.sort(key=lambda r: r["priority_score"], reverse=True)

    save_data(rows)
    log.info(f"Done. {len(rows)} researchers saved to {CSV_PATH}")
    if rows:
        log.info(f"Top 10 by priority score:")
        for r in rows[:10]:
            log.info(f"  score={r['priority_score']:>6}  h={r['h_index']:>3}  papers={r['paper_count_in_window']}  {r['name']:30s}  {r['institution']}")


def main():
    parser = argparse.ArgumentParser(description="Discover US-based industry ML researchers, ranked by impact")
    parser.add_argument("--days", type=int, default=2, help="Days to look back (default: 2)")
    parser.add_argument("--backfill-months", type=int, default=0, help="Backfill N months")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM relevance filter")
    parser.add_argument("--conferences-only", action="store_true", help="Only scrape conference websites (skip OpenAlex)")
    args = parser.parse_args()

    end_date = datetime.now(timezone.utc).replace(tzinfo=None)

    if args.backfill_months > 0:
        start_date = end_date - timedelta(days=args.backfill_months * 30)
        log.info(f"Backfilling {args.backfill_months} months: {start_date.date()} to {end_date.date()}")
    else:
        start_date = end_date - timedelta(days=args.days)
        log.info(f"Scraping last {args.days} day(s): {start_date.date()} to {end_date.date()}")

    scrape(start_date, end_date, skip_llm=args.skip_llm, conferences_only=args.conferences_only)


if __name__ == "__main__":
    main()
