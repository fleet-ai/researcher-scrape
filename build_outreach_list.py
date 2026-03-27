#!/usr/bin/env python3
"""
Build the final outreach list: industry researchers, h < 40, with emails.

Reads researchers.csv + email_cache.json + enrich_cache.json,
enriches missing h-index via OpenAlex, applies filters, outputs CSV.

Usage:
    python build_outreach_list.py                    # defaults: industry, h<40, 2+ papers
    python build_outreach_list.py --max-h 60         # raise h-index cap
    python build_outreach_list.py --min-papers 1     # include single-paper researchers
    python build_outreach_list.py --include-academic  # don't filter by industry
"""

import argparse
import csv
import json
import logging
import math
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "researchers.csv"
EMAIL_CACHE_PATH = DATA_DIR / "email_cache.json"
ENRICH_CACHE_PATH = DATA_DIR / "enrich_cache.json"
INST_CACHE_PATH = DATA_DIR / "institution_domains.json"
OUTPUT_PATH = DATA_DIR / "outreach_list.csv"

OPENALEX_API = "https://api.openalex.org"
OPENALEX_EMAIL = "scraper@fleet.ai"
ROR_API = "https://api.ror.org/organizations"

# --- Industry classification ---

INDUSTRY_KEYWORDS = [
    "google", "deepmind", "microsoft", "meta ai", "facebook", "fair",
    "amazon", "aws", "apple", "nvidia", "salesforce", "alibaba",
    "bytedance", "tiktok", "tencent", "huawei", "baidu", "ibm",
    "intel", "samsung", "adobe", "openai", "anthropic", "cohere",
    "together ai", "together.ai", "mistral", "snap", "oracle",
    "qualcomm", "bosch", "sony", "instadeep", "kuaishou", "nebius",
    "metr", "eleutherai", "lakera", "zhipu", "moonshot", "minimax",
    "stepfun", "xai", "character.ai", "sakana", "sea ai", "uber",
    "contextual", "adept", "inflection", "runway", "stability",
    "ai21", "cerebras", "groq", "hugging face", "huggingface",
    "lightning ai", "mosaic", "deci", "weights & biases", "wandb",
    "allen institute for a",  # AI2
    " inc", " ltd", " corp", " llc",
    " lab", " labs",
]

ACADEMIC_KEYWORDS = [
    "university", "universit", "college", "institute of technology",
    "école", "ecole", "technion", "polytechnic", "kaist", "epfl",
    "eth zurich", "eth ", "mit ", "caltech", "oxford", "cambridge",
    "stanford", "berkeley", "princeton", "harvard", "yale", "columbia",
    "academy of science", "school of", "department of",
]

# Industry orgs that also match academic keywords (whitelist)
INDUSTRY_OVERRIDES = {
    "microsoft research", "google research", "meta research",
    "allen institute for artificial intelligence",
    "shanghai artificial intelligence laboratory",
    "shanghai ai laboratory",
    "beijing academy of artificial intelligence",
}


def is_industry(institution: str) -> bool:
    if not institution:
        return False
    low = institution.lower().strip()

    # Check overrides first
    for override in INDUSTRY_OVERRIDES:
        if override in low:
            return True

    # Exclude academic
    for kw in ACADEMIC_KEYWORDS:
        if kw in low:
            return False

    # Match industry
    for kw in INDUSTRY_KEYWORDS:
        if kw in low:
            return True

    return False


# --- OpenAlex enrichment ---

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def _openalex_get(path: str, params: dict | None = None) -> dict | None:
    url = f"{OPENALEX_API}{path}"
    if params is None:
        params = {}
    params["mailto"] = OPENALEX_EMAIL
    for attempt in range(5):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = min(30 * (2 ** attempt), 300)
                log.warning(f"OpenAlex rate limited, sleeping {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception:
            if attempt < 4:
                time.sleep(2)
    return None


def enrich_researcher(name: str, institution: str, cache: dict) -> dict:
    """Look up h-index from OpenAlex. Returns enrichment dict."""
    cache_key = _normalize(name)
    if cache_key in cache:
        return cache[cache_key]

    data = _openalex_get("/authors", {
        "filter": f"display_name.search:{name}",
        "per_page": 5,
        "select": "id,display_name,summary_stats,last_known_institutions",
    })
    if not data or not data.get("results"):
        cache[cache_key] = {}
        return {}

    name_norm = _normalize(name)
    inst_norm = _normalize(institution) if institution else ""

    best = None
    for candidate in data["results"]:
        if _normalize(candidate.get("display_name", "")) != name_norm:
            continue
        if inst_norm:
            for inst in (candidate.get("last_known_institutions") or []):
                if inst_norm in _normalize(inst.get("display_name", "")):
                    best = candidate
                    break
        if best is None:
            best = candidate
        if best:
            break

    if not best:
        cache[cache_key] = {}
        return {}

    stats = best.get("summary_stats", {})
    enriched = {
        "h_index": stats.get("h_index", 0),
        "cited_by_count": stats.get("cited_by_count", 0),
        "works_count": stats.get("works_count", 0),
        "2yr_mean_citedness": stats.get("2yr_mean_citedness", 0),
    }
    cache[cache_key] = enriched
    return enriched


# --- Email from caches ---

COMPANY_DOMAINS = {
    "google": "google.com", "google deepmind": "google.com",
    "google research": "google.com", "deepmind": "google.com",
    "microsoft": "microsoft.com", "microsoft research": "microsoft.com",
    "meta": "meta.com", "meta ai": "meta.com", "facebook": "meta.com",
    "amazon": "amazon.com", "apple": "apple.com", "nvidia": "nvidia.com",
    "salesforce": "salesforce.com", "salesforce research": "salesforce.com",
    "alibaba": "alibaba-inc.com", "alibaba group": "alibaba-inc.com",
    "bytedance": "bytedance.com", "tencent": "tencent.com",
    "tencent ai lab": "tencent.com", "huawei": "huawei.com",
    "baidu": "baidu.com", "ibm": "ibm.com", "intel": "intel.com",
    "samsung": "samsung.com", "adobe": "adobe.com",
    "openai": "openai.com", "anthropic": "anthropic.com",
    "tiktok": "bytedance.com", "instadeep": "instadeep.com",
    "kuaishou": "kuaishou.com", "nebius": "nebius.com",
    "snap": "snap.com", "oracle": "oracle.com",
    "sony": "sony.com", "sony ai": "sony.com",
    "cohere": "cohere.com", "mistral": "mistral.ai",
    "together ai": "together.ai", "groq": "groq.com",
    "cerebras": "cerebras.ai", "hugging face": "huggingface.co",
    "huggingface": "huggingface.co",
}


def _strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def _parse_name(full_name: str) -> tuple[str, str]:
    name = _strip_accents(full_name.strip())
    parts = name.split()
    if len(parts) < 2:
        return (parts[0].lower() if parts else "", "")
    return (parts[0].lower(), parts[-1].lower())


def get_email(name: str, institution: str, email_cache: dict, inst_cache: dict) -> tuple[str, str]:
    """Get email from cache or construct from institution domain. Returns (email, source)."""
    cache_key = _normalize(name)

    # Check email cache first
    if cache_key in email_cache:
        entry = email_cache[cache_key]
        if entry.get("email"):
            return entry["email"], entry.get("source", "cached")

    # Construct from institution domain
    inst_lower = institution.lower() if institution else ""
    domain = ""

    # Check company domains
    for key, d in COMPANY_DOMAINS.items():
        if key in inst_lower:
            domain = d
            break

    # Check institution cache
    if not domain:
        inst_key = _normalize(institution.split(",")[0].strip()) if institution else ""
        domain = inst_cache.get(inst_key, "")

    if domain:
        first, last = _parse_name(name)
        if first and last:
            return f"{first}.{last}@{domain}", "inferred"

    return "", ""


# --- Main ---

OUTPUT_FIELDS = [
    "name", "institution", "email", "email_source",
    "h_index", "cited_by_count", "paper_count",
    "priority_score", "relevant_papers", "venues",
    "linkedin_search_url", "google_scholar_url",
]


def main():
    parser = argparse.ArgumentParser(description="Build filtered outreach list")
    parser.add_argument("--max-h", type=int, default=40, help="Max h-index (exclusive, default: 40)")
    parser.add_argument("--min-papers", type=int, default=2, help="Min paper count (default: 2)")
    parser.add_argument("--include-academic", action="store_true", help="Include academic researchers")
    parser.add_argument("--skip-enrichment", action="store_true", help="Skip OpenAlex enrichment")
    args = parser.parse_args()

    # Load data
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    log.info(f"Loaded {len(rows)} researchers")

    # Load caches
    email_cache = {}
    if EMAIL_CACHE_PATH.exists():
        email_cache = json.loads(EMAIL_CACHE_PATH.read_text())
    inst_cache = {}
    if INST_CACHE_PATH.exists():
        inst_cache = json.loads(INST_CACHE_PATH.read_text())
    enrich_cache = {}
    if ENRICH_CACHE_PATH.exists():
        enrich_cache = json.loads(ENRICH_CACHE_PATH.read_text())
    log.info(f"Caches: {len(email_cache)} emails, {len(enrich_cache)} enriched, {len(inst_cache)} institutions")

    # Filter: industry + min papers
    filtered = []
    for r in rows:
        if int(r.get("paper_count", 0)) < args.min_papers:
            continue
        if not args.include_academic and not is_industry(r.get("institution", "")):
            continue
        filtered.append(r)
    log.info(f"After filters (industry={'no' if args.include_academic else 'yes'}, papers>={args.min_papers}): {len(filtered)}")

    # Enrich h-index
    if not args.skip_enrichment:
        need_enrich = [r for r in filtered if _normalize(r["name"]) not in enrich_cache]
        log.info(f"Enriching {len(need_enrich)} researchers via OpenAlex...")
        for i, r in enumerate(need_enrich):
            enrich_researcher(r["name"], r.get("institution", ""), enrich_cache)
            if (i + 1) % 100 == 0:
                log.info(f"  {i+1}/{len(need_enrich)} enriched")
                ENRICH_CACHE_PATH.write_text(json.dumps(enrich_cache))
            time.sleep(0.15)
        DATA_DIR.mkdir(exist_ok=True)
        ENRICH_CACHE_PATH.write_text(json.dumps(enrich_cache))

    # Apply h-index filter + build output
    output = []
    skipped_h = 0
    for r in filtered:
        enrichment = enrich_cache.get(_normalize(r["name"]), {})
        h = enrichment.get("h_index", 0)
        if h >= args.max_h:
            skipped_h += 1
            continue

        email, email_source = get_email(r["name"], r.get("institution", ""), email_cache, inst_cache)

        cited = enrichment.get("cited_by_count", 0)
        citedness = enrichment.get("2yr_mean_citedness", 0)
        n_papers = int(r.get("paper_count", 0))
        score = round(h + n_papers * 10 + math.log(1 + citedness) * 15, 1)

        output.append({
            "name": r["name"],
            "institution": r.get("institution", ""),
            "email": email,
            "email_source": email_source,
            "h_index": h,
            "cited_by_count": cited,
            "paper_count": n_papers,
            "priority_score": score,
            "relevant_papers": r.get("relevant_papers", ""),
            "venues": r.get("venues", ""),
            "linkedin_search_url": r.get("linkedin_search_url", ""),
            "google_scholar_url": r.get("google_scholar_url", ""),
        })

    output.sort(key=lambda x: x["priority_score"], reverse=True)
    log.info(f"Skipped {skipped_h} researchers with h >= {args.max_h}")
    log.info(f"Final list: {len(output)} researchers")

    # Save
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output)
    log.info(f"Saved to {OUTPUT_PATH}")

    # Summary
    with_email = sum(1 for r in output if r["email"])
    log.info(f"  {with_email}/{len(output)} have emails")
    if output:
        log.info(f"  Top 15:")
        for r in output[:15]:
            log.info(f"    h={r['h_index']:>3d}  papers={r['paper_count']}  {r['name']:30s}  {r['institution'][:30]}")


if __name__ == "__main__":
    main()
