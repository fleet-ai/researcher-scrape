#!/usr/bin/env python3
"""
Email scraper for researchers discovered by the main pipeline.

Reads researchers.csv, looks up emails via three strategies:
1. Semantic Scholar → homepage → scrape for emails  (verified)
2. ORCID public API → email                         (verified)
3. ROR institution lookup → domain → email pattern   (inferred)

Results cached in data/email_cache.json (per-researcher) and
data/institution_domains.json (institution → domain mapping).

Usage:
    python scrape_emails.py                  # All researchers
    python scrape_emails.py --top 500        # Top 500 by priority score
    python scrape_emails.py --min-papers 3   # Only 3+ papers
"""

import argparse
import csv
import json
import logging
import os
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
OUTPUT_PATH = DATA_DIR / "researchers_with_emails.csv"
EMAIL_CACHE_PATH = DATA_DIR / "email_cache.json"
INST_CACHE_PATH = DATA_DIR / "institution_domains.json"

S2_API = "https://api.semanticscholar.org/graph/v1"
ORCID_API = "https://pub.orcid.org/v3.0"
ROR_API = "https://api.ror.org/organizations"

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

EXCLUDED_EMAIL_DOMAINS = {
    "example.com", "example.org", "test.com", "localhost",
    "sentry.io", "wixpress.com", "w3.org", "schema.org",
    "googleusercontent.com", "gstatic.com", "googleapis.com",
    "facebook.com", "twitter.com", "linkedin.com", "github.com",
    "jquery.com", "cloudflare.com", "gravatar.com", "wp.com",
    "wordpress.com", "amazonaws.com", "shields.io", "badge.fury.io",
    "duckduckgo.com", "duck.com",
}

JUNK_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".css", ".js", ".woff", ".ico")

# Well-known company email domains (not in ROR)
COMPANY_DOMAINS = {
    "google": "google.com",
    "google deepmind": "google.com",
    "google research": "google.com",
    "deepmind": "google.com",
    "microsoft": "microsoft.com",
    "microsoft research": "microsoft.com",
    "meta": "meta.com",
    "meta ai": "meta.com",
    "facebook": "meta.com",
    "fair": "meta.com",
    "amazon": "amazon.com",
    "amazon web services": "amazon.com",
    "aws": "amazon.com",
    "apple": "apple.com",
    "nvidia": "nvidia.com",
    "salesforce": "salesforce.com",
    "salesforce research": "salesforce.com",
    "alibaba": "alibaba-inc.com",
    "alibaba group": "alibaba-inc.com",
    "bytedance": "bytedance.com",
    "bytedance inc.": "bytedance.com",
    "tencent": "tencent.com",
    "tencent ai lab": "tencent.com",
    "huawei": "huawei.com",
    "huawei technologies ltd.": "huawei.com",
    "baidu": "baidu.com",
    "ibm": "ibm.com",
    "ibm research": "ibm.com",
    "intel": "intel.com",
    "intel labs": "intel.com",
    "samsung": "samsung.com",
    "adobe": "adobe.com",
    "adobe research": "adobe.com",
    "tiktok": "bytedance.com",
    "sea ai lab": "sea.com",
    "allen institute for ai": "allenai.org",
    "ai2": "allenai.org",
    "openai": "openai.com",
    "anthropic": "anthropic.com",
    "uber": "uber.com",
    "uber ai": "uber.com",
    "snap": "snap.com",
    "snap inc": "snap.com",
    "oracle": "oracle.com",
    "qualcomm": "qualcomm.com",
    "bosch": "bosch.com",
    "sony": "sony.com",
    "sony ai": "sony.com",
    "instadeep": "instadeep.com",
    "kuaishou": "kuaishou.com",
    "nebius": "nebius.com",
    "metr": "metr.org",
    "eleutherai": "eleuther.ai",
    "lakera": "lakera.ai",
    "cohere": "cohere.com",
    "together ai": "together.ai",
    "together.ai": "together.ai",
    "mistral": "mistral.ai",
    "mistral ai": "mistral.ai",
    "zhipu ai": "zhipuai.cn",
    "zhipu": "zhipuai.cn",
    "moonshot ai": "moonshot.cn",
    "minimax": "minimaxi.com",
    "stepfun": "stepfun.com",
    "xai": "x.ai",
    "character.ai": "character.ai",
    "sakana ai": "sakana.ai",
}


def load_researchers(csv_path: Path, top_n: int = 0, min_papers: int = 1) -> list[dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row.get("paper_count", 0)) >= min_papers:
                rows.append(row)
    if top_n > 0:
        rows = rows[:top_n]
    return rows


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


# ---------------------------------------------------------------------------
# Name parsing helpers
# ---------------------------------------------------------------------------

def _strip_accents(s: str) -> str:
    """Remove diacritical marks (é→e, ü→u, etc.)."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def _parse_name(full_name: str) -> tuple[str, str]:
    """Split a full name into (first, last). Returns lowercase ASCII."""
    name = _strip_accents(full_name.strip())
    parts = name.split()
    if len(parts) < 2:
        return (parts[0].lower() if parts else "", "")
    return (parts[0].lower(), parts[-1].lower())


def _build_email_patterns(first: str, last: str, domain: str) -> list[str]:
    """Generate common academic email patterns."""
    if not first or not last or not domain:
        return []
    patterns = [
        f"{first}.{last}@{domain}",       # john.smith@uni.edu
        f"{first[0]}{last}@{domain}",      # jsmith@uni.edu
        f"{first}{last[0]}@{domain}",      # johns@uni.edu
        f"{first}@{domain}",               # john@uni.edu
        f"{last}.{first}@{domain}",        # smith.john@uni.edu
        f"{first}_{last}@{domain}",        # john_smith@uni.edu
    ]
    return patterns


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------

def _s2_get(path: str, params: dict | None = None) -> dict | None:
    url = f"{S2_API}{path}"
    headers = {}
    api_key = os.environ.get("S2_API_KEY", "")
    if api_key:
        headers["x-api-key"] = api_key

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception:
            if attempt < 2:
                time.sleep(2)
    return None


def find_author_info(name: str, institution: str) -> dict:
    """Search S2 for author. Returns {homepage, orcid}."""
    data = _s2_get("/author/search", {
        "query": name,
        "fields": "name,homepage,externalIds,affiliations",
        "limit": 5,
    })
    if not data or not data.get("data"):
        return {}

    name_norm = _normalize(name)
    inst_norm = _normalize(institution) if institution else ""

    best = None
    for author in data["data"]:
        if _normalize(author.get("name", "")) != name_norm:
            continue
        if inst_norm:
            for aff in (author.get("affiliations") or []):
                aff_norm = _normalize(aff)
                if inst_norm in aff_norm or aff_norm in inst_norm:
                    best = author
                    break
        if best is None:
            best = author
        if best:
            break

    if not best:
        return {}

    ext_ids = best.get("externalIds") or {}
    return {
        "homepage": best.get("homepage") or "",
        "orcid": ext_ids.get("ORCID", ""),
    }


# ---------------------------------------------------------------------------
# Homepage email scraping
# ---------------------------------------------------------------------------

def _extract_emails(html: str) -> list[str]:
    raw = EMAIL_RE.findall(html)
    seen = set()
    results = []
    for email in raw:
        low = email.lower()
        domain = low.split("@")[1]
        if domain in EXCLUDED_EMAIL_DOMAINS:
            continue
        if any(low.endswith(s) for s in JUNK_SUFFIXES):
            continue
        if domain.count(".") > 3:
            continue
        if low not in seen:
            seen.add(low)
            results.append(email)
    return results


def scrape_page_emails(url: str) -> list[str]:
    if not url:
        return []
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }, allow_redirects=True)
        if resp.status_code != 200:
            return []
        return _extract_emails(resp.text)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# ORCID
# ---------------------------------------------------------------------------

def lookup_orcid_email(orcid_id: str) -> str:
    if not orcid_id:
        return ""
    try:
        resp = requests.get(f"{ORCID_API}/{orcid_id}/email", headers={"Accept": "application/json"}, timeout=10)
        if resp.status_code != 200:
            return ""
        for e in resp.json().get("email", []):
            addr = e.get("email")
            if addr:
                return addr
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# DuckDuckGo web search for verified emails
# ---------------------------------------------------------------------------

DDG_URL = "https://html.duckduckgo.com/html/"


def search_email_ddg(name: str, institution: str) -> str:
    """Search DuckDuckGo for researcher email. Returns first valid email found."""
    query = f"{name} {institution} email".strip()
    try:
        resp = requests.get(DDG_URL, params={"q": query}, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        if resp.status_code != 200:
            return ""
        emails = _extract_emails(resp.text)
        # Prefer emails that contain part of the researcher's name
        name_parts = {p.lower() for p in name.split() if len(p) > 2}
        for email in emails:
            local = email.split("@")[0].lower()
            if any(part in local for part in name_parts):
                return email
        # Fall back to first non-junk email
        return emails[0] if emails else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Institution → email domain (ROR API + company fallback)
# ---------------------------------------------------------------------------

def _load_inst_cache() -> dict:
    if INST_CACHE_PATH.exists():
        try:
            return json.loads(INST_CACHE_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_inst_cache(cache: dict):
    DATA_DIR.mkdir(exist_ok=True)
    INST_CACHE_PATH.write_text(json.dumps(cache))


def lookup_institution_domain(institution: str, inst_cache: dict) -> str:
    """Get email domain for an institution. Checks company list, then ROR API."""
    if not institution:
        return ""

    # Clean up duplicated/compound institution names
    # "Tsinghua University, Tsinghua University" → try each part
    # "IIIS, Tsinghua University" → try "IIIS" then "Tsinghua University"
    # "University of Washington/AI2" → try each part
    parts = [p.strip() for p in re.split(r"[,/]", institution) if p.strip()]
    if not parts:
        return ""

    # Check company domains first (any part)
    for part in parts:
        part_lower = part.lower()
        for company_key, domain in COMPANY_DOMAINS.items():
            if company_key in part_lower:
                return domain

    # Try each part against cache and ROR
    for part in parts:
        cache_key = _normalize(part)
        if cache_key in inst_cache:
            if inst_cache[cache_key]:
                return inst_cache[cache_key]
            continue

        domain = _ror_lookup(part)
        inst_cache[cache_key] = domain
        if domain:
            return domain

    return ""


def _ror_lookup(inst_name: str) -> str:
    """Query ROR API for an institution's email domain."""
    try:
        resp = requests.get(ROR_API, params={"query": inst_name}, timeout=8)
        if resp.status_code == 200:
            items = resp.json().get("items", [])
            if items:
                domains = items[0].get("domains", [])
                if domains:
                    return domains[0]
                links = items[0].get("links", [])
                if links:
                    homepage = links[0].get("value", "") if isinstance(links[0], dict) else links[0]
                    if homepage:
                        parsed = urlparse(homepage)
                        domain = parsed.hostname or ""
                        if domain.startswith("www."):
                            domain = domain[4:]
                        if domain:
                            return domain
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def scrape_emails(rows: list[dict], s2_limit: int = 0, ddg_limit: int = 0) -> list[dict]:
    # Load caches
    cache: dict = {}
    if EMAIL_CACHE_PATH.exists():
        try:
            cache = json.loads(EMAIL_CACHE_PATH.read_text())
            log.info(f"Loaded email cache with {len(cache)} entries")
        except Exception:
            pass

    inst_cache = _load_inst_cache()
    log.info(f"Loaded institution domain cache with {len(inst_cache)} entries")

    has_s2_key = bool(os.environ.get("S2_API_KEY"))
    delay = 0.15 if has_s2_key else 1.05
    if not has_s2_key:
        log.info("No S2_API_KEY — using 1 req/sec rate limit. Set S2_API_KEY for 10x speed.")

    found_verified = 0
    found_inferred = 0
    cached_hits = 0

    for i, row in enumerate(rows):
        name = row["name"]
        institution = row.get("institution", "")
        cache_key = _normalize(name)

        # Check cache
        if cache_key in cache:
            row["email"] = cache[cache_key].get("email", "")
            row["email_source"] = cache[cache_key].get("source", "")
            row["homepage"] = cache[cache_key].get("homepage", "")
            if row["email"]:
                if cache[cache_key].get("source") == "inferred":
                    found_inferred += 1
                else:
                    found_verified += 1
                cached_hits += 1
            continue

        # Strategy 1 & 2: Semantic Scholar → homepage/ORCID (only for top researchers)
        homepage = ""
        orcid = ""
        email = ""
        source = ""

        use_s2 = (s2_limit <= 0 or i < s2_limit)
        if use_s2:
            info = find_author_info(name, institution)
            homepage = info.get("homepage", "")
            orcid = info.get("orcid", "")

            if homepage:
                emails = scrape_page_emails(homepage)
                if emails:
                    email = emails[0]
                    source = "homepage"

            if not email and orcid:
                email = lookup_orcid_email(orcid)
                if email:
                    source = "orcid"

        # Strategy 3: DuckDuckGo web search for verified email
        use_ddg = (ddg_limit <= 0 or i < ddg_limit)
        if not email and use_ddg:
            email = search_email_ddg(name, institution)
            if email:
                source = "web_search"
            time.sleep(1.5)  # Be polite to DDG

        # Strategy 4: Institution domain → email pattern (fallback)
        if not email:
            inst_domain = lookup_institution_domain(institution, inst_cache)
            if inst_domain:
                first, last = _parse_name(name)
                patterns = _build_email_patterns(first, last, inst_domain)
                if patterns:
                    email = patterns[0]  # firstname.lastname@domain
                    source = "inferred"

        row["email"] = email
        row["email_source"] = source
        row["homepage"] = homepage

        cache[cache_key] = {"email": email, "homepage": homepage, "orcid": orcid, "source": source}

        if email:
            if source == "inferred":
                found_inferred += 1
            else:
                found_verified += 1

        if (i + 1) % 100 == 0:
            log.info(f"  {i+1}/{len(rows)} | verified={found_verified} inferred={found_inferred} (cached={cached_hits})")
            EMAIL_CACHE_PATH.write_text(json.dumps(cache))
            _save_inst_cache(inst_cache)

        if use_s2 and not use_ddg:
            time.sleep(delay)

    DATA_DIR.mkdir(exist_ok=True)
    EMAIL_CACHE_PATH.write_text(json.dumps(cache))
    _save_inst_cache(inst_cache)
    total = found_verified + found_inferred
    log.info(f"Done: {total}/{len(rows)} emails ({found_verified} verified, {found_inferred} inferred, {cached_hits} cached)")
    return rows


def save_results(rows: list[dict], output_path: Path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for f in ("email", "email_source", "homepage"):
        if f not in fieldnames:
            fieldnames.append(f)

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Saved {len(rows)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Scrape emails for discovered ML researchers")
    parser.add_argument("--top", type=int, default=0, help="Only top N researchers by priority score")
    parser.add_argument("--min-papers", type=int, default=1, help="Minimum paper count to include (default: 1)")
    parser.add_argument("--s2-limit", type=int, default=0, help="Only query Semantic Scholar for top N researchers (0=all)")
    parser.add_argument("--ddg-limit", type=int, default=0, help="DuckDuckGo search for top N researchers (0=all)")
    args = parser.parse_args()

    rows = load_researchers(CSV_PATH, top_n=args.top, min_papers=args.min_papers)
    log.info(f"Loaded {len(rows)} researchers from {CSV_PATH}")

    rows = scrape_emails(rows, s2_limit=args.s2_limit, ddg_limit=args.ddg_limit)
    save_results(rows, OUTPUT_PATH)


if __name__ == "__main__":
    main()
