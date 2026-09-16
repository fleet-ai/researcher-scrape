#!/usr/bin/env python3
"""Import verified shortlist candidates into the Lighthouse ATS.

Reads the shortlist_*.csv files under data/expand_output, keeps Yes-verdict
rows, merges the three benchmark tabs into one group, dedupes people that
appear in several groups (best rank wins), finds each person's LinkedIn URL
via one cached web-search LLM call (the ATS requires it), and sources them
with POST /v1/ats/actions/source-external-candidates/bulk.

Sourcing creates a silent ATS application record. It sends no email and no
Slack message. The API dedupes by candidate email and rejects a second
active application per person (per-row 422 in the bulk response).

Env: ATS_API_KEY, OPENROUTER_API_KEY. Actor defaults to deniz@fleet.so.

Usage:
    python import_ats.py --dry-run     # build the list + LinkedIn lookups only
    python import_ats.py               # do the import
"""

import argparse
import csv
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = DATA_DIR / "expand_output"
LINKEDIN_CACHE_PATH = DATA_DIR / "linkedin_cache.json"
RESULTS_PATH = DATA_DIR / "ats_import_results.json"

ATS_API_BASE = os.environ.get(
    "ATS_API_BASE", "https://api.internal.fleet-platform.fleetai.com"
)
ATS_ACTOR_EMAIL = os.environ.get("ATS_ACTOR_EMAIL", "deniz@fleet.so")
OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"
LINKEDIN_MODEL = os.environ.get("LINKEDIN_MODEL", "anthropic/claude-sonnet-4.5:online")

# group -> (ATS job slug, shortlist CSVs merged into the group)
GROUPS: dict[str, tuple[str, list[str]]] = {
    "benchmarks": (
        "research-scientist-benchmarks",
        [
            "shortlist_benchmarks.csv",
            "shortlist_agentic_benchmarks.csv",
            "shortlist_stem_benchmarks.csv",
        ],
    ),
    "post_training": ("research-scientist-post-training", ["shortlist_post_training.csv"]),
    "environment_generation": (
        "research-scientist-environment-scaling",
        ["shortlist_environment_generation.csv"],
    ),
}


class Candidate(BaseModel):
    name: str
    email: str
    website: str = ""
    career_stage: str = ""
    key_work: str = ""
    group: str
    rank: int  # shortlist row number, 1 = best
    linkedin_url: str = ""


class SourceRequest(BaseModel):
    full_name: str
    email: str
    linkedin_url: str
    job_slug: str
    source_platform: str = "researcher-scrape"
    source_detail: str


class BulkResult(BaseModel):
    index: int
    ok: bool
    application_id: str | None = None
    result: dict | None = None  # per-row response object (applicationId, stage, reactivated)
    error: str | None = None


class BulkResponse(BaseModel):
    ok: bool
    sourced_count: int
    failed_count: int
    results: list[BulkResult] = Field(default_factory=list)


def load_candidates() -> list[Candidate]:
    """Yes-verdict rows, one group per person (best rank wins)."""
    best: dict[str, Candidate] = {}
    for group, (_slug, files) in GROUPS.items():
        for fname in files:
            for i, row in enumerate(csv.DictReader(open(OUT_DIR / fname)), start=1):
                if not row["Recruitable?"].strip().startswith("Yes"):
                    continue
                email = row["Personal Email"].strip()
                if not email:
                    continue
                cand = Candidate(
                    name=row["Name"].strip(),
                    email=email,
                    website=row.get("Website", "").strip(),
                    career_stage=row.get("Career Stage", "").strip(),
                    key_work=row.get("Key Work", "").strip(),
                    group=group,
                    rank=int(row.get("#") or i),
                )
                key = cand.name.lower()
                if key not in best or cand.rank < best[key].rank:
                    best[key] = cand
    return sorted(best.values(), key=lambda c: (c.group, c.rank))


LINKEDIN_PROMPT = """Find the LinkedIn profile URL of this ML researcher by searching the web.

Name: {name}
Career stage: {stage}
Key work: {work}
Personal website: {website}

Confirm identity by matching the work above. Respond with ONLY a JSON object:
{{"linkedin_url": "https://www.linkedin.com/in/<handle>", "confident": true}}
If you cannot find or confirm the profile, return {{"linkedin_url": "", "confident": false}}."""

LINKEDIN_RE = re.compile(r"^https?://([a-z]{2,3}\.)?linkedin\.com/in/[^\s/?]+$")


def _linkedin_one(c: Candidate, api_key: str) -> str:
    prompt = LINKEDIN_PROMPT.format(
        name=c.name, stage=c.career_stage[:120], work=c.key_work[:200], website=c.website
    )
    for attempt in range(3):
        try:
            resp = requests.post(
                OPENROUTER_API,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": LINKEDIN_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 400,
                },
                timeout=120,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            m = re.search(r"\{.*\}", content, re.DOTALL)
            data = json.loads(m.group(0)) if m else {}
            url = (data.get("linkedin_url") or "").strip().rstrip("/")
            if data.get("confident") and LINKEDIN_RE.match(url):
                return url
            return ""
        except Exception as exc:  # noqa: BLE001 — retry any transport/parse error
            if attempt == 2:
                log.warning(f"  LinkedIn lookup failed for {c.name}: {exc}")
            time.sleep(2 * (attempt + 1))
    return ""


def resolve_linkedin(cands: list[Candidate], api_key: str) -> None:
    cache: dict[str, str] = (
        json.loads(LINKEDIN_CACHE_PATH.read_text()) if LINKEDIN_CACHE_PATH.exists() else {}
    )
    todo = [c for c in cands if c.name.lower() not in cache]
    log.info(f"LinkedIn lookups: {len(todo)} to fetch ({len(cands) - len(todo)} cached)")
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_linkedin_one, c, api_key): c for c in todo}
        done = 0
        for fut, c in futures.items():
            cache[c.name.lower()] = fut.result()
            done += 1
            if done % 25 == 0:
                log.info(f"  {done}/{len(todo)}")
                LINKEDIN_CACHE_PATH.write_text(json.dumps(cache, indent=1))
    LINKEDIN_CACHE_PATH.write_text(json.dumps(cache, indent=1))
    for c in cands:
        c.linkedin_url = cache.get(c.name.lower(), "")


def source_bulk(batch: list[SourceRequest], api_key: str) -> BulkResponse:
    resp = requests.post(
        f"{ATS_API_BASE}/v1/ats/actions/source-external-candidates/bulk",
        headers={
            "X-API-Key": api_key,
            "X-ATS-Actor-Email": ATS_ACTOR_EMAIL,
            "Content-Type": "application/json",
        },
        json={"candidates": [r.model_dump(mode="json") for r in batch]},
        timeout=300,
    )
    resp.raise_for_status()
    return BulkResponse.model_validate(resp.json())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=25)
    args = parser.parse_args()

    or_key = os.environ["OPENROUTER_API_KEY"]
    cands = load_candidates()
    by_group: dict[str, int] = {}
    for c in cands:
        by_group[c.group] = by_group.get(c.group, 0) + 1
    log.info(f"Import list: {len(cands)} people {by_group}")

    resolve_linkedin(cands, or_key)
    with_li = [c for c in cands if c.linkedin_url]
    log.info(f"LinkedIn confirmed: {len(with_li)}/{len(cands)}")
    for c in cands:
        if not c.linkedin_url:
            log.info(f"  no LinkedIn, skipped: {c.name} ({c.group} #{c.rank})")

    if args.dry_run:
        log.info("Dry run — no import.")
        return

    ats_key = os.environ["ATS_API_KEY"]
    requests_out = [
        SourceRequest(
            full_name=c.name,
            email=c.email,
            linkedin_url=c.linkedin_url,
            job_slug=GROUPS[c.group][0],
            source_detail=f"researcher-scrape shortlist 2026-08-29, {c.group} rank {c.rank}",
        )
        for c in with_li
    ]
    all_results: list[dict] = []
    sourced = failed = 0
    for i in range(0, len(requests_out), args.batch_size):
        batch = requests_out[i : i + args.batch_size]
        resp = source_bulk(batch, ats_key)
        sourced += resp.sourced_count
        failed += resp.failed_count
        for r in resp.results:
            cand = batch[r.index]
            all_results.append({"name": cand.full_name, "job_slug": cand.job_slug, **r.model_dump()})
            if not r.ok:
                log.info(f"  FAILED {cand.full_name}: {r.error}")
        log.info(f"batch {i // args.batch_size + 1}: sourced {resp.sourced_count}, failed {resp.failed_count}")
    RESULTS_PATH.write_text(json.dumps(all_results, indent=1))
    log.info(f"Done: {sourced} sourced, {failed} failed. Results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
