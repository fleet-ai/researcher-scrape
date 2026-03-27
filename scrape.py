#!/usr/bin/env python3
"""
ML researcher discovery pipeline.

1. Scrape all accepted papers from NeurIPS/ICML/ICLR 2025 conference websites
2. LLM filter: classify each paper as relevant (RL, post-training, world models, etc.) or not
3. Extract all authors from relevant papers only
4. Populate researcher profiles (institution, h-index, citation metrics)
5. Output researcher-centric CSV ranked by priority score
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
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "researchers.csv"

# --- Conference website JSON endpoints (NeurIPS/ICML/ICLR) ---
CONFERENCE_JSONS = {
    "NeurIPS 2025": "https://neurips.cc/static/virtual/data/neurips-2025-orals-posters.json",
    "ICML 2025": "https://icml.cc/static/virtual/data/icml-2025-orals-posters.json",
    "ICLR 2025": "https://iclr.cc/static/virtual/data/iclr-2025-orals-posters.json",
}

# Regex to detect non-human "author" names
_BAD_AUTHOR_RE = re.compile(r"\(.*\)|^\d|^[A-Z]{2,}\d|GPT|Gemini|Claude|LLaMA|Llama|Mistral|Copilot", re.IGNORECASE)

# LLM config
OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "anthropic/claude-sonnet-4.5"

PAPER_RELEVANCE_PROMPT = """You are classifying ML conference papers for a hiring pipeline. We want papers about:

1. **Reinforcement Learning** (RL, RLHF, GRPO, DPO, PPO, policy optimization, reward modeling, multi-agent RL, offline RL, inverse RL)
2. **Post-training & alignment** (preference optimization, instruction tuning, constitutional AI, red-teaming, safety training, RLHF pipelines)
3. **World models** (model-based RL, learned simulators, predictive models of environments, video prediction for planning)
4. **Environment simulation** (sim-to-real, robotics environments, physics simulation for RL, procedural generation of tasks/environments)
5. **LLM training & optimization** (pretraining at scale, efficient training, distributed training, training infrastructure, scaling laws, data curation for training)
6. **Agentic AI** (tool use, code generation agents, autonomous agents, planning, reasoning chains, multi-step decision making)
7. **Reward modeling & evaluation** (reward hacking, reward shaping, LLM-as-judge, automated evaluation, benchmarks for agents)

We do NOT want: pure computer vision (detection, segmentation, generation with no agent/RL component), security/privacy, hardware/chip design, quantum computing, HCI/UX, bioinformatics, networking, databases, pure NLP linguistics, speech processing, medical imaging, graph neural networks (unless for RL), theoretical optimization (unless for RL/training).

For each paper below, respond with ONLY a JSON array of objects: {"id": <number>, "relevant": true/false}. No explanation.

Papers:
"""

FIELDNAMES = [
    "priority_score",
    "name",
    "h_index",
    "cited_by_count",
    "works_count",
    "2yr_mean_citedness",
    "institution",
    "paper_count",
    "relevant_papers",
    "venues",
    "linkedin_search_url",
    "google_scholar_url",
]


# ---------------------------------------------------------------------------
# Conference website scraping
# ---------------------------------------------------------------------------

def fetch_conference_website_papers(conf_name: str, url: str) -> list[dict]:
    """Fetch accepted papers from a conference virtual site JSON endpoint."""
    log.info(f"[Conference] Fetching {conf_name}")
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


# ---------------------------------------------------------------------------
# LLM paper classification
# ---------------------------------------------------------------------------

def _llm_call(api_key: str, prompt: str) -> str | None:
    """Single OpenRouter API call with retries."""
    for attempt in range(3):
        try:
            resp = requests.post(
                OPENROUTER_API,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": OPENROUTER_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 4000},
                timeout=90,
            )
            if resp.status_code == 429:
                log.warning("OpenRouter rate limited, sleeping 10s")
                time.sleep(10)
                continue
            if resp.status_code != 200:
                log.warning(f"OpenRouter {resp.status_code}: {resp.text[:200]}")
                return None
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            log.warning(f"OpenRouter request failed (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(3)
    return None


def _parse_llm_json(content: str) -> list[dict]:
    """Parse JSON array from LLM response, handling markdown code blocks."""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    return json.loads(content)


def filter_relevant_papers(papers: list[dict], api_key: str) -> set[int]:
    """Use LLM to classify papers by topic relevance. Returns set of relevant paper indices."""
    batch_size = 50  # Paper titles are short, can fit more per batch
    relevant_indices = set()
    total = len(papers)

    for i in range(0, total, batch_size):
        batch = papers[i:i + batch_size]

        lines = []
        for j, p in enumerate(batch):
            title = (p.get("name") or "").replace("\n", " ").strip()
            lines.append(f"{j+1}. {title}")

        prompt = PAPER_RELEVANCE_PROMPT + "\n".join(lines)

        content = _llm_call(api_key, prompt)
        if content is None:
            # Fail open: include all papers in batch
            log.warning(f"  LLM call failed, including entire batch as fallback")
            for j in range(len(batch)):
                relevant_indices.add(i + j)
        else:
            try:
                results = _parse_llm_json(content)
                for item in results:
                    if item.get("relevant"):
                        idx = item["id"] - 1
                        if 0 <= idx < len(batch):
                            relevant_indices.add(i + idx)
            except (json.JSONDecodeError, KeyError) as e:
                log.warning(f"  Failed to parse LLM response: {e}. Including entire batch.")
                for j in range(len(batch)):
                    relevant_indices.add(i + j)

        kept = sum(1 for j in range(len(batch)) if (i + j) in relevant_indices)
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        log.info(f"  Paper filter batch {batch_num}/{total_batches}: {kept}/{len(batch)} relevant")
        time.sleep(0.5)

    return relevant_indices


# ---------------------------------------------------------------------------
# Author extraction from relevant papers
# ---------------------------------------------------------------------------

def collect_authors_from_relevant_papers(all_papers: list[dict], relevant_indices: set[int], venue: str, authors: dict):
    """Extract all authors from relevant papers into the authors dict."""
    for idx in relevant_indices:
        paper = all_papers[idx]
        title = (paper.get("name") or "").replace("\n", " ").strip()
        if not title:
            continue

        paper_info = {"title": title, "venue": venue}

        for auth in paper.get("authors", []):
            name = (auth.get("fullname") or "").strip()
            institution = (auth.get("institution") or "").strip()
            if not name or _BAD_AUTHOR_RE.search(name):
                continue

            # Key by lowercase name + institution for dedup
            key = f"{name.lower()}:{institution.lower()}"

            if key not in authors:
                authors[key] = {
                    "name": name,
                    "institution": institution if institution and institution.lower() != "none" else "",
                    "papers": [paper_info],
                }
            else:
                authors[key]["papers"].append(paper_info)


# ---------------------------------------------------------------------------
# OpenAlex profile enrichment
# ---------------------------------------------------------------------------

OPENALEX_API = "https://api.openalex.org"
OPENALEX_EMAIL = "scraper@fleet.ai"


def _openalex_get(path: str, params: dict | None = None) -> dict | None:
    """Single OpenAlex API call with rate limit handling."""
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


def _normalize(s: str) -> str:
    """Lowercase, collapse whitespace for fuzzy matching."""
    return re.sub(r"\s+", " ", s.lower().strip())


def enrich_profiles(authors: dict, min_papers: int = 2):
    """Look up OpenAlex author profiles for researchers with >= min_papers."""
    candidates = [(k, v) for k, v in authors.items() if len(v["papers"]) >= min_papers]
    log.info(f"Enriching {len(candidates)} researchers (with {min_papers}+ papers) via OpenAlex...")

    # Load enrichment cache
    cache_path = DATA_DIR / "enrich_cache.json"
    cache: dict = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
            log.info(f"  Loaded enrichment cache with {len(cache)} entries")
        except Exception:
            pass

    resolved = 0
    for i, (key, info) in enumerate(candidates):
        name = info["name"]
        institution = info.get("institution", "")
        cache_key = _normalize(name)

        # Check cache first
        if cache_key in cache:
            info.update(cache[cache_key])
            resolved += 1
            continue

        data = _openalex_get("/authors", {"filter": f"display_name.search:{name}", "per_page": 5, "select": "id,display_name,summary_stats,last_known_institutions"})
        if not data or not data.get("results"):
            time.sleep(0.15)
            continue

        # Match by name, prefer institution match
        name_norm = _normalize(name)
        best = None
        for candidate in data["results"]:
            if _normalize(candidate.get("display_name", "")) != name_norm:
                continue
            if institution:
                inst_norm = _normalize(institution)
                for inst in (candidate.get("last_known_institutions") or []):
                    if inst_norm in _normalize(inst.get("display_name", "")):
                        best = candidate
                        break
            if best is None:
                best = candidate
            if best:
                break

        if best:
            stats = best.get("summary_stats", {})
            enriched = {
                "openalex_id": best.get("id", ""),
                "h_index": stats.get("h_index", 0),
                "i10_index": stats.get("i10_index", 0),
                "cited_by_count": stats.get("cited_by_count", 0),
                "works_count": stats.get("works_count", 0),
                "2yr_mean_citedness": stats.get("2yr_mean_citedness", 0),
            }
            info.update(enriched)
            cache[cache_key] = enriched
            if not institution:
                insts = best.get("last_known_institutions") or []
                if insts:
                    info["institution"] = insts[0].get("display_name", "")
            resolved += 1

        if (i + 1) % 100 == 0:
            log.info(f"  Resolved {resolved}/{i+1} authors so far...")
            cache_path.write_text(json.dumps(cache))
        time.sleep(0.15)  # ~7 req/sec, under polite pool limit of 10/sec

    cache_path.write_text(json.dumps(cache))
    log.info(f"  Enrichment done: {resolved}/{len(candidates)} resolved")


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

def scrape(skip_llm: bool = False, skip_enrichment: bool = False, min_papers_enrich: int = 2):
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not skip_llm and not openrouter_key:
        log.error("OPENROUTER_API_KEY not set. Use --skip-llm to skip paper classification.")
        return

    # {name_lower:inst_lower: {name, institution, papers: [...]}}
    authors: dict[str, dict] = {}
    cache_path = DATA_DIR / "llm_cache.json"

    # Load LLM classification cache
    llm_cache: dict[str, list[int]] = {}
    if cache_path.exists():
        try:
            llm_cache = json.loads(cache_path.read_text())
            log.info(f"Loaded LLM cache with {len(llm_cache)} conferences")
        except Exception:
            pass

    for conf_name, json_url in CONFERENCE_JSONS.items():
        papers = fetch_conference_website_papers(conf_name, json_url)
        if not papers:
            continue

        if skip_llm:
            # No filtering — take all papers
            relevant = set(range(len(papers)))
            log.info(f"  Skipping LLM filter, all {len(papers)} papers included")
        elif conf_name in llm_cache:
            relevant = set(llm_cache[conf_name])
            log.info(f"  Using cached LLM results: {len(relevant)}/{len(papers)} papers relevant")
        else:
            log.info(f"  Classifying {len(papers)} papers via LLM...")
            relevant = filter_relevant_papers(papers, openrouter_key)
            log.info(f"  {len(relevant)}/{len(papers)} papers are relevant")
            llm_cache[conf_name] = sorted(relevant)
            DATA_DIR.mkdir(exist_ok=True)
            cache_path.write_text(json.dumps(llm_cache))

        before = len(authors)
        collect_authors_from_relevant_papers(papers, relevant, conf_name, authors)
        log.info(f"  -> {len(authors) - before} new researchers extracted")

    log.info(f"Total unique researchers: {len(authors)}")

    # Enrich with OpenAlex profiles
    if not skip_enrichment:
        enrich_profiles(authors, min_papers=min_papers_enrich)

    # Build final rows
    rows = []
    for key, info in authors.items():
        papers = info["papers"]
        venues = sorted(set(p["venue"] for p in papers if p.get("venue")))
        paper_titles = [p["title"] for p in papers]
        # Deduplicate titles (same paper at multiple venues)
        seen = set()
        unique_titles = []
        for t in paper_titles:
            if t not in seen:
                seen.add(t)
                unique_titles.append(t)

        n_papers = len(unique_titles)
        h_index = info.get("h_index", 0)
        cited = info.get("cited_by_count", 0)
        citedness_2yr = info.get("2yr_mean_citedness", 0)
        score = round(h_index + n_papers * 10 + math.log(1 + citedness_2yr) * 15, 1)

        rows.append({
            "priority_score": score,
            "name": info["name"],
            "h_index": h_index,
            "cited_by_count": cited,
            "works_count": info.get("works_count", 0),
            "2yr_mean_citedness": citedness_2yr,
            "institution": info["institution"],
            "paper_count": n_papers,
            "relevant_papers": " | ".join(unique_titles[:5]),
            "venues": "; ".join(venues),
            "linkedin_search_url": build_linkedin_search_url(info["name"]),
            "google_scholar_url": build_google_scholar_url(info["name"]),
        })

    rows.sort(key=lambda r: r["priority_score"], reverse=True)

    save_data(rows)
    log.info(f"Done. {len(rows)} researchers saved to {CSV_PATH}")
    if rows:
        log.info(f"Top 15 by priority score:")
        for r in rows[:15]:
            log.info(f"  score={r['priority_score']:>5}  papers={r['paper_count']}  {r['name']:30s}  {r['institution']}")


def main():
    parser = argparse.ArgumentParser(description="Discover ML researchers from top conferences, filtered by topic relevance")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM paper classification (include all papers)")
    parser.add_argument("--skip-enrichment", action="store_true", help="Skip OpenAlex profile enrichment")
    parser.add_argument("--min-papers", type=int, default=2, help="Minimum papers for OpenAlex enrichment (default: 2)")
    args = parser.parse_args()

    scrape(skip_llm=args.skip_llm, skip_enrichment=args.skip_enrichment, min_papers_enrich=args.min_papers)


if __name__ == "__main__":
    main()
