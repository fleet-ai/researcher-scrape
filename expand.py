#!/usr/bin/env python3
"""
Seed-and-expand researcher discovery pipeline.

Starts from known papers/researchers, expands via citation graph (Semantic Scholar),
classifies by career stage and category fit (LLM), filters, enriches with emails.

Usage:
    python expand.py --seeds seeds.yaml --config config.yaml
    python expand.py --seeds seeds.yaml --config config.yaml --skip-emails
    python expand.py --seeds seeds.yaml --config config.yaml --dry-run
"""

import argparse
import csv
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import requests
import yaml

from s2_client import S2Client
from scrape import _llm_call, _parse_llm_json, OPENROUTER_API, OPENROUTER_MODEL
from scrape_emails import scrape_emails as run_email_cascade, _normalize
from build_outreach_list import enrich_researcher as openalex_enrich

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
EXPAND_CACHE_PATH = DATA_DIR / "expand_cache.json"

# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Researcher:
    name: str
    s2_author_id: str = ""
    institution: str = ""
    homepage: str = ""
    h_index: int = 0
    cited_by_count: int = 0
    works_count: int = 0
    paper_count: int = 0
    key_papers: list = field(default_factory=list)
    career_stage: str = ""
    categories: list = field(default_factory=list)
    category_scores: dict = field(default_factory=dict)
    email: str = ""
    email_source: str = ""
    source_type: str = ""   # seed, coauthor, citation, reference, topic_search
    depth: int = 0
    recruitable: str = "Yes"


def _cache_key(name: str) -> str:
    return _normalize(name)


def _extract_affiliation(author: dict) -> str:
    affiliations = author.get("affiliations") or []
    return affiliations[0] if affiliations else ""


def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def _save_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ── Phase 1: Resolve seeds ───────────────────────────────────────────────────

def _serialize_researchers(researchers: dict) -> list[dict]:
    """Convert researchers dict to JSON-serializable list."""
    out = []
    for key, r in researchers.items():
        out.append({
            "key": key, "name": r.name, "s2_author_id": r.s2_author_id,
            "institution": r.institution, "homepage": r.homepage,
            "h_index": r.h_index, "cited_by_count": r.cited_by_count,
            "works_count": r.works_count, "paper_count": r.paper_count,
            "key_papers": r.key_papers[:10], "career_stage": r.career_stage,
            "categories": r.categories, "category_scores": r.category_scores,
            "email": r.email, "email_source": r.email_source,
            "source_type": r.source_type, "depth": r.depth,
            "recruitable": r.recruitable,
        })
    return out


def _deserialize_researchers(data: list[dict]) -> dict:
    """Restore researchers dict from cached JSON."""
    researchers = {}
    for d in data:
        r = Researcher(
            name=d["name"], s2_author_id=d.get("s2_author_id", ""),
            institution=d.get("institution", ""), homepage=d.get("homepage", ""),
            h_index=d.get("h_index", 0), cited_by_count=d.get("cited_by_count", 0),
            works_count=d.get("works_count", 0), paper_count=d.get("paper_count", 0),
            key_papers=d.get("key_papers", []), career_stage=d.get("career_stage", ""),
            categories=d.get("categories", []), category_scores=d.get("category_scores", {}),
            email=d.get("email", ""), email_source=d.get("email_source", ""),
            source_type=d.get("source_type", ""), depth=d.get("depth", 0),
            recruitable=d.get("recruitable", "Yes"),
        )
        researchers[d["key"]] = r
    return researchers


def resolve_seeds(seeds_cfg: dict, s2: S2Client) -> tuple[deque, dict]:
    """Resolve seed papers and researchers. Returns (queue, researchers)."""
    queue = deque()  # (paper_id, depth, source_type)
    researchers = {}
    seen_papers = set()

    # Seed papers (arxiv IDs)
    for arxiv_id in seeds_cfg.get("papers", []):
        paper = s2.paper_by_arxiv_id(arxiv_id)
        if paper and paper.get("paperId"):
            queue.append((paper["paperId"], 0, "seed"))
            seen_papers.add(paper["paperId"])
            log.info(f"  Seed paper: {paper.get('title', arxiv_id)[:80]}")
        else:
            log.warning(f"  Could not resolve arxiv:{arxiv_id}")

    # Seed papers (by title)
    for title in seeds_cfg.get("paper_titles", []):
        paper = s2.paper_by_title(title)
        if paper and paper.get("paperId"):
            queue.append((paper["paperId"], 0, "seed"))
            seen_papers.add(paper["paperId"])
            log.info(f"  Seed paper (title): {paper.get('title', title)[:80]}")
        else:
            log.warning(f"  Could not resolve title: {title[:60]}")

    # Seed researchers
    for seed in seeds_cfg.get("researchers", []):
        name = seed["name"]
        institution = seed.get("institution", "")
        author = s2.search_author(name)
        if not author:
            log.warning(f"  Could not find researcher: {name}")
            continue

        key = _cache_key(name)
        researchers[key] = Researcher(
            name=name,
            s2_author_id=author.get("authorId", ""),
            institution=institution or _extract_affiliation(author),
            h_index=author.get("hIndex") or 0,
            cited_by_count=author.get("citationCount") or 0,
            paper_count=author.get("paperCount") or 0,
            source_type="seed",
            depth=0,
        )

        # Add their recent papers to queue
        if author.get("authorId"):
            papers = s2.get_author_papers(author["authorId"], limit=10)
            for p in papers:
                pid = p.get("paperId")
                if pid and pid not in seen_papers:
                    queue.append((pid, 0, "seed"))
                    seen_papers.add(pid)
                    researchers[key].key_papers.append(p.get("title", ""))
        log.info(f"  Seed researcher: {name} ({institution}), {len(researchers[key].key_papers)} papers queued")

    return queue, researchers


# ── Phase 2: BFS expansion ───────────────────────────────────────────────────

def expand_graph(queue: deque, researchers: dict, s2: S2Client,
                 max_depth: int = 2, max_coauthors: int = 10,
                 max_citations: int = 50, max_references: int = 30,
                 prefer_recent: bool = True) -> set:
    """BFS over the citation graph. Modifies researchers in-place. Returns seen_papers."""
    seen_papers = {pid for pid, _, _ in queue}
    processed = 0

    while queue:
        paper_id, depth, source_type = queue.popleft()
        processed += 1
        if processed % 50 == 0:
            log.info(f"  Processed {processed} papers, {len(researchers)} researchers, queue={len(queue)}")

        # Get co-authors
        authors = s2.get_paper_authors(paper_id)
        paper_info = s2.paper_by_s2_id(paper_id)
        paper_title = paper_info.get("title", "") if paper_info else ""

        for author in authors[:max_coauthors]:
            name = author.get("name", "")
            if not name or len(name) < 3:
                continue
            key = _cache_key(name)
            if key not in researchers:
                researchers[key] = Researcher(
                    name=name,
                    s2_author_id=author.get("authorId", ""),
                    institution=_extract_affiliation(author),
                    h_index=author.get("hIndex") or 0,
                    cited_by_count=author.get("citationCount") or 0,
                    paper_count=author.get("paperCount") or 0,
                    key_papers=[paper_title] if paper_title else [],
                    source_type=source_type if source_type != "seed" else "coauthor",
                    depth=depth,
                )
            else:
                if paper_title and paper_title not in researchers[key].key_papers:
                    researchers[key].key_papers.append(paper_title)
                researchers[key].depth = min(researchers[key].depth, depth)

        # Expand further if not at max depth
        if depth < max_depth:
            citations = s2.get_citations(paper_id, limit=max_citations)
            if prefer_recent:
                citations.sort(key=lambda c: c.get("year") or 0, reverse=True)
            for cit in citations:
                cit_id = cit.get("paperId")
                if cit_id and cit_id not in seen_papers:
                    seen_papers.add(cit_id)
                    queue.append((cit_id, depth + 1, "citation"))

            references = s2.get_references(paper_id, limit=max_references)
            for ref in references:
                ref_id = ref.get("paperId")
                if ref_id and ref_id not in seen_papers:
                    seen_papers.add(ref_id)
                    queue.append((ref_id, depth + 1, "reference"))

    log.info(f"  BFS done: {processed} papers processed, {len(researchers)} researchers found")
    return seen_papers


# ── Phase 2b: Topic search expansion ─────────────────────────────────────────

def expand_via_topics(config: dict, s2: S2Client, researchers: dict, seen_papers: set):
    """Keyword search on S2 to find additional researchers."""
    categories = config.get("categories", [])
    topic_limit = config.get("expansion", {}).get("topic_search_limit", 50)

    for cat in categories:
        for kw in cat.get("keywords", []):
            papers = s2.search_papers(kw, year_range="2024-2026", limit=topic_limit)
            added = 0
            for p in papers:
                pid = p.get("paperId")
                if not pid or pid in seen_papers:
                    continue
                seen_papers.add(pid)
                for author in (p.get("authors") or []):
                    name = author.get("name", "")
                    if not name or len(name) < 3:
                        continue
                    key = _cache_key(name)
                    if key not in researchers:
                        researchers[key] = Researcher(
                            name=name,
                            s2_author_id=author.get("authorId", ""),
                            key_papers=[p.get("title", "")],
                            source_type="topic_search",
                            depth=1,
                        )
                        added += 1
                    elif p.get("title") and p["title"] not in researchers[key].key_papers:
                        researchers[key].key_papers.append(p["title"])
            log.info(f"  Topic '{kw}': {added} new researchers")


# ── Phase 3: LLM career stage classification ─────────────────────────────────

CAREER_STAGE_PROMPT = """Classify each ML researcher by career stage. You have: name, institution, h-index, paper count, and recent paper titles.

Categories (pick ONE per person):
- graduating_phd: PhD student in 4th+ year, likely graduating 2025-2026
- recent_grad: Graduated PhD 2024-2025, now in first industry role or postdoc
- junior_industry: 2-6 years post-PhD at a company (not a professor)
- mid_industry: 6-10 years experience, IC/senior IC role at a company
- senior: 10+ years, staff/principal level
- professor: Faculty at a university
- early_phd: 1st-3rd year PhD student
- founder: CEO/CTO/co-founder of a company
- unknown: Cannot determine

Respond with ONLY a JSON array: [{"id": 1, "stage": "graduating_phd"}, ...]. No explanation.

Researchers:
"""


def classify_career_stages(researchers: dict, api_key: str, cache: dict):
    """Batch-classify career stages via LLM. Modifies researchers in-place."""
    cached_stages = cache.get("career_stages", {})
    to_classify = []
    for key, r in researchers.items():
        if key in cached_stages:
            r.career_stage = cached_stages[key]
        elif not r.career_stage:
            to_classify.append((key, r))

    if not to_classify:
        return
    log.info(f"  Classifying {len(to_classify)} researchers ({len(cached_stages)} cached)")

    batch_size = 20
    for i in range(0, len(to_classify), batch_size):
        batch = to_classify[i:i + batch_size]
        lines = []
        for j, (key, r) in enumerate(batch):
            papers_str = " | ".join(r.key_papers[:5])
            lines.append(f"{j+1}. Name: {r.name}, Institution: {r.institution or 'unknown'}, "
                         f"h-index: {r.h_index}, Papers: {r.paper_count}, "
                         f"Recent: {papers_str[:200]}")

        content = _llm_call(api_key, CAREER_STAGE_PROMPT + "\n".join(lines))
        if content:
            try:
                results = _parse_llm_json(content)
                for item in results:
                    idx = item.get("id", 0) - 1
                    if 0 <= idx < len(batch):
                        key, r = batch[idx]
                        r.career_stage = item.get("stage", "unknown")
                        cached_stages[key] = r.career_stage
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                log.warning(f"  Career stage parse error: {e}")

        batch_num = i // batch_size + 1
        total_batches = (len(to_classify) + batch_size - 1) // batch_size
        log.info(f"  Career stage batch {batch_num}/{total_batches}")
        time.sleep(0.5)

    cache["career_stages"] = cached_stages


# ── Phase 4: LLM category fit classification ─────────────────────────────────

CATEGORY_FIT_PROMPT = """Rate how well each researcher fits each category based on their recent paper titles.

Categories:
{categories_block}

Score each researcher 0.0-1.0 per category (0 = no fit, 1 = perfect fit).

Respond with ONLY a JSON array: [{{"id": 1, "scores": {{"Category A": 0.8, "Category B": 0.1}}}}]. No explanation.

Researchers:
"""


def classify_categories(researchers: dict, categories: list, api_key: str, cache: dict):
    """Batch-classify category fit via LLM. Modifies researchers in-place."""
    cached_fit = cache.get("category_fit", {})
    cat_names = [c["name"] for c in categories]
    categories_block = "\n".join(f"- {c['name']}: {', '.join(c.get('keywords', []))}" for c in categories)

    to_classify = []
    for key, r in researchers.items():
        if key in cached_fit:
            r.category_scores = cached_fit[key]
            r.categories = [name for name, score in cached_fit[key].items() if score >= 0.5]
        elif not r.categories:
            to_classify.append((key, r))

    if not to_classify:
        return
    log.info(f"  Classifying {len(to_classify)} researchers into {len(categories)} categories ({len(cached_fit)} cached)")

    batch_size = 15
    for i in range(0, len(to_classify), batch_size):
        batch = to_classify[i:i + batch_size]
        lines = []
        for j, (key, r) in enumerate(batch):
            papers_str = " | ".join(r.key_papers[:8])
            lines.append(f"{j+1}. {r.name}: {papers_str[:250]}")

        prompt = CATEGORY_FIT_PROMPT.format(categories_block=categories_block) + "\n".join(lines)
        content = _llm_call(api_key, prompt)
        if content:
            try:
                results = _parse_llm_json(content)
                for item in results:
                    idx = item.get("id", 0) - 1
                    if 0 <= idx < len(batch):
                        key, r = batch[idx]
                        scores = item.get("scores", {})
                        r.category_scores = scores
                        r.categories = [name for name, score in scores.items() if score >= 0.5]
                        cached_fit[key] = scores
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                log.warning(f"  Category fit parse error: {e}")

        batch_num = i // batch_size + 1
        total_batches = (len(to_classify) + batch_size - 1) // batch_size
        log.info(f"  Category fit batch {batch_num}/{total_batches}")
        time.sleep(0.5)

    cache["category_fit"] = cached_fit


# ── Phase 5: Filtering ────────────────────────────────────────────────────────

def apply_filters(researchers: dict, config: dict, exclude_names: set) -> dict:
    """Apply configured filters. Returns filtered dict."""
    filters = config.get("filters", {})
    max_h = filters.get("max_h_index", 40)
    min_papers = filters.get("min_papers", 1)
    allowed_stages = set(filters.get("career_stages", [
        "graduating_phd", "recent_grad", "junior_industry", "mid_industry"
    ]))
    exclude_institutions = {i.lower() for i in filters.get("exclude_institutions", [])}
    user_exclude_names = {n.lower() for n in filters.get("exclude_names", [])}

    filtered = {}
    removed = {"h_index": 0, "papers": 0, "stage": 0, "institution": 0, "name": 0, "category": 0, "dedup": 0}

    for key, r in researchers.items():
        if key in exclude_names or r.name.lower() in user_exclude_names:
            removed["dedup"] += 1
            continue
        if r.h_index and r.h_index >= max_h:
            removed["h_index"] += 1
            continue
        if r.paper_count and r.paper_count < min_papers:
            removed["papers"] += 1
            continue
        if r.career_stage and r.career_stage not in allowed_stages and r.career_stage != "unknown":
            removed["stage"] += 1
            continue
        if any(exc in (r.institution or "").lower() for exc in exclude_institutions):
            removed["institution"] += 1
            continue
        if not r.categories:
            removed["category"] += 1
            continue

        # Flag recruitability
        hard_recruit = {"openai", "anthropic", "google deepmind", "deepmind"}
        if any(hr in (r.institution or "").lower() for hr in hard_recruit):
            r.recruitable = "Unlikely"
        elif r.career_stage in ("senior", "professor", "founder"):
            r.recruitable = "Unlikely"

        filtered[key] = r

    log.info(f"  Filtered: {len(researchers)} -> {len(filtered)} (removed: {removed})")
    return filtered


# ── Phase 6: Enrichment ───────────────────────────────────────────────────────

def enrich(researchers: dict, skip_emails: bool = False):
    """OpenAlex enrichment + email cascade."""
    # OpenAlex
    enrich_cache = _load_json(DATA_DIR / "enrich_cache.json")
    enriched = 0
    for key, r in researchers.items():
        if r.h_index == 0:
            result = openalex_enrich(r.name, r.institution, enrich_cache)
            if result:
                r.h_index = result.get("h_index", r.h_index)
                r.cited_by_count = result.get("cited_by_count", r.cited_by_count)
                r.works_count = result.get("works_count", r.works_count)
                enriched += 1
    _save_json(DATA_DIR / "enrich_cache.json", enrich_cache)
    log.info(f"  OpenAlex enriched {enriched} researchers")

    if skip_emails:
        return

    # Email cascade — convert to row format
    rows = [{"name": r.name, "institution": r.institution or "",
             "paper_count": str(r.paper_count), "priority_score": "0"}
            for r in researchers.values()]

    enriched_rows = run_email_cascade(rows)

    # Map back
    for row in enriched_rows:
        key = _cache_key(row["name"])
        if key in researchers:
            researchers[key].email = row.get("email", "")
            researchers[key].email_source = row.get("email_source", "")
            researchers[key].homepage = row.get("homepage", "")

    with_email = sum(1 for r in researchers.values() if r.email)
    log.info(f"  Emails found: {with_email}/{len(researchers)}")


# ── Phase 7: Output ───────────────────────────────────────────────────────────

OUTPUT_FIELDS = [
    "name", "career_stage", "institution", "categories", "key_papers",
    "email", "email_source", "homepage", "h_index", "cited_by_count",
    "paper_count", "source_type", "depth", "recruitable",
]


def write_output(researchers: dict, categories: list, output_dir: Path):
    """Write per-category CSVs and combined.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for key, r in sorted(researchers.items(), key=lambda x: x[1].h_index, reverse=True):
        row = {
            "name": r.name,
            "career_stage": r.career_stage,
            "institution": r.institution,
            "categories": "; ".join(r.categories),
            "key_papers": " | ".join(r.key_papers[:5]),
            "email": r.email,
            "email_source": r.email_source,
            "homepage": r.homepage,
            "h_index": r.h_index,
            "cited_by_count": r.cited_by_count,
            "paper_count": r.paper_count,
            "source_type": r.source_type,
            "depth": r.depth,
            "recruitable": r.recruitable,
        }
        all_rows.append(row)

    # Combined
    _write_csv(output_dir / "combined.csv", all_rows)
    log.info(f"  combined.csv: {len(all_rows)} researchers")

    # Per-category
    for cat in categories:
        cat_name = cat["name"]
        cat_rows = [r for r in all_rows if cat_name in r["categories"]]
        safe_name = re.sub(r"[^a-z0-9_]", "_", cat_name.lower())
        _write_csv(output_dir / f"{safe_name}.csv", cat_rows)
        log.info(f"  {safe_name}.csv: {len(cat_rows)} researchers")


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ── Dedup helpers ─────────────────────────────────────────────────────────────

def load_existing_names(csv_path: str) -> set:
    """Load normalized names from an existing CSV for deduplication."""
    names = set()
    path = Path(csv_path)
    if not path.exists():
        return names
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name", "")
            if name:
                names.add(_cache_key(name))
    return names


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Seed-and-expand researcher discovery")
    parser.add_argument("--seeds", required=True, help="Path to seeds YAML")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="Override max expansion depth (0=seeds only, 1=one hop, 2=two hops)")
    parser.add_argument("--skip-emails", action="store_true", help="Skip email discovery")
    parser.add_argument("--skip-classify", action="store_true", help="Skip LLM classification")
    parser.add_argument("--skip-topics", action="store_true", help="Skip topic keyword search")
    parser.add_argument("--dry-run", action="store_true", help="Expand only, skip classify/enrich/email")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from cached state (skip seed resolution + BFS, go straight to classify/filter/enrich)")
    parser.add_argument("--output-dir", default="data/expand_output", help="Output directory")
    parser.add_argument("--dedup-csv", default="", help="CSV to dedup against")
    args = parser.parse_args()

    with open(args.seeds) as f:
        seeds_cfg = yaml.safe_load(f)["seeds"]
    with open(args.config) as f:
        config = yaml.safe_load(f)

    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    s2 = S2Client()
    expand_cache = _load_json(EXPAND_CACHE_PATH)

    # Depth override
    exp = config.get("expansion", {})
    max_depth = args.max_depth if args.max_depth is not None else exp.get("max_depth", 2)

    # Load existing names for dedup
    exclude_names = set()
    if args.dedup_csv:
        exclude_names = load_existing_names(args.dedup_csv)
    log.info(f"Dedup against {len(exclude_names)} existing names")

    # ── Resume from cache or expand fresh ──
    if args.resume and "researchers" in expand_cache:
        researchers = _deserialize_researchers(expand_cache["researchers"])
        seen_papers = set(expand_cache.get("seen_papers", []))
        log.info(f"=== Resumed: {len(researchers)} researchers, {len(seen_papers)} seen papers ===")
    else:
        # Phase 1: Resolve seeds
        log.info("=== Phase 1: Resolve Seeds ===")
        queue, researchers = resolve_seeds(seeds_cfg, s2)
        log.info(f"  Seeds resolved: {len(queue)} papers queued, {len(researchers)} seed researchers")

        # Phase 2: BFS expansion
        log.info(f"=== Phase 2: Graph Expansion (depth={max_depth}) ===")
        seen_papers = expand_graph(
            queue, researchers, s2,
            max_depth=max_depth,
            max_coauthors=exp.get("max_coauthors_per_paper", 10),
            max_citations=exp.get("max_citations_per_paper", 50),
            max_references=exp.get("max_references_per_paper", 30),
            prefer_recent=exp.get("prefer_recent", True),
        )

        # Phase 2b: Topic search
        if not args.skip_topics:
            log.info("=== Phase 2b: Topic Search ===")
            expand_via_topics(config, s2, researchers, seen_papers)

        log.info(f"  Total researchers after expansion: {len(researchers)}")

        # Save expansion state for resume
        expand_cache["researchers"] = _serialize_researchers(researchers)
        expand_cache["seen_papers"] = list(seen_papers)
        expand_cache["last_depth"] = max_depth
        _save_json(EXPAND_CACHE_PATH, expand_cache)
        log.info(f"  Expansion state saved ({len(researchers)} researchers, depth={max_depth})")

    if args.dry_run:
        log.info("Dry run — writing raw expansion output")
        write_output(researchers, config.get("categories", []), Path(args.output_dir))
        return

    # Phase 3: Career stage classification
    if not args.skip_classify and openrouter_key:
        log.info("=== Phase 3: Career Stage Classification ===")
        classify_career_stages(researchers, openrouter_key, expand_cache)
        _save_json(EXPAND_CACHE_PATH, expand_cache)

    # Phase 4: Category classification
    if not args.skip_classify and openrouter_key:
        log.info("=== Phase 4: Category Classification ===")
        classify_categories(researchers, config.get("categories", []), openrouter_key, expand_cache)
        _save_json(EXPAND_CACHE_PATH, expand_cache)

    # Phase 5: Filter
    log.info("=== Phase 5: Filtering ===")
    researchers = apply_filters(researchers, config, exclude_names)

    # Phase 6: Enrich
    log.info("=== Phase 6: Enrichment ===")
    enrich(researchers, skip_emails=args.skip_emails)

    # Phase 7: Output
    log.info("=== Phase 7: Output ===")
    write_output(researchers, config.get("categories", []), Path(args.output_dir))
    _save_json(EXPAND_CACHE_PATH, expand_cache)

    log.info("Done.")


if __name__ == "__main__":
    main()
