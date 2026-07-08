# researcher-scrape

Discovery pipeline for ML researchers, organized around five hiring positions: **World Models, Agentic Benchmarks, STEM Benchmarks, Environment Generation, Post-Training.**

Two pipelines live here:

| Script | What it does | When to use |
|---|---|---|
| `expand.py` | **Primary.** Seed papers/researchers → citation-graph BFS (OpenAlex) → LLM career-stage + position-fit classification → filter → profile + email enrichment → **xlsx with one tab per position** | Weekly hiring list. Runs on a cron. |
| `scrape.py` | Legacy conference sweep: NeurIPS/ICML/ICLR 2025 full paper lists → LLM topic filter → ranked flat CSV | One-off conference-wide sweeps |

See **[PIPELINE.md](PIPELINE.md)** for the full phase-by-phase flow, sequence diagram, latest run counts, and design decisions (why enrichment is by author ID, why topic search was removed, why filters run twice).

## Usage — expand.py (primary)

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="sk-or-..."

# Full pipeline: BFS + LLM classify + filter + enrich + emails + xlsx
python expand.py --seeds seeds_fleet.yaml --config config_fleet.yaml

# Resume from cached state (skips everything already computed)
python expand.py --seeds seeds_fleet.yaml --config config_fleet.yaml --resume

# Useful flags
#   --max-depth 0|1|2     BFS hops (default from config; keep at 1)
#   --skip-emails         skip the email cascade
#   --skip-enrichment     skip profile enrichment entirely
#   --skip-classify       skip LLM phases
#   --dry-run             expand only, write raw output
#   --dedup-csv FILE      drop names already in FILE
#   --slack-channel NAME  upload xlsx to Slack after the run (needs SLACK_BOT_TOKEN)
```

**Inputs:**
- `seeds_fleet.yaml` — seed papers (arxiv IDs, one commented section per position) and seed researchers. Add papers here; IDs should be verified (wrong IDs walk the wrong citation graph).
- `config_fleet.yaml` — the five categories (names drive LLM classification + xlsx tabs), filters (h-index cap, career stages, excluded institutions), expansion fan-out.

**Output:** `data/expand_output/researchers.xlsx` — Combined tab + one tab per position. Also per-position CSVs.

## Weekly cron

`.github/workflows/weekly.yml` runs the full pipeline Mondays 13:00 UTC, uploads the xlsx as an artifact, optionally posts to `#hiring`, and commits caches back. Repo secrets needed: `OPENROUTER_API_KEY`, `SLACK_BOT_TOKEN`.

## Caching

All under `data/`, all committed back by the cron so repeat runs only pay for the delta:

- `expand_cache.json` — graph state + LLM labels (drives `--resume`)
- `enrich_cache_by_id.json` — OpenAlex profiles keyed by author ID (collision-proof)
- `enrich_cache.json` — name-based profile lookups (fallback path)
- `email_cache.json` — email results incl. misses
- `institution_domains.json` — institution → email-domain mappings
- `llm_cache.json` — `scrape.py` conference classifications

## API notes

- **OpenAlex** is the graph + enrichment backbone (no key needed; polite pool via `mailto`). All callers share one throttle (`oa_throttle.py`, 2 RPS sustained); a single 429 triggers a 15-minute cooldown and an immediate fallback to Semantic Scholar rather than retries.
- **Semantic Scholar** is the enrichment fallback (`s2_client.py`). Unauthenticated ~0.3 RPS; set `S2_API_KEY` for 10x.
- **OpenRouter** (Sonnet 4.5) does paper/researcher classification.

## Usage — scrape.py (legacy conference sweep)

```bash
python scrape.py                    # full: LLM filter + enrichment → data/researchers.csv
python scrape.py --skip-enrichment  # no OpenAlex
python scrape.py --skip-llm         # include all papers
```
