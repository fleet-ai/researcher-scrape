# researcher-scrape

Discovery pipeline for ML researchers, organized around six hiring positions: **World Models, Agentic Benchmarks, STEM Benchmarks, Benchmarks (model cards), Environment Generation, Post-Training.**

The Benchmarks role (added 2026-08-29) is seeded from the benchmark papers cited in the eval tables of ten 2026 frontier model cards; the Post-Training role runs a narrower stage filter in its dedicated crawl (`config_pt.yaml`): graduating PhDs and industry only, professors/seniors/founders dropped.

Three scripts live here:

| Script | What it does | When to use |
|---|---|---|
| `expand.py` | **Primary.** Seed papers/researchers → citation-graph BFS (OpenAlex) → ID-batch profile enrichment → LLM career-stage + position-fit classification → geography/stage filter → emails → **wide pool xlsx** (tab per position) | Weekly, via cron |
| `verify_shortlist.py` | **Final mile.** Top-40 per position → per-person **live web verification** (current employer with transitions, PhD status, personal email, key work, reasoned verdict) → **recruiter shortlist xlsx** | After every expand run |
| `scrape.py` | Legacy conference sweep: NeurIPS/ICML/ICLR 2025 full paper lists → LLM topic filter → ranked flat CSV | One-off sweeps |

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

**Output:** `data/expand_output/researchers.xlsx` — Combined tab + one tab per position (sorted recruitable Yes → Stretch → Unlikely). Also per-position CSVs.

## Usage — verify_shortlist.py (recruiter shortlist)

```bash
export OPENROUTER_API_KEY="sk-or-..."
python verify_shortlist.py                    # all tabs, 40 confirmed Yes/Maybe rows each
python verify_shortlist.py --top 25 --tabs world_models post_training
```

For each candidate an LLM with live web search (OpenRouter `:online`) checks homepage / Scholar / LinkedIn and returns: specific career stage ("Recent grad (PhD 24, UPenn) -> AI2"), personal email, website, one-line key work with venues, and a verdict with reason ("Unlikely - at OpenAI", "Yes - just switched labs"). Verification walks the ranked list and keeps filling until `--top` slots hold identity-confirmed Yes/Maybe candidates (verify-then-cut): unconfirmed identities and verified-Unlikely rows are excluded and never consume a slot. `--max-verify` caps the walk (default 4x top). Results cached in `data/verify_cache.json`.

**Output:** `data/expand_output/shortlist.xlsx` (tab per position, Yes → Maybe → Unlikely) + per-position `shortlist_*.csv`.

## Weekly cron

`.github/workflows/weekly.yml`, Mondays 13:00 UTC (or manual: Actions → weekly-researcher-discovery → Run workflow): runs `expand.py`, then `verify_shortlist.py --top 40`, uploads **both** xlsx files as 90-day artifacts, posts the shortlist to `#hiring`, and commits caches back. Repo secrets (set): `OPENROUTER_API_KEY`, `SLACK_BOT_TOKEN`.

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
