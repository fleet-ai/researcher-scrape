# researcher-scrape

Discovery pipeline for ML researchers publishing at top conferences, ranked by impact and filtered for relevance to RL, post-training, and agentic AI.

## Pipeline

```
Conference website JSONs (NeurIPS, ICML, ICLR 2025)
        │
        ▼
LLM paper classification (Sonnet 4.5 via OpenRouter)
        │
        ▼
Extract ALL authors from relevant papers
        │
        ▼
OpenAlex profile enrichment (h-index, citations)
        │
        ▼
Ranked CSV (priority_score = h_index + papers*10 + log(1 + 2yr_citedness)*15)
```

1. **Fetch papers** from conference virtual site JSON endpoints (NeurIPS, ICML, ICLR 2025)
2. **LLM paper filter** (Sonnet 4.5 via OpenRouter): classifies each paper as relevant or not, in batches of 50 titles
3. **Extract authors** from relevant papers only (no keyword/institution filtering)
4. **OpenAlex enrichment** (optional): resolves researcher profiles (h-index, citation count, works count, 2yr citedness)
5. **Priority scoring**: `h_index + paper_count * 10 + log(1 + 2yr_mean_citedness) * 15`

## Topics of Interest

RL, RLHF, GRPO, DPO, PPO, policy optimization, reward modeling, multi-agent RL, offline RL, post-training, alignment, preference optimization, instruction tuning, world models, model-based RL, environment simulation, sim-to-real, LLM training, distributed training, scaling laws, agentic AI, tool use, code generation agents, planning, reasoning chains, reward hacking, LLM-as-judge.

## Output

`data/researchers.csv` — one row per researcher, sorted by priority score.

Columns: `priority_score`, `name`, `h_index`, `cited_by_count`, `works_count`, `2yr_mean_citedness`, `institution`, `paper_count`, `relevant_papers`, `venues`, `linkedin_search_url`, `google_scholar_url`

## Caching

- **LLM cache** (`data/llm_cache.json`): paper classification results are cached per conference, so re-runs skip the expensive LLM step
- **Enrichment cache** (`data/enrich_cache.json`): OpenAlex profile lookups are cached per researcher name

## Usage

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="sk-or-..."

# Full pipeline (LLM classification + OpenAlex enrichment)
python scrape.py

# Skip enrichment (faster, no OpenAlex API needed)
python scrape.py --skip-enrichment

# Skip LLM classification (include all papers)
python scrape.py --skip-llm

# Only enrich researchers with 3+ papers (default: 2)
python scrape.py --min-papers 3
```
