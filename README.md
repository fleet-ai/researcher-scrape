# researcher-scrape

Discovery pipeline for **US-based industry/lab ML researchers**, ranked by impact and filtered for relevance to RL, post-training, and agentic AI.

## Data Sources

Three complementary approaches:

1. **Conference websites** — scrape accepted paper JSONs directly from NeurIPS, ICML, and ICLR virtual sites (these conferences moved to OpenReview and are no longer indexed by OpenAlex)
2. **OpenAlex conferences** — other venues via the OpenAlex API (AAAI, AISTATS, CoRL, RSS, ACL, EMNLP)
3. **OpenAlex lab papers** — AI/ML papers from known research labs (Google, Meta, OpenAI, NVIDIA, Apple, Amazon, Allen AI, Toyota Research, IBM Research, Hugging Face)

## Pipeline

```
Conference JSONs ─┐
                  ├─> US industry filter ─> OpenAlex profile lookup ─> h-index filter ─> LLM relevance filter ─> ranked CSV
OpenAlex API ─────┘
```

1. **Collect papers + authors** from all three sources
2. **US industry filter**: conference website authors matched against 50+ known US industry/lab name patterns; OpenAlex authors filtered by country code + institution type
3. **OpenAlex profile resolution**: conference website authors looked up by name to get OpenAlex IDs and metrics
4. **h-index filter** (5–80): removes very junior researchers and untouchable senior whales
5. **LLM relevance filter** (Sonnet 4.5 via OpenRouter): batches of 25 researchers with paper titles, keeps only those working on RL, post-training, world models, environment simulation, LLM training, or agentic AI
6. **Priority scoring**: `h_index + paper_count * 10 + log(1 + 2yr_citedness) * 15`

## Filters Detail

- **Country**: US-affiliated institutions only
- **Institution type**: Industry, labs, research institutes, nonprofits (excludes universities and hospitals)
- **Topic**: Lab papers filtered to AI + Computer Vision subfields
- **Venue**: Excludes repositories (Zenodo) and non-CS journals (medical, quantum, etc.)
- **LLM relevance**: Accepts RL, RLHF, post-training, alignment, world models, sim-to-real, LLM training, agentic AI. Rejects security, hardware, quantum, HCI, bioinformatics, pure NLP linguistics, pure CV.

## Output

`data/researchers.csv` — one row per researcher, sorted by priority score.

Columns: `priority_score`, `name`, `openalex_id`, `h_index`, `i10_index`, `cited_by_count`, `works_count`, `2yr_mean_citedness`, `institution`, `institution_type`, `city`, `paper_count_in_window`, `top_paper_title`, `top_paper_link`, `venues`, `linkedin_search_url`, `google_scholar_url`

## Usage

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="sk-or-..."

# Last 2 days (default) — uses OpenAlex date range + conference website data
python scrape.py

# Backfill last 12 months
python scrape.py --backfill-months 12

# Skip LLM relevance filter (faster, less precise)
python scrape.py --backfill-months 12 --skip-llm
```
