# arxiv-profile-scrape

Nightly scraper that finds researchers publishing on arXiv in:
- Reinforcement Learning (RL, RLHF, GRPO, DPO, policy optimization)
- Post-training alignment & optimization
- World models & model-based RL
- Environment simulation & sim-to-real

## Output

`data/researchers.csv` — columns: `name`, `paper_link`, `paper_title`, `arxiv_category`, `published_date`, `linkedin`, `personal_website`

Author homepages are enriched via the Semantic Scholar API. LinkedIn is best-effort (manual review recommended).

## Usage

```bash
pip install -r requirements.txt

# Nightly run (last 24h)
python scrape.py

# Backfill last 3 months
python scrape.py --backfill-months 3

# Skip Semantic Scholar enrichment (faster)
python scrape.py --no-enrich --backfill-months 3
```

## GitHub Actions

Runs daily at 6 AM UTC. Can also be triggered manually with an optional `backfill_months` input.
