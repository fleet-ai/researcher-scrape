# arxiv-profile-scrape

Nightly scraper for finding researchers publishing on arXiv in:
- Reinforcement Learning (RL, RLHF, GRPO, DPO, policy optimization)
- Post-training alignment & optimization
- World models & model-based RL
- Environment simulation & sim-to-real

Built as a hiring pipeline — enriches authors with institution, country, and search links.

## Output

| File | Description |
|------|-------------|
| `data/researchers.csv` | All researchers |
| `data/researchers_us.csv` | US-based researchers only |

Columns: `name`, `paper_link`, `paper_title`, `arxiv_category`, `published_date`, `institution`, `institution_type`, `country`, `city`, `linkedin_search_url`, `google_scholar_url`, `personal_website`

## Data Sources

- **arXiv API** — paper metadata + author names
- **OpenAlex API** — institution, country, city (structured, ~35% coverage for recent papers)
- **Semantic Scholar API** — author homepage URLs
- **LinkedIn** — constructed Google search URLs (`site:linkedin.com/in "Name"`)
- **Google Scholar** — constructed search URLs

## Usage

```bash
pip install -r requirements.txt

# Nightly run (last 24h, with enrichment)
python scrape.py

# Backfill last 3 months
python scrape.py --backfill-months 3

# Fast mode: skip OpenAlex/S2 enrichment
python scrape.py --no-enrich --backfill-months 3
```

## GitHub Actions

Runs daily at 6 AM UTC with full enrichment. Can also be triggered manually with an optional `backfill_months` input.
