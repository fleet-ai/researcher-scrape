# researcher-scrape

Scraper for finding **US-based industry/lab ML researchers** using the [OpenAlex API](https://docs.openalex.org/).

## Strategy

Two complementary approaches:

1. **Conference authors** — all authors from top ML/RL venues (AAAI, NeurIPS, ICML, ICLR, AISTATS, CoRL, RSS, ACL, EMNLP)
2. **Lab authors** — AI/ML papers from known research labs (Google, Meta, OpenAI, NVIDIA, Apple, Amazon, Allen AI, Toyota Research, IBM Research, Hugging Face)

## Filters

- **Country**: US-affiliated institutions only
- **Institution type**: Industry, labs, research institutes, nonprofits (excludes universities and hospitals)
- **Topic**: Lab papers filtered to AI + Computer Vision subfields
- **Venue**: Excludes repositories (Zenodo) and non-CS journals (medical, quantum, etc.)

## Output

`data/researchers.csv`

Columns: `name`, `paper_link`, `paper_title`, `published_date`, `venue`, `institution`, `institution_type`, `city`, `linkedin_search_url`, `google_scholar_url`

## Usage

```bash
pip install -r requirements.txt

# Last 2 days (default)
python scrape.py

# Backfill last 12 months (full conference cycle)
python scrape.py --backfill-months 12
```
