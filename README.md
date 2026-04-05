# Jamil Strategic Tweet Engine

A **Streamlit** dashboard for exploring your tweet history: engagement leaderboards, posting heatmaps, AI-driven topic pillars, brand-fit scoring, and news-to-tweet generation in your voice.

## What it does

- **Leaderboard** — Rank tweets by engagement rate (favorites ÷ views) with filters (minimum views, hide replies, top N, keyword search).
- **Activity heatmap** — Posting frequency by day of week and hour (Plotly).
- **Topic modeler** — OpenAI analyzes your corpus and returns five “core pillar” topics as structured JSON.
- **Brand compatibility** — Scores how well a named brand fits your content, with reasoning (and optional tips when the score is low).
- **News reactor** — Scrapes a news URL (`newspaper3k`) and drafts one reactive tweet styled like your past posts.

## Tech stack

- Python 3.10+ (recommended)
- [Streamlit](https://streamlit.io/)
- [pandas](https://pandas.pydata.org/), [Plotly](https://plotly.com/python/)
- [OpenAI Python SDK](https://github.com/openai/openai-python) + [python-dotenv](https://pypi.org/project/python-dotenv/)
- [newspaper3k](https://github.com/codelucas/newspaper) (+ `lxml_html_clean`) for article extraction

## Prerequisites

- An **OpenAI API key** with access to the models you select in the app (e.g. `gpt-4o-mini`).
- A **CSV export** of tweets with at least: `text`, `created_at`, `favorite_count`, `view_count` (column names are normalized to lowercase).

## Local setup

1. **Clone the repository** (or download the project folder).

2. **Create and activate a virtual environment** (from the project root):

   ```bash
   python -m venv .venv
   ```

   - **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
   - **macOS / Linux:** `source .venv/bin/activate`

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   - Copy `.env.example` to `.env` in the **same directory as `app.py`**.
   - Set `OPENAI_API_KEY` to your real key (never commit `.env`).

5. **Run the app:**

   ```bash
   streamlit run app.py
   ```

   Open the URL shown in the terminal (usually `http://localhost:8501`).

## Environment variables

| Variable            | Required | Description                          |
|---------------------|----------|--------------------------------------|
| `OPENAI_API_KEY`    | Yes*     | OpenAI API key (`sk-...`).           |

\*Required for **Topic Modeler**, **Brand Compatibility Agent**, and **The News Reactor**. Upload and leaderboard/heatmap work with CSV only.

See `.env.example` for a template.

## Project layout

- `app.py` — Single-file Streamlit application
- `requirements.txt` — Python dependencies
- `.env.example` — Example environment file (no secrets)

## Notes

- Article scraping depends on the target site; paywalled or heavily scripted pages may fail.
- AI features send tweet text (and scraped article text) to OpenAI; use data you are allowed to process.

## License

Add a license file if you publish this repo publicly (e.g. MIT).
