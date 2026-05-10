"""
fetch_news.py
=============
Fetches recent football news for each WC2026 team using NewsAPI.org.
Applies simple keyword-based sentiment scoring and caches results for
6 hours so repeat runs do not burn through the free-tier rate limit.

When NEWSAPI_KEY environment variable is not set (or quota is exhausted),
the script falls back to deterministic mock data that exercises the full
downstream pipeline identically to live data.

Outputs
-------
data/raw/team_news_{team_slug}.json   - per-team article list with sentiment

Usage
-----
    export NEWSAPI_KEY="your_key_here"   # optional; mock used if absent
    python fetch_news.py

    # Fetch only specific teams:
    python fetch_news.py --teams "France,Brazil,Argentina"

    # Force refresh (ignore cache):
    python fetch_news.py --refresh
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

NEWSAPI_BASE = "https://newsapi.org/v2/everything"
CACHE_TTL_SECONDS = 6 * 3600   # 6 hours
REQUEST_DELAY = 0.25            # seconds between API calls (rate-limit courtesy)

POSITIVE_KEYWORDS = {
    "win", "victory", "wins", "won", "triumph", "form", "fitness",
    "confident", "prepared", "ready", "strong", "impressive", "dominant",
    "goal", "scored", "excellent", "unbeaten", "qualif", "advance",
}

NEGATIVE_KEYWORDS = {
    "injury", "injured", "suspend", "suspended", "suspension", "ban", "banned",
    "loss", "lost", "defeat", "crisis", "doubt", "concern", "struggle",
    "miss", "absent", "ruled out", "withdraw", "withdrawn", "poor",
    "eliminated", "shocking", "dismal",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# All 48 WC2026 teams (deduplicated from wc2026_fixtures.py)
# ---------------------------------------------------------------------------

WC2026_TEAMS: list[str] = [
    "USA", "Panama", "Bolivia", "Uruguay",
    "Mexico", "South Korea", "Croatia", "Poland",
    "Canada", "Morocco", "Belgium", "Ukraine",
    "France", "Brazil", "Japan", "Saudi Arabia",
    "Spain", "Australia", "Portugal", "Nigeria",
    "Argentina", "Chile", "Ecuador", "Philippines",
    "England", "Algeria", "Senegal", "New Zealand",
    "Netherlands", "Colombia", "Iran", "Jamaica",
    "Germany", "Cameroon", "Switzerland", "Paraguay",
    "Italy", "Hungary", "Tunisia", "South Africa",
    "Turkey", "Ivory Coast", "Venezuela", "Montenegro",
    "Qatar",
]


# ---------------------------------------------------------------------------
# Caching utilities
# ---------------------------------------------------------------------------

def team_cache_path(team: str) -> Path:
    slug = team.lower().replace(" ", "_").replace("/", "_")
    return RAW_DIR / f"team_news_{slug}.json"


def is_cache_fresh(path: Path, ttl: int = CACHE_TTL_SECONDS) -> bool:
    """Return True if *path* exists and was modified less than *ttl* seconds ago."""
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < ttl


def load_cache(path: Path) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def save_cache(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Sentiment analysis
# ---------------------------------------------------------------------------

def score_sentiment(text: str) -> dict:
    """
    Return a simple sentiment assessment based on keyword matching.

    Returns
    -------
    {
        "label": "positive" | "negative" | "neutral",
        "positive_count": int,
        "negative_count": int,
        "score": float  # positive: >0, negative: <0, neutral: 0
    }
    """
    text_lower = text.lower()
    pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
    neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
    net = pos - neg

    if net > 0:
        label = "positive"
    elif net < 0:
        label = "negative"
    else:
        label = "neutral"

    return {
        "label": label,
        "positive_count": pos,
        "negative_count": neg,
        "score": float(net),
    }


def enrich_articles(articles: list[dict]) -> list[dict]:
    """Add sentiment scores to each article dict."""
    enriched = []
    for article in articles:
        combined_text = " ".join(
            filter(None, [article.get("title", ""), article.get("description", "")])
        )
        sentiment = score_sentiment(combined_text)
        enriched.append(
            {
                "headline": article.get("title", ""),
                "description": article.get("description", ""),
                "url": article.get("url", ""),
                "source": article.get("source", {}).get("name", ""),
                "published_at": article.get("publishedAt", ""),
                "sentiment": sentiment,
            }
        )
    return enriched


# ---------------------------------------------------------------------------
# NewsAPI fetcher
# ---------------------------------------------------------------------------

def fetch_from_newsapi(team: str, api_key: str, max_articles: int = 10) -> list[dict]:
    """
    Query NewsAPI for recent news about *team* + 'World Cup 2026'.

    Returns a list of raw article dicts from NewsAPI.
    Raises requests.HTTPError on non-2xx responses.
    """
    query = f'"{team}" AND ("World Cup 2026" OR "FIFA 2026" OR "football")'
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max_articles,
        "apiKey": api_key,
    }
    resp = requests.get(NEWSAPI_BASE, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("articles", [])


# ---------------------------------------------------------------------------
# Mock data generator
# ---------------------------------------------------------------------------

MOCK_HEADLINES: list[dict] = [
    {
        "title": "{team} look in fine form ahead of World Cup 2026 opener",
        "description": "{team} trained with confidence and all key players are fit for the tournament.",
        "sentiment_hint": "positive",
    },
    {
        "title": "{team} star player faces late fitness test",
        "description": "A key {team} midfielder is doubtful after picking up a minor injury in training.",
        "sentiment_hint": "negative",
    },
    {
        "title": "{team} coach announces final 26-man World Cup squad",
        "description": "The {team} manager confirmed the group for the World Cup 2026 with no major surprises.",
        "sentiment_hint": "neutral",
    },
    {
        "title": "{team} win pre-tournament friendly to build momentum",
        "description": "{team} secured a comfortable victory in their final warm-up match before WC2026.",
        "sentiment_hint": "positive",
    },
    {
        "title": "{team} captain backs young squad to go far at World Cup 2026",
        "description": "The {team} skipper expressed confidence in the blend of experience and youth in the squad.",
        "sentiment_hint": "positive",
    },
]


def generate_mock_articles(team: str) -> list[dict]:
    """
    Return a set of deterministic mock articles for *team*.
    Uses the team name's hash to vary which templates are selected.
    """
    import hashlib
    seed = int(hashlib.md5(team.encode()).hexdigest(), 16) % len(MOCK_HEADLINES)

    articles = []
    for i in range(3):
        template = MOCK_HEADLINES[(seed + i) % len(MOCK_HEADLINES)]
        pub_offset_days = i + 1
        pub_date = (
            datetime.now(timezone.utc).replace(microsecond=0)
            .__class__(
                datetime.now(timezone.utc).year,
                datetime.now(timezone.utc).month,
                max(1, datetime.now(timezone.utc).day - pub_offset_days),
                10, 0, 0,
                tzinfo=timezone.utc,
            )
        )
        articles.append(
            {
                "title": template["title"].replace("{team}", team),
                "description": template["description"].replace("{team}", team),
                "url": f"https://mock-sports-news.example.com/{team.lower().replace(' ', '-')}-{i}",
                "source": {"name": "Mock Sports News"},
                "publishedAt": pub_date.isoformat().replace("+00:00", "Z"),
            }
        )
    return articles


# ---------------------------------------------------------------------------
# Per-team orchestration
# ---------------------------------------------------------------------------

def fetch_team_news(
    team: str,
    api_key: Optional[str],
    force_refresh: bool = False,
) -> dict:
    """
    Fetch (or load from cache) news for a single team.

    Returns the full result dict that will be written to disk.
    """
    cache_path = team_cache_path(team)

    if not force_refresh and is_cache_fresh(cache_path):
        cached = load_cache(cache_path)
        if cached is not None:
            log.debug("Cache hit for %s", team)
            return cached

    # Try live API first; fall back to mock on any failure
    raw_articles: list[dict] = []
    source = "newsapi"
    error_msg: Optional[str] = None

    if api_key:
        try:
            raw_articles = fetch_from_newsapi(team, api_key)
            time.sleep(REQUEST_DELAY)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            error_msg = f"HTTP {status}: {exc}"
            log.warning("NewsAPI HTTP error for %s: %s — falling back to mock", team, error_msg)
        except requests.RequestException as exc:
            error_msg = str(exc)
            log.warning("NewsAPI request error for %s: %s — falling back to mock", team, error_msg)

    if not raw_articles:
        raw_articles = generate_mock_articles(team)
        source = "mock"

    enriched = enrich_articles(raw_articles)

    # Aggregate team-level sentiment
    labels = [a["sentiment"]["label"] for a in enriched]
    avg_score = (
        sum(a["sentiment"]["score"] for a in enriched) / len(enriched)
        if enriched else 0.0
    )
    if avg_score > 0:
        team_sentiment = "positive"
    elif avg_score < 0:
        team_sentiment = "negative"
    else:
        team_sentiment = "neutral"

    result = {
        "team": team,
        "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": source,
        "article_count": len(enriched),
        "team_sentiment": {
            "label": team_sentiment,
            "average_score": round(avg_score, 3),
            "positive_articles": labels.count("positive"),
            "negative_articles": labels.count("negative"),
            "neutral_articles": labels.count("neutral"),
        },
        "articles": enriched,
    }
    if error_msg:
        result["api_error"] = error_msg

    save_cache(cache_path, result)
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch WC2026 team news from NewsAPI with caching and sentiment scoring."
    )
    parser.add_argument(
        "--teams",
        type=str,
        default=None,
        help="Comma-separated list of team names to fetch (default: all 48 teams).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-fetch even if cache is fresh.",
    )
    parser.add_argument(
        "--list-teams",
        action="store_true",
        help="Print all team names and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.list_teams:
        for t in sorted(WC2026_TEAMS):
            print(f"  {t}")
        return

    api_key: Optional[str] = os.environ.get("NEWSAPI_KEY")
    if api_key:
        log.info("NewsAPI key detected — will attempt live fetch.")
    else:
        log.info("NEWSAPI_KEY not set — using mock data for all teams.")

    teams_to_fetch: list[str] = (
        [t.strip() for t in args.teams.split(",")]
        if args.teams
        else WC2026_TEAMS
    )

    unknown = [t for t in teams_to_fetch if t not in WC2026_TEAMS]
    if unknown:
        log.warning("Unknown team(s) — will still attempt fetch: %s", unknown)

    results_summary: list[dict] = []

    log.info("Fetching news for %d teams …", len(teams_to_fetch))
    for team in tqdm(teams_to_fetch, desc="Teams", ncols=80):
        result = fetch_team_news(team, api_key, force_refresh=args.refresh)
        results_summary.append(
            {
                "team": team,
                "articles": result["article_count"],
                "sentiment": result["team_sentiment"]["label"],
                "avg_score": result["team_sentiment"]["average_score"],
                "source": result["source"],
                "cache_path": str(team_cache_path(team).relative_to(ROOT)),
            }
        )

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    log.info("\n%-25s  %-8s  %-10s  %8s  %s", "Team", "Articles", "Sentiment", "Score", "Source")
    log.info("-" * 70)
    for row in sorted(results_summary, key=lambda r: r["avg_score"], reverse=True):
        log.info(
            "%-25s  %-8d  %-10s  %+8.2f  %s",
            row["team"],
            row["articles"],
            row["sentiment"],
            row["avg_score"],
            row["source"],
        )

    # ------------------------------------------------------------------
    # Save a consolidated index
    # ------------------------------------------------------------------
    index_path = RAW_DIR / "team_news_index.json"
    index = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "total_teams": len(results_summary),
        "api_key_used": bool(api_key),
        "cache_ttl_hours": CACHE_TTL_SECONDS // 3600,
        "teams": results_summary,
    }
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, ensure_ascii=False)
    log.info("\nSaved consolidated index → %s", index_path)

    # Positive/negative/neutral breakdown
    sentiments = [r["sentiment"] for r in results_summary]
    log.info(
        "Sentiment breakdown: positive=%d  negative=%d  neutral=%d",
        sentiments.count("positive"),
        sentiments.count("negative"),
        sentiments.count("neutral"),
    )


if __name__ == "__main__":
    main()
