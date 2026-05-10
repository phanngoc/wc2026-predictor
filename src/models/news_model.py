"""
news_model.py
=============
The News Hawk — sentiment + keyword analyser that produces a per-team
"news factor" used to nudge baseline predictions up or down.

Reads the per-team JSON files written by ``src/data/fetch_news.py``
(``data/raw/team_news_<team_slug>.json``) and turns them into:

  * ``get_team_news_factor(team)`` -> float in [0.7, 1.3]
  * ``predict_adjustment(home, away)`` -> dict with adjustment + reasoning
  * ``predict(home, away, base_probs=None)`` -> 3-way probability dict
    matching the StatisticalModel/FormModel API contract

The class is *not* a learned model — it's deterministic — but it
maintains the same interface (predict / save / load) so the debate
engine can treat all three "analysts" uniformly.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib


# ---------------------------------------------------------------------------
# Paths & logging
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "news_model.pkl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [news]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("news_model")


# ---------------------------------------------------------------------------
# Keyword scoring tables
# ---------------------------------------------------------------------------

# Multipliers on the raw "win-prob lift" each keyword contributes.
# Positive lifts boost the home team's win probability; negative lifts
# suppress it. The aggregator clamps and re-normalises afterwards.

POSITIVE_KEYWORDS: dict[str, float] = {
    "fitness": 0.04,
    "fit": 0.03,
    "ready": 0.03,
    "strong squad": 0.05,
    "in form": 0.04,
    "confident": 0.03,
    "winning streak": 0.05,
    "unbeaten": 0.05,
    "impressive": 0.03,
    "dominant": 0.04,
    "record": 0.02,
    "back to": 0.03,    # "back to fitness"
    "scoring": 0.02,
    "clean sheet": 0.03,
    "strong form": 0.04,
}
# Cap on positive contribution per team (~ +8% on win prob).
POSITIVE_CAP = 0.08

NEGATIVE_KEYWORDS: dict[str, float] = {
    "injury": -0.05,
    "injured": -0.05,
    "doubt": -0.04,
    "miss": -0.04,
    "missed": -0.04,
    "missing": -0.04,
    "ruled out": -0.06,
    "out of": -0.04,
    "withdrawn": -0.05,
    "withdraw": -0.04,
    "concern": -0.03,
    "struggle": -0.03,
    "struggling": -0.03,
    "fitness test": -0.04,
    "suspended": -0.07,
    "suspension": -0.06,
    "ban": -0.06,
    "banned": -0.06,
    "red card": -0.07,
    "crisis": -0.05,
    "poor": -0.03,
    "defeat": -0.03,
    "lost": -0.02,
    "shocking": -0.04,
    "absent": -0.05,
    "doping": -0.08,
}
# Cap on negative contribution per team (~ -10% on win prob).
NEGATIVE_CAP = -0.10


def _slug(team: str) -> str:
    """Match fetch_news.py's slug convention for cache file names."""
    return team.lower().replace(" ", "_").replace("/", "_")


def _team_news_path(team: str) -> Path:
    return RAW_DIR / f"team_news_{_slug(team)}.json"


# ---------------------------------------------------------------------------
# News-derived team factor
# ---------------------------------------------------------------------------

def _score_team_news(payload: dict) -> tuple[float, list[str]]:
    """
    Convert a raw news payload into a *win-prob delta* in approximately
    [-0.20, +0.15] plus a list of human-readable signal strings.

    The signal blends:
      - average sentiment score (article level)
      - team_sentiment.label
      - keyword hits across headlines / descriptions
    """
    if not payload:
        return 0.0, ["no news data on file"]

    signals: list[str] = []
    delta = 0.0

    # 1) Aggregate sentiment baseline (-0.06 .. +0.06)
    team_sentiment = payload.get("team_sentiment", {}) or {}
    avg_score = float(team_sentiment.get("average_score", 0.0) or 0.0)
    sentiment_delta = max(-0.06, min(0.06, avg_score * 0.03))
    delta += sentiment_delta
    signals.append(
        f"avg sentiment={avg_score:+.2f} ({team_sentiment.get('label', 'neutral')}) "
        f"→ {sentiment_delta:+.2%}"
    )

    pos_articles = int(team_sentiment.get("positive_articles", 0) or 0)
    neg_articles = int(team_sentiment.get("negative_articles", 0) or 0)
    if pos_articles or neg_articles:
        signals.append(
            f"articles: +{pos_articles} positive / -{neg_articles} negative"
        )

    # 2) Keyword scan over articles
    pos_total = 0.0
    neg_total = 0.0
    pos_hits: dict[str, int] = {}
    neg_hits: dict[str, int] = {}

    for art in payload.get("articles", []) or []:
        blob = " ".join(filter(None, [
            art.get("headline", ""),
            art.get("description", ""),
        ])).lower()
        if not blob:
            continue
        for kw, lift in POSITIVE_KEYWORDS.items():
            if kw in blob:
                pos_total += lift
                pos_hits[kw] = pos_hits.get(kw, 0) + 1
        for kw, lift in NEGATIVE_KEYWORDS.items():
            if kw in blob:
                neg_total += lift   # already negative
                neg_hits[kw] = neg_hits.get(kw, 0) + 1

    # Apply per-direction caps
    pos_total = min(POSITIVE_CAP, pos_total)
    neg_total = max(NEGATIVE_CAP, neg_total)
    delta += pos_total + neg_total

    if pos_hits:
        top_pos = sorted(pos_hits.items(), key=lambda kv: -kv[1])[:3]
        signals.append(
            "positive cues: "
            + ", ".join(f"'{k}'×{n}" for k, n in top_pos)
            + f" (+{pos_total:.2%})"
        )
    if neg_hits:
        top_neg = sorted(neg_hits.items(), key=lambda kv: -kv[1])[:3]
        signals.append(
            "negative cues: "
            + ", ".join(f"'{k}'×{n}" for k, n in top_neg)
            + f" ({neg_total:.2%})"
        )

    # Hard floor / ceiling on combined delta
    delta = max(-0.20, min(0.15, delta))
    return delta, signals


def _delta_to_factor(delta: float) -> float:
    """
    Convert a [-0.20..+0.15] delta into a multiplier in [0.7..1.3].

    delta = -0.20 → factor 0.70
    delta =  0.00 → factor 1.00
    delta = +0.15 → factor ~1.225 (we extend symmetrically up to 1.30)
    """
    if delta >= 0:
        # Map [0..0.15] → [1.0..1.30]
        factor = 1.0 + (delta / 0.15) * 0.30
    else:
        # Map [-0.20..0] → [0.70..1.0]
        factor = 1.0 + (delta / 0.20) * 0.30
    return float(max(0.70, min(1.30, factor)))


# ---------------------------------------------------------------------------
# NewsModel
# ---------------------------------------------------------------------------

@dataclass
class NewsModel:
    """Sentiment-driven adjustment model for the debate engine."""

    model_path: Path = field(default=DEFAULT_MODEL_PATH)
    news_dir: Path = field(default=RAW_DIR)
    cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    base_probs: dict[str, float] = field(default_factory=lambda: {
        "home_win": 0.42, "draw": 0.27, "away_win": 0.31,
    })

    # ------------------------------------------------------------------
    # Loading per-team news payloads
    # ------------------------------------------------------------------

    def _load_team_payload(self, team: str) -> Optional[dict]:
        if team in self.cache:
            return self.cache[team]
        path = self.news_dir / f"team_news_{_slug(team)}.json"
        if not path.exists():
            log.debug("No news file for %s at %s", team, path)
            self.cache[team] = {}
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.cache[team] = data
            return data
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to read news for %s: %s", team, exc)
            self.cache[team] = {}
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_team_news_factor(self, team_name: str) -> float:
        """
        Return a multiplier in [0.7, 1.3] capturing how news around
        ``team_name`` should adjust their baseline win probability.
        """
        payload = self._load_team_payload(team_name)
        delta, _ = _score_team_news(payload or {})
        return _delta_to_factor(delta)

    def predict_adjustment(
        self,
        home_team: str,
        away_team: str,
    ) -> dict[str, Any]:
        """
        Build the news-only "vote" used by the debate engine. Returns the
        adjusted 3-way probabilities, the per-team factors, signals, and a
        natural-language reasoning string.
        """
        home_payload = self._load_team_payload(home_team) or {}
        away_payload = self._load_team_payload(away_team) or {}

        home_delta, home_signals = _score_team_news(home_payload)
        away_delta, away_signals = _score_team_news(away_payload)

        home_factor = _delta_to_factor(home_delta)
        away_factor = _delta_to_factor(away_delta)

        # Apply factors to a baseline: home factor lifts/suppresses
        # home_win, away factor does the opposite. Draw absorbs leftover
        # mass.
        base = self.base_probs
        home_win = base["home_win"] * home_factor / max(0.5, away_factor)
        away_win = base["away_win"] * away_factor / max(0.5, home_factor)
        # Draw probability grows when both teams have negative news (chaotic
        # match, low quality) and shrinks when both have positive news.
        draw_shift = -(home_delta + away_delta) * 0.5
        draw = base["draw"] + draw_shift
        draw = max(0.10, min(0.45, draw))

        total = home_win + away_win + draw
        norm = {
            "home_win": float(home_win / total),
            "draw": float(draw / total),
            "away_win": float(away_win / total),
        }

        reasoning = self._build_reasoning(
            home_team, away_team,
            home_factor, away_factor,
            home_signals, away_signals,
            home_delta, away_delta,
            norm,
        )

        sorted_p = sorted(norm.values(), reverse=True)
        gap = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else float(sorted_p[0])
        # News confidence is inherently lower than statistical models —
        # cap it well below 1.0 so the debate engine treats it as a softer
        # signal.
        confidence = max(0.0, min(0.85, 0.30 + gap * 1.2 + abs(home_delta - away_delta) * 1.5))

        return {
            "home_win": round(norm["home_win"], 4),
            "draw": round(norm["draw"], 4),
            "away_win": round(norm["away_win"], 4),
            "confidence": round(confidence, 4),
            "reasoning": reasoning,
            "features_used": {
                "home_news_factor": round(home_factor, 4),
                "away_news_factor": round(away_factor, 4),
                "home_delta": round(home_delta, 4),
                "away_delta": round(away_delta, 4),
                "home_signals": home_signals,
                "away_signals": away_signals,
            },
        }

    # Provide the same surface as the other two models so the debate
    # engine can call all three uniformly.
    def predict(
        self,
        home_team: str,
        away_team: str,
        elo_ratings_dict: Optional[dict[str, float]] = None,  # accepted for API symmetry
        match_date=None,
        base_probs: Optional[dict[str, float]] = None,
    ) -> dict[str, Any]:
        if base_probs is not None:
            saved = self.base_probs
            self.base_probs = base_probs
            try:
                return self.predict_adjustment(home_team, away_team)
            finally:
                self.base_probs = saved
        return self.predict_adjustment(home_team, away_team)

    # ------------------------------------------------------------------
    # Persistence (lightweight: just config + cache)
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path) if path else self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "news_dir": str(self.news_dir),
            "cache": self.cache,
            "base_probs": self.base_probs,
        }
        joblib.dump(payload, path)
        log.info("Saved NewsModel → %s", path)
        return path

    def load(self, path: Optional[Path] = None) -> bool:
        path = Path(path) if path else self.model_path
        if not path.exists():
            log.warning("NewsModel file not found: %s", path)
            return False
        payload = joblib.load(path)
        self.news_dir = Path(payload.get("news_dir", str(RAW_DIR)))
        self.cache = payload.get("cache", {}) or {}
        self.base_probs = payload.get("base_probs", self.base_probs) or self.base_probs
        log.info("Loaded NewsModel ← %s", path)
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_reasoning(
        self,
        home: str, away: str,
        home_factor: float, away_factor: float,
        home_signals: list[str], away_signals: list[str],
        home_delta: float, away_delta: float,
        prob_dict: dict[str, float],
    ) -> str:
        def _team_summary(team: str, factor: float, delta: float, signals: list[str]) -> str:
            if math.isclose(factor, 1.0, abs_tol=0.02):
                tone = "neutral news"
            elif factor > 1.05:
                tone = f"positive news (×{factor:.2f})"
            elif factor < 0.95:
                tone = f"negative news (×{factor:.2f})"
            else:
                tone = f"mildly mixed news (×{factor:.2f})"
            if signals:
                return f"{team}: {tone} — " + "; ".join(signals[:3])
            return f"{team}: {tone}"

        home_blurb = _team_summary(home, home_factor, home_delta, home_signals)
        away_blurb = _team_summary(away, away_factor, away_delta, away_signals)

        best = max(prob_dict.items(), key=lambda kv: kv[1])
        outcome = {"home_win": f"{home} win",
                   "draw": "draw",
                   "away_win": f"{away} win"}[best[0]]
        return (
            f"News Hawk read the press wires. {home_blurb}. {away_blurb}. "
            f"Net news adjustment leans {outcome} at {best[1]:.0%}."
        )


# ---------------------------------------------------------------------------
# CLI helper — quick adjustment dump for a pair
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="News-only match adjustment.")
    parser.add_argument("home", help="Home team name (must match news file slug).")
    parser.add_argument("away", help="Away team name.")
    args = parser.parse_args()

    model = NewsModel()
    out = model.predict_adjustment(args.home, args.away)
    log.info("Prediction: %s", json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
