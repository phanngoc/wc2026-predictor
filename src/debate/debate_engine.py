"""
debate_engine.py
================
3-Model debate ensemble for World Cup 2026 match prediction.

Three analysts each produce a 3-way (home / draw / away) probability
distribution along with a written argument. The DebateEngine then:

  1. Pulls each prediction with reasoning.
  2. Computes a weighted ensemble (default 40/35/25).
  3. Detects disagreement (when models pick different winners or when
     probability spread is high).
  4. Generates a transcript that reads like a panel discussion.
  5. Emits a single consensus prediction with calibrated confidence.

Public surface
--------------
    DebateEngine().predict_match("France", "Brazil", "2026-06-15")

Returns the JSON-serialisable dict described in the task spec.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Imports work both when invoked as a module and when run directly.
try:
    from src.models.statistical_model import StatisticalModel
    from src.models.form_model import FormModel
    from src.models.news_model import NewsModel
except ImportError:  # pragma: no cover - script-mode fallback
    _here = Path(__file__).resolve().parents[2]
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from src.models.statistical_model import StatisticalModel  # type: ignore
    from src.models.form_model import FormModel  # type: ignore
    from src.models.news_model import NewsModel  # type: ignore


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [debate]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("debate_engine")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROB_KEYS = ("home_win", "draw", "away_win")

DEFAULT_WEIGHTS = {
    "historian": 0.40,
    "form_expert": 0.35,
    "news_hawk": 0.25,
}

# Disagreement thresholds
DISAGREEMENT_PROB_GAP = 0.15   # if any two models differ by >= 15% on any
                                # outcome, flag disagreement
DISAGREEMENT_PICK_DIVERGENCE = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _argmax_outcome(prob_dict: dict[str, Any]) -> str:
    return max(
        ((k, v) for k, v in prob_dict.items() if k in PROB_KEYS and isinstance(v, (int, float))),
        key=lambda kv: kv[1],
    )[0]


def _outcome_label(outcome: str, home: str, away: str) -> str:
    return {
        "home_win": f"{home} Win",
        "draw": "Draw",
        "away_win": f"{away} Win",
    }.get(outcome, outcome)


def _normalise(p: dict[str, float]) -> dict[str, float]:
    total = sum(p.get(k, 0.0) for k in PROB_KEYS) or 1.0
    return {k: float(p.get(k, 0.0)) / total for k in PROB_KEYS}


def _confidence_label(conf: float) -> str:
    pct = max(0.0, min(1.0, conf)) * 100.0
    if pct >= 75:
        tier = "High"
    elif pct >= 55:
        tier = "Medium"
    elif pct >= 40:
        tier = "Low"
    else:
        tier = "Very Low"
    return f"{tier} ({pct:.0f}%)"


# ---------------------------------------------------------------------------
# DebateEngine
# ---------------------------------------------------------------------------

@dataclass
class DebateEngine:
    """Synthesises predictions from three analyst models."""

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    historian: StatisticalModel = field(default_factory=StatisticalModel)
    form_expert: FormModel = field(default_factory=FormModel)
    news_hawk: NewsModel = field(default_factory=NewsModel)
    auto_load: bool = True

    def __post_init__(self) -> None:
        self._normalise_weights()
        if self.auto_load:
            self._load_models()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _normalise_weights(self) -> None:
        total = sum(self.weights.values()) or 1.0
        self.weights = {k: float(v) / total for k, v in self.weights.items()}

    def _load_models(self) -> None:
        """Try to load each model from disk; warn but do not fail."""
        loaded = []
        if self.historian.load():
            loaded.append("historian")
        else:
            log.warning("StatisticalModel not loaded — will use ELO fallback.")
        if self.form_expert.load():
            loaded.append("form_expert")
        else:
            log.warning("FormModel not loaded — will use neutral fallback.")
        # News model has near-zero load cost; load() is a no-op if no pkl.
        self.news_hawk.load()
        loaded.append("news_hawk")
        log.info("DebateEngine ready. Loaded analysts: %s", ", ".join(loaded))

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[str | datetime | pd.Timestamp] = None,
        is_world_cup: bool = True,
        is_neutral: bool = False,
    ) -> dict[str, Any]:
        """
        Run all three analysts, synthesise, and return the structured
        prediction described in the task spec.
        """
        match_ts = pd.Timestamp(match_date) if match_date else pd.Timestamp("today")

        # ---- 1. Each analyst speaks --------------------------------
        try:
            historian_out = self.historian.predict(
                home_team, away_team,
                is_world_cup=is_world_cup, is_neutral=is_neutral,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.error("Historian failed: %s", exc, exc_info=True)
            historian_out = self._neutral_vote(home_team, away_team, "Historian unavailable")

        try:
            form_out = self.form_expert.predict(
                home_team, away_team, match_date=match_ts,
            )
        except Exception as exc:  # pragma: no cover
            log.error("Form expert failed: %s", exc, exc_info=True)
            form_out = self._neutral_vote(home_team, away_team, "Form expert unavailable")

        try:
            news_out = self.news_hawk.predict_adjustment(home_team, away_team)
        except Exception as exc:  # pragma: no cover
            log.error("News hawk failed: %s", exc, exc_info=True)
            news_out = self._neutral_vote(home_team, away_team, "News hawk unavailable")

        analysts = {
            "historian": historian_out,
            "form_expert": form_out,
            "news_hawk": news_out,
        }

        # ---- 2. Weighted ensemble ---------------------------------
        ensemble = self._weighted_ensemble(analysts)

        # ---- 3. Detect disagreement -------------------------------
        disagreement_flags = self._detect_disagreement(analysts)

        # ---- 4. Build debate transcript ---------------------------
        transcript, consensus_text = self._build_transcript(
            home_team, away_team, analysts, ensemble, disagreement_flags,
        )

        # ---- 5. Final packaging ------------------------------------
        recommended = _argmax_outcome(ensemble)
        recommended_label = _outcome_label(recommended, home_team, away_team)
        confidence_score = self._calibrated_confidence(
            ensemble, analysts, disagreement_flags,
        )

        debate_block = {
            "historian": self._analyst_block(analysts["historian"], home_team, away_team),
            "form_expert": self._analyst_block(analysts["form_expert"], home_team, away_team),
            "news_hawk": self._analyst_block(analysts["news_hawk"], home_team, away_team),
            "consensus": consensus_text,
            "transcript": transcript,
        }
        if disagreement_flags["any_disagreement"]:
            debate_block["disagreements"] = disagreement_flags["details"]

        return {
            "match": f"{home_team} vs {away_team}",
            "date": match_ts.date().isoformat(),
            "predictions": {
                "home_win": round(ensemble["home_win"], 4),
                "draw": round(ensemble["draw"], 4),
                "away_win": round(ensemble["away_win"], 4),
            },
            "recommended_bet": recommended_label,
            "confidence": _confidence_label(confidence_score),
            "confidence_score": round(confidence_score, 4),
            "debate": debate_block,
            "model_weights": dict(self.weights),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _weighted_ensemble(
        self, analysts: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        """Combine 3 probability dicts with the configured weights."""
        agg = {k: 0.0 for k in PROB_KEYS}
        for name, weight in self.weights.items():
            probs = _normalise(analysts[name])
            for k in PROB_KEYS:
                agg[k] += weight * probs[k]
        return _normalise(agg)

    def _detect_disagreement(
        self, analysts: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Return a dict describing how/where the analysts disagree."""
        picks = {name: _argmax_outcome(out) for name, out in analysts.items()}
        unique_picks = set(picks.values())

        details: list[str] = []
        any_dis = False

        if DISAGREEMENT_PICK_DIVERGENCE and len(unique_picks) > 1:
            any_dis = True
            details.append(
                "Pick divergence: " + ", ".join(
                    f"{name}={pick}" for name, pick in picks.items()
                )
            )

        # Probability gap on any single outcome
        for outcome in PROB_KEYS:
            vals = [_normalise(out)[outcome] for out in analysts.values()]
            spread = max(vals) - min(vals)
            if spread >= DISAGREEMENT_PROB_GAP:
                any_dis = True
                details.append(
                    f"Spread on {outcome}: {min(vals):.0%}–{max(vals):.0%} "
                    f"(Δ={spread:.0%})"
                )

        return {
            "any_disagreement": any_dis,
            "picks": picks,
            "unique_picks": list(unique_picks),
            "details": details,
        }

    def _build_transcript(
        self,
        home: str, away: str,
        analysts: dict[str, dict[str, Any]],
        ensemble: dict[str, float],
        disagreement: dict[str, Any],
    ) -> tuple[str, str]:
        """Produce a panel-style transcript + a one-paragraph consensus."""
        h = analysts["historian"]
        f = analysts["form_expert"]
        n = analysts["news_hawk"]

        h_pick = _argmax_outcome(h)
        f_pick = _argmax_outcome(f)
        n_pick = _argmax_outcome(n)

        h_label = _outcome_label(h_pick, home, away)
        f_label = _outcome_label(f_pick, home, away)
        n_label = _outcome_label(n_pick, home, away)

        lines: list[str] = []
        lines.append(
            f"HISTORIAN: I lean {h_label} at {h[h_pick]:.0%}. {h.get('reasoning', '')}"
        )

        # Form expert reacts to historian
        if f_pick == h_pick:
            lines.append(
                f"FORM EXPERT: I agree on {f_label}, my model gives it "
                f"{f[f_pick]:.0%}. {f.get('reasoning', '')}"
            )
        else:
            lines.append(
                f"FORM EXPERT: I have to push back — my read is {f_label} at "
                f"{f[f_pick]:.0%}, not {h_label}. {f.get('reasoning', '')}"
            )

        # News hawk weighs in
        if n_pick == h_pick == f_pick:
            lines.append(
                f"NEWS HAWK: Press wires confirm — {n_label} at {n[n_pick]:.0%}. "
                f"{n.get('reasoning', '')}"
            )
        elif n_pick != h_pick or n_pick != f_pick:
            lines.append(
                f"NEWS HAWK: Hold on — current news points more towards "
                f"{n_label} at {n[n_pick]:.0%}. {n.get('reasoning', '')}"
            )
        else:
            lines.append(
                f"NEWS HAWK: News confirms {n_label} at {n[n_pick]:.0%}. "
                f"{n.get('reasoning', '')}"
            )

        # Resolution line
        ens_pick = _argmax_outcome(ensemble)
        ens_label = _outcome_label(ens_pick, home, away)
        ens_pct = ensemble[ens_pick] * 100

        if disagreement["any_disagreement"]:
            lines.append(
                f"MODERATOR: Panel split — {self._disagreement_summary(disagreement)}. "
                f"Weighted ensemble settles on {ens_label} at {ens_pct:.0f}%."
            )
        else:
            lines.append(
                f"MODERATOR: All three analysts agree on {ens_label}. "
                f"Weighted ensemble: {ens_pct:.0f}%."
            )

        transcript = "\n".join(lines)

        # ---- consensus paragraph ----
        agree_count = sum(1 for p in (h_pick, f_pick, n_pick) if p == ens_pick)
        if agree_count == 3:
            consensus = (
                f"All 3 analysts converge on {ens_label}. "
                f"Confidence is high; no model dissents."
            )
        elif agree_count == 2:
            dissenters = [name for name, pick in disagreement["picks"].items() if pick != ens_pick]
            dissent_label = ", ".join(dissenters)
            key_concern = ""
            if "news_hawk" in dissenters:
                key_concern = " Key debate: news_hawk raised current-news concerns."
            elif "form_expert" in dissenters:
                key_concern = " Key debate: form_expert sees momentum pointing the other way."
            elif "historian" in dissenters:
                key_concern = " Key debate: historian's longer-term ELO view diverges."
            consensus = (
                f"2 of 3 analysts favor {ens_label}; {dissent_label} dissents."
                f"{key_concern}"
            )
        else:
            consensus = (
                f"Analysts split three ways. Weighted ensemble breaks the tie "
                f"on {ens_label} at {ens_pct:.0f}%, but treat with caution."
            )

        return transcript, consensus

    def _disagreement_summary(self, disagreement: dict[str, Any]) -> str:
        if not disagreement["details"]:
            return "no major disagreement"
        return disagreement["details"][0]

    def _calibrated_confidence(
        self,
        ensemble: dict[str, float],
        analysts: dict[str, dict[str, Any]],
        disagreement: dict[str, Any],
    ) -> float:
        """
        Confidence is a blend of:
          - ensemble margin (top - second)
          - per-analyst confidence (weighted average)
          - penalty when analysts disagree
        """
        ordered = sorted(ensemble.values(), reverse=True)
        margin = ordered[0] - ordered[1]
        margin_score = min(1.0, 0.40 + margin * 1.6)

        weighted_conf = 0.0
        for name, w in self.weights.items():
            weighted_conf += w * float(analysts[name].get("confidence", 0.5))

        base = 0.6 * margin_score + 0.4 * weighted_conf

        # Disagreement penalty
        if disagreement["any_disagreement"]:
            n_unique = len(disagreement["unique_picks"])
            if n_unique == 3:
                base *= 0.75
            elif n_unique == 2:
                base *= 0.88

        return max(0.0, min(1.0, base))

    def _analyst_block(
        self,
        out: dict[str, Any],
        home: str, away: str,
    ) -> dict[str, Any]:
        pick = _argmax_outcome(out)
        return {
            "prediction": pick,
            "prediction_label": _outcome_label(pick, home, away),
            "probability": round(float(out[pick]), 4),
            "probabilities": {
                "home_win": round(float(out["home_win"]), 4),
                "draw": round(float(out["draw"]), 4),
                "away_win": round(float(out["away_win"]), 4),
            },
            "confidence": round(float(out.get("confidence", 0.5)), 4),
            "reasoning": out.get("reasoning", ""),
        }

    def _neutral_vote(self, home: str, away: str, why: str) -> dict[str, Any]:
        return {
            "home_win": 0.40,
            "draw": 0.28,
            "away_win": 0.32,
            "confidence": 0.30,
            "reasoning": f"{why}; using neutral fallback distribution.",
            "features_used": {},
        }


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Run the 3-model debate engine for a single match.")
    parser.add_argument("home", help="Home team name")
    parser.add_argument("away", help="Away team name")
    parser.add_argument("--date", default=None, help="Match date YYYY-MM-DD")
    parser.add_argument("--neutral", action="store_true", help="Match on neutral ground")
    parser.add_argument("--no-wc", action="store_true", help="Treat as friendly, not WC")
    args = parser.parse_args()

    engine = DebateEngine()
    out = engine.predict_match(
        args.home, args.away,
        match_date=args.date,
        is_world_cup=not args.no_wc,
        is_neutral=args.neutral,
    )
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
