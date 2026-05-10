"""
form_model.py
=============
The Form Expert — recent-form / momentum based prediction model.

Trains a Gradient Boosting Classifier on engineered features derived
from each team's last 5 matches, with an exponential-decay weighting
that emphasizes the most recent results, and a tournament-importance
weight (World Cup > continental cups > qualifiers > friendlies).

Loads ``data/processed/historical_features.csv`` and filters to
matches from 2018 onwards (recent form is the focus).

Outputs
-------
models/form_model.pkl
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Paths & logging
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
HISTORICAL_FEATURES_PATH = PROCESSED_DIR / "historical_features.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "form_model.pkl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [form]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("form_model")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSES = ["L", "D", "W"]
CLASS_TO_KEY = {"W": "home_win", "D": "draw", "L": "away_win"}

LOOKBACK = 5            # last N matches considered
RECENT_CUTOFF = "2018-01-01"

# Tournament importance multipliers (used to weight matches in form calc)
TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 1.50,
    "World Cup qualification": 1.20,
    "UEFA Euro": 1.30,
    "UEFA Euro qualification": 1.10,
    "Copa America": 1.30,
    "African Cup of Nations": 1.20,
    "AFC Asian Cup": 1.20,
    "CONCACAF": 1.15,
    "Friendly": 0.70,
    "UEFA Nations League": 1.05,
}

FEATURE_COLUMNS: list[str] = [
    # Per-team last-5 derived
    "home_form_score",
    "home_momentum",
    "home_goal_diff_avg",
    "home_goals_scored_avg",
    "home_goals_conceded_avg",
    "home_home_form",         # form when playing at home
    "away_form_score",
    "away_momentum",
    "away_goal_diff_avg",
    "away_goals_scored_avg",
    "away_goals_conceded_avg",
    "away_away_form",         # form when playing away
    # Differentials
    "form_diff",
    "momentum_diff",
    "goal_diff_diff",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result_to_class(result: str) -> str:
    return result if result in CLASSES else "D"


def _result_to_points(result: str) -> int:
    return {"W": 3, "D": 1, "L": 0}.get(result, 0)


def _tournament_weight(tournament: Optional[str]) -> float:
    """Return the importance multiplier for a tournament name."""
    if not tournament:
        return 1.0
    name = str(tournament)
    for key, weight in TOURNAMENT_WEIGHTS.items():
        if key.lower() in name.lower():
            return weight
    return 1.0


def _exponential_weights(n: int, decay: float = 0.65) -> np.ndarray:
    """
    Return a length-n array of weights that *decay* into the past.
    Index n-1 is the most-recent match (highest weight).
    With decay=0.65: [0.07, 0.10, 0.16, 0.25, 0.39, ...] roughly.
    """
    if n <= 0:
        return np.array([])
    w = np.array([decay ** (n - 1 - i) for i in range(n)], dtype=float)
    return w / w.sum()


# ---------------------------------------------------------------------------
# Per-team match history view
# ---------------------------------------------------------------------------

def _build_team_history(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build, for each team, a sorted DataFrame of their matches with
    columns: date, opponent, goals_for, goals_against, result, is_home,
    tournament, weight (tournament weight).
    """
    log.info("Building per-team match history …")
    teams = set(df["home_team"]).union(set(df["away_team"]))
    history: dict[str, pd.DataFrame] = {}

    home_view = df[["date", "home_team", "away_team",
                    "home_score", "away_score", "tournament"]].copy()
    home_view.columns = ["date", "team", "opponent", "gf", "ga", "tournament"]
    home_view["is_home"] = True

    away_view = df[["date", "away_team", "home_team",
                    "away_score", "home_score", "tournament"]].copy()
    away_view.columns = ["date", "team", "opponent", "gf", "ga", "tournament"]
    away_view["is_home"] = False

    full = pd.concat([home_view, away_view], ignore_index=True)
    full["result"] = np.where(
        full["gf"] > full["ga"], "W",
        np.where(full["gf"] < full["ga"], "L", "D"),
    )
    full["weight"] = full["tournament"].apply(_tournament_weight)

    for team in teams:
        team_rows = full[full["team"] == team].sort_values("date").reset_index(drop=True)
        history[team] = team_rows

    log.info("Built history for %d teams.", len(history))
    return history


# ---------------------------------------------------------------------------
# Form feature computation
# ---------------------------------------------------------------------------

def _compute_form_features(
    team_history: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    lookback: int = LOOKBACK,
) -> dict[str, float]:
    """
    Given a single team's full chronological history and a cutoff date,
    return a dict of last-N form features for use as model inputs.
    """
    past = team_history[team_history["date"] < cutoff_date]
    last_n = past.tail(lookback)

    if len(last_n) == 0:
        return {
            "form_score": 0.5,
            "momentum": 0.5,
            "goal_diff_avg": 0.0,
            "goals_scored_avg": 1.0,
            "goals_conceded_avg": 1.0,
            "home_form": 0.5,
            "away_form": 0.5,
        }

    n = len(last_n)
    weights_recency = _exponential_weights(n)
    tournament_weights = last_n["weight"].to_numpy()
    combined = weights_recency * tournament_weights
    combined_n = combined / combined.sum()

    points = np.array([_result_to_points(r) for r in last_n["result"]], dtype=float)
    form_score = float(np.sum(points * combined_n) / 3.0)   # normalised 0..1

    # Momentum: recent matches weight even more heavily
    sharper = _exponential_weights(n, decay=0.45)
    momentum = float(np.sum(points * sharper) / 3.0)

    gf = last_n["gf"].astype(float).to_numpy()
    ga = last_n["ga"].astype(float).to_numpy()
    goal_diff_avg = float(np.mean(gf - ga))
    goals_scored_avg = float(np.mean(gf))
    goals_conceded_avg = float(np.mean(ga))

    home_subset = last_n[last_n["is_home"] == True]   # noqa: E712
    away_subset = last_n[last_n["is_home"] == False]  # noqa: E712

    def _subset_form(subset: pd.DataFrame) -> float:
        if len(subset) == 0:
            return 0.5
        pts = np.array([_result_to_points(r) for r in subset["result"]], dtype=float)
        return float(np.mean(pts) / 3.0)

    return {
        "form_score": form_score,
        "momentum": momentum,
        "goal_diff_avg": goal_diff_avg,
        "goals_scored_avg": goals_scored_avg,
        "goals_conceded_avg": goals_conceded_avg,
        "home_form": _subset_form(home_subset),
        "away_form": _subset_form(away_subset),
    }


# ---------------------------------------------------------------------------
# FormModel
# ---------------------------------------------------------------------------

@dataclass
class FormModel:
    """Recent-form / momentum based predictor."""

    model_path: Path = field(default=DEFAULT_MODEL_PATH)
    classifier: Optional[Any] = None
    label_encoder: Optional[LabelEncoder] = None
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    team_history: dict[str, pd.DataFrame] = field(default_factory=dict)
    training_metrics: dict[str, float] = field(default_factory=dict)
    last_known_date: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> dict[str, float]:
        log.info("Preparing training data (rows=%d) …", len(df))
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Filter to recent matches — recent form is the focus
        recent_mask = df["date"] >= pd.Timestamp(RECENT_CUTOFF)
        df = df[recent_mask].sort_values("date").reset_index(drop=True)
        log.info("Filtered to %d matches since %s", len(df), RECENT_CUTOFF)

        if len(df) < 200:
            log.warning("Very small training set (%d rows). "
                        "Model may be noisy.", len(df))

        df["home_result"] = df["home_result"].map(_result_to_class)

        # Build full team history (uses the *unfiltered* df so we have
        # enough lookback even for matches near the start of 2018).
        # We rebuild here from the filtered df; for lookback purposes the
        # caller can also supply pre-built history via train_models.py.
        self.team_history = _build_team_history(df)

        # Compute features for each match using the team history
        log.info("Computing per-match form features …")
        rows: list[dict[str, float]] = []
        targets: list[str] = []
        for _, match in df.iterrows():
            home_hist = self.team_history.get(match["home_team"])
            away_hist = self.team_history.get(match["away_team"])
            if home_hist is None or away_hist is None:
                continue
            home_feats = _compute_form_features(home_hist, match["date"])
            away_feats = _compute_form_features(away_hist, match["date"])

            row = {
                "home_form_score": home_feats["form_score"],
                "home_momentum": home_feats["momentum"],
                "home_goal_diff_avg": home_feats["goal_diff_avg"],
                "home_goals_scored_avg": home_feats["goals_scored_avg"],
                "home_goals_conceded_avg": home_feats["goals_conceded_avg"],
                "home_home_form": home_feats["home_form"],
                "away_form_score": away_feats["form_score"],
                "away_momentum": away_feats["momentum"],
                "away_goal_diff_avg": away_feats["goal_diff_avg"],
                "away_goals_scored_avg": away_feats["goals_scored_avg"],
                "away_goals_conceded_avg": away_feats["goals_conceded_avg"],
                "away_away_form": away_feats["away_form"],
                "form_diff": home_feats["form_score"] - away_feats["form_score"],
                "momentum_diff": home_feats["momentum"] - away_feats["momentum"],
                "goal_diff_diff": home_feats["goal_diff_avg"] - away_feats["goal_diff_avg"],
            }
            rows.append(row)
            targets.append(match["home_result"])

        if not rows:
            raise ValueError("No usable training rows after feature computation. "
                             "Check that the dataset contains matches >= 2018.")

        feat_df = pd.DataFrame(rows)
        X = feat_df[self.feature_columns].astype(float).values
        y_raw = np.array(targets)

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(CLASSES)
        y = self.label_encoder.transform(y_raw)

        log.info("Train matrix: X=%s  y=%s", X.shape, y.shape)

        # ----------------- Time-series cross-validation -----------------
        tscv = TimeSeriesSplit(n_splits=5)
        cv_acc: list[float] = []
        cv_logloss: list[float] = []
        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), start=1):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            clf = self._make_classifier()
            clf.fit(X_tr, y_tr)
            proba = clf.predict_proba(X_va)
            preds = np.argmax(proba, axis=1)
            cv_acc.append(accuracy_score(y_va, preds))
            try:
                cv_logloss.append(log_loss(y_va, proba, labels=list(range(len(CLASSES)))))
            except ValueError:
                cv_logloss.append(float("nan"))
            log.info("  fold %d: acc=%.4f  logloss=%.4f", fold, cv_acc[-1], cv_logloss[-1])

        self.classifier = self._make_classifier()
        self.classifier.fit(X, y)

        self.last_known_date = df["date"].max()

        self.training_metrics = {
            "cv_accuracy_mean": float(np.nanmean(cv_acc)) if cv_acc else 0.0,
            "cv_accuracy_std": float(np.nanstd(cv_acc)) if cv_acc else 0.0,
            "cv_logloss_mean": float(np.nanmean(cv_logloss)) if cv_logloss else 0.0,
            "n_train_rows": int(len(rows)),
            "n_features": int(len(self.feature_columns)),
        }
        log.info("Training complete. CV-acc=%.4f  CV-logloss=%.4f",
                 self.training_metrics["cv_accuracy_mean"],
                 self.training_metrics["cv_logloss_mean"])
        return self.training_metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[pd.Timestamp] = None,
        elo_ratings_dict: Optional[dict[str, float]] = None,  # accepted for API symmetry
    ) -> dict[str, Any]:
        """
        Predict W/D/L probabilities given the two teams. The match date
        defaults to "today" — the model uses each team's last 5 matches
        on or before that date.
        """
        match_date = pd.Timestamp(match_date) if match_date else (self.last_known_date or pd.Timestamp("today"))

        if self.classifier is None or self.label_encoder is None:
            log.debug("FormModel not trained — using neutral fallback.")
            return self._fallback_prediction(home_team, away_team)

        home_hist = self.team_history.get(home_team)
        away_hist = self.team_history.get(away_team)

        if home_hist is None or away_hist is None:
            log.warning("No history for %s or %s — using fallback.", home_team, away_team)
            return self._fallback_prediction(home_team, away_team)

        home_feats = _compute_form_features(home_hist, match_date)
        away_feats = _compute_form_features(away_hist, match_date)
        feat_dict = {
            "home_form_score": home_feats["form_score"],
            "home_momentum": home_feats["momentum"],
            "home_goal_diff_avg": home_feats["goal_diff_avg"],
            "home_goals_scored_avg": home_feats["goals_scored_avg"],
            "home_goals_conceded_avg": home_feats["goals_conceded_avg"],
            "home_home_form": home_feats["home_form"],
            "away_form_score": away_feats["form_score"],
            "away_momentum": away_feats["momentum"],
            "away_goal_diff_avg": away_feats["goal_diff_avg"],
            "away_goals_scored_avg": away_feats["goals_scored_avg"],
            "away_goals_conceded_avg": away_feats["goals_conceded_avg"],
            "away_away_form": away_feats["away_form"],
            "form_diff": home_feats["form_score"] - away_feats["form_score"],
            "momentum_diff": home_feats["momentum"] - away_feats["momentum"],
            "goal_diff_diff": home_feats["goal_diff_avg"] - away_feats["goal_diff_avg"],
        }
        x = np.array([[feat_dict[c] for c in self.feature_columns]], dtype=float)
        proba = self.classifier.predict_proba(x)[0]

        prob_dict: dict[str, float] = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
        for i, cls_label in enumerate(self.label_encoder.classes_):
            prob_dict[CLASS_TO_KEY[cls_label]] = float(proba[i])

        reasoning = self._build_reasoning(home_team, away_team, home_feats, away_feats, prob_dict)
        return self._package(prob_dict, reasoning=reasoning, features=feat_dict)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path) if path else self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        # Serialize per-team history as records (DataFrames don't pickle as
        # nicely cross-version, but joblib handles them fine; we still
        # squash to dicts to keep the payload portable).
        history_records = {
            team: hist.to_dict(orient="list")
            for team, hist in self.team_history.items()
        }
        payload = {
            "classifier": self.classifier,
            "label_encoder": self.label_encoder,
            "feature_columns": self.feature_columns,
            "team_history": history_records,
            "last_known_date": self.last_known_date.isoformat() if self.last_known_date is not None else None,
            "training_metrics": self.training_metrics,
        }
        joblib.dump(payload, path)
        log.info("Saved FormModel → %s", path)
        return path

    def load(self, path: Optional[Path] = None) -> bool:
        path = Path(path) if path else self.model_path
        if not path.exists():
            log.warning("FormModel file not found: %s", path)
            return False
        payload = joblib.load(path)
        self.classifier = payload.get("classifier")
        self.label_encoder = payload.get("label_encoder")
        self.feature_columns = payload.get("feature_columns", list(FEATURE_COLUMNS))
        history_records = payload.get("team_history", {})
        self.team_history = {
            team: pd.DataFrame(records) for team, records in history_records.items()
        }
        # Re-coerce date columns to Timestamp
        for hist in self.team_history.values():
            if "date" in hist.columns:
                hist["date"] = pd.to_datetime(hist["date"])
        last_date = payload.get("last_known_date")
        self.last_known_date = pd.to_datetime(last_date) if last_date else None
        self.training_metrics = payload.get("training_metrics", {})
        log.info("Loaded FormModel ← %s  (CV-acc=%.4f)",
                 path, self.training_metrics.get("cv_accuracy_mean", 0.0))
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_classifier(self):
        return GradientBoostingClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            random_state=42,
        )

    def _fallback_prediction(self, home: str, away: str) -> dict[str, Any]:
        """Neutral 3-way prediction when team history is unavailable."""
        probs = {"home_win": 0.40, "draw": 0.28, "away_win": 0.32}
        reasoning = (
            f"FormModel fallback: insufficient recent history for "
            f"{home}/{away}. Using neutral baseline (slight home edge)."
        )
        return self._package(probs, reasoning=reasoning, features={})

    def _build_reasoning(
        self,
        home: str, away: str,
        h_feats: dict[str, float], a_feats: dict[str, float],
        prob_dict: dict[str, float],
    ) -> str:
        h_form_pct = h_feats["form_score"] * 100
        a_form_pct = a_feats["form_score"] * 100
        h_gd = h_feats["goal_diff_avg"]
        a_gd = a_feats["goal_diff_avg"]
        h_mom = h_feats["momentum"]
        a_mom = a_feats["momentum"]

        if h_form_pct > a_form_pct + 15:
            form_blurb = f"{home} clearly hotter ({h_form_pct:.0f}% form vs {a_form_pct:.0f}%)"
        elif a_form_pct > h_form_pct + 15:
            form_blurb = f"{away} carrying better form ({a_form_pct:.0f}% vs {h_form_pct:.0f}%)"
        else:
            form_blurb = f"form is close — {home} {h_form_pct:.0f}% / {away} {a_form_pct:.0f}%"

        mom_diff = h_mom - a_mom
        if abs(mom_diff) < 0.05:
            mom_blurb = "momentum even"
        elif mom_diff > 0:
            mom_blurb = f"{home} has fresher momentum (+{mom_diff:.2f})"
        else:
            mom_blurb = f"{away} has fresher momentum ({-mom_diff:+.2f})"

        gd_blurb = (
            f"avg GD: {home} {h_gd:+.2f} (scored {h_feats['goals_scored_avg']:.2f}, "
            f"conceded {h_feats['goals_conceded_avg']:.2f}) "
            f"vs {away} {a_gd:+.2f} (scored {a_feats['goals_scored_avg']:.2f}, "
            f"conceded {a_feats['goals_conceded_avg']:.2f})"
        )

        best = max(prob_dict.items(), key=lambda kv: kv[1])
        outcome = {"home_win": f"{home} win",
                   "draw": "draw",
                   "away_win": f"{away} win"}[best[0]]
        return (
            f"Recent form analysis (last 5 weighted by recency + tournament tier): "
            f"{form_blurb}. {mom_blurb}. {gd_blurb}. "
            f"Form model leans {outcome} at {best[1]:.0%}."
        )

    def _package(
        self,
        prob_dict: dict[str, float],
        *,
        reasoning: str,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        total = sum(prob_dict.values()) or 1.0
        norm = {k: v / total for k, v in prob_dict.items()}
        sorted_p = sorted(norm.values(), reverse=True)
        gap = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else float(sorted_p[0])
        confidence = max(0.0, min(1.0, 0.4 + gap * 1.5))
        return {
            "home_win": round(float(norm["home_win"]), 4),
            "draw": round(float(norm["draw"]), 4),
            "away_win": round(float(norm["away_win"]), 4),
            "confidence": round(confidence, 4),
            "reasoning": reasoning,
            "features_used": features,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    if not HISTORICAL_FEATURES_PATH.exists():
        log.error("Historical features not found at %s. "
                  "Run download_historical.py first.", HISTORICAL_FEATURES_PATH)
        sys.exit(1)
    df = pd.read_csv(HISTORICAL_FEATURES_PATH)
    model = FormModel()
    metrics = model.train(df)
    model.save()
    log.info("Done. Metrics: %s", metrics)


if __name__ == "__main__":
    main()
