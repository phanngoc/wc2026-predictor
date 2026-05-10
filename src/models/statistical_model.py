"""
statistical_model.py
====================
The Historian — historical statistics & ELO based prediction model.

Loads ``data/processed/historical_features.csv`` (produced by
``src/data/download_historical.py``), trains an ensemble of XGBoost +
RandomForest classifiers on a rich set of ELO / rolling-form / H2H
features, and exposes a thin ``StatisticalModel`` class that the
debate engine consumes.

The class can also operate without a trained model by providing an
ELO-only fallback so downstream code (and the debate engine) never
crashes when historical data is missing.

Outputs
-------
models/statistical_model.pkl   - joblib-serialized trained ensemble +
                                 metadata (feature list, ELO snapshot,
                                 team form snapshot, training metrics).

Usage
-----
    from src.models.statistical_model import StatisticalModel
    m = StatisticalModel()
    m.train(df)              # df = historical_features.csv as DataFrame
    m.save()
    m.load()
    m.predict("France", "Brazil")
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

# Optional dependency — XGBoost. Fall back gracefully if missing.
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # pragma: no cover - import-time fallback
    XGBClassifier = None  # type: ignore[assignment]
    _HAS_XGB = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Paths & logging
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]   # wc2026-predictor/
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
HISTORICAL_FEATURES_PATH = PROCESSED_DIR / "historical_features.csv"
ELO_RATINGS_PATH = PROCESSED_DIR / "elo_ratings.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "statistical_model.pkl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [statistical]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("statistical_model")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ordered class labels (target = home_result mapped to W/D/L).
CLASSES = ["L", "D", "W"]   # 0=L (away win), 1=D, 2=W (home win)
CLASS_TO_KEY = {"W": "home_win", "D": "draw", "L": "away_win"}

FEATURE_COLUMNS: list[str] = [
    "elo_home_before",
    "elo_away_before",
    "elo_diff",
    "home_win_rate_10",
    "away_win_rate_10",
    "home_goals_scored_avg_5",
    "away_goals_scored_avg_5",
    "home_goals_conceded_avg_5",
    "away_goals_conceded_avg_5",
    "h2h_home_win_rate",
    "h2h_matches_count",
    "is_world_cup",
    "has_home_advantage",
]

DEFAULT_ELO = 1500.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result_to_class(result: str) -> str:
    """Map the dataset's W/D/L home-perspective result to our class label."""
    if result in CLASSES:
        return result
    # Fallback (defensive): unknown → draw
    return "D"


def _elo_only_probabilities(elo_home: float, elo_away: float,
                            home_advantage_pts: float = 60.0) -> dict[str, float]:
    """
    Pure-ELO 3-way probability fallback.

    Uses the standard ELO expected-score then carves out a draw mass that
    grows when the two teams are evenly matched.
    """
    diff = (elo_home + home_advantage_pts) - elo_away
    p_home = 1.0 / (1.0 + 10 ** (-diff / 400.0))
    p_away = 1.0 - p_home

    # Draw probability peaks when teams are equal; shrinks for big gaps.
    closeness = math.exp(-(diff ** 2) / (2 * (180.0 ** 2)))
    p_draw = 0.18 + 0.20 * closeness   # 0.18-0.38 range
    scale = 1.0 - p_draw
    p_home_final = p_home * scale
    p_away_final = p_away * scale

    return {
        "home_win": float(p_home_final),
        "draw": float(p_draw),
        "away_win": float(p_away_final),
    }


# ---------------------------------------------------------------------------
# Form snapshot — last-known rolling stats per team
# ---------------------------------------------------------------------------

@dataclass
class TeamFormSnapshot:
    """Per-team most-recent rolling stats used at predict-time."""
    win_rate_10: float = 0.5
    goals_scored_avg_5: float = 1.2
    goals_conceded_avg_5: float = 1.2
    matches_seen: int = 0


def _build_form_snapshot(df: pd.DataFrame) -> dict[str, TeamFormSnapshot]:
    """
    Walk the (already chronologically sorted) DataFrame keeping the *latest*
    rolling-form values seen for each team. Both home and away rows
    contribute. Result: dict team -> TeamFormSnapshot.
    """
    snap: dict[str, TeamFormSnapshot] = {}
    df_sorted = df.sort_values("date")

    for _, row in df_sorted.iterrows():
        for side, team in (("home", row["home_team"]), ("away", row["away_team"])):
            wr = row.get(f"{side}_win_rate_10")
            gs = row.get(f"{side}_goals_scored_avg_5")
            gc = row.get(f"{side}_goals_conceded_avg_5")
            if pd.notna(wr) and pd.notna(gs) and pd.notna(gc):
                snap[team] = TeamFormSnapshot(
                    win_rate_10=float(wr),
                    goals_scored_avg_5=float(gs),
                    goals_conceded_avg_5=float(gc),
                    matches_seen=snap.get(team, TeamFormSnapshot()).matches_seen + 1,
                )
    return snap


def _build_h2h_snapshot(df: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    """
    Build a head-to-head lookup keyed by *unordered* team pair so we can
    recover historical H2H stats at inference time.

    Returns dict (team_a, team_b) -> {"home_win_rate": x, "matches_count": n}
    where the rate is from team_a's perspective (alphabetically first team).
    """
    h2h: dict[tuple[str, str], dict[str, float]] = {}
    if "h2h_home_win_rate" not in df.columns:
        return h2h
    df_sorted = df.sort_values("date")
    for _, row in df_sorted.iterrows():
        home, away = row["home_team"], row["away_team"]
        rate = row.get("h2h_home_win_rate")
        cnt = row.get("h2h_matches_count", 0)
        if pd.isna(rate):
            continue
        # Store from the home team's perspective for this row
        key = tuple(sorted([home, away]))
        # Determine perspective: rate is wrt 'home' on this row.
        if key[0] == home:
            perspective_rate = float(rate)
        else:
            perspective_rate = 1.0 - float(rate)
        h2h[key] = {
            "home_win_rate": perspective_rate,
            "matches_count": float(cnt or 0),
        }
    return h2h


# ---------------------------------------------------------------------------
# StatisticalModel
# ---------------------------------------------------------------------------

@dataclass
class StatisticalModel:
    """
    Historical / ELO-driven match outcome predictor.

    Public methods
    --------------
    train(df)        -> training metrics dict
    predict(home, away, elo_ratings_dict=None) -> prediction dict
    save(path=None)  -> persisted model path
    load(path=None)  -> True on success
    """

    model_path: Path = field(default=DEFAULT_MODEL_PATH)
    xgb_model: Optional[Any] = None
    rf_model: Optional[Any] = None
    label_encoder: Optional[LabelEncoder] = None
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    elo_snapshot: dict[str, float] = field(default_factory=dict)
    form_snapshot: dict[str, TeamFormSnapshot] = field(default_factory=dict)
    h2h_snapshot: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)
    training_metrics: dict[str, float] = field(default_factory=dict)
    _ensemble_weights: tuple[float, float] = (0.55, 0.45)   # xgb, rf

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> dict[str, float]:
        """Fit the XGBoost + RF ensemble. Returns mean CV metrics."""
        log.info("Preparing training data (rows=%d) …", len(df))
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Drop matches without rolling-form features (early history)
        required = ["home_win_rate_10", "away_win_rate_10",
                    "home_goals_scored_avg_5", "away_goals_scored_avg_5",
                    "home_goals_conceded_avg_5", "away_goals_conceded_avg_5"]
        df = df.dropna(subset=required).reset_index(drop=True)

        # Fill optional columns with sensible defaults
        df["h2h_home_win_rate"] = df.get("h2h_home_win_rate", pd.Series([np.nan] * len(df))).fillna(0.5)
        df["h2h_matches_count"] = df.get("h2h_matches_count", pd.Series([0] * len(df))).fillna(0)
        df["is_world_cup"] = df.get("is_world_cup", pd.Series([False] * len(df))).astype(int)
        df["has_home_advantage"] = df.get("has_home_advantage", pd.Series([True] * len(df))).astype(int)

        df["home_result"] = df["home_result"].map(_result_to_class)

        X = df[self.feature_columns].astype(float).values
        y_raw = df["home_result"].values

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(CLASSES)
        y = self.label_encoder.transform(y_raw)

        log.info("Train matrix: X=%s  y=%s  classes=%s",
                 X.shape, y.shape, list(self.label_encoder.classes_))

        # ----------------- Time-series cross-validation -----------------
        tscv = TimeSeriesSplit(n_splits=5)
        cv_acc: list[float] = []
        cv_logloss: list[float] = []
        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), start=1):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            xgb = self._make_xgb()
            rf = self._make_rf()
            xgb.fit(X_tr, y_tr)
            rf.fit(X_tr, y_tr)

            proba = self._weighted_proba(xgb, rf, X_va)
            preds = np.argmax(proba, axis=1)
            cv_acc.append(accuracy_score(y_va, preds))
            try:
                cv_logloss.append(log_loss(y_va, proba, labels=list(range(len(CLASSES)))))
            except ValueError:
                cv_logloss.append(float("nan"))
            log.info("  fold %d: acc=%.4f  logloss=%.4f", fold, cv_acc[-1], cv_logloss[-1])

        # ----------------- Final fit on all data -----------------
        self.xgb_model = self._make_xgb()
        self.rf_model = self._make_rf()
        self.xgb_model.fit(X, y)
        self.rf_model.fit(X, y)

        # ----------------- Build inference snapshots -----------------
        self.elo_snapshot = self._build_elo_snapshot(df)
        self.form_snapshot = _build_form_snapshot(df)
        self.h2h_snapshot = _build_h2h_snapshot(df)

        self.training_metrics = {
            "cv_accuracy_mean": float(np.nanmean(cv_acc)) if cv_acc else 0.0,
            "cv_accuracy_std": float(np.nanstd(cv_acc)) if cv_acc else 0.0,
            "cv_logloss_mean": float(np.nanmean(cv_logloss)) if cv_logloss else 0.0,
            "n_train_rows": int(len(df)),
            "n_features": int(len(self.feature_columns)),
            "has_xgb": _HAS_XGB,
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
        elo_ratings_dict: Optional[dict[str, float]] = None,
        is_world_cup: bool = True,
        is_neutral: bool = False,
    ) -> dict[str, Any]:
        """
        Predict W/D/L probabilities for a single match.

        Parameters
        ----------
        home_team / away_team : team names (must match dataset spelling)
        elo_ratings_dict      : optional override of {team -> elo}; falls
                                back to the snapshot captured at train-time.
        is_world_cup          : if True, marks the match as WC for the model
        is_neutral            : if True, removes home advantage flag

        Returns dict:
        {
            "home_win": float, "draw": float, "away_win": float,
            "confidence": float,  # max(p) - second-best gap
            "reasoning": str,
            "features_used": dict
        }
        """
        elo_ratings_dict = elo_ratings_dict or self.elo_snapshot
        elo_h = float(elo_ratings_dict.get(home_team, DEFAULT_ELO))
        elo_a = float(elo_ratings_dict.get(away_team, DEFAULT_ELO))

        # If the model is not trained, return ELO-only fallback.
        if self.xgb_model is None or self.rf_model is None or self.label_encoder is None:
            log.debug("Model not trained — using ELO-only fallback.")
            probs = _elo_only_probabilities(elo_h, elo_a,
                                            home_advantage_pts=0.0 if is_neutral else 60.0)
            return self._package(
                probs,
                reasoning=(
                    f"Untrained fallback: ELO {home_team}={elo_h:.0f}, "
                    f"{away_team}={elo_a:.0f}. ELO diff={elo_h - elo_a:+.0f}."
                ),
                features={"elo_home": elo_h, "elo_away": elo_a, "elo_diff": elo_h - elo_a},
            )

        # Build feature vector from snapshots
        home_form = self.form_snapshot.get(home_team, TeamFormSnapshot())
        away_form = self.form_snapshot.get(away_team, TeamFormSnapshot())
        h2h_key = tuple(sorted([home_team, away_team]))
        h2h_info = self.h2h_snapshot.get(h2h_key, {"home_win_rate": 0.5, "matches_count": 0})
        # Re-orient to current home perspective
        if h2h_key[0] == home_team:
            h2h_rate = h2h_info["home_win_rate"]
        else:
            h2h_rate = 1.0 - h2h_info["home_win_rate"]

        feature_dict = {
            "elo_home_before": elo_h,
            "elo_away_before": elo_a,
            "elo_diff": elo_h - elo_a,
            "home_win_rate_10": home_form.win_rate_10,
            "away_win_rate_10": away_form.win_rate_10,
            "home_goals_scored_avg_5": home_form.goals_scored_avg_5,
            "away_goals_scored_avg_5": away_form.goals_scored_avg_5,
            "home_goals_conceded_avg_5": home_form.goals_conceded_avg_5,
            "away_goals_conceded_avg_5": away_form.goals_conceded_avg_5,
            "h2h_home_win_rate": float(h2h_rate),
            "h2h_matches_count": float(h2h_info["matches_count"]),
            "is_world_cup": int(bool(is_world_cup)),
            "has_home_advantage": int(not is_neutral),
        }
        x = np.array([[feature_dict[c] for c in self.feature_columns]], dtype=float)

        proba = self._weighted_proba(self.xgb_model, self.rf_model, x)[0]
        prob_dict: dict[str, float] = {}
        for i, cls_label in enumerate(self.label_encoder.classes_):
            prob_dict[CLASS_TO_KEY[cls_label]] = float(proba[i])
        # Ensure all 3 keys present
        for k in ("home_win", "draw", "away_win"):
            prob_dict.setdefault(k, 0.0)

        reasoning = self._build_reasoning(
            home_team, away_team, elo_h, elo_a,
            home_form, away_form, h2h_rate, h2h_info["matches_count"],
            prob_dict,
        )
        return self._package(prob_dict, reasoning=reasoning, features=feature_dict)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        """Serialize the model + metadata via joblib."""
        path = Path(path) if path else self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "xgb_model": self.xgb_model,
            "rf_model": self.rf_model,
            "label_encoder": self.label_encoder,
            "feature_columns": self.feature_columns,
            "elo_snapshot": self.elo_snapshot,
            "form_snapshot": {k: v.__dict__ for k, v in self.form_snapshot.items()},
            "h2h_snapshot": {f"{a}|{b}": v for (a, b), v in self.h2h_snapshot.items()},
            "training_metrics": self.training_metrics,
            "ensemble_weights": self._ensemble_weights,
        }
        joblib.dump(payload, path)
        log.info("Saved StatisticalModel → %s", path)
        return path

    def load(self, path: Optional[Path] = None) -> bool:
        """Load a previously saved model. Returns True on success."""
        path = Path(path) if path else self.model_path
        if not path.exists():
            log.warning("StatisticalModel file not found: %s", path)
            return False
        payload = joblib.load(path)
        self.xgb_model = payload.get("xgb_model")
        self.rf_model = payload.get("rf_model")
        self.label_encoder = payload.get("label_encoder")
        self.feature_columns = payload.get("feature_columns", list(FEATURE_COLUMNS))
        self.elo_snapshot = payload.get("elo_snapshot", {})
        self.form_snapshot = {
            k: TeamFormSnapshot(**v)
            for k, v in payload.get("form_snapshot", {}).items()
        }
        h2h_raw = payload.get("h2h_snapshot", {})
        self.h2h_snapshot = {
            tuple(k.split("|", 1)): v for k, v in h2h_raw.items()
        }
        self.training_metrics = payload.get("training_metrics", {})
        self._ensemble_weights = payload.get("ensemble_weights", (0.55, 0.45))
        log.info("Loaded StatisticalModel ← %s  (CV-acc=%.4f)",
                 path, self.training_metrics.get("cv_accuracy_mean", 0.0))
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_xgb(self):
        """Construct a fresh XGBoost classifier (or sklearn GBM fallback)."""
        if _HAS_XGB:
            return XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                num_class=len(CLASSES),
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
            )
        # XGBoost not installed — fall back to GradientBoosting
        from sklearn.ensemble import GradientBoostingClassifier
        log.warning("xgboost not installed — using GradientBoostingClassifier as drop-in.")
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42,
        )

    def _make_rf(self):
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )

    def _weighted_proba(self, xgb, rf, X) -> np.ndarray:
        """Weighted average of probabilities across XGB + RF."""
        wx, wr = self._ensemble_weights
        p_xgb = xgb.predict_proba(X)
        p_rf = rf.predict_proba(X)
        # Align column order (both should match label_encoder.classes_).
        return wx * p_xgb + wr * p_rf

    def _build_elo_snapshot(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Reconstruct each team's *latest* ELO from the historical features
        DataFrame (we recorded elo_*_before per match — the after-update ELO
        for the home/away team is implicit in subsequent matches' before
        ratings, so we approximate by taking the most recent before value
        seen, then nudging based on the most recent match outcome).
        """
        snap: dict[str, float] = {}
        df_sorted = df.sort_values("date")
        for _, row in df_sorted.iterrows():
            snap[row["home_team"]] = float(row["elo_home_before"])
            snap[row["away_team"]] = float(row["elo_away_before"])

        # If a static elo_ratings.csv is present, prefer it (more accurate
        # because download_historical.py wrote post-match ratings there).
        if ELO_RATINGS_PATH.exists():
            try:
                elo_df = pd.read_csv(ELO_RATINGS_PATH)
                for _, r in elo_df.iterrows():
                    snap[str(r["team"])] = float(r["elo_rating"])
                log.info("Augmented ELO snapshot from %s (%d teams)",
                         ELO_RATINGS_PATH, len(elo_df))
            except Exception as exc:  # pragma: no cover
                log.warning("Could not read elo_ratings.csv: %s", exc)
        return snap

    def _build_reasoning(
        self,
        home: str, away: str,
        elo_h: float, elo_a: float,
        home_form: TeamFormSnapshot, away_form: TeamFormSnapshot,
        h2h_rate: float, h2h_n: float,
        prob_dict: dict[str, float],
    ) -> str:
        diff = elo_h - elo_a
        if diff > 80:
            elo_blurb = f"clear ELO edge to {home} ({elo_h:.0f} vs {elo_a:.0f}, +{diff:.0f})"
        elif diff < -80:
            elo_blurb = f"clear ELO edge to {away} ({elo_h:.0f} vs {elo_a:.0f}, {diff:.0f})"
        else:
            elo_blurb = f"ELO is close: {home} {elo_h:.0f} vs {away} {elo_a:.0f} ({diff:+.0f})"

        form_blurb = (
            f"{home} 10-match win-rate {home_form.win_rate_10:.0%} "
            f"(scored {home_form.goals_scored_avg_5:.2f}/g, conceded "
            f"{home_form.goals_conceded_avg_5:.2f}/g); "
            f"{away} {away_form.win_rate_10:.0%} "
            f"(scored {away_form.goals_scored_avg_5:.2f}, "
            f"conceded {away_form.goals_conceded_avg_5:.2f})"
        )

        if h2h_n >= 1:
            h2h_blurb = f"H2H: {home} won {h2h_rate:.0%} of last {int(h2h_n)} meetings"
        else:
            h2h_blurb = "H2H: no recent meetings on record"

        best = max(prob_dict.items(), key=lambda kv: kv[1])
        outcome = {"home_win": f"{home} win",
                   "draw": "draw",
                   "away_win": f"{away} win"}[best[0]]
        return (
            f"ELO advantage analysis: {elo_blurb}. Form check: {form_blurb}. "
            f"{h2h_blurb}. Model leans {outcome} at {best[1]:.0%}."
        )

    def _package(
        self,
        prob_dict: dict[str, float],
        *,
        reasoning: str,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        # Renormalize defensively
        total = sum(prob_dict.values()) or 1.0
        norm = {k: v / total for k, v in prob_dict.items()}
        sorted_p = sorted(norm.values(), reverse=True)
        confidence = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else float(sorted_p[0])
        # Map [0..1] gap to a more intuitive 0..1 score
        confidence = max(0.0, min(1.0, 0.4 + confidence * 1.5))
        return {
            "home_win": round(float(norm["home_win"]), 4),
            "draw": round(float(norm["draw"]), 4),
            "away_win": round(float(norm["away_win"]), 4),
            "confidence": round(confidence, 4),
            "reasoning": reasoning,
            "features_used": features,
        }


# ---------------------------------------------------------------------------
# CLI helper — train when run directly
# ---------------------------------------------------------------------------

def main() -> None:
    if not HISTORICAL_FEATURES_PATH.exists():
        log.error("Historical features not found at %s. "
                  "Run download_historical.py first.", HISTORICAL_FEATURES_PATH)
        sys.exit(1)

    df = pd.read_csv(HISTORICAL_FEATURES_PATH)
    model = StatisticalModel()
    metrics = model.train(df)
    model.save()
    log.info("Done. Metrics: %s", metrics)


if __name__ == "__main__":
    main()
