"""
train_models.py
===============
End-to-end training orchestrator for the WC2026 prediction stack.

What it does
------------
1. Ensures historical features exist; if not, runs
   ``src/data/download_historical.py``.
2. If the dataset is still missing (e.g. offline), generates a
   deterministic synthetic dataset so the full pipeline still runs.
3. Splits the data into train (≤2021) / holdout (2022-2024).
4. Trains StatisticalModel, FormModel on the train split.
5. Persists the NewsModel config (deterministic — no fitting).
6. Evaluates each model + the DebateEngine ensemble on the holdout.
7. Prints a metrics summary table and saves the trained models.

Run
---
    python -m src.models.train_models
    # or
    python src/models/train_models.py
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Local imports (work both as module and as script)
try:
    from src.models.statistical_model import StatisticalModel
    from src.models.form_model import FormModel
    from src.models.news_model import NewsModel
    from src.debate.debate_engine import DebateEngine
except ImportError:  # pragma: no cover - script-mode fallback
    _here = Path(__file__).resolve().parents[2]
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from src.models.statistical_model import StatisticalModel  # type: ignore
    from src.models.form_model import FormModel  # type: ignore
    from src.models.news_model import NewsModel  # type: ignore
    from src.debate.debate_engine import DebateEngine  # type: ignore


# ---------------------------------------------------------------------------
# Paths & logging
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT / "models"
HISTORICAL_FEATURES_PATH = PROCESSED_DIR / "historical_features.csv"
DOWNLOAD_SCRIPT = ROOT / "src" / "data" / "download_historical.py"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [train]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("train_models")


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------

def ensure_historical_features() -> pd.DataFrame:
    """
    Make sure ``historical_features.csv`` exists. Strategy:
      1. If file exists → load and return.
      2. Otherwise try to run download_historical.py.
      3. If that fails (no network etc.) → fall back to synthetic data.
    """
    if HISTORICAL_FEATURES_PATH.exists():
        log.info("Loading existing historical features: %s", HISTORICAL_FEATURES_PATH)
        return pd.read_csv(HISTORICAL_FEATURES_PATH)

    log.warning("Historical features not found. Attempting to run %s", DOWNLOAD_SCRIPT)
    if DOWNLOAD_SCRIPT.exists():
        try:
            result = subprocess.run(
                [sys.executable, str(DOWNLOAD_SCRIPT)],
                cwd=str(ROOT),
                check=False,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0 and HISTORICAL_FEATURES_PATH.exists():
                log.info("download_historical.py succeeded.")
                return pd.read_csv(HISTORICAL_FEATURES_PATH)
            log.warning(
                "download_historical.py exited with code %d. stderr=%s",
                result.returncode,
                (result.stderr or "")[:500],
            )
        except Exception as exc:
            log.warning("Could not run download script: %s", exc)
    else:
        log.warning("download script not found at %s", DOWNLOAD_SCRIPT)

    log.warning("Falling back to synthetic dataset for pipeline testing.")
    df = generate_synthetic_dataset()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(HISTORICAL_FEATURES_PATH, index=False)
    log.info("Synthetic dataset saved to %s (%d rows).", HISTORICAL_FEATURES_PATH, len(df))
    return df


def generate_synthetic_dataset(
    n_matches: int = 4000,
    start_year: int = 2010,
    end_year: int = 2024,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a deterministic synthetic dataset that mirrors the schema of
    historical_features.csv. Useful when the real download is unavailable
    (offline CI, sandboxes). Probabilities are correlated with simulated
    ELO ratings so the trained models still learn a real signal.
    """
    rng = np.random.default_rng(seed)

    teams = [
        "USA", "Mexico", "Canada", "Brazil", "Argentina", "Uruguay",
        "France", "Germany", "Spain", "England", "Italy", "Netherlands",
        "Portugal", "Belgium", "Croatia", "Switzerland", "Poland",
        "Morocco", "Senegal", "Nigeria", "Cameroon", "Algeria",
        "Japan", "South Korea", "Australia", "Saudi Arabia", "Iran",
        "Colombia", "Chile", "Ecuador", "Paraguay", "Venezuela",
        "Denmark", "Sweden", "Turkey", "Hungary", "Tunisia", "Qatar",
    ]
    elos = {team: rng.normal(1700, 120) for team in teams}

    rows: list[dict[str, Any]] = []
    base_date = pd.Timestamp(f"{start_year}-01-01")
    days_per_match = max(1, ((end_year - start_year) * 365) // n_matches)

    # Track per-team rolling stats as we go
    history: dict[str, list[dict[str, Any]]] = {t: [] for t in teams}

    for i in range(n_matches):
        date = base_date + pd.Timedelta(days=i * days_per_match)
        home, away = rng.choice(teams, size=2, replace=False)
        elo_h = elos[home]
        elo_a = elos[away]

        # Tournament selection
        roll = rng.random()
        if roll < 0.05:
            tournament = "FIFA World Cup"
            is_wc = True
        elif roll < 0.15:
            tournament = "UEFA Euro"
            is_wc = False
        elif roll < 0.30:
            tournament = "World Cup qualification"
            is_wc = False
        elif roll < 0.55:
            tournament = "Friendly"
            is_wc = False
        else:
            tournament = "UEFA Nations League"
            is_wc = False

        neutral = bool(rng.random() < 0.30)
        home_advantage = 60.0 if not neutral else 0.0
        diff = (elo_h + home_advantage) - elo_a
        p_home = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        # Sample goals via Poisson with means tied to ELO
        mu_home = max(0.4, 1.4 + (diff / 400.0))
        mu_away = max(0.4, 1.4 - (diff / 400.0))
        hs = int(rng.poisson(mu_home))
        as_ = int(rng.poisson(mu_away))

        if hs > as_:
            result = "W"
            score_h, score_a = 1.0, 0.0
        elif hs < as_:
            result = "L"
            score_h, score_a = 0.0, 1.0
        else:
            result = "D"
            score_h, score_a = 0.5, 0.5

        # Update ELO
        K = 32.0 * (1.5 if is_wc else 1.0)
        exp_h = p_home
        elos[home] = elo_h + K * (score_h - exp_h)
        elos[away] = elo_a + K * (score_a - (1 - exp_h))

        def _rolling(team: str, w: int) -> dict[str, float]:
            past = history[team][-w:]
            if not past:
                return {"win_rate": np.nan, "gf_avg": np.nan, "ga_avg": np.nan}
            wins = sum(1 for p in past if p["result"] == "W")
            return {
                "win_rate": wins / len(past),
                "gf_avg": float(np.mean([p["gf"] for p in past])),
                "ga_avg": float(np.mean([p["ga"] for p in past])),
            }

        h_form_10 = _rolling(home, 10)
        a_form_10 = _rolling(away, 10)
        h_form_5 = _rolling(home, 5)
        a_form_5 = _rolling(away, 5)

        # Simple H2H lookup via history scan
        h2h_past = [
            r for r in rows
            if (
                ((r["home_team"] == home and r["away_team"] == away)
                 or (r["home_team"] == away and r["away_team"] == home))
                and pd.Timestamp(r["date"]) > date - pd.DateOffset(years=10)
            )
        ]
        if h2h_past:
            h2h_wins = sum(
                1 for r in h2h_past
                if (r["home_team"] == home and r["home_score"] > r["away_score"])
                or (r["away_team"] == home and r["away_score"] > r["home_score"])
            )
            h2h_rate = h2h_wins / len(h2h_past)
        else:
            h2h_rate = np.nan

        rows.append({
            "date": date,
            "home_team": home,
            "away_team": away,
            "home_score": hs,
            "away_score": as_,
            "tournament": tournament,
            "neutral": neutral,
            "has_home_advantage": not neutral,
            "home_result": result,
            "is_world_cup": is_wc,
            "is_wc_qualifier": tournament == "World Cup qualification",
            "year": date.year,
            "decade": (date.year // 10) * 10,
            "elo_home_before": elo_h,
            "elo_away_before": elo_a,
            "elo_diff": elo_h - elo_a,
            "home_win_rate_10": h_form_10["win_rate"],
            "away_win_rate_10": a_form_10["win_rate"],
            "home_goals_scored_avg_5": h_form_5["gf_avg"],
            "away_goals_scored_avg_5": a_form_5["gf_avg"],
            "home_goals_conceded_avg_5": h_form_5["ga_avg"],
            "away_goals_conceded_avg_5": a_form_5["ga_avg"],
            "h2h_home_win_rate": h2h_rate,
            "h2h_matches_count": len(h2h_past),
        })

        history[home].append({"result": result, "gf": hs, "ga": as_})
        history[away].append({
            "result": "W" if result == "L" else ("L" if result == "W" else "D"),
            "gf": as_, "ga": hs,
        })

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _result_to_outcome_key(result: str) -> str:
    return {"W": "home_win", "D": "draw", "L": "away_win"}.get(result, "draw")


def evaluate_model_on_holdout(
    model_name: str,
    predict_fn,
    holdout: pd.DataFrame,
    max_rows: int = 500,
) -> dict[str, float]:
    """Evaluate a single ``predict_fn(home, away, date)`` against a holdout."""
    correct = 0
    n = 0
    log_loss_sum = 0.0
    eps = 1e-9

    sample = holdout.sample(min(len(holdout), max_rows), random_state=7) \
        if len(holdout) > max_rows else holdout

    for _, row in sample.iterrows():
        try:
            pred = predict_fn(row["home_team"], row["away_team"], row["date"])
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("%s prediction failed: %s", model_name, exc)
            continue
        actual_key = _result_to_outcome_key(row["home_result"])
        # Top-1 accuracy
        top1 = max(("home_win", "draw", "away_win"), key=lambda k: pred[k])
        if top1 == actual_key:
            correct += 1
        # Log-loss
        p = max(eps, min(1 - eps, float(pred[actual_key])))
        log_loss_sum += -np.log(p)
        n += 1

    if n == 0:
        return {"accuracy": 0.0, "log_loss": float("nan"), "n_eval": 0}
    return {
        "accuracy": correct / n,
        "log_loss": log_loss_sum / n,
        "n_eval": n,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 72)
    log.info("WC2026 predictor — full training pipeline starting")
    log.info("=" * 72)

    # 1) Acquire data --------------------------------------------------
    df = ensure_historical_features()
    df["date"] = pd.to_datetime(df["date"])
    log.info("Dataset: %d matches | range %s → %s",
             len(df), df["date"].min().date(), df["date"].max().date())

    # 2) Train / holdout split -----------------------------------------
    train_mask = df["date"] < pd.Timestamp("2022-01-01")
    test_mask = (df["date"] >= pd.Timestamp("2022-01-01")) & \
                (df["date"] <= pd.Timestamp("2024-12-31"))

    train_df = df[train_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    if len(test_df) == 0:
        log.warning("Holdout window 2022-2024 is empty; using last 15%% of data instead.")
        cutoff = int(len(df) * 0.85)
        train_df = df.iloc[:cutoff].reset_index(drop=True)
        test_df = df.iloc[cutoff:].reset_index(drop=True)

    log.info("Train rows: %d | Holdout rows: %d", len(train_df), len(test_df))

    # 3) Train StatisticalModel ----------------------------------------
    log.info("--- Training StatisticalModel (Historian) ---")
    stat_model = StatisticalModel()
    stat_metrics = stat_model.train(train_df)
    stat_model.save()

    # 4) Train FormModel -----------------------------------------------
    log.info("--- Training FormModel (Form Expert) ---")
    form_model = FormModel()
    try:
        form_metrics = form_model.train(train_df)
        form_model.save()
    except ValueError as exc:
        log.warning("FormModel training skipped: %s", exc)
        form_metrics = {"cv_accuracy_mean": 0.0, "n_train_rows": 0, "skipped": True}

    # 5) Persist NewsModel ---------------------------------------------
    log.info("--- Persisting NewsModel (News Hawk) ---")
    news_model = NewsModel()
    # Warm up the cache for any teams with files on disk so save() captures it.
    for f in RAW_DIR.glob("team_news_*.json"):
        # Reverse-engineer team name from slug — best-effort, NewsModel
        # otherwise loads on demand.
        slug = f.stem.replace("team_news_", "")
        team_guess = slug.replace("_", " ").title()
        news_model._load_team_payload(team_guess)
    news_model.save()

    # 6) Holdout evaluation --------------------------------------------
    log.info("--- Evaluating on holdout (2022-2024) ---")
    if len(test_df) == 0:
        log.warning("No holdout matches available; skipping evaluation.")
        eval_stat = eval_form = eval_debate = {"accuracy": 0.0, "log_loss": float("nan"), "n_eval": 0}
    else:
        eval_stat = evaluate_model_on_holdout(
            "Historian",
            lambda h, a, d: stat_model.predict(h, a),
            test_df,
        )
        eval_form = evaluate_model_on_holdout(
            "FormExpert",
            lambda h, a, d: form_model.predict(h, a, match_date=d),
            test_df,
        )

        debate = DebateEngine(
            historian=stat_model,
            form_expert=form_model,
            news_hawk=news_model,
            auto_load=False,   # already in memory
        )
        eval_debate = evaluate_model_on_holdout(
            "DebateEnsemble",
            lambda h, a, d: {
                k: v for k, v in debate.predict_match(
                    h, a, match_date=d,
                )["predictions"].items()
            },
            test_df,
        )

    # 7) Print summary --------------------------------------------------
    log.info("=" * 72)
    log.info("TRAINING & EVALUATION SUMMARY")
    log.info("=" * 72)
    log.info("Historian (StatisticalModel):")
    log.info("  CV accuracy:     %.4f", stat_metrics.get("cv_accuracy_mean", 0.0))
    log.info("  CV logloss:      %.4f", stat_metrics.get("cv_logloss_mean", 0.0))
    log.info("  Holdout acc:     %.4f (n=%d)", eval_stat["accuracy"], eval_stat["n_eval"])
    log.info("  Holdout logloss: %.4f", eval_stat["log_loss"])

    log.info("Form Expert (FormModel):")
    log.info("  CV accuracy:     %.4f", form_metrics.get("cv_accuracy_mean", 0.0))
    log.info("  CV logloss:      %.4f", form_metrics.get("cv_logloss_mean", 0.0))
    log.info("  Holdout acc:     %.4f (n=%d)", eval_form["accuracy"], eval_form["n_eval"])
    log.info("  Holdout logloss: %.4f", eval_form["log_loss"])

    log.info("Debate Ensemble (40/35/25 default weights):")
    log.info("  Holdout acc:     %.4f (n=%d)", eval_debate["accuracy"], eval_debate["n_eval"])
    log.info("  Holdout logloss: %.4f", eval_debate["log_loss"])

    log.info("Models saved under: %s", MODELS_DIR)
    log.info("  - %s", MODELS_DIR / "statistical_model.pkl")
    log.info("  - %s", MODELS_DIR / "form_model.pkl")
    log.info("  - %s", MODELS_DIR / "news_model.pkl")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
