"""
download_historical.py
======================
Downloads international football results (1872-2025) from the martj42
GitHub dataset, engineers features (ELO ratings, rolling form, H2H stats),
and saves processed outputs for use by the WC2026 prediction model.

Outputs
-------
data/raw/results.csv               - raw match results
data/raw/goalscorers.csv           - raw goalscorer records
data/processed/historical_features.csv  - enriched match-level features
data/processed/elo_ratings.csv     - current ELO rating per team

Usage
-----
    python download_historical.py
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master"
RESULTS_URL = f"{BASE_URL}/results.csv"
GOALSCORERS_URL = f"{BASE_URL}/goalscorers.csv"

ROOT = Path(__file__).resolve().parents[2]   # wc2026-predictor/
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

ELO_START = 1500
K_FACTOR = 32
WC_WEIGHT_MULTIPLIER = 1.5   # World-Cup matches count more
ROLLING_WINDOWS = [5, 10]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_csv(url: str, dest: Path, chunk_size: int = 65536) -> pd.DataFrame:
    """Stream-download a CSV to *dest* and return it as a DataFrame."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s", url)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(
        desc=dest.name,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        ncols=80,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            fh.write(chunk)
            bar.update(len(chunk))

    log.info("Saved %s (%.1f KB)", dest, dest.stat().st_size / 1024)
    return pd.read_csv(dest)


# ---------------------------------------------------------------------------
# ELO calculation
# ---------------------------------------------------------------------------

def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for team A against team B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(
    rating_home: float,
    rating_away: float,
    home_goals: int,
    away_goals: int,
    k: float = K_FACTOR,
) -> tuple[float, float]:
    """Return updated (home_elo, away_elo) after a match result."""
    exp_home = expected_score(rating_home, rating_away)
    exp_away = 1.0 - exp_home

    if home_goals > away_goals:
        score_home, score_away = 1.0, 0.0
    elif home_goals < away_goals:
        score_home, score_away = 0.0, 1.0
    else:
        score_home, score_away = 0.5, 0.5

    new_home = rating_home + k * (score_home - exp_home)
    new_away = rating_away + k * (score_away - exp_away)
    return new_home, new_away


def compute_elo_ratings(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Iterate chronologically through all matches and build an ELO history.

    Returns
    -------
    df_enriched : match DataFrame with columns elo_home_before, elo_away_before
    elo_df      : current ELO per team (after last match)
    """
    df = df.sort_values("date").reset_index(drop=True)
    ratings: dict[str, float] = {}

    elo_home_before: list[float] = []
    elo_away_before: list[float] = []

    log.info("Computing ELO ratings over %d matches …", len(df))
    for _, row in tqdm(df.iterrows(), total=len(df), desc="ELO", ncols=80):
        home = row["home_team"]
        away = row["away_team"]
        rh = ratings.get(home, ELO_START)
        ra = ratings.get(away, ELO_START)

        elo_home_before.append(rh)
        elo_away_before.append(ra)

        k = K_FACTOR * WC_WEIGHT_MULTIPLIER if row.get("is_world_cup") else K_FACTOR
        new_rh, new_ra = update_elo(rh, ra, int(row["home_score"]), int(row["away_score"]), k)
        ratings[home] = new_rh
        ratings[away] = new_ra

    df["elo_home_before"] = elo_home_before
    df["elo_away_before"] = elo_away_before
    df["elo_diff"] = df["elo_home_before"] - df["elo_away_before"]

    elo_df = (
        pd.DataFrame.from_dict(ratings, orient="index", columns=["elo_rating"])
        .rename_axis("team")
        .reset_index()
        .sort_values("elo_rating", ascending=False)
        .reset_index(drop=True)
    )
    return df, elo_df


# ---------------------------------------------------------------------------
# Rolling form features
# ---------------------------------------------------------------------------

def _team_match_sequence(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """
    Return all matches for *team* as a unified home/away view with columns:
        date, opponent, goals_for, goals_against, result ('W'/'D'/'L'),
        is_home, match_idx (original df index)
    """
    home_mask = df["home_team"] == team
    away_mask = df["away_team"] == team

    home_rows = df[home_mask][["date", "away_team", "home_score", "away_score"]].copy()
    home_rows.columns = ["date", "opponent", "goals_for", "goals_against"]
    home_rows["is_home"] = True
    home_rows.index = df[home_mask].index

    away_rows = df[away_mask][["date", "home_team", "away_score", "home_score"]].copy()
    away_rows.columns = ["date", "opponent", "goals_for", "goals_against"]
    away_rows["is_home"] = False
    away_rows.index = df[away_mask].index

    combined = pd.concat([home_rows, away_rows]).sort_values("date")
    combined["result"] = combined.apply(
        lambda r: "W" if r["goals_for"] > r["goals_against"]
        else ("D" if r["goals_for"] == r["goals_against"] else "L"),
        axis=1,
    )
    return combined


def compute_rolling_form(df: pd.DataFrame, windows: list[int] = ROLLING_WINDOWS) -> pd.DataFrame:
    """
    For each match row, compute rolling form for both home and away teams
    using only matches *before* the current date (no data leakage).

    Adds columns (example for window=5):
        home_win_rate_5, home_goals_scored_avg_5, home_goals_conceded_avg_5
        away_win_rate_5, away_goals_scored_avg_5, away_goals_conceded_avg_5
    """
    df = df.sort_values("date").reset_index(drop=True)
    all_teams = set(df["home_team"]) | set(df["away_team"])

    # Pre-build team sequences (dict: team -> sorted DataFrame)
    log.info("Building team match sequences …")
    team_seqs: dict[str, pd.DataFrame] = {}
    for team in tqdm(all_teams, desc="Team seqs", ncols=80):
        team_seqs[team] = _team_match_sequence(df, team)

    # For each window, initialise output columns
    for w in windows:
        for side in ("home", "away"):
            df[f"{side}_win_rate_{w}"] = np.nan
            df[f"{side}_goals_scored_avg_{w}"] = np.nan
            df[f"{side}_goals_conceded_avg_{w}"] = np.nan

    log.info("Computing rolling form for %d matches …", len(df))
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Rolling form", ncols=80):
        match_date = row["date"]
        for side, team in (("home", row["home_team"]), ("away", row["away_team"])):
            seq = team_seqs[team]
            past = seq[seq["date"] < match_date]
            for w in windows:
                last_w = past.tail(w)
                if len(last_w) == 0:
                    continue
                wins = (last_w["result"] == "W").sum()
                df.at[idx, f"{side}_win_rate_{w}"] = wins / len(last_w)
                df.at[idx, f"{side}_goals_scored_avg_{w}"] = last_w["goals_for"].mean()
                df.at[idx, f"{side}_goals_conceded_avg_{w}"] = last_w["goals_against"].mean()

    return df


# ---------------------------------------------------------------------------
# Head-to-head features
# ---------------------------------------------------------------------------

def compute_h2h(df: pd.DataFrame, lookback_years: int = 10) -> pd.DataFrame:
    """
    For each match, compute head-to-head win rate for the home team
    against the away team in the last *lookback_years* years (before match date).

    Adds columns: h2h_home_win_rate, h2h_matches_count
    """
    df = df.sort_values("date").reset_index(drop=True)
    df["h2h_home_win_rate"] = np.nan
    df["h2h_matches_count"] = 0

    log.info("Computing head-to-head stats …")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="H2H", ncols=80):
        home, away = row["home_team"], row["away_team"]
        cutoff = row["date"] - pd.DateOffset(years=lookback_years)

        # Past meetings between these two teams in either direction
        mask = (
            (df["date"] >= cutoff)
            & (df["date"] < row["date"])
            & (
                ((df["home_team"] == home) & (df["away_team"] == away))
                | ((df["home_team"] == away) & (df["away_team"] == home))
            )
        )
        past = df[mask]
        n = len(past)
        df.at[idx, "h2h_matches_count"] = n

        if n == 0:
            continue

        # Count wins for the 'home' team regardless of which side they were on
        home_wins = (
            ((past["home_team"] == home) & (past["home_score"] > past["away_score"])).sum()
            + ((past["away_team"] == home) & (past["away_score"] > past["home_score"])).sum()
        )
        df.at[idx, "h2h_home_win_rate"] = home_wins / n

    return df


# ---------------------------------------------------------------------------
# Core processing pipeline
# ---------------------------------------------------------------------------

def process_results(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, type-cast, and add derived columns to the raw results DataFrame."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce").fillna(0).astype(int)
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce").fillna(0).astype(int)
    df["goal_difference"] = df["home_score"] - df["away_score"]

    # Neutral venue flag (already present in dataset, but coerce to bool)
    if "neutral" in df.columns:
        df["neutral"] = df["neutral"].astype(bool)
    else:
        df["neutral"] = False

    # Home-advantage flag (home team has advantage when match is NOT on neutral ground)
    df["has_home_advantage"] = ~df["neutral"]

    # Result from home team perspective
    df["home_result"] = np.select(
        [df["goal_difference"] > 0, df["goal_difference"] < 0],
        ["W", "L"],
        default="D",
    )

    # World Cup match flag
    wc_keywords = ["FIFA World Cup", "World Cup"]
    df["is_world_cup"] = df["tournament"].str.contains(
        "|".join(wc_keywords), case=False, na=False
    )
    df["is_wc_qualifier"] = df["tournament"].str.contains(
        "World Cup qualification", case=False, na=False
    )

    # Year / decade for temporal slicing
    df["year"] = df["date"].dt.year
    df["decade"] = (df["year"] // 10) * 10

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Download raw data
    # ------------------------------------------------------------------
    results_path = RAW_DIR / "results.csv"
    goalscorers_path = RAW_DIR / "goalscorers.csv"

    if results_path.exists():
        log.info("results.csv already present — loading from disk.")
        results_df = pd.read_csv(results_path)
    else:
        results_df = download_csv(RESULTS_URL, results_path)

    if goalscorers_path.exists():
        log.info("goalscorers.csv already present — loading from disk.")
        goalscorers_df = pd.read_csv(goalscorers_path)
    else:
        goalscorers_df = download_csv(GOALSCORERS_URL, goalscorers_path)

    log.info("Results shape: %s | Goalscorers shape: %s", results_df.shape, goalscorers_df.shape)

    # ------------------------------------------------------------------
    # 2. Basic cleaning & derived columns
    # ------------------------------------------------------------------
    log.info("Processing raw results …")
    df = process_results(results_df)

    # ------------------------------------------------------------------
    # 3. ELO ratings
    # ------------------------------------------------------------------
    df, elo_df = compute_elo_ratings(df)

    # ------------------------------------------------------------------
    # 4. Rolling form
    # ------------------------------------------------------------------
    df = compute_rolling_form(df)

    # ------------------------------------------------------------------
    # 5. Head-to-head statistics
    # ------------------------------------------------------------------
    df = compute_h2h(df)

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------
    features_path = PROCESSED_DIR / "historical_features.csv"
    elo_path = PROCESSED_DIR / "elo_ratings.csv"

    df.to_csv(features_path, index=False)
    log.info("Saved historical_features.csv  (%d rows, %d cols)", *df.shape)

    elo_df.to_csv(elo_path, index=False)
    log.info("Saved elo_ratings.csv  (%d teams)", len(elo_df))

    # ------------------------------------------------------------------
    # 7. Summary statistics
    # ------------------------------------------------------------------
    log.info("--- Summary ---")
    log.info("Date range : %s  →  %s", df["date"].min().date(), df["date"].max().date())
    log.info("Teams      : %d unique", len(set(df["home_team"]) | set(df["away_team"])))
    log.info("WC matches : %d", df["is_world_cup"].sum())
    top5 = elo_df.head(5)
    log.info("Top 5 ELO  :\n%s", top5.to_string(index=False))


if __name__ == "__main__":
    main()
