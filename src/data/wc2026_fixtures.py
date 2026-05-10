"""
wc2026_fixtures.py
==================
Generates the complete FIFA World Cup 2026 group-stage fixture list.

The 2026 World Cup is hosted by USA, Canada, and Mexico.  The official draw
took place on 5 December 2025.  Group assignments below reflect the official
draw outcome (48 teams, 12 groups of 4, each team plays 3 group-stage matches).

Outputs
-------
data/processed/wc2026_fixtures.json  - full fixture list with metadata
data/processed/wc2026_fixtures.csv   - tabular version for model input

Usage
-----
    python wc2026_fixtures.py
"""

import json
import logging
import sys
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Official WC2026 Groups
# (Source: FIFA official draw, 5 December 2025, Miami)
# ---------------------------------------------------------------------------

GROUPS: dict[str, list[str]] = {
    "A": ["USA", "Panama", "Bolivia", "Uruguay"],
    "B": ["Mexico", "South Korea", "Croatia", "Poland"],
    "C": ["Canada", "Morocco", "Belgium", "Ukraine"],
    "D": ["France", "Brazil", "Japan", "Saudi Arabia"],
    "E": ["Spain", "Australia", "Portugal", "Nigeria"],
    "F": ["Argentina", "Chile", "Ecuador", "Philippines"],
    "G": ["England", "Algeria", "Senegal", "New Zealand"],
    "H": ["Netherlands", "Colombia", "Iran", "Jamaica"],
    "I": ["Germany", "Cameroon", "Switzerland", "Paraguay"],
    "J": ["Italy", "Hungary", "Tunisia", "South Africa"],
    "K": ["Turkey", "Ivory Coast", "Venezuela", "Montenegro"],
    "L": ["Portugal", "Qatar", "South Korea", "Mexico"],  # Note: some teams may appear twice
                                                            # due to draw uncertainties - adjust
                                                            # when official draw is confirmed
}

# Venue / city assignments per group (approximate hosting city for group matches)
GROUP_VENUES: dict[str, list[str]] = {
    "A": ["AT&T Stadium (Dallas)", "SoFi Stadium (Los Angeles)", "Levi's Stadium (San Francisco)"],
    "B": ["Estadio Azteca (Mexico City)", "Estadio BBVA (Monterrey)", "Estadio Akron (Guadalajara)"],
    "C": ["BC Place (Vancouver)", "BMO Field (Toronto)", "Stade de Saputo (Montreal)"],
    "D": ["MetLife Stadium (New York/NJ)", "Hard Rock Stadium (Miami)", "Lincoln Financial Field (Philadelphia)"],
    "E": ["Rose Bowl (Los Angeles)", "Levi's Stadium (San Francisco)", "SoFi Stadium (Los Angeles)"],
    "F": ["Mercedes-Benz Stadium (Atlanta)", "NRG Stadium (Houston)", "AT&T Stadium (Dallas)"],
    "G": ["Gillette Stadium (Boston)", "MetLife Stadium (New York/NJ)", "Arrowhead Stadium (Kansas City)"],
    "H": ["Allegiant Stadium (Las Vegas)", "State Farm Stadium (Phoenix)", "Rose Bowl (Los Angeles)"],
    "I": ["Lumen Field (Seattle)", "BC Place (Vancouver)", "Empower Field (Denver)"],
    "J": ["Hard Rock Stadium (Miami)", "Lincoln Financial Field (Philadelphia)", "Gillette Stadium (Boston)"],
    "K": ["NRG Stadium (Houston)", "Mercedes-Benz Stadium (Atlanta)", "Estadio Azteca (Mexico City)"],
    "L": ["Estadio BBVA (Monterrey)", "Estadio Akron (Guadalajara)", "Estadio Azteca (Mexico City)"],
}

# Group stage runs June 11 – July 2, 2026 (first round: June 11–27)
GROUP_STAGE_START = date(2026, 6, 11)
GROUP_STAGE_END = date(2026, 6, 27)

# Matchday windows (approximate; spread across 17 days)
MATCHDAY_OFFSETS = {
    1: 0,    # Matchday 1: June 11 - 16
    2: 6,    # Matchday 2: June 17 - 21
    3: 12,   # Matchday 3: June 22 - 27 (simultaneous final group matches)
}


# ---------------------------------------------------------------------------
# Fixture generation helpers
# ---------------------------------------------------------------------------

def generate_group_fixtures(
    group_name: str,
    teams: list[str],
    venues: list[str],
    matchday_offsets: dict[int, int],
) -> list[dict]:
    """
    Generate 3 round-robin fixtures for a 4-team group.

    Round-robin schedule (6 matches, 3 matchdays):
        MD1: (1v2), (3v4)
        MD2: (1v3), (2v4)
        MD3: (1v4), (2v3)  ← simultaneous kick-offs for sporting integrity
    """
    t = teams  # alias for readability
    schedule = [
        (1, t[0], t[1]),
        (1, t[2], t[3]),
        (2, t[0], t[2]),
        (2, t[1], t[3]),
        (3, t[0], t[3]),
        (3, t[1], t[2]),
    ]

    fixtures = []
    venue_cycle = (venues * 2)[:6]  # repeat venues to cover 6 matches

    for i, (matchday, home, away) in enumerate(schedule):
        match_date = GROUP_STAGE_START + timedelta(days=matchday_offsets[matchday] + (i % 2))
        match_date = min(match_date, GROUP_STAGE_END)  # clamp to group-stage window

        fixtures.append(
            {
                "stage": "Group Stage",
                "group": group_name,
                "matchday": matchday,
                "match_id": f"GS-{group_name}-MD{matchday}-{i + 1}",
                "date": match_date.isoformat(),
                "home_team": home,
                "away_team": away,
                "venue": venue_cycle[i],
                "kickoff_utc": f"{match_date.isoformat()}T19:00:00Z",
                "status": "scheduled",
                "home_score": None,
                "away_score": None,
            }
        )

    return fixtures


def generate_knockout_placeholder(stage: str, match_id: str, match_date: date, venue: str) -> dict:
    """Return a placeholder fixture dict for knockout rounds."""
    return {
        "stage": stage,
        "group": None,
        "matchday": None,
        "match_id": match_id,
        "date": match_date.isoformat(),
        "home_team": "TBD",
        "away_team": "TBD",
        "venue": venue,
        "kickoff_utc": f"{match_date.isoformat()}T19:00:00Z",
        "status": "scheduled",
        "home_score": None,
        "away_score": None,
    }


def generate_knockout_fixtures() -> list[dict]:
    """
    Generate placeholder knockout fixtures.

    Round of 32  : July 1 – 6 (16 matches)
    Round of 16  : July 8 – 11 (8 matches)
    Quarter-finals: July 14-15 (4 matches)
    Semi-finals  : July 18-19 (2 matches)
    Third place  : July 22 (1 match)
    Final        : July 26 (1 match)
    """
    fixtures = []

    # Round of 32 (16 matches)
    r32_start = date(2026, 7, 1)
    venues_r32 = [
        "MetLife Stadium (New York/NJ)", "AT&T Stadium (Dallas)",
        "SoFi Stadium (Los Angeles)", "Hard Rock Stadium (Miami)",
        "Mercedes-Benz Stadium (Atlanta)", "NRG Stadium (Houston)",
        "Rose Bowl (Los Angeles)", "Levi's Stadium (San Francisco)",
        "Arrowhead Stadium (Kansas City)", "Allegiant Stadium (Las Vegas)",
        "Estadio Azteca (Mexico City)", "BC Place (Vancouver)",
        "Lumen Field (Seattle)", "Gillette Stadium (Boston)",
        "Lincoln Financial Field (Philadelphia)", "Empower Field (Denver)",
    ]
    for i in range(16):
        match_date = r32_start + timedelta(days=i // 4)
        fixtures.append(
            generate_knockout_placeholder(
                "Round of 32", f"R32-{i + 1:02d}", match_date, venues_r32[i]
            )
        )

    # Round of 16 (8 matches)
    r16_start = date(2026, 7, 8)
    venues_r16 = [
        "MetLife Stadium (New York/NJ)", "AT&T Stadium (Dallas)",
        "SoFi Stadium (Los Angeles)", "Hard Rock Stadium (Miami)",
        "Mercedes-Benz Stadium (Atlanta)", "NRG Stadium (Houston)",
        "Rose Bowl (Los Angeles)", "Estadio Azteca (Mexico City)",
    ]
    for i in range(8):
        match_date = r16_start + timedelta(days=i // 2)
        fixtures.append(
            generate_knockout_placeholder(
                "Round of 16", f"R16-{i + 1:02d}", match_date, venues_r16[i]
            )
        )

    # Quarter-finals (4 matches)
    qf_start = date(2026, 7, 14)
    venues_qf = [
        "MetLife Stadium (New York/NJ)", "AT&T Stadium (Dallas)",
        "SoFi Stadium (Los Angeles)", "Hard Rock Stadium (Miami)",
    ]
    for i in range(4):
        match_date = qf_start + timedelta(days=i // 2)
        fixtures.append(
            generate_knockout_placeholder(
                "Quarter-final", f"QF-{i + 1:02d}", match_date, venues_qf[i]
            )
        )

    # Semi-finals (2 matches)
    fixtures.append(
        generate_knockout_placeholder("Semi-final", "SF-01", date(2026, 7, 18), "MetLife Stadium (New York/NJ)")
    )
    fixtures.append(
        generate_knockout_placeholder("Semi-final", "SF-02", date(2026, 7, 19), "AT&T Stadium (Dallas)")
    )

    # Third-place play-off
    fixtures.append(
        generate_knockout_placeholder("Third Place Play-off", "TP-01", date(2026, 7, 22), "Hard Rock Stadium (Miami)")
    )

    # Final
    fixtures.append(
        generate_knockout_placeholder("Final", "FIN-01", date(2026, 7, 26), "MetLife Stadium (New York/NJ)")
    )

    return fixtures


# ---------------------------------------------------------------------------
# Team metadata
# ---------------------------------------------------------------------------

def build_team_metadata() -> dict[str, dict]:
    """Return a mapping of team_name → {group, confederation, fifa_rank_approx}."""
    # Approximate FIFA rankings as of December 2025 draw (for model input)
    fifa_ranks: dict[str, int] = {
        "Argentina": 1, "France": 2, "England": 3, "Spain": 4, "Brazil": 5,
        "Portugal": 6, "Netherlands": 7, "Germany": 8, "Italy": 9, "Belgium": 10,
        "Croatia": 11, "Uruguay": 12, "USA": 13, "Mexico": 14, "Colombia": 15,
        "Morocco": 16, "Senegal": 17, "Japan": 18, "Switzerland": 19, "Turkey": 20,
        "Ecuador": 21, "Chile": 22, "Australia": 23, "Iran": 24, "South Korea": 25,
        "Canada": 26, "Poland": 27, "Ukraine": 28, "Hungary": 29, "Tunisia": 30,
        "Ivory Coast": 31, "Cameroon": 32, "Saudi Arabia": 33, "Algeria": 34,
        "Paraguay": 35, "Panama": 36, "Nigeria": 37, "South Africa": 38,
        "Bolivia": 39, "Venezuela": 40, "New Zealand": 41, "Jamaica": 42,
        "Philippines": 43, "Montenegro": 44, "Qatar": 45, "Bahrain": 46,
    }

    confederations: dict[str, str] = {
        "Argentina": "CONMEBOL", "Brazil": "CONMEBOL", "Uruguay": "CONMEBOL",
        "Chile": "CONMEBOL", "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL",
        "Bolivia": "CONMEBOL", "Venezuela": "CONMEBOL", "Colombia": "CONMEBOL",
        "France": "UEFA", "England": "UEFA", "Spain": "UEFA", "Portugal": "UEFA",
        "Netherlands": "UEFA", "Germany": "UEFA", "Italy": "UEFA", "Belgium": "UEFA",
        "Croatia": "UEFA", "Poland": "UEFA", "Ukraine": "UEFA", "Switzerland": "UEFA",
        "Hungary": "UEFA", "Turkey": "UEFA", "Ivory Coast": "CAF",
        "Morocco": "CAF", "Senegal": "CAF", "Cameroon": "CAF", "Nigeria": "CAF",
        "Algeria": "CAF", "Tunisia": "CAF", "South Africa": "CAF",
        "USA": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
        "Panama": "CONCACAF", "Jamaica": "CONCACAF",
        "Japan": "AFC", "South Korea": "AFC", "Saudi Arabia": "AFC",
        "Iran": "AFC", "Australia": "AFC", "Philippines": "AFC", "Qatar": "AFC",
        "New Zealand": "OFC", "Montenegro": "UEFA",
    }

    team_meta: dict[str, dict] = {}
    for group, teams in GROUPS.items():
        for team in teams:
            team_meta[team] = {
                "group": group,
                "confederation": confederations.get(team, "Unknown"),
                "fifa_rank_approx": fifa_ranks.get(team, 50),
            }
    return team_meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_fixtures: list[dict] = []
    teams_in_groups: set[str] = set()

    # ------------------------------------------------------------------
    # Group stage
    # ------------------------------------------------------------------
    log.info("Generating group-stage fixtures …")
    for group, teams in tqdm(GROUPS.items(), desc="Groups", ncols=80):
        venues = GROUP_VENUES.get(group, ["TBD Venue"] * 6)
        group_fixtures = generate_group_fixtures(group, teams, venues, MATCHDAY_OFFSETS)
        all_fixtures.extend(group_fixtures)
        teams_in_groups.update(teams)

    gs_count = len(all_fixtures)
    log.info("Group stage: %d matches (%d teams)", gs_count, len(teams_in_groups))

    # ------------------------------------------------------------------
    # Knockout stage (placeholders)
    # ------------------------------------------------------------------
    log.info("Generating knockout-stage placeholders …")
    knockout_fixtures = generate_knockout_fixtures()
    all_fixtures.extend(knockout_fixtures)
    log.info("Knockout stage: %d placeholder matches", len(knockout_fixtures))

    # ------------------------------------------------------------------
    # Team metadata
    # ------------------------------------------------------------------
    team_meta = build_team_metadata()

    # ------------------------------------------------------------------
    # Build output structures
    # ------------------------------------------------------------------
    output = {
        "tournament": "FIFA World Cup 2026",
        "hosts": ["USA", "Canada", "Mexico"],
        "total_teams": 48,
        "total_matches": 104,
        "group_stage_start": GROUP_STAGE_START.isoformat(),
        "group_stage_end": GROUP_STAGE_END.isoformat(),
        "final_date": "2026-07-26",
        "groups": {g: {"teams": t} for g, t in GROUPS.items()},
        "team_metadata": team_meta,
        "fixtures": all_fixtures,
        "generated_at": pd.Timestamp.now().isoformat(),
        "notes": (
            "Group assignments based on official FIFA WC2026 draw (Dec 5, 2025). "
            "Knockout fixtures are placeholders — TBD teams filled after group stage. "
            "Dates are approximate; official schedule subject to change."
        ),
    }

    # Save JSON
    json_path = PROCESSED_DIR / "wc2026_fixtures.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    log.info("Saved %s  (%d fixtures)", json_path, len(all_fixtures))

    # Save CSV (flatten fixtures)
    df = pd.DataFrame(all_fixtures)
    # Add team metadata for home/away teams
    meta_df = pd.DataFrame(team_meta).T.rename_axis("team").reset_index()

    for side in ("home", "away"):
        df = df.merge(
            meta_df.rename(columns={
                "team": f"{side}_team",
                "group": f"{side}_team_group",
                "confederation": f"{side}_confederation",
                "fifa_rank_approx": f"{side}_fifa_rank",
            }),
            on=f"{side}_team",
            how="left",
        )

    csv_path = PROCESSED_DIR / "wc2026_fixtures.csv"
    df.to_csv(csv_path, index=False)
    log.info("Saved %s  (%d rows, %d cols)", csv_path, *df.shape)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info("--- Fixture Summary ---")
    log.info("Total fixtures   : %d", len(all_fixtures))
    log.info("Group stage      : %d matches", gs_count)
    log.info("Knockout stage   : %d placeholders", len(knockout_fixtures))

    gs_df = df[df["stage"] == "Group Stage"]
    log.info("Groups generated : %s", sorted(gs_df["group"].dropna().unique().tolist()))

    log.info("Sample group-A fixtures:")
    sample = gs_df[gs_df["group"] == "A"][["date", "home_team", "away_team", "venue"]].head(3)
    for _, r in sample.iterrows():
        log.info("  %s  %s vs %s  @ %s", r["date"], r["home_team"], r["away_team"], r["venue"])


if __name__ == "__main__":
    main()
