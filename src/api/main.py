"""
main.py
=======
FastAPI backend for the World Cup 2026 match prediction system.

Run from project root:
    uvicorn src.api.main:app --reload

Endpoints
---------
GET  /                        → redirect to /docs
GET  /health                  → service health + model status
GET  /fixtures                → all WC2026 fixtures (filters: ?group=A&stage=Group+Stage)
GET  /fixtures/{match_id}     → single fixture detail
GET  /teams                   → all 48 teams with group info
POST /predict                 → ad-hoc prediction for any two teams
GET  /predict/{match_id}      → predict a specific WC2026 fixture by match_id
GET  /group/{group_letter}    → predictions for every match in a group
GET  /upcoming                → next N fixtures within 7 days with auto-predictions
"""

from __future__ import annotations

import sys
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
import asyncio
from functools import partial

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path bootstrap — works whether launched from project root or elsewhere
# src/api/main.py  →  parents[0]=src/api  parents[1]=src  parents[2]=project_root
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.debate.debate_engine import DebateEngine  # noqa: E402 — after sys.path tweak

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [api]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("wc2026.api")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

FIXTURES_CSV = ROOT / "data" / "processed" / "wc2026_fixtures.csv"

# Filled at startup
_engine: Optional[DebateEngine] = None
_model_loaded: bool = False

# In-memory prediction cache: key → (timestamp, result)
_prediction_cache: dict[str, tuple[datetime, dict[str, Any]]] = {}
CACHE_TTL_SECONDS = 3600  # 1 hour


# ---------------------------------------------------------------------------
# Lifespan — load DebateEngine once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _model_loaded
    log.info("Loading DebateEngine …")
    try:
        loop = asyncio.get_event_loop()
        _engine = await loop.run_in_executor(None, DebateEngine)
        _model_loaded = True
        log.info("DebateEngine ready.")
    except Exception as exc:
        log.error("DebateEngine failed to load: %s", exc, exc_info=True)
        _engine = None
        _model_loaded = False
    yield
    log.info("API shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="WC2026 Prediction API",
    description=(
        "Three-model debate-ensemble predictions for every World Cup 2026 fixture. "
        "Predictions are cached for one hour per unique matchup."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    home_team: str = Field(..., examples=["France"])
    away_team: str = Field(..., examples=["Brazil"])
    match_date: str = Field(
        default="2026-06-15",
        description="ISO date string YYYY-MM-DD",
        examples=["2026-06-15"],
    )
    is_neutral: bool = Field(default=False, description="True if played on neutral ground")


class AnalystBlock(BaseModel):
    prediction: str
    prediction_label: str
    probability: float
    probabilities: dict[str, float]
    confidence: float
    reasoning: str


class DebateBlock(BaseModel):
    historian: AnalystBlock
    form_expert: AnalystBlock
    news_hawk: AnalystBlock
    consensus: str
    transcript: str
    disagreements: Optional[list[str]] = None


class PredictionResponse(BaseModel):
    match: str
    date: str
    predictions: dict[str, float]
    recommended_bet: str
    confidence: str
    confidence_score: float
    debate: DebateBlock
    model_weights: dict[str, float]


class FixtureResponse(BaseModel):
    match_id: str
    stage: str
    group: Optional[str]
    matchday: Optional[float]
    date: str
    home_team: str
    away_team: str
    venue: str
    kickoff_utc: str
    status: str
    home_score: Optional[float]
    away_score: Optional[float]
    home_confederation: Optional[str]
    away_confederation: Optional[str]
    home_fifa_rank: Optional[float]
    away_fifa_rank: Optional[float]


class TeamResponse(BaseModel):
    name: str
    group: Optional[str]
    confederation: Optional[str]
    fifa_rank: Optional[float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


class FixtureWithPrediction(BaseModel):
    fixture: FixtureResponse
    prediction: Optional[PredictionResponse] = None
    prediction_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_fixtures() -> pd.DataFrame:
    """Load WC2026 fixtures CSV. Raises RuntimeError if file missing."""
    if not FIXTURES_CSV.exists():
        raise RuntimeError(f"Fixtures CSV not found: {FIXTURES_CSV}")
    df = pd.read_csv(FIXTURES_CSV)
    # Convert numpy NaN → Python None for Pydantic compatibility
    return df.where(df.notna(), other=None)


def _nan_to_none(v: Any) -> Any:
    """Convert numpy NaN / pandas NA to Python None."""
    try:
        if v is None:
            return None
        if isinstance(v, float) and (v != v):  # NaN check
            return None
        return v
    except Exception:
        return v


def _row_to_fixture(row: pd.Series) -> FixtureResponse:
    """Convert a DataFrame row to a FixtureResponse."""
    def s(v: Any) -> Optional[str]:
        v = _nan_to_none(v)
        return str(v) if v is not None else None

    return FixtureResponse(
        match_id=str(row["match_id"]),
        stage=str(row["stage"]),
        group=s(row["group"]),
        matchday=_nan_to_none(row["matchday"]),
        date=str(row["date"]),
        home_team=str(row["home_team"]),
        away_team=str(row["away_team"]),
        venue=s(row["venue"]) or "",
        kickoff_utc=s(row["kickoff_utc"]) or "",
        status=s(row["status"]) or "unknown",
        home_score=_nan_to_none(row["home_score"]),
        away_score=_nan_to_none(row["away_score"]),
        home_confederation=s(row["home_confederation"]),
        away_confederation=s(row["away_confederation"]),
        home_fifa_rank=_nan_to_none(row["home_fifa_rank"]),
        away_fifa_rank=_nan_to_none(row["away_fifa_rank"]),
    )


def _cache_key(home: str, away: str, match_date: str) -> str:
    return f"{home.strip().lower()}|{away.strip().lower()}|{match_date}"


def _get_cached(key: str) -> Optional[dict[str, Any]]:
    entry = _prediction_cache.get(key)
    if entry is None:
        return None
    ts, result = entry
    if (datetime.now(tz=timezone.utc) - ts).total_seconds() > CACHE_TTL_SECONDS:
        del _prediction_cache[key]
        return None
    return result


def _set_cached(key: str, result: dict[str, Any]) -> None:
    _prediction_cache[key] = (datetime.now(tz=timezone.utc), result)


def _known_teams(df: pd.DataFrame) -> set[str]:
    """Return the set of all group-stage team names (predictable teams)."""
    gs = df[df["stage"] == "Group Stage"]
    return set(gs["home_team"].dropna().tolist()) | set(gs["away_team"].dropna().tolist())


async def _run_prediction(
    home_team: str,
    away_team: str,
    match_date: str,
    is_neutral: bool = False,
) -> dict[str, Any]:
    """
    Run DebateEngine.predict_match in a thread pool so we never block the
    event loop during the ML-heavy computation.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Prediction engine not available.")

    key = _cache_key(home_team, away_team, match_date)
    cached = _get_cached(key)
    if cached is not None:
        log.info("Cache hit: %s vs %s on %s", home_team, away_team, match_date)
        return cached

    log.info("Running prediction: %s vs %s on %s", home_team, away_team, match_date)
    loop = asyncio.get_event_loop()
    fn = partial(
        _engine.predict_match,
        home_team,
        away_team,
        match_date=match_date,
        is_world_cup=True,
        is_neutral=is_neutral,
    )
    result: dict[str, Any] = await loop.run_in_executor(None, fn)
    _set_cached(key, result)
    return result


def _validate_teams(home: str, away: str, known: set[str]) -> None:
    """Raise 404 if either team is not in the known WC2026 roster."""
    if home not in known:
        raise HTTPException(status_code=404, detail=f"Unknown team: '{home}'")
    if away not in known:
        raise HTTPException(status_code=404, detail=f"Unknown team: '{away}'")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Service liveness + model-load status."""
    return HealthResponse(
        status="ok",
        model_loaded=_model_loaded,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
    )


@app.get("/fixtures", response_model=list[FixtureResponse], tags=["Fixtures"])
async def list_fixtures(
    group: Optional[str] = Query(default=None, description="Filter by group letter, e.g. A"),
    stage: Optional[str] = Query(default=None, description="Filter by stage, e.g. 'Group Stage'"),
):
    """
    Return all WC2026 fixtures.
    Optionally filter by **group** (A-L) and/or **stage**
    (Group Stage, Round of 32, Round of 16, Quarter-final, Semi-final, Final).
    """
    try:
        df = _load_fixtures()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if group:
        df = df[df["group"].str.upper() == group.upper()]
    if stage:
        df = df[df["stage"].str.lower() == stage.lower()]

    return [_row_to_fixture(row) for _, row in df.iterrows()]


@app.get("/fixtures/{match_id}", response_model=FixtureResponse, tags=["Fixtures"])
async def get_fixture(match_id: str):
    """Return detail for a single WC2026 fixture by its **match_id** (e.g. GS-A-MD1-1)."""
    try:
        df = _load_fixtures()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    row = df[df["match_id"] == match_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Fixture '{match_id}' not found.")

    return _row_to_fixture(row.iloc[0])


@app.get("/teams", response_model=list[TeamResponse], tags=["Teams"])
async def list_teams():
    """
    Return all 48 WC2026 group-stage teams with their group, confederation, and FIFA rank.
    Teams are deduplicated — each appears once (home-team row preferred for rank/confederation).
    """
    try:
        df = _load_fixtures()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    gs = df[df["stage"] == "Group Stage"].copy()

    # Build a team registry from home-team rows (authoritative confederation/rank source)
    home_info = (
        gs[["home_team", "home_team_group", "home_confederation", "home_fifa_rank"]]
        .drop_duplicates(subset=["home_team"])
        .rename(columns={
            "home_team": "name",
            "home_team_group": "group",
            "home_confederation": "confederation",
            "home_fifa_rank": "fifa_rank",
        })
    )
    away_info = (
        gs[["away_team", "away_team_group", "away_confederation", "away_fifa_rank"]]
        .drop_duplicates(subset=["away_team"])
        .rename(columns={
            "away_team": "name",
            "away_team_group": "group",
            "away_confederation": "confederation",
            "away_fifa_rank": "fifa_rank",
        })
    )

    # Merge; home-team rows take precedence
    combined = pd.concat([home_info, away_info], ignore_index=True)
    combined = combined.drop_duplicates(subset=["name"], keep="first")
    combined = combined.sort_values("name").reset_index(drop=True)

    return [
        TeamResponse(
            name=str(r["name"]),
            group=r["group"],
            confederation=r["confederation"],
            fifa_rank=r["fifa_rank"],
        )
        for _, r in combined.iterrows()
    ]


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_match(body: PredictRequest):
    """
    Run a debate-ensemble prediction for any two WC2026 teams.

    - **home_team** / **away_team**: must be valid WC2026 team names
    - **match_date**: ISO date string (YYYY-MM-DD)
    - **is_neutral**: set True for matches on neutral ground

    Results are cached in-memory for 1 hour per unique matchup.
    """
    try:
        df = _load_fixtures()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    known = _known_teams(df)
    _validate_teams(body.home_team, body.away_team, known)

    result = await _run_prediction(
        body.home_team,
        body.away_team,
        body.match_date,
        body.is_neutral,
    )
    return result


@app.get("/predict/{match_id}", response_model=PredictionResponse, tags=["Predictions"])
async def predict_fixture(match_id: str):
    """
    Predict a specific WC2026 fixture by **match_id**.
    Only works for group-stage fixtures where both teams are known.
    Knockout-stage fixtures (TBD teams) return 404.
    """
    try:
        df = _load_fixtures()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    row_df = df[df["match_id"] == match_id]
    if row_df.empty:
        raise HTTPException(status_code=404, detail=f"Fixture '{match_id}' not found.")

    row = row_df.iloc[0]
    home = str(row["home_team"])
    away = str(row["away_team"])

    if home == "TBD" or away == "TBD":
        raise HTTPException(
            status_code=404,
            detail=f"Teams for fixture '{match_id}' are not yet determined (TBD).",
        )

    known = _known_teams(df)
    _validate_teams(home, away, known)

    # Neutral ground: any fixture outside of group-stage home venues for the host nations
    # Use is_neutral=True for knockout fixtures; group stage treats home_team as designated home
    is_neutral = str(row.get("stage", "")) != "Group Stage"

    result = await _run_prediction(
        home,
        away,
        str(row["date"]),
        is_neutral=is_neutral,
    )
    return result


@app.get("/group/{group_letter}", response_model=list[FixtureWithPrediction], tags=["Predictions"])
async def group_predictions(group_letter: str):
    """
    Return all 6 group-stage matches for the given group letter (A–L),
    each enriched with a debate-ensemble prediction.

    Predictions run concurrently and are cached individually.
    """
    group_upper = group_letter.upper()

    try:
        df = _load_fixtures()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    group_df = df[
        (df["stage"] == "Group Stage") & (df["group"].str.upper() == group_upper)
    ]
    if group_df.empty:
        raise HTTPException(status_code=404, detail=f"Group '{group_upper}' not found.")

    known = _known_teams(df)

    async def _predict_one(row: pd.Series) -> FixtureWithPrediction:
        fixture = _row_to_fixture(row)
        home = str(row["home_team"])
        away = str(row["away_team"])
        try:
            _validate_teams(home, away, known)
            pred = await _run_prediction(home, away, str(row["date"]), is_neutral=False)
            return FixtureWithPrediction(fixture=fixture, prediction=pred)
        except HTTPException as exc:
            return FixtureWithPrediction(
                fixture=fixture, prediction_error=exc.detail
            )
        except Exception as exc:
            log.error("Prediction error for %s vs %s: %s", home, away, exc, exc_info=True)
            return FixtureWithPrediction(
                fixture=fixture, prediction_error=str(exc)
            )

    tasks = [_predict_one(row) for _, row in group_df.iterrows()]
    results = await asyncio.gather(*tasks)
    return list(results)


@app.get("/upcoming", response_model=list[FixtureWithPrediction], tags=["Predictions"])
async def upcoming_fixtures(
    days: int = Query(default=7, ge=1, le=30, description="Look-ahead window in days"),
    limit: int = Query(default=10, ge=1, le=48, description="Maximum fixtures to return"),
):
    """
    Return the next **N** scheduled fixtures within the look-ahead window,
    each enriched with a debate-ensemble prediction.

    - **days**: how many days ahead to look (default 7, max 30)
    - **limit**: cap on number of fixtures (default 10, max 48)
    """
    try:
        df = _load_fixtures()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    now = datetime.now(tz=timezone.utc)
    cutoff = now + timedelta(days=days)

    # Parse kickoff_utc; fall back to date column
    df["_kickoff_dt"] = pd.to_datetime(df["kickoff_utc"], utc=True, errors="coerce")
    mask_has_dt = df["_kickoff_dt"].notna()
    df_with_dt = df[mask_has_dt].copy()

    upcoming = df_with_dt[
        (df_with_dt["_kickoff_dt"] >= now) & (df_with_dt["_kickoff_dt"] <= cutoff)
    ].sort_values("_kickoff_dt").head(limit)

    # Include TBD fixtures as fixtures-only (no prediction)
    known = _known_teams(df)

    async def _predict_one(row: pd.Series) -> FixtureWithPrediction:
        fixture = _row_to_fixture(row)
        home = str(row["home_team"])
        away = str(row["away_team"])

        if home == "TBD" or away == "TBD":
            return FixtureWithPrediction(
                fixture=fixture,
                prediction_error="Teams not yet determined (TBD).",
            )

        if home not in known or away not in known:
            return FixtureWithPrediction(
                fixture=fixture,
                prediction_error=f"Team(s) outside known WC2026 roster.",
            )

        is_neutral = str(row.get("stage", "")) != "Group Stage"
        try:
            pred = await _run_prediction(home, away, str(row["date"]), is_neutral=is_neutral)
            return FixtureWithPrediction(fixture=fixture, prediction=pred)
        except Exception as exc:
            log.error("Prediction error for %s vs %s: %s", home, away, exc, exc_info=True)
            return FixtureWithPrediction(fixture=fixture, prediction_error=str(exc))

    tasks = [_predict_one(row) for _, row in upcoming.iterrows()]
    results = await asyncio.gather(*tasks)
    return list(results)
