# WC 2026 Predictor

Match prediction system for the FIFA World Cup 2026 (USA, Canada, Mexico).

## Features

- **Statistical Model** — Elo ratings, head-to-head records, historical match data
- **Form Model** — Recent team performance and momentum analysis
- **News Model** — Sentiment from team news (injuries, transfers, morale)
- **Ensemble** — Combines all three models for final prediction
- **Dashboard** — Interactive HTML dashboard for browsing predictions
- **FastAPI Backend** — REST API for predictions and fixtures

## Quick Start

```bash
pip install -r requirements.txt

# Train models
python -m src.models.train_models

# Start API server
uvicorn src.api.main:app --reload

# Open dashboard
open dashboard/index.html
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/fixtures` | All WC2026 fixtures |
| GET | `/teams` | All 48 teams with group info |
| POST | `/predict` | Ad-hoc prediction for any two teams |
| GET | `/predict/{match_id}` | Predict a specific fixture |
| GET | `/group/{group}` | Predictions for a group |
| GET | `/upcoming` | Next matches with auto-predictions |

## Data Sources

- Historical international match results (1872-2025)
- FIFA/Elo ratings
- Team news and injury reports

## Project Structure

```
src/
├── api/          # FastAPI backend
├── data/         # Data fetching and processing
├── debate/       # AI debate engine for match analysis
└── models/       # Statistical, form, and news models
data/
├── raw/          # Raw CSVs and team news JSON
└── processed/    # Processed fixtures, features, Elo ratings
dashboard/        # Static HTML dashboard
```

## License

MIT
