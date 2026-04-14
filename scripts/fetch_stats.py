"""Fetch weekly player stats from nflreadpy and compute fantasy points per week.

Thin script: network I/O + per-week scoring only. Season aggregation, PPG, and
rank computation live in scripts/build_dataset.py so they can be iterated on
without re-fetching.

Run: uv run python scripts/fetch_stats.py
"""

import nflreadpy as nfl
import polars as pl
from pathlib import Path

from ff_ai_assistant.config import ALL_SCORING_SETTINGS

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SEASONS = list(range(2018, 2026))

weekly_stats = nfl.load_player_stats(SEASONS, summary_level="week")

# Keep all weeks (including week 18) and all fantasy positions.
# build_dataset.py decides which weeks and positions to include in aggregation.
weekly_stats = weekly_stats.filter(
    pl.col("position").is_in(["QB", "RB", "WR", "TE", "K", "DST"])
)

# Compute per-week fantasy points for every scoring setting.
for key, scoring in ALL_SCORING_SETTINGS.items():
    weekly_stats = weekly_stats.with_columns(
        pl.sum_horizontal(
            [pl.col(stat).fill_null(0) * weight for stat, weight in scoring.items()]
        ).alias(f"fantasy_points_{key}")
    )

weekly_stats.write_csv(
    DATA_DIR / "player_stats_2018_2025_weekly.csv", separator="\t"
)
print(f"Saved weekly stats: {weekly_stats.shape}")
