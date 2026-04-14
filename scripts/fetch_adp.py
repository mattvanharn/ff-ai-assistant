"""Process FantasyPros ADP CSV files.

You manually download the CSVs from FantasyPros (one per year) and save them as
FantasyPros_YYYY_Overall_ADP_Rankings.csv in data/raw/. This script strips them
down to: season, player_name, adp, pos, team.

`pos` is extracted from the FantasyPros POS column (e.g. "RB1" -> "RB") so we
have a fantasy position for every ADP row even when ff_playerids doesn't know
the player.

Run: uv run python scripts/fetch_adp.py
"""

import re
from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_PATH = DATA_DIR / "adp_combined.csv"


def load_fantasypros_adp(data_dir: Path = DATA_DIR) -> pl.DataFrame:
    """Load all FantasyPros ADP CSVs and produce a single combined frame."""
    pattern = re.compile(r"FantasyPros_(\d{4})_Overall_ADP_Rankings\.csv")
    dfs = []

    for f in sorted(data_dir.glob("FantasyPros_*_Overall_ADP_Rankings.csv")):
        match = pattern.match(f.name)
        if not match:
            continue
        season = int(match.group(1))
        df = pl.read_csv(f)
        df = df.with_columns(
            pl.lit(season).alias("season"),
            pl.col("Player").alias("player_name"),
            pl.col("AVG").cast(pl.Float64).alias("adp"),
            # "RB1" -> "RB"; DST rows come out as "DST"
            pl.col("POS").str.extract(r"^([A-Z]+)", 1).alias("pos"),
            pl.col("Team").alias("team"),
        )
        df = df.select(["season", "player_name", "adp", "pos", "team"])
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No FantasyPros ADP files found in {data_dir}")

    return pl.concat(dfs)


if __name__ == "__main__":
    df = load_fantasypros_adp()
    df.write_csv(OUTPUT_PATH)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
