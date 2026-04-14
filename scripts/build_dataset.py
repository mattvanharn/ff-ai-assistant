"""Build the processed dataset from raw CSVs.

Pipeline:
    raw CSVs (produced by fetch_stats.py, fetch_adp.py, fetch_rosters.py)
      -> weekly season aggregation (with games played + PPG)
      -> resolve ADP rows to stats player_ids (3-tier match + fuzzy)
      -> join stats + ADP + roster metadata
      -> compute value scores (pos_finish_rank, adp_pos_rank, value_over_adp)
      -> write 3 parquets to data/processed/

Run:
    uv run python scripts/build_dataset.py
"""

from __future__ import annotations

import nflreadpy as nfl
import polars as pl
from rapidfuzz import fuzz, process

from ff_ai_assistant.config import (
    ALL_SCORING_SETTINGS,
    ANALYSIS_PARQUET,
    COMBINED_PARQUET,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    WEEKLY_PARQUET,
)
from ff_ai_assistant.utils import normalize_player_name

# --- constants -------------------------------------------------------------

FANTASY_POSITIONS = ["QB", "RB", "WR", "TE", "K"]  # DST excluded: no weekly data
DEFAULT_SCORING = "sleeper_half_ppr"                # for ranks + value scores
SEASON_WEEK_CUTOFF = 17                              # weeks kept in season totals
MIN_GAMES_FOR_PPG_RANK = 8                           # filter for PPG-based rankings
FUZZ_THRESHOLD = 90                                  # rapidfuzz score ≥ this = accept

DST_NAME_PATTERN = (
    r"(?i)(jaguars|rams|bears|eagles|broncos|vikings|saints|49ers|ravens|steelers|"
    r"chiefs|chargers|seahawks|cowboys|lions|bengals|texans|jets|bills|colts|titans|"
    r"patriots|panthers|dolphins|giants|falcons|packers|cardinals|commanders|"
    r"raiders|browns|buccaneers)$"
)

SCORING_KEYS = list(ALL_SCORING_SETTINGS.keys())  # e.g. ["sleeper_half_ppr", ...]
ROW_KEY = ["season", "player_name", "adp"]         # uniquely identifies an ADP row


# --- loaders ---------------------------------------------------------------


def load_weekly_stats() -> pl.DataFrame:
    """Read weekly stats CSV written by fetch_stats.py (tab-separated)."""
    return pl.read_csv(
        RAW_DATA_DIR / "player_stats_2018_2025_weekly.csv", separator="\t"
    )


def load_adp() -> pl.DataFrame:
    """Read combined ADP CSV written by fetch_adp.py. Adds is_dst + match_key.

    The CSV now carries `pos` (from FantasyPros), which we use as a fallback
    position for ADP rows whose gsis_id never resolves.
    """
    df = pl.read_csv(RAW_DATA_DIR / "adp_combined.csv")
    df = df.with_columns(
        pl.col("player_name").str.contains(DST_NAME_PATTERN).alias("is_dst")
    )
    df = df.with_columns(
        pl.Series("match_key", [normalize_player_name(n) for n in df["player_name"]])
    )
    return df


def load_rosters() -> pl.DataFrame:
    """Read roster CSV written by fetch_rosters.py (tab-separated)."""
    return pl.read_csv(
        RAW_DATA_DIR / "rosters_2018_2025.csv", separator="\t"
    )


# --- season aggregation ----------------------------------------------------


def aggregate_to_season(weekly: pl.DataFrame) -> pl.DataFrame:
    """Aggregate weekly stats to one row per (player_id, season).

    - Filters to fantasy positions + weeks 1..SEASON_WEEK_CUTOFF.
    - Sums each fantasy_points_{key} column -> seasonal_fantasy_points_{key}.
    - Counts games played (distinct weeks with a stat row).
    - Computes PPG for each scoring column.
    - Team is the LAST team in the sorted week sequence (end-of-season team).
    """
    filtered = weekly.filter(
        pl.col("position").is_in(FANTASY_POSITIONS)
        & (pl.col("week") <= SEASON_WEEK_CUTOFF)
    ).sort(["season", "player_id", "week"])

    sum_exprs = [
        pl.col(f"fantasy_points_{k}").sum().alias(f"seasonal_fantasy_points_{k}")
        for k in SCORING_KEYS
    ]

    season = filtered.group_by(
        ["player_id", "player_display_name", "position", "season"]
    ).agg(
        *sum_exprs,
        pl.len().alias("games_played"),
        pl.col("team").last().alias("team"),
    )

    ppg_exprs = [
        (
            pl.col(f"seasonal_fantasy_points_{k}") / pl.col("games_played")
        ).round(2).alias(f"ppg_{k}")
        for k in SCORING_KEYS
    ]
    season = season.with_columns(*ppg_exprs)

    season = season.with_columns(
        pl.Series(
            "match_key",
            [normalize_player_name(n) for n in season["player_display_name"]],
        )
    )
    return season


def add_ranks(season: pl.DataFrame) -> pl.DataFrame:
    """Add overall_points_rank + position_points_rank using DEFAULT_SCORING column."""
    default_col = f"seasonal_fantasy_points_{DEFAULT_SCORING}"
    return season.with_columns(
        pl.col(default_col)
        .rank(method="min", descending=True)
        .over("season")
        .cast(pl.Int32)
        .alias("overall_points_rank"),
        pl.col(default_col)
        .rank(method="min", descending=True)
        .over(["season", "position"])
        .cast(pl.Int32)
        .alias("position_points_rank"),
    )


# --- name matching ---------------------------------------------------------


def build_id_bridge() -> pl.DataFrame:
    """Load ff_playerids -> (match_key, gsis_id, position, merge_name).

    Preserves duplicate match_keys; resolution happens during ADP join.
    """
    ff_ids = nfl.load_ff_playerids()
    bridge = (
        ff_ids.filter(pl.col("gsis_id").is_not_null())
        .select(["gsis_id", "merge_name", "position", "name"])
    )
    bridge = bridge.with_columns(
        pl.Series(
            "match_key",
            [normalize_player_name(n) for n in bridge["merge_name"]],
        )
    )
    return bridge.filter(pl.col("match_key") != "")


def resolve_adp_player_ids(
    adp: pl.DataFrame,
    bridge: pl.DataFrame,
    season_stats: pl.DataFrame,
) -> pl.DataFrame:
    """Resolve every non-DST ADP row to a gsis_id using 3 tiers.

    Tier 1 — exact match_key join to bridge (replicates for ambiguous keys)
    Tier 2 — disambiguate: prefer candidates with stats in that season
    Tier 3 — fuzzy match (rapidfuzz) for ADP rows still unmatched after T1/T2

    Output columns: all input ADP columns + gsis_id + bridge_position + match_tier
        match_tier in {"exact", "disambiguated", "fuzzy", "unresolved", "dst"}
    """
    adp_real = adp.filter(~pl.col("is_dst"))
    adp_dst = adp.filter(pl.col("is_dst"))

    # --- Tier 1 + 2: exact + disambiguation ---
    # Marker: does this gsis_id have stats in that season?
    stats_marker = (
        season_stats.select(["player_id", "season"])
        .unique()
        .rename({"player_id": "gsis_id"})
        .with_columns(pl.lit(True).alias("_has_stats"))
    )

    expanded = (
        adp_real.join(
            bridge.select(["match_key", "gsis_id", "position"]).rename(
                {"position": "bridge_position"}
            ),
            on="match_key",
            how="left",
        )
        .join(stats_marker, on=["gsis_id", "season"], how="left")
        .with_columns(pl.col("_has_stats").fill_null(False))
    )

    # For each ADP row, prefer the candidate that has stats that season.
    # If none do, keep whichever comes first (n_candidates == 1 or "no hits" case).
    preferred = (
        expanded.sort(
            ROW_KEY + ["_has_stats"],
            descending=[False, False, False, True],
        )
        .unique(subset=ROW_KEY, keep="first")
        .drop("_has_stats")
    )

    # Rows with a gsis_id found at this point are either "exact" (single candidate)
    # or "disambiguated" (multi-candidate resolved via _has_stats).
    # Count candidates per row to tell them apart.
    candidate_counts = (
        expanded.group_by(ROW_KEY)
        .agg(pl.col("gsis_id").drop_nulls().n_unique().alias("n_candidates"))
    )
    preferred = preferred.join(candidate_counts, on=ROW_KEY, how="left")
    preferred = preferred.with_columns(
        pl.when(pl.col("gsis_id").is_null())
        .then(pl.lit("unresolved"))
        .when(pl.col("n_candidates") <= 1)
        .then(pl.lit("exact"))
        .otherwise(pl.lit("disambiguated"))
        .alias("match_tier")
    ).drop("n_candidates")

    already_resolved = preferred.filter(pl.col("gsis_id").is_not_null())
    still_missing = preferred.filter(pl.col("gsis_id").is_null())

    # --- Tier 3: fuzzy match ---
    # Build a match_key -> preferred gsis_id lookup from the bridge.
    # Preference: bridge entries whose gsis_id has stats in any season.
    if still_missing.height > 0:
        has_stats_ever = (
            season_stats.select(["player_id"])
            .unique()
            .rename({"player_id": "gsis_id"})
            .with_columns(pl.lit(True).alias("_has_stats_ever"))
        )
        bridge_with_flag = bridge.join(
            has_stats_ever, on="gsis_id", how="left"
        ).with_columns(pl.col("_has_stats_ever").fill_null(False))
        bridge_lookup = (
            bridge_with_flag.sort(
                ["match_key", "_has_stats_ever"], descending=[False, True]
            )
            .unique(subset=["match_key"], keep="first")
            .select(["match_key", "gsis_id", "position"])
            .rename({"position": "bridge_position"})
        )

        bridge_keys = bridge_lookup["match_key"].to_list()
        key_to_gsis = dict(
            zip(bridge_keys, bridge_lookup["gsis_id"].to_list())
        )
        key_to_pos = dict(
            zip(bridge_keys, bridge_lookup["bridge_position"].to_list())
        )

        fuzzy_gsis: list[str | None] = []
        fuzzy_pos: list[str | None] = []
        for mk in still_missing["match_key"].to_list():
            match = process.extractOne(
                mk, bridge_keys, scorer=fuzz.ratio, score_cutoff=FUZZ_THRESHOLD
            )
            if match:
                matched_key = match[0]
                fuzzy_gsis.append(key_to_gsis[matched_key])
                fuzzy_pos.append(key_to_pos[matched_key])
            else:
                fuzzy_gsis.append(None)
                fuzzy_pos.append(None)

        still_missing = still_missing.with_columns(
            pl.Series("fuzzy_gsis", fuzzy_gsis, dtype=pl.Utf8),
            pl.Series("fuzzy_pos", fuzzy_pos, dtype=pl.Utf8),
        ).with_columns(
            pl.coalesce(["gsis_id", "fuzzy_gsis"]).alias("gsis_id"),
            pl.coalesce(["bridge_position", "fuzzy_pos"]).alias("bridge_position"),
            pl.when(pl.col("fuzzy_gsis").is_not_null())
            .then(pl.lit("fuzzy"))
            .otherwise(pl.lit("unresolved"))
            .alias("match_tier"),
        ).drop(["fuzzy_gsis", "fuzzy_pos"])

    all_real = pl.concat([already_resolved, still_missing])

    # DSTs: never resolved, flagged separately
    adp_dst = adp_dst.with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("gsis_id"),
        pl.lit(None, dtype=pl.Utf8).alias("bridge_position"),
        pl.lit("dst").alias("match_tier"),
    )

    return pl.concat([all_real, adp_dst.select(all_real.columns)])


# --- roster metadata -------------------------------------------------------


def build_roster_meta(rosters: pl.DataFrame) -> pl.DataFrame:
    """Collapse rosters to one row per (gsis_id, season)."""
    return (
        rosters.filter(
            pl.col("gsis_id").is_not_null() & (pl.col("gsis_id") != "")
        )
        .group_by(["gsis_id", "season"])
        .agg(
            pl.col("full_name").first().alias("roster_name"),
            pl.col("team").last().alias("roster_team"),
            pl.col("position").first().alias("roster_position"),
            pl.col("years_exp").max().alias("years_exp"),
            pl.col("birth_date").first().alias("birth_date"),
            pl.col("height").first().alias("height"),
            pl.col("weight").first().alias("weight"),
            pl.col("college").first().alias("college"),
            pl.col("draft_number").first().alias("draft_number"),
            pl.col("entry_year").first().alias("entry_year"),
        )
    )


# --- final assembly --------------------------------------------------------


def build_combined(
    season_stats: pl.DataFrame,
    adp_resolved: pl.DataFrame,
    roster_meta: pl.DataFrame,
) -> pl.DataFrame:
    """Join stats + ADP + rosters, then append drafted-no-stats holdouts.

    Starts from season_stats so grain is guaranteed one-per-(player_id, season).
    Holdouts (ADP rows with no matching stats) are appended with zero stats.
    Positions are constrained to FANTASY_POSITIONS for holdout rows to prevent
    CB/DT/LB etc. from leaking in.
    """
    # 1. ADP rows with a resolved gsis_id, one-per-(gsis_id, season)
    #    (multiple ADP spellings that resolved to the same gsis_id collapse)
    adp_for_join = (
        adp_resolved.filter(pl.col("gsis_id").is_not_null())
        .select(["gsis_id", "season", "adp", "player_name"])
        .rename({"gsis_id": "player_id", "player_name": "adp_player_name"})
        .sort("adp")
        .unique(subset=["player_id", "season"], keep="first")
    )

    combined = season_stats.join(
        adp_for_join, on=["player_id", "season"], how="left"
    ).join(
        roster_meta,
        left_on=["player_id", "season"],
        right_on=["gsis_id", "season"],
        how="left",
    )

    # 2. Drafted-no-stats holdouts: ADP rows that either had no resolved gsis_id,
    #    OR resolved to a gsis_id that has no stats row for that season.
    matched_ids = combined.filter(pl.col("adp").is_not_null()).select(
        ["player_id", "season"]
    )

    holdouts_resolved = (
        adp_resolved.filter(
            pl.col("gsis_id").is_not_null() & ~pl.col("is_dst")
        )
        .rename({"gsis_id": "player_id"})
        .join(matched_ids, on=["player_id", "season"], how="anti")
    )

    holdouts_unresolved = (
        adp_resolved.filter(
            pl.col("gsis_id").is_null() & ~pl.col("is_dst")
        )
        .rename({"gsis_id": "player_id"})
    )

    holdouts = pl.concat([holdouts_resolved, holdouts_unresolved])

    # Position resolution priority for holdouts:
    #   1. adp `pos` (from FantasyPros) — always a fantasy position when present
    #   2. bridge_position (from ff_playerids) — fallback; may use legacy codes
    #      like "PK" for kicker or "XX" for retired/deceased, so those need the
    #      FantasyPros pos to avoid being dropped by the fantasy filter below.
    holdouts = holdouts.with_columns(
        pl.coalesce(["pos", "bridge_position"]).alias("position")
    ).filter(pl.col("position").is_in(FANTASY_POSITIONS))

    # De-dup to one row per (player_id, season) — earlier ADP (lower number) wins
    # For unresolved rows player_id is null; keep one per (player_name, season).
    holdouts = holdouts.sort("adp").unique(
        subset=["player_name", "season"], keep="first"
    )

    # Build holdout rows matching `combined` schema
    holdout_rows = pl.DataFrame(
        {
            "player_id": holdouts["player_id"],
            "player_display_name": holdouts["player_name"],
            "position": holdouts["position"],
            "season": holdouts["season"],
            "match_key": holdouts["match_key"],
            "games_played": pl.Series(
                [0] * holdouts.height, dtype=pl.UInt32
            ),
            "team": pl.Series([None] * holdouts.height, dtype=pl.Utf8),
            "overall_points_rank": pl.Series(
                [None] * holdouts.height, dtype=pl.Int32
            ),
            "position_points_rank": pl.Series(
                [None] * holdouts.height, dtype=pl.Int32
            ),
            "adp": holdouts["adp"],
            "adp_player_name": holdouts["player_name"],
        }
    )
    # Zero out stat columns (seasonal + ppg for every scoring key)
    zero_cols = []
    for k in SCORING_KEYS:
        zero_cols.append(
            pl.lit(0.0).alias(f"seasonal_fantasy_points_{k}")
        )
        zero_cols.append(pl.lit(0.0).alias(f"ppg_{k}"))
    holdout_rows = holdout_rows.with_columns(*zero_cols)

    # Fill roster metadata columns with nulls (align schema with combined)
    for col, dtype in combined.schema.items():
        if col not in holdout_rows.columns:
            holdout_rows = holdout_rows.with_columns(
                pl.lit(None).cast(dtype).alias(col)
            )

    holdout_rows = holdout_rows.select(combined.columns)
    return pl.concat([combined, holdout_rows])


def compute_value_scores(combined: pl.DataFrame) -> pl.DataFrame:
    """Return analysis subset: matched QB/RB/WR/TE with ADP, enriched with ranks."""
    default_col = f"seasonal_fantasy_points_{DEFAULT_SCORING}"
    analysis = combined.filter(
        pl.col("adp").is_not_null()
        & pl.col(default_col).is_not_null()
        & (pl.col("games_played") > 0)
        & pl.col("position").is_in(["QB", "RB", "WR", "TE"])
    )

    analysis = analysis.with_columns(
        pl.col(default_col)
        .rank(method="ordinal", descending=True)
        .over(["season", "position"])
        .cast(pl.Int32)
        .alias("pos_finish_rank"),
        pl.col("adp")
        .rank(method="ordinal", descending=False)
        .over(["season", "position"])
        .cast(pl.Int32)
        .alias("adp_pos_rank"),
    ).with_columns(
        (pl.col("adp_pos_rank") - pl.col("pos_finish_rank")).alias("value_over_adp"),
        ((pl.col("adp") - 1) / 12).floor().cast(pl.Int32).add(1).alias("draft_round"),
    )

    starter_cutoff = (
        pl.when(pl.col("position") == "QB")
        .then(pl.col("pos_finish_rank") <= 12)
        .when(pl.col("position") == "RB")
        .then(pl.col("pos_finish_rank") <= 24)
        .when(pl.col("position") == "WR")
        .then(pl.col("pos_finish_rank") <= 24)
        .when(pl.col("position") == "TE")
        .then(pl.col("pos_finish_rank") <= 12)
        .otherwise(False)
    )
    return analysis.with_columns(starter_cutoff.alias("is_starter"))


# --- validation ------------------------------------------------------------


def validate(
    combined: pl.DataFrame,
    analysis: pl.DataFrame,
    weekly: pl.DataFrame,
    adp: pl.DataFrame,
    resolved: pl.DataFrame,
) -> None:
    """Assert grain + sanity checks. Prints a summary report."""
    checks_passed = 0
    checks_failed = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal checks_passed, checks_failed
        mark = "PASS" if condition else "FAIL"
        print(f"  [{mark}] {name}{(' -- ' + detail) if detail else ''}")
        if condition:
            checks_passed += 1
        else:
            checks_failed += 1

    print("Validation:")

    # Grain: one row per (player_id, season) for rows with non-null player_id
    with_id = combined.filter(pl.col("player_id").is_not_null())
    n_unique = with_id.select(["player_id", "season"]).n_unique()
    check(
        "combined grain is (player_id, season)",
        n_unique == with_id.height,
        f"{with_id.height - n_unique} duplicates",
    )

    # Positions constrained
    bad_positions = combined.filter(
        ~pl.col("position").is_in(FANTASY_POSITIONS)
    )
    check(
        "combined positions are in FANTASY_POSITIONS",
        bad_positions.height == 0,
        f"{bad_positions.height} rows with bad positions",
    )

    # ADP coverage: every non-DST ADP row should appear in combined
    adp_non_dst = adp.filter(~pl.col("is_dst"))
    combined_adp_rows = combined.filter(pl.col("adp").is_not_null()).select(
        ["season", "adp_player_name", "adp"]
    )
    coverage = combined_adp_rows.unique().height
    check(
        "every non-DST ADP row represented in combined",
        coverage >= adp_non_dst.unique(subset=ROW_KEY).height,
        f"{coverage}/{adp_non_dst.unique(subset=ROW_KEY).height}",
    )

    # analysis ranks start at 1
    rank_check = (
        analysis.group_by(["season", "position"])
        .agg(pl.col("pos_finish_rank").min().alias("rank_min"))
        .filter(pl.col("rank_min") != 1)
    )
    check(
        "analysis pos_finish_rank starts at 1 per (season, position)",
        rank_check.height == 0,
        f"{rank_check.height} groups violate",
    )

    # weekly has weeks 1-18 (we include full data in raw, even though season
    # aggregation uses cutoff)
    weeks = sorted(weekly["week"].unique().to_list())
    check(
        "weekly parquet contains weeks 1..17 at minimum",
        set(range(1, 18)).issubset(set(weeks)),
        f"weeks present: {weeks}",
    )

    # Match tier breakdown
    tier_counts = resolved.group_by("match_tier").agg(pl.len().alias("n")).sort("n", descending=True)
    print("\n  Match tier breakdown:")
    for row in tier_counts.iter_rows(named=True):
        print(f"    {row['match_tier']:15s}  {row['n']}")

    non_dst = resolved.filter(~pl.col("is_dst"))
    resolved_count = non_dst.filter(pl.col("gsis_id").is_not_null()).height
    print(
        f"\n  Resolved {resolved_count}/{non_dst.height} non-DST ADP rows "
        f"({resolved_count / non_dst.height:.1%})"
    )

    print(f"\n  {checks_passed} passed, {checks_failed} failed")
    if checks_failed > 0:
        raise AssertionError(f"{checks_failed} validation checks failed")


# --- main ------------------------------------------------------------------


def main() -> None:
    print("Loading raw data...")
    weekly = load_weekly_stats()
    adp = load_adp()
    rosters = load_rosters()

    print("Aggregating weekly -> season...")
    season_stats = add_ranks(aggregate_to_season(weekly))

    print("Resolving ADP -> gsis_id...")
    bridge = build_id_bridge()
    adp_resolved = resolve_adp_player_ids(adp, bridge, season_stats)

    print("Building roster metadata...")
    roster_meta = build_roster_meta(rosters)

    print("Assembling combined table...")
    combined = build_combined(season_stats, adp_resolved, roster_meta)
    analysis = compute_value_scores(combined)

    print()
    validate(combined, analysis, weekly, adp, adp_resolved)

    print("\nWriting parquets...")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(COMBINED_PARQUET)
    analysis.write_parquet(ANALYSIS_PARQUET)
    weekly.write_parquet(WEEKLY_PARQUET)

    print(f"  combined_stats_adp.parquet:  {combined.shape}")
    print(f"  analysis_with_value.parquet: {analysis.shape}")
    print(f"  weekly_stats.parquet:        {weekly.shape}")


if __name__ == "__main__":
    main()
