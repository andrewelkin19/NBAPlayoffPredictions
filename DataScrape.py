"""
NBA Playoff Outcome Prediction - Data Pipeline
CS830 Final Project - Andrew Elkin

Fetches regular season team stats and playoff game results from nba_api,
constructs a feature matrix suitable for training a game outcome classifier.

Features are based on Dean Oliver's Four Factors:
  Offense: eFG%, TOV%, ORB%, FTR
  Defense: opponent eFG%, opponent TOV%, DRB%, opponent FTR

Optional features (each requires a flag):
  --clutch       CLUTCH_WIN_PCT, CLUTCH_NET_RATING
  --star-players TOP1_PM, TOP2_AVG_PM, TOP3_AVG_PM
  --bpm          TOP1_BPM, TOP2_AVG_BPM, TOP3_AVG_BPM  (requires scrape_bpm.py)
  --ts           TS_PCT, OPP_TS_PCT (True Shooting % — includes free throws)
  --ratings      OFF_RATING, DEF_RATING, NET_RATING (official NBA per-100-possession ratings)

Output: data/training_data.csv, data/playoff_brackets.csv

Usage:
  python DataScrape.py                                  # Four Factors only
  python DataScrape.py --clutch --star-players --bpm    # all features
  python DataScrape.py --clutch --star-players --bpm --ts  # including TS%
  python DataScrape.py --ratings                        # include official ratings
"""

import argparse
import time
import os
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamefinder, leaguedashteamclutch, leaguedashplayerstats, leaguedashteamstats

# ── Config ────────────────────────────────────────────────────────────────────

SEASONS = [
    "2005-06", "2006-07", "2007-08", "2008-09", "2009-10",
    "2010-11",            "2012-13", "2013-14", "2014-15",
    "2015-16", "2016-17", "2017-18", "2018-19",
                          "2020-21", "2021-22", "2022-23",
                          "2023-24", "2024-25"
]

DATA_DIR    = "data"
SLEEP_SEC   = 2.5
TIMEOUT     = 60
MAX_RETRIES = 3

# ── Helpers ───────────────────────────────────────────────────────────────────

def sleep():
    time.sleep(SLEEP_SEC)


def season_to_year(season: str) -> int:
    return int(season[:4]) + 1


# ── Step 1: Fetch raw game logs ───────────────────────────────────────────────

def fetch_game_logs(season: str, season_type: str) -> pd.DataFrame:
    """
    Returns one row per team per game for a given season and type.
    Caches results to disk so re-runs don't hit the API again.
    """
    safe_type  = season_type.replace(" ", "_")
    cache_path = os.path.join(DATA_DIR, f"cache_{season}_{safe_type}.csv")

    if os.path.exists(cache_path):
        print(f"  Loading from cache: {cache_path}")
        df = pd.read_csv(cache_path)
        df["TEAM_ID"] = df["TEAM_ID"].astype(int)
        return df

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                league_id_nullable="00",
                season_type_nullable=season_type,
                timeout=TIMEOUT,
            )
            sleep()
            df = finder.get_data_frames()[0]
            df["SEASON"]  = season
            df["TEAM_ID"] = df["TEAM_ID"].astype(int)
            df.to_csv(cache_path, index=False)
            return df
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = SLEEP_SEC * (2 ** attempt)
            print(f"  Attempt {attempt} failed ({e}). Retrying in {wait:.1f}s...")
            time.sleep(wait)


# ── Step 1b: Fetch clutch stats ───────────────────────────────────────────────

def fetch_clutch_stats(season: str) -> pd.DataFrame:
    """
    Fetch per-team clutch stats for a season from nba_api.
    Clutch = last 5 minutes, game within 5 points (NBA standard definition).
    Results are cached to data/cache_{season}_clutch.csv.
    """
    cache_path = os.path.join(DATA_DIR, f"cache_{season}_clutch.csv")

    if os.path.exists(cache_path):
        print(f"  Loading clutch stats from cache: {cache_path}")
        df = pd.read_csv(cache_path)
        df["TEAM_ID"] = df["TEAM_ID"].astype(int)
        return df

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            endpoint = leaguedashteamclutch.LeagueDashTeamClutch(
                season=season,
                per_mode_detailed="PerGame",
                timeout=TIMEOUT,
            )
            sleep()
            df = endpoint.get_data_frames()[0]
            df["TEAM_ID"] = df["TEAM_ID"].astype(int)
            df.to_csv(cache_path, index=False)
            print(f"  Clutch stats cached: {cache_path}")
            return df
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = SLEEP_SEC * (2 ** attempt)
            print(f"  Attempt {attempt} failed ({e}). Retrying in {wait:.1f}s...")
            time.sleep(wait)


# ── Step 1c: Fetch per-player regular season stats ───────────────────────────

def fetch_player_stats(season: str) -> pd.DataFrame:
    """
    Fetch per-player regular season per-game stats for a season.
    Results cached to data/cache_{season}_player_stats.csv.
    """
    cache_path = os.path.join(DATA_DIR, f"cache_{season}_player_stats.csv")

    if os.path.exists(cache_path):
        print(f"  Loading player stats from cache: {cache_path}")
        df = pd.read_csv(cache_path)
        df["TEAM_ID"] = df["TEAM_ID"].astype(int)
        return df

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed="PerGame",
                timeout=TIMEOUT,
            )
            sleep()
            df = endpoint.get_data_frames()[0]
            df["TEAM_ID"] = df["TEAM_ID"].astype(int)
            df.to_csv(cache_path, index=False)
            print(f"  Player stats cached: {cache_path}")
            return df
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = SLEEP_SEC * (2 ** attempt)
            print(f"  Attempt {attempt} failed ({e}). Retrying in {wait:.1f}s...")
            time.sleep(wait)


def fetch_player_game_logs(season: str, season_type: str = "Playoffs") -> pd.DataFrame:
    """
    Fetch per-player per-game box scores for a season and game type.
    Results cached to data/cache_{season}_player_game_logs_{type}.csv.
    """
    from nba_api.stats.endpoints import playergamelogs

    safe_type  = season_type.replace(" ", "_")
    cache_path = os.path.join(DATA_DIR, f"cache_{season}_player_game_logs_{safe_type}.csv")

    if os.path.exists(cache_path):
        print(f"  Loading player game logs from cache: {cache_path}")
        df = pd.read_csv(cache_path)
        df["TEAM_ID"]  = df["TEAM_ID"].astype(int)
        df["GAME_ID"]  = df["GAME_ID"].astype(str)
        return df

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            endpoint = playergamelogs.PlayerGameLogs(
                season_nullable=season,
                season_type_nullable=season_type,
                timeout=TIMEOUT,
            )
            sleep()
            df = endpoint.get_data_frames()[0]
            df["TEAM_ID"] = df["TEAM_ID"].astype(int)
            df["GAME_ID"] = df["GAME_ID"].astype(str)
            df.to_csv(cache_path, index=False)
            print(f"  Player game logs cached: {cache_path}")
            return df
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = SLEEP_SEC * (2 ** attempt)
            print(f"  Attempt {attempt} failed ({e}). Retrying in {wait:.1f}s...")
            time.sleep(wait)


# ── Step 1d: Fetch official advanced team stats ───────────────────────────────

def fetch_advanced_team_stats(season: str) -> pd.DataFrame:
    """
    Fetch official per-team advanced stats (OFF_RATING, DEF_RATING, NET_RATING)
    from nba_api's LeagueDashTeamStats endpoint with measure_type='Advanced'.

    These are the NBA's official possession-normalized ratings, avoiding the
    approximation error in hand-computed possession estimates.
    Results cached to data/cache_{season}_advanced_stats.csv.
    """
    cache_path = os.path.join(DATA_DIR, f"cache_{season}_advanced_stats.csv")

    if os.path.exists(cache_path):
        print(f"  Loading advanced stats from cache: {cache_path}")
        df = pd.read_csv(cache_path)
        df["TEAM_ID"] = df["TEAM_ID"].astype(int)
        return df

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            endpoint = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense="Advanced",
                per_mode_detailed="PerGame",
                season_type_all_star="Regular Season",
                timeout=TIMEOUT,
            )
            sleep()
            df = endpoint.get_data_frames()[0]
            df["TEAM_ID"] = df["TEAM_ID"].astype(int)
            df.to_csv(cache_path, index=False)
            print(f"  Advanced stats cached: {cache_path}")
            return df
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = SLEEP_SEC * (2 ** attempt)
            print(f"  Attempt {attempt} failed ({e}). Retrying in {wait:.1f}s...")
            time.sleep(wait)


# ── Step 2: Compute season-level team features ────────────────────────────────

def compute_team_features(reg_season_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates per-team regular season stats into Dean Oliver's Four Factors,
    plus True Shooting % (always computed; included in training data only if
    --ts flag is passed).

    Four Factors — Offense:
      EFG_PCT  = (FGM + 0.5*FG3M) / FGA
      TOV_PCT  = TOV / (FGA + 0.44*FTA + TOV)
      ORB_PCT  = OREB / (OREB + OPP_DREB)
      FTR      = FTA / FGA

    Four Factors — Defense:
      OPP_EFG_PCT, OPP_TOV_PCT, DRB_PCT, OPP_FTR

    True Shooting %:
      TS_PCT     = PTS / (2 * (FGA + 0.44 * FTA))
      OPP_TS_PCT = OPP_PTS / (2 * (OPP_FGA + 0.44 * OPP_FTA))
      (always computed here; only added to training rows if --ts is passed)

    WIN_PCT is kept for the Finals home-court tiebreaker in bracket_simulator.
    """
    df = reg_season_logs.copy()
    df["HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs." in x else 0)
    df["WIN"]  = (df["WL"] == "W").astype(int)

    # Build opponent lookup
    opp_cols = ["GAME_ID", "TEAM_ID",
                "PTS", "FGM", "FGA", "FG3M", "FG3A",
                "FTA", "OREB", "DREB", "TOV"]
    opp = df[opp_cols].copy()
    opp.columns = ["GAME_ID", "OPP_TEAM_ID",
                   "OPP_PTS", "OPP_FGM", "OPP_FGA", "OPP_FG3M", "OPP_FG3A",
                   "OPP_FTA", "OPP_OREB", "OPP_DREB", "OPP_TOV"]

    df = df.merge(opp, on="GAME_ID", how="left")
    df = df[df["TEAM_ID"] != df["OPP_TEAM_ID"]]

    agg = df.groupby("TEAM_ID").agg(
        TEAM_NAME         = ("TEAM_NAME",        "first"),
        TEAM_ABBREVIATION = ("TEAM_ABBREVIATION", "first"),
        GAMES             = ("GAME_ID",           "count"),
        WINS              = ("WIN",               "sum"),
        NET_RATING        = ("PLUS_MINUS",        "mean"),
        # Offensive raw stats
        FGM_PG            = ("FGM",               "mean"),
        FGA_PG            = ("FGA",               "mean"),
        FG3M_PG           = ("FG3M",              "mean"),
        FTA_PG            = ("FTA",               "mean"),
        OREB_PG           = ("OREB",              "mean"),
        DREB_PG           = ("DREB",              "mean"),
        TOV_PG            = ("TOV",               "mean"),
        PTS_PG            = ("PTS",               "mean"),
        # Defensive raw stats
        OPP_FGM_PG        = ("OPP_FGM",           "mean"),
        OPP_FGA_PG        = ("OPP_FGA",           "mean"),
        OPP_FG3M_PG       = ("OPP_FG3M",          "mean"),
        OPP_FTA_PG        = ("OPP_FTA",           "mean"),
        OPP_OREB_PG       = ("OPP_OREB",          "mean"),
        OPP_DREB_PG       = ("OPP_DREB",          "mean"),
        OPP_TOV_PG        = ("OPP_TOV",           "mean"),
        OPP_PTS_PG        = ("OPP_PTS",           "mean"),
    ).reset_index()

    agg["WIN_PCT"] = agg["WINS"] / agg["GAMES"]

    # ── Four Factors — Offense ────────────────────────────────────────────────
    agg["EFG_PCT"] = (agg["FGM_PG"] + 0.5 * agg["FG3M_PG"]) / agg["FGA_PG"]
    agg["TOV_PCT"] = agg["TOV_PG"] / (agg["FGA_PG"] + 0.44 * agg["FTA_PG"] + agg["TOV_PG"])
    agg["ORB_PCT"] = agg["OREB_PG"] / (agg["OREB_PG"] + agg["OPP_DREB_PG"])
    agg["FTR"]     = agg["FTA_PG"] / agg["FGA_PG"]

    # ── Four Factors — Defense ────────────────────────────────────────────────
    agg["OPP_EFG_PCT"] = (agg["OPP_FGM_PG"] + 0.5 * agg["OPP_FG3M_PG"]) / agg["OPP_FGA_PG"]
    agg["OPP_TOV_PCT"] = agg["OPP_TOV_PG"] / (
        agg["OPP_FGA_PG"] + 0.44 * agg["OPP_FTA_PG"] + agg["OPP_TOV_PG"]
    )
    agg["DRB_PCT"] = agg["DREB_PG"] / (agg["DREB_PG"] + agg["OPP_OREB_PG"])
    agg["OPP_FTR"] = agg["OPP_FTA_PG"] / agg["OPP_FGA_PG"]

    # ── True Shooting % ───────────────────────────────────────────────────────
    # Always computed here; only pulled into training examples if --ts is passed.
    # TS% = PTS / (2 * (FGA + 0.44 * FTA))
    # Includes free throw value unlike eFG%, which only adjusts for 3-pointers.
    agg["TS_PCT"]     = agg["PTS_PG"]     / (2 * (agg["FGA_PG"]     + 0.44 * agg["FTA_PG"]))
    agg["OPP_TS_PCT"] = agg["OPP_PTS_PG"] / (2 * (agg["OPP_FGA_PG"] + 0.44 * agg["OPP_FTA_PG"]))

    agg["TEAM_ID"] = agg["TEAM_ID"].astype(int)
    return agg


# ── Step 2b: Merge clutch features ───────────────────────────────────────────

def compute_clutch_features(
    team_features: pd.DataFrame,
    clutch_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges clutch stats into team_features.
    Adds CLUTCH_WIN_PCT and CLUTCH_NET_RATING columns.
    """
    clutch = clutch_df[["TEAM_ID", "W_PCT", "PLUS_MINUS"]].copy()
    clutch = clutch.rename(columns={
        "W_PCT":      "CLUTCH_WIN_PCT",
        "PLUS_MINUS": "CLUTCH_NET_RATING",
    })
    clutch["TEAM_ID"] = clutch["TEAM_ID"].astype(int)

    merged = team_features.merge(clutch, on="TEAM_ID", how="left")
    merged["CLUTCH_WIN_PCT"]    = merged["CLUTCH_WIN_PCT"].fillna(0.5)
    merged["CLUTCH_NET_RATING"] = merged["CLUTCH_NET_RATING"].fillna(0.0)

    n_missing = merged["CLUTCH_WIN_PCT"].isna().sum()
    if n_missing:
        print(f"  [warning] {n_missing} teams missing clutch data — filled with defaults")

    return merged


# ── Step 2c: Compute star player features ────────────────────────────────────

def compute_star_player_features(
    team_features: pd.DataFrame,
    player_stats: pd.DataFrame,
    min_mpg: float = 20.0,
    min_games: int = 20,
) -> pd.DataFrame:
    """
    Adds TOP1_PM, TOP2_AVG_PM, TOP3_AVG_PM columns to team_features.
    Only players meeting minute and game thresholds are considered.
    """
    qualified = player_stats[
        (player_stats["MIN"] >= min_mpg) &
        (player_stats["GP"]  >= min_games)
    ].copy()

    star_rows = []
    for team_id in team_features["TEAM_ID"].unique():
        team_players = (
            qualified[qualified["TEAM_ID"] == int(team_id)]
            .sort_values("PLUS_MINUS", ascending=False)
        )

        if len(team_players) == 0:
            top1_pm, top2_avg_pm, top3_avg_pm = 0.0, 0.0, 0.0
        elif len(team_players) == 1:
            top1_pm = float(team_players.iloc[0]["PLUS_MINUS"])
            top2_avg_pm = top3_avg_pm = top1_pm
        elif len(team_players) == 2:
            top1_pm     = float(team_players.iloc[0]["PLUS_MINUS"])
            top2_avg_pm = float(team_players.head(2)["PLUS_MINUS"].mean())
            top3_avg_pm = top2_avg_pm
        else:
            top1_pm     = float(team_players.iloc[0]["PLUS_MINUS"])
            top2_avg_pm = float(team_players.head(2)["PLUS_MINUS"].mean())
            top3_avg_pm = float(team_players.head(3)["PLUS_MINUS"].mean())

        star_rows.append({
            "TEAM_ID":     int(team_id),
            "TOP1_PM":     top1_pm,
            "TOP2_AVG_PM": top2_avg_pm,
            "TOP3_AVG_PM": top3_avg_pm,
        })

    star_df = pd.DataFrame(star_rows)
    merged  = team_features.merge(star_df, on="TEAM_ID", how="left")
    merged["TOP1_PM"]     = merged["TOP1_PM"].fillna(0.0)
    merged["TOP2_AVG_PM"] = merged["TOP2_AVG_PM"].fillna(0.0)
    merged["TOP3_AVG_PM"] = merged["TOP3_AVG_PM"].fillna(0.0)
    return merged


# ── Step 2d: Merge BPM features ──────────────────────────────────────────────

from scrape_bpm import load_bpm_season, compute_star_bpm_features


# ── Step 2e: Merge official advanced ratings ──────────────────────────────────

def compute_advanced_ratings(
    team_features: pd.DataFrame,
    advanced_df: pd.DataFrame,
) -> pd.DataFrame:
    ratings = advanced_df[["TEAM_ID", "OFF_RATING", "DEF_RATING", "NET_RATING"]].copy()
    ratings["TEAM_ID"] = ratings["TEAM_ID"].astype(int)

    # Drop the approximated NET_RATING (mean PLUS_MINUS per game) so the
    # official per-100-possession version from advanced stats takes its place
    # without pandas creating NET_RATING_x / NET_RATING_y suffixes.
    tf = team_features.drop(columns=["NET_RATING"], errors="ignore")

    merged = tf.merge(ratings, on="TEAM_ID", how="left")

    n_missing = merged["OFF_RATING"].isna().sum()
    if n_missing:
        print(f"  [warning] {n_missing} teams missing advanced ratings — filled with 0.0")

    merged["OFF_RATING"] = merged["OFF_RATING"].fillna(0.0)
    merged["DEF_RATING"] = merged["DEF_RATING"].fillna(0.0)
    merged["NET_RATING"] = merged["NET_RATING"].fillna(0.0)
    return merged


# ── Step 3: Compute head-to-head regular season win rate ──────────────────────

def compute_h2h(reg_season_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Returns [TEAM_ID, OPP_TEAM_ID, H2H_WIN_PCT].
    """
    df = reg_season_logs.copy()
    df["WIN"] = (df["WL"] == "W").astype(int)

    opp = df[["GAME_ID", "TEAM_ID"]].copy()
    opp.columns = ["GAME_ID", "OPP_TEAM_ID"]

    df = df.merge(opp, on="GAME_ID", how="left")
    df = df[df["TEAM_ID"] != df["OPP_TEAM_ID"]]

    h2h = df.groupby(["TEAM_ID", "OPP_TEAM_ID"]).agg(
        H2H_WINS  = ("WIN", "sum"),
        H2H_GAMES = ("WIN", "count"),
    ).reset_index()

    h2h["H2H_WIN_PCT"] = h2h["H2H_WINS"] / h2h["H2H_GAMES"]
    h2h["TEAM_ID"]     = h2h["TEAM_ID"].astype(int)
    h2h["OPP_TEAM_ID"] = h2h["OPP_TEAM_ID"].astype(int)
    return h2h[["TEAM_ID", "OPP_TEAM_ID", "H2H_WIN_PCT"]]


# ── Step 4: Build training examples ──────────────────────────────────────────

def build_training_examples(
    playoff_logs:   pd.DataFrame,
    team_features:  pd.DataFrame,
    h2h:            pd.DataFrame,
    season:         str,
    include_ts:     bool = False,
) -> pd.DataFrame:
    """
    For each playoff game, creates one training row anchored on the home team.
    Features are differentials (home - away). Label = 1 if home team wins.

    Optional feature groups are included only if the corresponding columns
    exist in team_features AND the relevant flag was passed:
      - clutch:   CLUTCH_WIN_PCT, CLUTCH_NET_RATING   (auto-detected)
      - star PM:  TOP1_PM, TOP2_AVG_PM, TOP3_AVG_PM   (auto-detected)
      - BPM:      TOP1_BPM, TOP2_AVG_BPM, TOP3_AVG_BPM (auto-detected)
      - TS%:      TS_PCT, OPP_TS_PCT                   (requires include_ts=True)
      - ratings:  OFF_RATING, DEF_RATING, NET_RATING   (auto-detected)
    """
    df = playoff_logs.copy()
    df["WIN"]  = (df["WL"] == "W").astype(int)
    df["HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs." in x else 0)

    home = df[df["HOME"] == 1][["GAME_ID", "TEAM_ID", "WIN"]].copy()
    away = df[df["HOME"] == 0][["GAME_ID", "TEAM_ID"]].copy()
    away.columns = ["GAME_ID", "OPP_TEAM_ID"]

    games = home.merge(away, on="GAME_ID", how="inner")
    games["SEASON"] = season

    games["TEAM_ID"]         = games["TEAM_ID"].astype(int)
    games["OPP_TEAM_ID"]     = games["OPP_TEAM_ID"].astype(int)
    team_features            = team_features.copy()
    team_features["TEAM_ID"] = team_features["TEAM_ID"].astype(int)
    h2h                      = h2h.copy()
    h2h["TEAM_ID"]           = h2h["TEAM_ID"].astype(int)
    h2h["OPP_TEAM_ID"]       = h2h["OPP_TEAM_ID"].astype(int)

    # ── Determine which columns to pull from team_features ────────────────────
    base_feat_cols = [
        "TEAM_ID",
        "EFG_PCT", "OPP_EFG_PCT",
        "TOV_PCT", "OPP_TOV_PCT",
        "ORB_PCT", "DRB_PCT",
        "FTR",     "OPP_FTR",
        "WIN_PCT",
    ]

    # TS% is always computed in team_features but only included if --ts passed
    has_ts = include_ts and "TS_PCT" in team_features.columns
    if has_ts:
        base_feat_cols += ["TS_PCT", "OPP_TS_PCT"]

    has_clutch = "CLUTCH_WIN_PCT" in team_features.columns
    if has_clutch:
        base_feat_cols += ["CLUTCH_WIN_PCT", "CLUTCH_NET_RATING"]

    has_star = "TOP1_PM" in team_features.columns
    if has_star:
        base_feat_cols += ["TOP1_PM", "TOP2_AVG_PM"]
        if "TOP3_AVG_PM" in team_features.columns:
            base_feat_cols += ["TOP3_AVG_PM"]

    has_bpm = "TOP1_BPM" in team_features.columns
    if has_bpm:
        base_feat_cols += ["TOP1_BPM", "TOP2_AVG_BPM", "TOP3_AVG_BPM"]

    # Official advanced ratings are included whenever --ratings was passed
    has_ratings = "OFF_RATING" in team_features.columns
    if has_ratings:
        base_feat_cols += ["OFF_RATING", "DEF_RATING", "NET_RATING"]

    home_feats = team_features[base_feat_cols].add_prefix("HOME_").rename(
        columns={"HOME_TEAM_ID": "TEAM_ID"}
    )
    away_feats = team_features[base_feat_cols].add_prefix("AWAY_").rename(
        columns={"AWAY_TEAM_ID": "OPP_TEAM_ID"}
    )

    games = games.merge(home_feats, on="TEAM_ID",     how="left")
    games = games.merge(away_feats, on="OPP_TEAM_ID", how="left")
    games = games.merge(h2h, on=["TEAM_ID", "OPP_TEAM_ID"], how="left")
    games["H2H_WIN_PCT"] = games["H2H_WIN_PCT"].fillna(0.5)

    # ── Four Factor differentials ─────────────────────────────────────────────
    games["DIFF_EFG_PCT"]     = games["HOME_EFG_PCT"]     - games["AWAY_EFG_PCT"]
    games["DIFF_OPP_EFG_PCT"] = games["HOME_OPP_EFG_PCT"] - games["AWAY_OPP_EFG_PCT"]
    games["DIFF_TOV_PCT"]     = games["HOME_TOV_PCT"]     - games["AWAY_TOV_PCT"]
    games["DIFF_OPP_TOV_PCT"] = games["HOME_OPP_TOV_PCT"] - games["AWAY_OPP_TOV_PCT"]
    games["DIFF_ORB_PCT"]     = games["HOME_ORB_PCT"]     - games["AWAY_ORB_PCT"]
    games["DIFF_DRB_PCT"]     = games["HOME_DRB_PCT"]     - games["AWAY_DRB_PCT"]
    games["DIFF_FTR"]         = games["HOME_FTR"]         - games["AWAY_FTR"]
    games["DIFF_OPP_FTR"]     = games["HOME_OPP_FTR"]     - games["AWAY_OPP_FTR"]
    games["DIFF_WIN_PCT"]     = games["HOME_WIN_PCT"]     - games["AWAY_WIN_PCT"]

    # ── TS% differentials (only if --ts passed) ───────────────────────────────
    if has_ts:
        games["DIFF_TS_PCT"]     = games["HOME_TS_PCT"]     - games["AWAY_TS_PCT"]
        games["DIFF_OPP_TS_PCT"] = games["HOME_OPP_TS_PCT"] - games["AWAY_OPP_TS_PCT"]

    # ── Clutch differentials ──────────────────────────────────────────────────
    if has_clutch:
        games["DIFF_CLUTCH_WIN_PCT"]    = (
            games["HOME_CLUTCH_WIN_PCT"]    - games["AWAY_CLUTCH_WIN_PCT"]
        )
        games["DIFF_CLUTCH_NET_RATING"] = (
            games["HOME_CLUTCH_NET_RATING"] - games["AWAY_CLUTCH_NET_RATING"]
        )

    # ── Star player differentials ─────────────────────────────────────────────
    if has_star:
        games["DIFF_TOP1_PM"]     = games["HOME_TOP1_PM"]     - games["AWAY_TOP1_PM"]
        games["DIFF_TOP2_AVG_PM"] = games["HOME_TOP2_AVG_PM"] - games["AWAY_TOP2_AVG_PM"]
        if "TOP3_AVG_PM" in team_features.columns:
            games["DIFF_TOP3_AVG_PM"] = games["HOME_TOP3_AVG_PM"] - games["AWAY_TOP3_AVG_PM"]

    # ── BPM differentials ─────────────────────────────────────────────────────
    if has_bpm:
        games["DIFF_TOP1_BPM"]     = games["HOME_TOP1_BPM"]     - games["AWAY_TOP1_BPM"]
        games["DIFF_TOP2_AVG_BPM"] = games["HOME_TOP2_AVG_BPM"] - games["AWAY_TOP2_AVG_BPM"]
        games["DIFF_TOP3_AVG_BPM"] = games["HOME_TOP3_AVG_BPM"] - games["AWAY_TOP3_AVG_BPM"]

    # ── Advanced rating differentials ─────────────────────────────────────────
    if has_ratings:
        games["DIFF_OFF_RATING"] = games["HOME_OFF_RATING"] - games["AWAY_OFF_RATING"]
        games["DIFF_DEF_RATING"] = games["HOME_DEF_RATING"] - games["AWAY_DEF_RATING"]
        games["DIFF_NET_RATING"] = games["HOME_NET_RATING"] - games["AWAY_NET_RATING"]

    games.rename(columns={"WIN": "LABEL"}, inplace=True)
    return games


# ── Step 5: Build playoff bracket summary ─────────────────────────────────────

def build_bracket_summary(playoff_logs: pd.DataFrame, season: str) -> pd.DataFrame:
    """
    Collapses individual playoff games into series results.
    Returns one row per series with winner and game count.
    """
    df = playoff_logs.copy()
    df["WIN"]  = (df["WL"] == "W").astype(int)
    df["HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs." in x else 0)

    home = df[df["HOME"] == 1][["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "WIN"]].copy()
    away = df[df["HOME"] == 0][["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION"]].copy()
    away.columns = ["GAME_ID", "OPP_TEAM_ID", "OPP_ABBREVIATION"]

    games = home.merge(away, on="GAME_ID")
    games["SERIES_KEY"] = games.apply(
        lambda r: "_".join(sorted([str(r["TEAM_ID"]), str(r["OPP_TEAM_ID"])])), axis=1
    )

    series = games.groupby("SERIES_KEY").agg(
        HOME_TEAM_ID  = ("TEAM_ID",           "first"),
        AWAY_TEAM_ID  = ("OPP_TEAM_ID",        "first"),
        HOME_ABBREV   = ("TEAM_ABBREVIATION",  "first"),
        AWAY_ABBREV   = ("OPP_ABBREVIATION",   "first"),
        HOME_WINS     = ("WIN",                "sum"),
        TOTAL_GAMES   = ("GAME_ID",            "count"),
    ).reset_index()

    series["AWAY_WINS"]     = series["TOTAL_GAMES"] - series["HOME_WINS"]
    series["SERIES_WINNER"] = series.apply(
        lambda r: r["HOME_ABBREV"] if r["HOME_WINS"] > r["AWAY_WINS"] else r["AWAY_ABBREV"],
        axis=1,
    )
    series["SEASON"] = season
    return series


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NBA playoff data pipeline")
    parser.add_argument(
        "--clutch", action="store_true",
        help="Fetch clutch stats and add DIFF_CLUTCH_* columns"
    )
    parser.add_argument(
        "--star-players", action="store_true",
        help="Fetch player stats and add DIFF_TOP1_PM / DIFF_TOP2_AVG_PM / DIFF_TOP3_AVG_PM columns"
    )
    parser.add_argument(
        "--bpm", action="store_true",
        help="Merge BPM from data/bpm/ and add DIFF_TOP*_BPM columns. "
             "Requires scrape_bpm.py to have been run first."
    )
    parser.add_argument(
        "--ts", action="store_true",
        help="Include True Shooting %% columns: DIFF_TS_PCT, DIFF_OPP_TS_PCT. "
             "TS%% = PTS / (2*(FGA + 0.44*FTA)) — incorporates free throw value "
             "unlike eFG%% which only adjusts for 3-pointers."
    )
    parser.add_argument(
        "--ratings", action="store_true",
        help="Fetch official NBA advanced stats and add DIFF_OFF_RATING, "
             "DIFF_DEF_RATING, DIFF_NET_RATING columns."
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Clutch features     : {'ENABLED' if args.clutch        else 'disabled  (pass --clutch)'}")
    print(f"Star player features: {'ENABLED' if args.star_players  else 'disabled  (pass --star-players)'}")
    print(f"BPM features        : {'ENABLED' if args.bpm           else 'disabled  (pass --bpm, requires scrape_bpm.py)'}")
    print(f"True Shooting %%     : {'ENABLED' if args.ts            else 'disabled  (pass --ts)'}")
    print(f"Advanced ratings    : {'ENABLED' if args.ratings       else 'disabled  (pass --ratings)'}")
    print()

    all_training = []
    all_brackets = []

    for season in SEASONS:
        print(f"\n{'='*50}")
        print(f"Processing season: {season}")
        print(f"{'='*50}")

        try:
            print("  Fetching regular season logs...")
            reg_logs = fetch_game_logs(season, "Regular Season")
            print(f"  → {len(reg_logs)} team-game rows")

            print("  Fetching playoff logs...")
            playoff_logs = fetch_game_logs(season, "Playoffs")
            print(f"  → {len(playoff_logs)} team-game rows")

            if playoff_logs.empty:
                print("  No playoff data — skipping.")
                continue

            print("  Computing team features...")
            team_features = compute_team_features(reg_logs)

            if args.clutch:
                print("  Fetching clutch stats...")
                clutch_df     = fetch_clutch_stats(season)
                team_features = compute_clutch_features(team_features, clutch_df)
                print(f"  → clutch features merged for {len(team_features)} teams")

            if args.star_players:
                print("  Fetching player stats...")
                player_stats  = fetch_player_stats(season)
                team_features = compute_star_player_features(team_features, player_stats)
                print(f"  → star player features merged for {len(team_features)} teams")

            if args.bpm:
                bpm_df = load_bpm_season(season)
                if bpm_df is not None:
                    team_features = compute_star_bpm_features(team_features, bpm_df)
                    print(f"  → BPM features merged for {len(team_features)} teams")
                else:
                    print(f"  [skip] No BPM data for {season} — run scrape_bpm.py first")

            if args.ts:
                print(f"  → TS%% will be included in training examples")

            if args.ratings:
                print("  Fetching official advanced team stats...")
                advanced_df   = fetch_advanced_team_stats(season)
                team_features = compute_advanced_ratings(team_features, advanced_df)
                print(f"  → OFF_RATING / DEF_RATING / NET_RATING merged for {len(team_features)} teams")

            print("  Computing head-to-head records...")
            h2h = compute_h2h(reg_logs)

            print("  Building training examples...")
            training = build_training_examples(
                playoff_logs, team_features, h2h, season,
                include_ts=args.ts,
            )
            all_training.append(training)
            print(f"  → {len(training)} training examples")

            print("  Building bracket summary...")
            bracket = build_bracket_summary(playoff_logs, season)
            all_brackets.append(bracket)
            print(f"  → {len(bracket)} series")

        except Exception as e:
            print(f"  ERROR on {season}: {e}")
            continue

    if all_training:
        training_df   = pd.concat(all_training, ignore_index=True)
        training_path = os.path.join(DATA_DIR, "training_data.csv")
        training_df.to_csv(training_path, index=False)
        n_cols = len(training_df.columns)
        print(f"\n✓ Training data saved: {training_path} ({len(training_df)} rows, {n_cols} cols)")

        extra_cols = [c for c in training_df.columns
                      if any(k in c for k in ("CLUTCH", "TOP", "TS_PCT", "RATING"))]
        if extra_cols:
            print(f"  Extra feature columns: {extra_cols}")

    if all_brackets:
        bracket_df   = pd.concat(all_brackets, ignore_index=True)
        bracket_path = os.path.join(DATA_DIR, "playoff_brackets.csv")
        bracket_df.to_csv(bracket_path, index=False)
        print(f"✓ Bracket data saved: {bracket_path} ({len(bracket_df)} rows)")

    print("\nDone!")


if __name__ == "__main__":
    main()