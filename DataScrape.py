"""
NBA Playoff Outcome Prediction - Data Pipeline
CS830 Final Project - Andrew Elkin

Fetches regular season team stats and playoff game results from nba_api,
constructs a feature matrix suitable for training a game outcome classifier.

Features are based on Dean Oliver's Four Factors:
  Offense: eFG%, TOV%, ORB%, FTR
  Defense: opponent eFG%, opponent TOV%, DRB%, opponent FTR

Output: data/training_data.csv, data/playoff_brackets.csv
"""

import time
import os
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamefinder

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


# ── Step 2: Compute season-level team features ────────────────────────────────

def compute_team_features(reg_season_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates per-team regular season stats into Dean Oliver's Four Factors.

    Four Factors — Offense:
      EFG_PCT     = (FGM + 0.5*FG3M) / FGA          effective field goal %
      TOV_PCT     = TOV / (FGA + 0.44*FTA + TOV)    turnover rate per possession
      ORB_PCT     = OREB / (OREB + OPP_DREB)         offensive rebound rate
      FTR         = FTA / FGA                         free throw attempt rate

    Four Factors — Defense (opponent's offensive stats against this team):
      OPP_EFG_PCT = (OPP_FGM + 0.5*OPP_FG3M) / OPP_FGA
      OPP_TOV_PCT = OPP_TOV / (OPP_FGA + 0.44*OPP_FTA + OPP_TOV)
      DRB_PCT     = DREB / (DREB + OPP_OREB)
      OPP_FTR     = OPP_FTA / OPP_FGA

    WIN_PCT is also kept for the Finals home-court tiebreaker in bracket_simulator.
    """
    df = reg_season_logs.copy()
    df["HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs." in x else 0)
    df["WIN"]  = (df["WL"] == "W").astype(int)

    # Build opponent lookup — all stats needed for Four Factors
    opp_cols = ["GAME_ID", "TEAM_ID",
                "PTS", "FGM", "FGA", "FG3M", "FG3A",
                "FTA", "OREB", "DREB", "TOV"]
    opp = df[opp_cols].copy()
    opp.columns = ["GAME_ID", "OPP_TEAM_ID",
                   "OPP_PTS", "OPP_FGM", "OPP_FGA", "OPP_FG3M", "OPP_FG3A",
                   "OPP_FTA", "OPP_OREB", "OPP_DREB", "OPP_TOV"]

    df = df.merge(opp, on="GAME_ID", how="left")
    df = df[df["TEAM_ID"] != df["OPP_TEAM_ID"]]

    # Aggregate to season-level per-game averages
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
        # Defensive raw stats (opponent's offensive stats against this team)
        OPP_FGM_PG        = ("OPP_FGM",           "mean"),
        OPP_FGA_PG        = ("OPP_FGA",           "mean"),
        OPP_FG3M_PG       = ("OPP_FG3M",          "mean"),
        OPP_FTA_PG        = ("OPP_FTA",           "mean"),
        OPP_OREB_PG       = ("OPP_OREB",          "mean"),
        OPP_DREB_PG       = ("OPP_DREB",          "mean"),
        OPP_TOV_PG        = ("OPP_TOV",           "mean"),
    ).reset_index()

    agg["WIN_PCT"] = agg["WINS"] / agg["GAMES"]

    # ── Four Factors — Offense ────────────────────────────────────────────────
    # eFG%: weights 3-pointers at 1.5× since they're worth 50% more than a 2.
    agg["EFG_PCT"] = (agg["FGM_PG"] + 0.5 * agg["FG3M_PG"]) / agg["FGA_PG"]

    # TOV%: turnovers per possession. 0.44 is Oliver's coefficient for the
    # fraction of FTA trips that actually end a possession.
    agg["TOV_PCT"] = agg["TOV_PG"] / (agg["FGA_PG"] + 0.44 * agg["FTA_PG"] + agg["TOV_PG"])

    # ORB%: fraction of available offensive boards the team captures.
    agg["ORB_PCT"] = agg["OREB_PG"] / (agg["OREB_PG"] + agg["OPP_DREB_PG"])

    # FTR: how aggressively the team gets to the line.
    agg["FTR"] = agg["FTA_PG"] / agg["FGA_PG"]

    # ── Four Factors — Defense ────────────────────────────────────────────────
    # Opponent's eFG% allowed — lower is better defense.
    agg["OPP_EFG_PCT"] = (agg["OPP_FGM_PG"] + 0.5 * agg["OPP_FG3M_PG"]) / agg["OPP_FGA_PG"]

    # Turnovers forced per opponent possession — higher is better.
    agg["OPP_TOV_PCT"] = agg["OPP_TOV_PG"] / (
        agg["OPP_FGA_PG"] + 0.44 * agg["OPP_FTA_PG"] + agg["OPP_TOV_PG"]
    )

    # DRB%: fraction of available defensive boards the team secures.
    agg["DRB_PCT"] = agg["DREB_PG"] / (agg["DREB_PG"] + agg["OPP_OREB_PG"])

    # Opponent FTR — how often the team fouls. Lower is better defensively.
    agg["OPP_FTR"] = agg["OPP_FTA_PG"] / agg["OPP_FGA_PG"]

    agg["TEAM_ID"] = agg["TEAM_ID"].astype(int)
    return agg


# ── Step 3: Compute head-to-head regular season win rate ──────────────────────

def compute_h2h(reg_season_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Returns [TEAM_ID, OPP_TEAM_ID, H2H_WIN_PCT] — each team's win rate
    against each opponent in the regular season.
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


# ── Step 4: Build training examples from playoff games ────────────────────────

def build_training_examples(
    playoff_logs:   pd.DataFrame,
    team_features:  pd.DataFrame,
    h2h:            pd.DataFrame,
    season:         str,
) -> pd.DataFrame:
    """
    For each playoff game, creates one training row anchored on the home team.
    Features are Four Factor differentials (home - away).
    Label = 1 if home team wins.
    """
    df = playoff_logs.copy()
    df["WIN"]  = (df["WL"] == "W").astype(int)
    df["HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs." in x else 0)

    home = df[df["HOME"] == 1][["GAME_ID", "TEAM_ID", "WIN"]].copy()
    away = df[df["HOME"] == 0][["GAME_ID", "TEAM_ID"]].copy()
    away.columns = ["GAME_ID", "OPP_TEAM_ID"]

    games = home.merge(away, on="GAME_ID", how="inner")
    games["SEASON"] = season

    # Normalize types before merging
    games["TEAM_ID"]     = games["TEAM_ID"].astype(int)
    games["OPP_TEAM_ID"] = games["OPP_TEAM_ID"].astype(int)
    team_features        = team_features.copy()
    team_features["TEAM_ID"] = team_features["TEAM_ID"].astype(int)
    h2h = h2h.copy()
    h2h["TEAM_ID"]     = h2h["TEAM_ID"].astype(int)
    h2h["OPP_TEAM_ID"] = h2h["OPP_TEAM_ID"].astype(int)

    # Columns to pull from team_features for each side
    feat_cols = ["TEAM_ID",
                 "EFG_PCT", "OPP_EFG_PCT",
                 "TOV_PCT", "OPP_TOV_PCT",
                 "ORB_PCT", "DRB_PCT",
                 "FTR",     "OPP_FTR",
                 "WIN_PCT"]   # kept for Finals HCA tiebreaker, not a model feature

    home_feats = team_features[feat_cols].add_prefix("HOME_").rename(
        columns={"HOME_TEAM_ID": "TEAM_ID"}
    )
    away_feats = team_features[feat_cols].add_prefix("AWAY_").rename(
        columns={"AWAY_TEAM_ID": "OPP_TEAM_ID"}
    )

    games = games.merge(home_feats, on="TEAM_ID",     how="left")
    games = games.merge(away_feats, on="OPP_TEAM_ID", how="left")

    # Head-to-head win pct (home team's record vs away team in regular season)
    games = games.merge(h2h, on=["TEAM_ID", "OPP_TEAM_ID"], how="left")
    games["H2H_WIN_PCT"] = games["H2H_WIN_PCT"].fillna(0.5)

    # ── Four Factor differentials (home − away) ───────────────────────────────
    games["DIFF_EFG_PCT"]     = games["HOME_EFG_PCT"]     - games["AWAY_EFG_PCT"]
    games["DIFF_OPP_EFG_PCT"] = games["HOME_OPP_EFG_PCT"] - games["AWAY_OPP_EFG_PCT"]
    games["DIFF_TOV_PCT"]     = games["HOME_TOV_PCT"]     - games["AWAY_TOV_PCT"]
    games["DIFF_OPP_TOV_PCT"] = games["HOME_OPP_TOV_PCT"] - games["AWAY_OPP_TOV_PCT"]
    games["DIFF_ORB_PCT"]     = games["HOME_ORB_PCT"]     - games["AWAY_ORB_PCT"]
    games["DIFF_DRB_PCT"]     = games["HOME_DRB_PCT"]     - games["AWAY_DRB_PCT"]
    games["DIFF_FTR"]         = games["HOME_FTR"]         - games["AWAY_FTR"]
    games["DIFF_OPP_FTR"]     = games["HOME_OPP_FTR"]     - games["AWAY_OPP_FTR"]
    # Also keep WIN_PCT diff for the baseline comparison in logistic_regression.py
    games["DIFF_WIN_PCT"]     = games["HOME_WIN_PCT"]     - games["AWAY_WIN_PCT"]

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
    os.makedirs(DATA_DIR, exist_ok=True)

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

            print("  Computing head-to-head records...")
            h2h = compute_h2h(reg_logs)

            print("  Building training examples...")
            training = build_training_examples(playoff_logs, team_features, h2h, season)
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
        print(f"\n✓ Training data saved: {training_path} ({len(training_df)} rows)")

    if all_brackets:
        bracket_df   = pd.concat(all_brackets, ignore_index=True)
        bracket_path = os.path.join(DATA_DIR, "playoff_brackets.csv")
        bracket_df.to_csv(bracket_path, index=False)
        print(f"✓ Bracket data saved: {bracket_path} ({len(bracket_df)} rows)")

    print("\nDone!")


if __name__ == "__main__":
    main()