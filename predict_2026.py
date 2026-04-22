"""
NBA Playoff Outcome Prediction - 2025-26 Season Prediction
CS830 Final Project - Andrew Elkin

Predicts the 2025-26 NBA playoff bracket using:
  - A model trained on all prior seasons (2005-06 through 2024-25)
  - 2025-26 regular season stats fetched live from nba_api

Because the 2025-26 playoffs are still in progress, this season is NOT
added to the training data (there are no complete playoff results to use
as labels). Instead, we use the trained model purely for inference.

NOTE: Run this AFTER completing the Four Factor feature updates to
data_pipeline.py. This script uses the same updated compute_team_features()
and compute_h2h() functions from that file.
"""

import os
import numpy as np
import pandas as pd

from DataScrape import fetch_game_logs, compute_team_features, compute_h2h
from logistic_regression import LogisticRegression, FEATURE_COLS
from bracket_simulator   import simulate_bracket, print_results

# ── 2025-26 playoff bracket ───────────────────────────────────────────────────
#
# Seedings confirmed from the 2026 NBA Playoffs bracket (play-in complete).
# Team IDs are nba_api integer identifiers.
#
# WEST                                    EAST
# 1. OKC Thunder       1610612760        1. Detroit Pistons    1610612765
# 2. San Antonio Spurs 1610612759        2. Boston Celtics     1610612738
# 3. Denver Nuggets    1610612743        3. New York Knicks    1610612752
# 4. LA Lakers         1610612747        4. Cleveland Cavaliers 1610612739
# 5. Houston Rockets   1610612745        5. Toronto Raptors    1610612761
# 6. Minnesota T-Wolves 1610612750       6. Atlanta Hawks      1610612737
# 7. Portland T-Blazers 1610612757       7. Philadelphia 76ers 1610612755
# 8. Phoenix Suns      1610612756        8. Orlando Magic      1610612753
#
# First-round matchups (already underway as of April 22, 2026):
#   West: OKC vs PHX (OKC leads 1-0), LAL vs HOU (LAL leads 2-0),
#         DEN vs MIN (tied 1-1),       SA  vs POR (tied 1-1)
#   East: DET vs ORL (ORL leads 1-0), BOS vs PHI (BOS leads 1-0),
#         NYK vs TOR,                  CLE vs ATL (CLE leads 2-0)

WEST_SEEDS = {
    1: 1610612760,   # OKC Thunder
    2: 1610612759,   # San Antonio Spurs
    3: 1610612743,   # Denver Nuggets
    4: 1610612747,   # LA Lakers
    5: 1610612745,   # Houston Rockets
    6: 1610612750,   # Minnesota Timberwolves
    7: 1610612757,   # Portland Trail Blazers
    8: 1610612756,   # Phoenix Suns
}

EAST_SEEDS = {
    1: 1610612765,   # Detroit Pistons
    2: 1610612738,   # Boston Celtics
    3: 1610612752,   # New York Knicks
    4: 1610612739,   # Cleveland Cavaliers
    5: 1610612761,   # Toronto Raptors
    6: 1610612737,   # Atlanta Hawks
    7: 1610612755,   # Philadelphia 76ers
    8: 1610612753,   # Orlando Magic
}

NAME_LOOKUP = {
    1610612760: "OKC Thunder",
    1610612759: "SA Spurs",
    1610612743: "DEN Nuggets",
    1610612747: "LAL Lakers",
    1610612745: "HOU Rockets",
    1610612750: "MIN Timberwolves",
    1610612757: "POR Trail Blazers",
    1610612756: "PHX Suns",
    1610612765: "DET Pistons",
    1610612738: "BOS Celtics",
    1610612752: "NYK Knicks",
    1610612739: "CLE Cavaliers",
    1610612761: "TOR Raptors",
    1610612737: "ATL Hawks",
    1610612755: "PHI 76ers",
    1610612753: "ORL Magic",
}

CURRENT_SEASON = "2025-26"
DATA_PATH      = "data/training_data.csv"


# ── Step 1: Train model on all prior seasons ──────────────────────────────────

def train_model():
    print("Loading training data from all prior seasons...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} games across {df['SEASON'].nunique()} seasons.\n")

    X = df[FEATURE_COLS].values
    y = df["LABEL"].values

    print("Training logistic regression model...")
    model = LogisticRegression(learning_rate=0.1, epochs=1000, lambda_=0.01)
    model.fit(X, y)
    print("  Done.\n")
    return model


# ── Step 2: Fetch 2025-26 regular season stats ────────────────────────────────

def load_current_season_data():
    """
    Fetch the completed 2025-26 regular season game logs from nba_api,
    compute Four Factor team features and H2H records.

    The regular season ended April 12, 2026 so all data is available.
    Results are cached to data/cache_2025-26_Regular_Season.csv after
    the first fetch, so re-runs are instant.
    """
    print(f"Fetching {CURRENT_SEASON} regular season data...")
    reg_logs = fetch_game_logs(CURRENT_SEASON, "Regular Season")
    print(f"  {len(reg_logs)} team-game rows loaded.\n")

    print("Computing team features...")
    team_features = compute_team_features(reg_logs)
    print(f"  {len(team_features)} teams.\n")

    print("Computing head-to-head records...")
    h2h_df = compute_h2h(reg_logs)

    # Convert H2H DataFrame to lookup dict for the simulator
    h2h_lookup = {
        (int(row["TEAM_ID"]), int(row["OPP_TEAM_ID"])): float(row["H2H_WIN_PCT"])
        for _, row in h2h_df.iterrows()
    }
    print(f"  {len(h2h_lookup)} head-to-head records.\n")

    return team_features, h2h_lookup


# ── Step 3: Build bracket dicts ───────────────────────────────────────────────

def build_bracket(seed_map, team_features_df):
    """
    Convert {seed: team_id} mapping into the {seed: team_dict} format
    expected by simulate_bracket().

    Pulls each team's Four Factor stats from the computed features DataFrame.
    """
    # Index by TEAM_ID for fast lookup
    features_by_id = team_features_df.set_index("TEAM_ID")

    bracket = {}
    for seed, team_id in seed_map.items():
        if team_id not in features_by_id.index:
            raise ValueError(
                f"Team ID {team_id} ({NAME_LOOKUP.get(team_id, '?')}) "
                f"not found in {CURRENT_SEASON} regular season data."
            )
        row = features_by_id.loc[team_id]
        bracket[seed] = {
            "TEAM_ID":     int(team_id),
            # Four Factor offensive
            "EFG_PCT":     float(row["EFG_PCT"]),
            "TOV_PCT":     float(row["TOV_PCT"]),
            "ORB_PCT":     float(row["ORB_PCT"]),
            "FTR":         float(row["FTR"]),
            # Four Factor defensive
            "OPP_EFG_PCT": float(row["OPP_EFG_PCT"]),
            "OPP_TOV_PCT": float(row["OPP_TOV_PCT"]),
            "DRB_PCT":     float(row["DRB_PCT"]),
            "OPP_FTR":     float(row["OPP_FTR"]),
            # Kept for Finals home-court tiebreaker (not used as a model feature)
            "WIN_PCT":     float(row["WIN_PCT"]),
        }
    return bracket


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  2025-26 NBA Playoff Prediction")
    print("=" * 60 + "\n")

    # Train on all historical seasons
    model = train_model()

    # Fetch current season data
    team_features, h2h_lookup = load_current_season_data()

    # Verify all 16 playoff teams appear in the data
    all_playoff_ids = list(WEST_SEEDS.values()) + list(EAST_SEEDS.values())
    found_ids = set(team_features["TEAM_ID"].astype(int))
    missing = [tid for tid in all_playoff_ids if tid not in found_ids]
    if missing:
        print("WARNING: Missing team IDs in fetched data:")
        for tid in missing:
            print(f"  {tid} → {NAME_LOOKUP.get(tid, '?')}")
        print("Check that nba_api has complete 2025-26 data and retry.\n")
        return

    # Build bracket structures
    west_bracket = build_bracket(WEST_SEEDS, team_features)
    east_bracket = build_bracket(EAST_SEEDS, team_features)

    # Run the simulation
    print("Simulating 2025-26 playoff bracket...\n")
    champ_probs, conf_probs = simulate_bracket(
        west_bracket, east_bracket, model, h2h_lookup
    )

    # Print full results
    print_results(champ_probs, conf_probs, west_bracket, east_bracket, NAME_LOOKUP)

    # Print first-round series probabilities for context
    print("\nFirst-round series win probabilities (higher seed):")
    print("-" * 50)
    first_round = [
        (1, 8, "West"), (4, 5, "West"), (3, 6, "West"), (2, 7, "West"),
        (1, 8, "East"), (4, 5, "East"), (3, 6, "East"), (2, 7, "East"),
    ]
    from bracket_simulator import series_win_prob
    for h_seed, l_seed, conf in first_round:
        bracket = west_bracket if conf == "West" else east_bracket
        h_team  = bracket[h_seed]
        l_team  = bracket[l_seed]
        p = series_win_prob(model, h_team, l_team, h2h_lookup)
        h_name = NAME_LOOKUP.get(h_team["TEAM_ID"], str(h_team["TEAM_ID"]))
        l_name = NAME_LOOKUP.get(l_team["TEAM_ID"], str(l_team["TEAM_ID"]))
        bar = "█" * int(p * 20)
        print(f"  [{conf}] ({h_seed}) {h_name:<20} vs ({l_seed}) {l_name:<20}  {p:.1%}  {bar}")

    print(f"\nSanity check — probs sum to: {sum(champ_probs.values()):.6f}")


if __name__ == "__main__":
    main()