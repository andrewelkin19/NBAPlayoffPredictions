"""
NBA Playoff Outcome Prediction - Bracket Simulator
CS830 Final Project - Andrew Elkin

Given a trained logistic regression model and team stats for a given season,
computes win probabilities for every possible playoff series and the
championship probability for each of the 16 playoff teams.

Algorithm overview:
  1. Per-game probability: the model outputs P(home team wins one game)
     given the Four Factor differential feature vector for that matchup.

  2. Per-series probability: computed via recursive DP over all possible
     game sequences in a best-of-7, accounting for the NBA's alternating
     home court schedule (higher seed hosts games 1, 2, 5, 7).

  3. Bracket propagation: we maintain P(each team reaches each round)
     and marginalize over all possible opponents at each stage. Teams in
     the same bracket half are mutually exclusive; teams from different
     halves are independent, so P(matchup A vs B) = P(A reached) x P(B reached).

Implemented from scratch: steps 1-3 above (NumPy + pure Python).
Library usage: pandas for data loading, functools.lru_cache for DP memoization.
"""

import os
import numpy as np
import pandas as pd
import time
from functools import lru_cache

from logistic_regression import LogisticRegression, FEATURE_COLS
from nba_api.stats.endpoints import leaguestandingsv3
from nba_api.stats.library.parameters import Season

headers = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Referer': 'https://www.nba.com/',
    'Connection': 'keep-alive',
}

# ── NBA home court schedule ───────────────────────────────────────────────────
# Higher seed hosts games 1, 2, 5, 7. Lower seed hosts games 3, 4, 6.
HOME_GAMES = {1, 2, 5, 7}

# ── Per-game win probability ──────────────────────────────────────────────────

def game_win_prob(model, home_team, away_team, h2h_lookup):
    """
    P(home_team wins a single game) using the trained logistic regression.

    Constructs the same Four Factor differential feature vector that
    build_training_examples() produces in data_pipeline.py.

    Args:
        model:      Trained LogisticRegression instance.
        home_team:  dict with keys EFG_PCT, OPP_EFG_PCT, TOV_PCT, OPP_TOV_PCT,
                    ORB_PCT, DRB_PCT, FTR, OPP_FTR, WIN_PCT, TEAM_ID.
        away_team:  Same structure as home_team.
        h2h_lookup: dict {(team_id_a, team_id_b): h2h_win_pct}.

    Returns:
        float in (0, 1).
    """
    h2h = h2h_lookup.get((home_team["TEAM_ID"], away_team["TEAM_ID"]), 0.5)

    features = np.array([[
        home_team["EFG_PCT"]     - away_team["EFG_PCT"],      # DIFF_EFG_PCT
        home_team["OPP_EFG_PCT"] - away_team["OPP_EFG_PCT"],  # DIFF_OPP_EFG_PCT
        home_team["TOV_PCT"]     - away_team["TOV_PCT"],      # DIFF_TOV_PCT
        home_team["OPP_TOV_PCT"] - away_team["OPP_TOV_PCT"],  # DIFF_OPP_TOV_PCT
        home_team["ORB_PCT"]     - away_team["ORB_PCT"],      # DIFF_ORB_PCT
        home_team["DRB_PCT"]     - away_team["DRB_PCT"],      # DIFF_DRB_PCT
        home_team["FTR"]         - away_team["FTR"],          # DIFF_FTR
        home_team["OPP_FTR"]     - away_team["OPP_FTR"],      # DIFF_OPP_FTR
        h2h,                                                   # H2H_WIN_PCT
    ]])

    return float(model.predict_proba(features)[0])


# ── Per-series win probability ────────────────────────────────────────────────

def series_win_prob(model, higher_seed, lower_seed, h2h_lookup):
    """
    P(higher_seed wins a best-of-7 series) against lower_seed.

    Because home court alternates game-by-game, we compute two probabilities:
      p_home = P(higher seed wins a game at their own court)
      p_away = P(higher seed wins a game at the opponent's court)
             = 1 - P(lower seed wins at their own court)

    Then use recursive DP:
      dp(wins_high, wins_low) = P(higher seed wins the series from this state)

    Game number = wins_high + wins_low + 1, which determines home court.
    Series ends when either team reaches 4 wins.
    """
    p_home = game_win_prob(model, higher_seed, lower_seed, h2h_lookup)
    p_away = 1.0 - game_win_prob(model, lower_seed, higher_seed, h2h_lookup)

    @lru_cache(maxsize=None)
    def dp(wins_high, wins_low):
        if wins_high == 4:
            return 1.0
        if wins_low == 4:
            return 0.0
        game_num = wins_high + wins_low + 1
        p = p_home if game_num in HOME_GAMES else p_away
        return (p       * dp(wins_high + 1, wins_low)
              + (1 - p) * dp(wins_high,     wins_low + 1))

    result = dp(0, 0)
    dp.cache_clear()
    return result


# ── Conference bracket simulation ─────────────────────────────────────────────

def simulate_conference(teams_by_seed, model, h2h_lookup):
    """
    Compute P(each team wins the conference) for an 8-team bracket.

    NBA bracket structure (higher seed = home court):
      Round 1:     (1v8), (2v7), (3v6), (4v5)
      Semifinals:  winner(1/8) vs winner(4/5),  winner(2/7) vs winner(3/6)
      Conf Finals: winner of top half vs winner of bottom half

    At each round we maintain prob_reach[seed] = P(this team is still alive).
    For any two teams from DIFFERENT bracket halves, their reach probabilities
    are independent, so P(matchup a vs b) = prob_reach[a] * prob_reach[b].

    Args:
        teams_by_seed: dict {seed (1-8): team_dict}
        model:         Trained LogisticRegression instance.
        h2h_lookup:    dict {(team_id_a, team_id_b): h2h_win_pct}.

    Returns:
        dict {seed: P(wins conference)}, values sum to ~1.0.
    """
    prob_reach = {seed: 1.0 for seed in range(1, 9)}

    # ── Round 1 ───────────────────────────────────────────────────────────────
    for h_seed, l_seed in [(1, 8), (2, 7), (3, 6), (4, 5)]:
        p = series_win_prob(
            model, teams_by_seed[h_seed], teams_by_seed[l_seed], h2h_lookup
        )
        prob_reach[h_seed] = p
        prob_reach[l_seed] = 1.0 - p

    # ── Semifinals ────────────────────────────────────────────────────────────
    semifinal_pairs = [
        ([1, 8], [4, 5]),
        ([2, 7], [3, 6]),
    ]

    prob_reach_semi = {seed: 0.0 for seed in range(1, 9)}

    for group_a, group_b in semifinal_pairs:
        for a_seed in group_a:
            for b_seed in group_b:
                p_matchup = prob_reach[a_seed] * prob_reach[b_seed]
                if p_matchup < 1e-12:
                    continue

                h_seed = min(a_seed, b_seed)
                l_seed = max(a_seed, b_seed)

                p_series = series_win_prob(
                    model, teams_by_seed[h_seed], teams_by_seed[l_seed], h2h_lookup
                )
                prob_reach_semi[h_seed] += p_matchup * p_series
                prob_reach_semi[l_seed] += p_matchup * (1.0 - p_series)

    prob_reach = prob_reach_semi

    # ── Conference Finals ─────────────────────────────────────────────────────
    top_half = [1, 8, 4, 5]
    bot_half = [2, 7, 3, 6]

    prob_conf_champ = {seed: 0.0 for seed in range(1, 9)}

    for a_seed in top_half:
        for b_seed in bot_half:
            p_matchup = prob_reach[a_seed] * prob_reach[b_seed]
            if p_matchup < 1e-12:
                continue

            h_seed = min(a_seed, b_seed)
            l_seed = max(a_seed, b_seed)

            p_series = series_win_prob(
                model, teams_by_seed[h_seed], teams_by_seed[l_seed], h2h_lookup
            )
            prob_conf_champ[h_seed] += p_matchup * p_series
            prob_conf_champ[l_seed] += p_matchup * (1.0 - p_series)

    return prob_conf_champ


# ── Full playoff bracket simulation ──────────────────────────────────────────

def simulate_bracket(west_by_seed, east_by_seed, model, h2h_lookup):
    """
    Compute P(each team wins the NBA championship) for the full 16-team field.

    Simulates each conference independently, then runs the NBA Finals as a
    weighted sum over all possible conference champion matchups.

    Home court in the Finals goes to whichever conference champion had the
    higher regular-season WIN_PCT.

    Returns:
        champ_probs: dict {team_id: P(wins championship)}, sums to ~1.0.
        conf_probs:  dict {team_id: P(wins conference)}.
    """
    print("  Simulating Western Conference bracket...")
    west_conf = simulate_conference(west_by_seed, model, h2h_lookup)

    print("  Simulating Eastern Conference bracket...")
    east_conf = simulate_conference(east_by_seed, model, h2h_lookup)

    west_by_id = {west_by_seed[s]["TEAM_ID"]: p for s, p in west_conf.items()}
    east_by_id = {east_by_seed[s]["TEAM_ID"]: p for s, p in east_conf.items()}
    conf_probs = {**west_by_id, **east_by_id}

    all_team_ids = list(conf_probs.keys())
    champ_probs  = {tid: 0.0 for tid in all_team_ids}

    print("  Simulating NBA Finals...")

    for w_seed, w_p_conf in west_conf.items():
        for e_seed, e_p_conf in east_conf.items():
            p_finals = w_p_conf * e_p_conf
            if p_finals < 1e-12:
                continue

            west_team = west_by_seed[w_seed]
            east_team = east_by_seed[e_seed]

            if west_team["WIN_PCT"] >= east_team["WIN_PCT"]:
                higher, lower = west_team, east_team
            else:
                higher, lower = east_team, west_team

            p_series = series_win_prob(model, higher, lower, h2h_lookup)
            champ_probs[higher["TEAM_ID"]] += p_finals * p_series
            champ_probs[lower["TEAM_ID"]]  += p_finals * (1.0 - p_series)

    return champ_probs, conf_probs


# ── Data loading ──────────────────────────────────────────────────────────────

def load_season_data(season, path="data/training_data.csv"):
    """
    Extract per-team Four Factor stats and H2H records for a given season
    from training_data.csv.

    training_data.csv stores features as HOME_* and AWAY_* columns per game.
    We recover per-team stats by taking the first occurrence of each team.

    Returns:
        team_stats:  dict {team_id: team_dict}.
        h2h_lookup:  dict {(team_id_a, team_id_b): h2h_win_pct}.
    """
    df = pd.read_csv(path)
    season_df = df[df["SEASON"] == season].copy()

    if season_df.empty:
        raise ValueError(f"No data found for season {season!r} in {path}")

    team_stats = {}

    for _, row in season_df.iterrows():
        # Home team
        tid = int(row["TEAM_ID"])
        if tid not in team_stats:
            team_stats[tid] = {
                "TEAM_ID":     tid,
                "EFG_PCT":     float(row["HOME_EFG_PCT"]),
                "OPP_EFG_PCT": float(row["HOME_OPP_EFG_PCT"]),
                "TOV_PCT":     float(row["HOME_TOV_PCT"]),
                "OPP_TOV_PCT": float(row["HOME_OPP_TOV_PCT"]),
                "ORB_PCT":     float(row["HOME_ORB_PCT"]),
                "DRB_PCT":     float(row["HOME_DRB_PCT"]),
                "FTR":         float(row["HOME_FTR"]),
                "OPP_FTR":     float(row["HOME_OPP_FTR"]),
                "WIN_PCT":     float(row["HOME_WIN_PCT"]),
            }

        # Away team
        opp_tid = int(row["OPP_TEAM_ID"])
        if opp_tid not in team_stats:
            team_stats[opp_tid] = {
                "TEAM_ID":     opp_tid,
                "EFG_PCT":     float(row["AWAY_EFG_PCT"]),
                "OPP_EFG_PCT": float(row["AWAY_OPP_EFG_PCT"]),
                "TOV_PCT":     float(row["AWAY_TOV_PCT"]),
                "OPP_TOV_PCT": float(row["AWAY_OPP_TOV_PCT"]),
                "ORB_PCT":     float(row["AWAY_ORB_PCT"]),
                "DRB_PCT":     float(row["AWAY_DRB_PCT"]),
                "FTR":         float(row["AWAY_FTR"]),
                "OPP_FTR":     float(row["AWAY_OPP_FTR"]),
                "WIN_PCT":     float(row["AWAY_WIN_PCT"]),
            }

    # H2H lookup
    h2h_lookup = {}
    for _, row in season_df.iterrows():
        tid     = int(row["TEAM_ID"])
        opp_tid = int(row["OPP_TEAM_ID"])
        h2h_lookup[(tid, opp_tid)] = float(row["H2H_WIN_PCT"])

    return team_stats, h2h_lookup

def load_playoff_bracket(season: str, team_stats: dict, data_dir="data") -> tuple[dict, dict, dict]:
    cache_path = os.path.join(data_dir, f"cache_{season}_standings.csv")

    if os.path.exists(cache_path):
        print(f"  Loading standings from cache: {cache_path}")
        df = pd.read_csv(cache_path)
    else:
        MAX_RETRIES = 3
        SLEEP_SEC = 2.5
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                standings = leaguestandingsv3.LeagueStandingsV3(
                    season=season,
                    season_type="Regular Season",
                    league_id="00",
                    headers=headers,
                    timeout=120,   # bumped from 60
                )
                time.sleep(SLEEP_SEC)
                df = standings.get_data_frames()[0]
                df.to_csv(cache_path, index=False)
                print(f"  Standings cached: {cache_path}")
                break
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise
                wait = SLEEP_SEC * (2 ** attempt)
                print(f"  Attempt {attempt} failed ({e}). Retrying in {wait:.1f}s...")
                time.sleep(wait)

    df["TeamID"] = df["TeamID"].astype(int)
    
    east = df[df["Conference"] == "East"].nsmallest(8, "PlayoffRank")
    west = df[df["Conference"] == "West"].nsmallest(8, "PlayoffRank")

    east_bracket, west_bracket, name_lookup = {}, {}, {}

    for _, row in east.iterrows():
        tid, seed = int(row["TeamID"]), int(row["PlayoffRank"])
        if tid not in team_stats:
            raise KeyError(f"No stats for East seed {seed} ({row['TeamAbbreviation']}, ID {tid}). "
                           f"Check that {season} training data exists.")
        east_bracket[seed] = team_stats[tid]
        name_lookup[tid] = f"{row['TeamCity']} {row['TeamName']}"

    for _, row in west.iterrows():
        tid, seed = int(row["TeamID"]), int(row["PlayoffRank"])
        if tid not in team_stats:
            raise KeyError(f"No stats for West seed {seed} ({row['TeamAbbreviation']}, ID {tid}). "
                           f"Check that {season} training data exists.")
        west_bracket[seed] = team_stats[tid]
        name_lookup[tid] = f"{row['TeamCity']} {row['TeamName']}"

    return east_bracket, west_bracket, name_lookup

# ── Results display ───────────────────────────────────────────────────────────

def print_results(champ_probs, conf_probs, west_by_seed, east_by_seed,
                  name_lookup=None):
    def team_label(team_id):
        if name_lookup and team_id in name_lookup:
            return name_lookup[team_id]
        return str(team_id)

    print("\n" + "=" * 60)
    print(f"{'TEAM':<25} {'CONF WIN%':>10} {'CHAMP%':>10}")
    print("=" * 60)

    for team_id, champ_p in sorted(champ_probs.items(), key=lambda x: -x[1]):
        conf_p  = conf_probs.get(team_id, 0.0)
        bar_len = int(champ_p * 30)
        bar     = "█" * bar_len
        print(f"  {team_label(team_id):<23} {conf_p:>9.1%} {champ_p:>9.1%}  {bar}")

    print("=" * 60)
    print(f"  {'Total':23}  {sum(conf_probs.values()):>9.3f} {sum(champ_probs.values()):>9.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    DATA_PATH   = "data/training_data.csv"
    DEMO_SEASON = "2023-24"

    df       = pd.read_csv(DATA_PATH)
    train_df = df[df["SEASON"] != DEMO_SEASON]

    print(f"Training on {len(train_df)} games, holding out {DEMO_SEASON}...")
    model = LogisticRegression(learning_rate=0.1, epochs=1000, lambda_=0.01)
    model.fit(train_df[FEATURE_COLS].values, train_df["LABEL"].values)
    print("Done.\n")

    team_stats, h2h_lookup = load_season_data(DEMO_SEASON, path=DATA_PATH)
    print(f"Loaded stats for {len(team_stats)} teams in {DEMO_SEASON}.\n")
    
    HOLDOUT_SEASON = "2023-24"   # season you're simulating
    
    east_bracket, west_bracket, name_lookup = load_playoff_bracket(HOLDOUT_SEASON, team_stats)

    print(f"Simulating {DEMO_SEASON} playoff bracket...\n")
    champ_probs, conf_probs = simulate_bracket(
        west_bracket, east_bracket, model, h2h_lookup
    )

    print_results(champ_probs, conf_probs, west_bracket, east_bracket, name_lookup)

    total = sum(champ_probs.values())
    print(f"\nSanity check — champ probs sum to: {total:.6f}")


if __name__ == "__main__":
    main()