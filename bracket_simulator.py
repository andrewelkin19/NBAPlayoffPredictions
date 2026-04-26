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
from feature_sets import DEFAULT_PRESET, get_preset
from nba_api.stats.endpoints import leaguestandingsv3

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


# ── Feature vector construction ───────────────────────────────────────────────

def build_feature_vector(home_team: dict, away_team: dict,
                         h2h_lookup: dict, feature_cols: list[str]) -> np.ndarray:
    """
    Build the feature vector for a single game from team dicts.

    Each column in feature_cols maps to one of two forms:
      - "DIFF_<KEY>" → home_team[KEY] - away_team[KEY]
      - "H2H_WIN_PCT" → looked up directly from h2h_lookup

    This is the inverse of the build_training_examples() logic in DataScrape.py,
    so any feature added there will automatically work here as long as the
    corresponding key exists in the team dict.

    Args:
        home_team:    dict with per-team stat keys (EFG_PCT, OPP_EFG_PCT, etc.)
        away_team:    same structure as home_team
        h2h_lookup:   dict {(team_id_a, team_id_b): h2h_win_pct}
        feature_cols: ordered list of column names (e.g. FEATURE_COLS from feature_sets)

    Returns:
        np.ndarray of shape (1, len(feature_cols))
    """
    values = []
    for col in feature_cols:
        if col == "H2H_WIN_PCT":
            val = h2h_lookup.get(
                (home_team["TEAM_ID"], away_team["TEAM_ID"]), 0.5
            )
        elif col.startswith("DIFF_"):
            key = col[5:]  # strip "DIFF_" prefix
            val = home_team[key] - away_team[key]
        else:
            raise ValueError(
                f"Don't know how to build feature '{col}' from team dicts. "
                f"Expected 'DIFF_<key>' or 'H2H_WIN_PCT'."
            )
        values.append(val)
    return np.array([values])


# ── Per-game win probability ──────────────────────────────────────────────────

def game_win_prob(model, home_team: dict, away_team: dict,
                  h2h_lookup: dict, feature_cols: list[str]) -> float:
    """
    P(home_team wins a single game) using the trained logistic regression.

    Constructs the feature vector dynamically from feature_cols so that
    any preset (Four Factors, Four Factors + Clutch, etc.) works without
    changes to this function.

    Args:
        model:        Trained LogisticRegression instance.
        home_team:    dict with keys matching the stat names used in feature_cols.
        away_team:    Same structure as home_team.
        h2h_lookup:   dict {(team_id_a, team_id_b): h2h_win_pct}.
        feature_cols: Ordered list of feature column names.

    Returns:
        float in (0, 1).
    """
    features = build_feature_vector(home_team, away_team, h2h_lookup, feature_cols)
    return float(model.predict_proba(features)[0])


# ── Per-series win probability ────────────────────────────────────────────────

def series_win_prob(model, higher_seed: dict, lower_seed: dict,
                    h2h_lookup: dict, feature_cols: list[str] = None) -> float:
    """
    P(higher_seed wins a best-of-7 series) against lower_seed.

    feature_cols defaults to FEATURE_COLS (the default preset) for backward
    compatibility with any call sites that don't pass it explicitly.

    Because home court alternates game-by-game, we compute two probabilities:
      p_home = P(higher seed wins a game at their own court)
      p_away = P(higher seed wins a game at the opponent's court)
             = 1 - P(lower seed wins at their own court)

    Then use recursive DP:
      dp(wins_high, wins_low) = P(higher seed wins the series from this state)

    Game number = wins_high + wins_low + 1, which determines home court.
    Series ends when either team reaches 4 wins.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    p_home = game_win_prob(model, higher_seed, lower_seed, h2h_lookup, feature_cols)
    p_away = 1.0 - game_win_prob(model, lower_seed, higher_seed, h2h_lookup, feature_cols)

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

def simulate_conference(teams_by_seed: dict, model, h2h_lookup: dict,
                        feature_cols: list[str]) -> dict:
    """
    Compute P(each team wins the conference) for an 8-team bracket.

    NBA bracket structure (higher seed = home court):
      Round 1:     (1v8), (2v7), (3v6), (4v5)
      Semifinals:  winner(1/8) vs winner(4/5),  winner(2/7) vs winner(3/6)
      Conf Finals: winner of top half vs winner of bottom half

    Args:
        teams_by_seed: dict {seed (1-8): team_dict}
        model:         Trained LogisticRegression instance.
        h2h_lookup:    dict {(team_id_a, team_id_b): h2h_win_pct}.
        feature_cols:  Ordered list of feature column names.

    Returns:
        dict {seed: P(wins conference)}, values sum to ~1.0.
    """
    prob_reach = {seed: 1.0 for seed in range(1, 9)}

    # ── Round 1 ───────────────────────────────────────────────────────────────
    for h_seed, l_seed in [(1, 8), (2, 7), (3, 6), (4, 5)]:
        p = series_win_prob(
            model, teams_by_seed[h_seed], teams_by_seed[l_seed],
            h2h_lookup, feature_cols
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
                    model, teams_by_seed[h_seed], teams_by_seed[l_seed],
                    h2h_lookup, feature_cols
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
                model, teams_by_seed[h_seed], teams_by_seed[l_seed],
                h2h_lookup, feature_cols
            )
            prob_conf_champ[h_seed] += p_matchup * p_series
            prob_conf_champ[l_seed] += p_matchup * (1.0 - p_series)

    return prob_conf_champ


# ── Full playoff bracket simulation ──────────────────────────────────────────

def simulate_bracket(west_by_seed: dict, east_by_seed: dict, model,
                     h2h_lookup: dict, feature_cols: list[str] = None):
    """
    Compute P(each team wins the NBA championship) for the full 16-team field.

    feature_cols defaults to FEATURE_COLS for backward compatibility.

    Returns:
        champ_probs: dict {team_id: P(wins championship)}, sums to ~1.0.
        conf_probs:  dict {team_id: P(wins conference)}.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    print("  Simulating Western Conference bracket...")
    west_conf = simulate_conference(west_by_seed, model, h2h_lookup, feature_cols)

    print("  Simulating Eastern Conference bracket...")
    east_conf = simulate_conference(east_by_seed, model, h2h_lookup, feature_cols)

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

            p_series = series_win_prob(model, higher, lower, h2h_lookup, feature_cols)
            champ_probs[higher["TEAM_ID"]] += p_finals * p_series
            champ_probs[lower["TEAM_ID"]]  += p_finals * (1.0 - p_series)

    return champ_probs, conf_probs


# ── Data loading ──────────────────────────────────────────────────────────────

def load_season_data(season: str, path: str = "data/training_data.csv"):
    """
    Extract per-team Four Factor stats and H2H records for a given season
    from training_data.csv.

    Recovers per-team stats by taking the first occurrence of each team.
    Any HOME_* column that exists in the CSV is pulled into the team dict,
    so clutch columns (HOME_CLUTCH_WIN_PCT, etc.) are included automatically
    when present.

    Returns:
        team_stats:  dict {team_id: team_dict}.
        h2h_lookup:  dict {(team_id_a, team_id_b): h2h_win_pct}.
    """
    df = pd.read_csv(path)
    season_df = df[df["SEASON"] == season].copy()

    if season_df.empty:
        raise ValueError(f"No data found for season {season!r} in {path}")

    # Identify all HOME_* stat columns (excludes DIFF_* and metadata columns)
    home_stat_cols = [c for c in season_df.columns if c.startswith("HOME_")]

    team_stats = {}

    for _, row in season_df.iterrows():
        # Home team — pull all HOME_* stats into the dict
        tid = int(row["TEAM_ID"])
        if tid not in team_stats:
            entry = {"TEAM_ID": tid}
            for col in home_stat_cols:
                key = col[5:]  # strip "HOME_"
                entry[key] = float(row[col])
            team_stats[tid] = entry

        # Away team — pull all AWAY_* stats
        opp_tid = int(row["OPP_TEAM_ID"])
        if opp_tid not in team_stats:
            entry = {"TEAM_ID": opp_tid}
            for col in home_stat_cols:
                away_col = "AWAY_" + col[5:]
                if away_col in row:
                    key = col[5:]
                    entry[key] = float(row[away_col])
            team_stats[opp_tid] = entry

    # H2H lookup
    h2h_lookup = {}
    for _, row in season_df.iterrows():
        tid     = int(row["TEAM_ID"])
        opp_tid = int(row["OPP_TEAM_ID"])
        h2h_lookup[(tid, opp_tid)] = float(row["H2H_WIN_PCT"])

    return team_stats, h2h_lookup


def load_playoff_bracket(season: str, team_stats: dict,
                         data_dir: str = "data") -> tuple[dict, dict, dict]:
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
                    league_id="00",
                    season_type="Regular Season",
                    season=season,
                    timeout=120,
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
    df["PlayoffSeeding"] = pd.to_numeric(df["PlayoffSeeding"], errors="coerce")
    df["PlayoffRank"]    = pd.to_numeric(df["PlayoffRank"],    errors="coerce")

    use_col = "PlayoffSeeding" if df["PlayoffSeeding"].gt(0).any() else "PlayoffRank"

    east = df[df["Conference"] == "East"].dropna(subset=[use_col])
    east = east[east[use_col] > 0].nsmallest(8, use_col)

    west = df[df["Conference"] == "West"].dropna(subset=[use_col])
    west = west[west[use_col] > 0].nsmallest(8, use_col)

    print("\n  Standings top 8 East:")
    for _, row in east.iterrows():
        tid = int(row["TeamID"])
        in_stats = tid in team_stats
        print(f"    Rank {row['PlayoffRank']}: {row['TeamCity']} {row['TeamName']} "
              f"(ID {tid}) — in team_stats: {in_stats}")
    print("  Standings top 8 West:")
    for _, row in west.iterrows():
        tid = int(row["TeamID"])
        in_stats = tid in team_stats
        print(f"    Rank {row['PlayoffRank']}: {row['TeamCity']} {row['TeamName']} "
              f"(ID {tid}) — in team_stats: {in_stats}")
    print(f"\n  team_stats keys: {sorted(team_stats.keys())}")

    east_bracket, west_bracket, name_lookup = {}, {}, {}

    for _, row in east.iterrows():
        tid, seed = int(row["TeamID"]), int(row[use_col])
        if tid not in team_stats:
            raise KeyError(
                f"No stats for East seed {seed} ({row['TeamSlug']}, ID {tid}). "
                f"Check that {season} training data exists."
            )
        east_bracket[seed] = team_stats[tid]
        name_lookup[tid] = f"{row['TeamCity']} {row['TeamName']}"

    for _, row in west.iterrows():
        tid, seed = int(row["TeamID"]), int(row[use_col])
        if tid not in team_stats:
            raise KeyError(
                f"No stats for West seed {seed} ({row['TeamSlug']}, ID {tid}). "
                f"Check that {season} training data exists."
            )
        west_bracket[seed] = team_stats[tid]
        name_lookup[tid] = f"{row['TeamCity']} {row['TeamName']}"

    return east_bracket, west_bracket, name_lookup


# ── Results display ───────────────────────────────────────────────────────────

def print_results(champ_probs: dict, conf_probs: dict,
                  west_by_seed: dict, east_by_seed: dict,
                  name_lookup: dict = None):
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
    print(f"  {'Total':23}  {sum(conf_probs.values()):>9.3f} "
          f"{sum(champ_probs.values()):>9.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    from feature_sets import PRESETS

    parser = argparse.ArgumentParser(description="Bracket simulator standalone")
    parser.add_argument(
        "--preset", default=DEFAULT_PRESET,
        choices=list(PRESETS.keys()),
        help="Feature preset to use (default: %(default)s)"
    )
    args = parser.parse_args()

    feature_cols   = get_preset(args.preset)
    DATA_PATH      = "data/training_data.csv"
    HOLDOUT_SEASON = "2021-22"

    df       = pd.read_csv(DATA_PATH)
    train_df = df[df["SEASON"] != HOLDOUT_SEASON]

    print(f"Training on {len(train_df)} games, holding out {HOLDOUT_SEASON}...")
    print(f"Preset: {args.preset} ({len(feature_cols)} features)")
    model = LogisticRegression(learning_rate=0.1, epochs=1000, lambda_=0.01)
    model.fit(train_df[feature_cols].values, train_df["LABEL"].values)
    print("Done.\n")

    team_stats, h2h_lookup = load_season_data(HOLDOUT_SEASON, path=DATA_PATH)
    print(f"Loaded stats for {len(team_stats)} teams in {HOLDOUT_SEASON}.\n")

    east_bracket, west_bracket, name_lookup = load_playoff_bracket(
        HOLDOUT_SEASON, team_stats
    )

    print(f"Simulating {HOLDOUT_SEASON} playoff bracket...\n")
    champ_probs, conf_probs = simulate_bracket(
        west_bracket, east_bracket, model, h2h_lookup, feature_cols
    )

    print_results(champ_probs, conf_probs, west_bracket, east_bracket, name_lookup)

    total = sum(champ_probs.values())
    print(f"\nSanity check — champ probs sum to: {total:.6f}")

    test_df  = df[df["SEASON"] == HOLDOUT_SEASON]
    X_test   = test_df[feature_cols].values
    y_test   = test_df["LABEL"].values
    accuracy = (model.predict(X_test) == y_test).mean()
    print(f"Test Season Accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()