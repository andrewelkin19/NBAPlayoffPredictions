"""
NBA Playoff Outcome Prediction - Evaluation Script
CS830 Final Project - Andrew Elkin

Runs leave-one-season-out cross-validation across all seasons.
Outputs:
  - Console: live output as each season is evaluated
  - results/evaluation_report_{preset}.txt: full readable report

Usage:
  python evaluate.py                          # default preset (four_factors)
  python evaluate.py --preset four_factors_clutch
  python evaluate.py --compare-all            # run every preset and print a summary table
  python evaluate.py --list-presets           # show available presets
"""

import os
import argparse
import numpy as np
import pandas as pd
from logistic_regression import LogisticRegression
from feature_sets import get_preset, list_presets, DEFAULT_PRESET, PRESETS
from bracket_simulator import (
    simulate_bracket, load_season_data, load_playoff_bracket, series_win_prob
)

SEASONS = [
    "2005-06", "2006-07", "2007-08", "2008-09", "2009-10",
    "2010-11",            "2012-13", "2013-14", "2014-15",
    "2015-16", "2016-17", "2017-18", "2018-19",
                          "2020-21", "2021-22", "2022-23",
                          "2023-24", "2024-25"
]

# Actual NBA champions for each season in our dataset.
# Used by --compare-all to evaluate championship prediction accuracy.
ACTUAL_CHAMPIONS = {
    "2005-06": 1610612748,  # Miami Heat
    "2006-07": 1610612759,  # San Antonio Spurs
    "2007-08": 1610612738,  # Boston Celtics
    "2008-09": 1610612747,  # Los Angeles Lakers
    "2009-10": 1610612747,  # Los Angeles Lakers
    "2010-11": 1610612742,  # Dallas Mavericks
    "2012-13": 1610612748,  # Miami Heat
    "2013-14": 1610612759,  # San Antonio Spurs
    "2014-15": 1610612744,  # Golden State Warriors
    "2015-16": 1610612739,  # Cleveland Cavaliers
    "2016-17": 1610612744,  # Golden State Warriors
    "2017-18": 1610612744,  # Golden State Warriors
    "2018-19": 1610612761,  # Toronto Raptors
    "2020-21": 1610612749,  # Milwaukee Bucks
    "2021-22": 1610612744,  # Golden State Warriors
    "2022-23": 1610612743,  # Denver Nuggets
    "2023-24": 1610612738,  # Boston Celtics
    "2024-25": 1610612760,  # Oklahoma City Thunder
}

DATA_PATH   = "data/training_data.csv"
RESULTS_DIR = "results"


# ── Output helper ─────────────────────────────────────────────────────────────

class Writer:
    """Prints to console and collects all output for file writing."""
    def __init__(self):
        self.lines = []

    def __call__(self, line=""):
        print(line)
        self.lines.append(line)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))
        print(f"\n✓ Report saved to {path}")


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Accuracy, log-loss, and Brier score."""
    y_pred   = (y_proba >= 0.5).astype(int)
    accuracy = float((y_pred == y_true).mean())

    eps      = 1e-7
    p        = np.clip(y_proba, eps, 1 - eps)
    log_loss = float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
    brier    = float(np.mean((y_proba - y_true) ** 2))

    return {"accuracy": accuracy, "log_loss": log_loss, "brier": brier}


# ── Seeding-accurate baseline ─────────────────────────────────────────────────

def build_seed_lookup(east_bracket: dict, west_bracket: dict) -> dict:
    lookup = {}
    for seed, team in east_bracket.items():
        lookup[int(team["TEAM_ID"])] = seed
    for seed, team in west_bracket.items():
        lookup[int(team["TEAM_ID"])] = seed
    return lookup


def higher_seed_baseline(test_df: pd.DataFrame, seed_lookup: dict,
                         team_stats: dict = None) -> tuple[float, int]:
    correct = 0
    total   = 0

    for _, row in test_df.iterrows():
        home_tid  = int(row["TEAM_ID"])
        away_tid  = int(row["OPP_TEAM_ID"])
        home_seed = seed_lookup.get(home_tid)
        away_seed = seed_lookup.get(away_tid)

        if home_seed is None or away_seed is None:
            # Team missing from seed lookup (e.g. relocated franchise edge case).
            # Fall back to win percentage from team_stats if available.
            if team_stats is not None:
                home_wp = team_stats.get(home_tid, {}).get("WIN_PCT", 0.5)
                away_wp = team_stats.get(away_tid, {}).get("WIN_PCT", 0.5)
                higher_seed_is_home = home_wp >= away_wp
            else:
                continue  # no fallback available — skip
        elif home_seed != away_seed:
            higher_seed_is_home = home_seed < away_seed
        else:
            # Equal seeds (Finals tiebreak) — use WIN_PCT
            if team_stats is not None:
                home_wp = team_stats.get(home_tid, {}).get("WIN_PCT", 0.5)
                away_wp = team_stats.get(away_tid, {}).get("WIN_PCT", 0.5)
                higher_seed_is_home = home_wp >= away_wp
            else:
                continue

        home_won        = int(row["LABEL"]) == 1
        higher_seed_won = higher_seed_is_home == home_won

        correct += int(higher_seed_won)
        total   += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, total


# ── Bracket visualization ─────────────────────────────────────────────────────

def predict_series(model, higher_seed, lower_seed, h2h_lookup, name_lookup,
                   feature_cols):
    p      = series_win_prob(model, higher_seed, lower_seed, h2h_lookup, feature_cols)
    winner = higher_seed if p >= 0.5 else lower_seed
    prob   = p if p >= 0.5 else 1.0 - p

    wname = name_lookup.get(winner["TEAM_ID"],      str(winner["TEAM_ID"]))
    hname = name_lookup.get(higher_seed["TEAM_ID"], str(higher_seed["TEAM_ID"]))
    lname = name_lookup.get(lower_seed["TEAM_ID"],  str(lower_seed["TEAM_ID"]))
    return winner, prob, wname, hname, lname


def home_court(team_a, team_b, seed_lookup):
    seed_a = seed_lookup.get(int(team_a["TEAM_ID"]))
    seed_b = seed_lookup.get(int(team_b["TEAM_ID"]))

    if seed_a is not None and seed_b is not None and seed_a != seed_b:
        return (team_a, team_b) if seed_a < seed_b else (team_b, team_a)

    return (team_a, team_b) if team_a["WIN_PCT"] >= team_b["WIN_PCT"] else (team_b, team_a)


def display_bracket(west_by_seed, east_by_seed, model, h2h_lookup,
                    name_lookup, seed_lookup, w, feature_cols):

    def run_conference(conf_name, by_seed):
        w(f"  ── {conf_name} " + "─" * (54 - len(conf_name)))

        w(f"\n  First Round")
        r1_winners = {}
        for h_s, l_s in [(1, 8), (2, 7), (3, 6), (4, 5)]:
            winner, prob, wname, hname, lname = predict_series(
                model, by_seed[h_s], by_seed[l_s], h2h_lookup, name_lookup, feature_cols
            )
            w(f"    ({h_s}) {hname} vs ({l_s}) {lname:<22}  → {wname} ({prob:.0%})")
            r1_winners[h_s] = winner

        w(f"\n  Semifinals")
        semi_pairs   = [(r1_winners[1], r1_winners[4]), (r1_winners[2], r1_winners[3])]
        semi_winners = []
        for team_a, team_b in semi_pairs:
            higher, lower = home_court(team_a, team_b, seed_lookup)
            winner, prob, wname, hname, lname = predict_series(
                model, higher, lower, h2h_lookup, name_lookup, feature_cols
            )
            w(f"    {hname} vs {lname:<26}  → {wname} ({prob:.0%})")
            semi_winners.append(winner)

        w(f"\n  Conference Finals")
        higher, lower = home_court(semi_winners[0], semi_winners[1], seed_lookup)
        winner, prob, wname, hname, lname = predict_series(
            model, higher, lower, h2h_lookup, name_lookup, feature_cols
        )
        w(f"    {hname} vs {lname:<26}  → {wname} ({prob:.0%})")
        w()
        return winner

    w()
    west_champ = run_conference("WESTERN CONFERENCE", west_by_seed)
    east_champ = run_conference("EASTERN CONFERENCE", east_by_seed)

    w(f"  ── NBA FINALS " + "─" * 54)
    higher, lower = home_court(west_champ, east_champ, seed_lookup)
    winner, prob, wname, hname, lname = predict_series(
        model, higher, lower, h2h_lookup, name_lookup, feature_cols
    )
    w(f"    {hname} vs {lname:<26}  → {wname} ({prob:.0%})")
    w()
    w(f"  ★  PREDICTED CHAMPION: {wname} ({prob:.0%} to win Finals)")
    w()


def display_champ_table(champ_probs, conf_probs, name_lookup, w):
    w("  " + "=" * 62)
    w(f"  {'TEAM':<26} {'CONF WIN%':>10} {'CHAMP%':>10}")
    w("  " + "=" * 62)
    for tid, champ_p in sorted(champ_probs.items(), key=lambda x: -x[1]):
        conf_p = conf_probs.get(tid, 0.0)
        bar    = "█" * int(champ_p * 30)
        name   = name_lookup.get(tid, str(tid))
        w(f"  {name:<26} {conf_p:>9.1%} {champ_p:>9.1%}  {bar}")
    w("  " + "=" * 62)


# ── Per-season evaluation ─────────────────────────────────────────────────────

def evaluate_season(
    holdout: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    w: Writer,
) -> dict | None:
    train_df = df[df["SEASON"] != holdout]
    test_df  = df[df["SEASON"] == holdout]

    if test_df.empty:
        w(f"  [skip] No test data for {holdout}")
        return None

    try:
        team_stats, h2h_lookup = load_season_data(holdout, path=DATA_PATH)
        east_bracket, west_bracket, name_lookup = load_playoff_bracket(holdout, team_stats)
        seed_lookup = build_seed_lookup(east_bracket, west_bracket)
    except Exception as e:
        w(f"  [skip] Could not load bracket for {holdout}: {e}")
        return None

    X_train = train_df[feature_cols].values
    y_train = train_df["LABEL"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["LABEL"].values

    model = LogisticRegression(learning_rate=0.1, epochs=1000, lambda_=0.01)
    model.fit(X_train, y_train)

    y_proba              = model.predict_proba(X_test)
    metrics              = compute_metrics(y_test, y_proba)
    baseline, n_matched  = higher_seed_baseline(test_df, seed_lookup, team_stats)
    n_games              = len(test_df)

    if n_matched < n_games:
        w(f"  [warning] {n_games - n_matched}/{n_games} games had unmatched team IDs in seed lookup")

    delta = metrics["accuracy"] - baseline
    sign  = "+" if delta >= 0 else ""

    w(f"  Games in holdout : {n_games}")
    w(f"  Model accuracy   : {metrics['accuracy']:.3f}")
    w(f"  Baseline (seed)  : {baseline:.3f}  ({sign}{delta:.3f} vs baseline)")
    w(f"  Log-loss         : {metrics['log_loss']:.3f}")
    w(f"  Brier score      : {metrics['brier']:.3f}")

    try:
        w(f"\n  Predicted bracket ({holdout}):")
        display_bracket(
            west_bracket, east_bracket, model, h2h_lookup,
            name_lookup, seed_lookup, w, feature_cols
        )
        champ_probs, conf_probs = simulate_bracket(
            west_bracket, east_bracket, model, h2h_lookup, feature_cols
        )
        w(f"  Championship probabilities:")
        display_champ_table(champ_probs, conf_probs, name_lookup, w)

    except Exception as e:
        w(f"\n  [bracket simulation skipped: {e}]")

    return {**metrics, "baseline": baseline, "n_games": n_games, "season": holdout}


# ── Summary table ─────────────────────────────────────────────────────────────

def display_summary(results: list[dict], feature_cols: list[str], w: Writer):
    w()
    w("=" * 74)
    w("  CROSS-VALIDATION SUMMARY — ALL SEASONS")
    w("=" * 74)
    w(f"  {'Season':<10} {'N':>5} {'Accuracy':>10} {'Baseline':>10} "
      f"{'Δ Acc':>8} {'Log-Loss':>10} {'Brier':>8}")
    w("-" * 74)

    accs, bases, losses, briers = [], [], [], []
    for r in results:
        delta = r["accuracy"] - r["baseline"]
        sign  = "+" if delta >= 0 else ""
        w(f"  {r['season']:<10} {r['n_games']:>5} {r['accuracy']:>10.3f} "
          f"{r['baseline']:>10.3f} {sign}{delta:>7.3f} "
          f"{r['log_loss']:>10.3f} {r['brier']:>8.3f}")
        accs.append(r["accuracy"])
        bases.append(r["baseline"])
        losses.append(r["log_loss"])
        briers.append(r["brier"])

    w("-" * 74)
    mean_delta = np.mean(accs) - np.mean(bases)
    sign = "+" if mean_delta >= 0 else ""
    w(f"  {'MEAN':<10} {'':>5} {np.mean(accs):>10.3f} "
      f"{np.mean(bases):>10.3f} {sign}{mean_delta:>7.3f} "
      f"{np.mean(losses):>10.3f} {np.mean(briers):>8.3f}")
    w(f"  {'STD':<10} {'':>5} {np.std(accs):>10.3f} "
      f"{np.std(bases):>10.3f} {'':>8} "
      f"{np.std(losses):>10.3f} {np.std(briers):>8.3f}")
    w("=" * 74)

    w()
    w(f"  Features used in model ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        w(f"    {i:2}. {col}")
    w()
    w("  Baseline: always predict the actual higher seed wins, using verified")
    w("  bracket seedings joined per game. Lower seed number = higher seed.")
    w("  Log-loss and Brier score: lower is better.")
    w()


# ── Compare-all mode ──────────────────────────────────────────────────────────

def get_actual_series_results(season_df: pd.DataFrame) -> dict:
    """
    Derive actual series winners from game-level playoff data.

    Groups games by unique {team_a, team_b} pairs and counts wins for each side.
    Works for all rounds since each pair only meets once per playoff bracket.

    Returns:
        dict mapping frozenset({team_id_a, team_id_b}) -> winning_team_id
    """
    pairs = set()
    for _, row in season_df.iterrows():
        pairs.add(frozenset([int(row["TEAM_ID"]), int(row["OPP_TEAM_ID"])]))

    results = {}
    for pair in pairs:
        a, b = list(pair)
        a_wins = (
            len(season_df[(season_df["TEAM_ID"] == a) &
                          (season_df["OPP_TEAM_ID"] == b) &
                          (season_df["LABEL"] == 1)]) +
            len(season_df[(season_df["TEAM_ID"] == b) &
                          (season_df["OPP_TEAM_ID"] == a) &
                          (season_df["LABEL"] == 0)])
        )
        b_wins = (
            len(season_df[(season_df["TEAM_ID"] == b) &
                          (season_df["OPP_TEAM_ID"] == a) &
                          (season_df["LABEL"] == 1)]) +
            len(season_df[(season_df["TEAM_ID"] == a) &
                          (season_df["OPP_TEAM_ID"] == b) &
                          (season_df["LABEL"] == 0)])
        )
        results[pair] = a if a_wins >= b_wins else b

    return results

def run_compare_all(df: pd.DataFrame):
    """
    Run cross-validation for every preset and save a comparison table.
    Includes champion and series prediction accuracy via bracket simulation.
    Also computes model-free baselines for comparison.
    Output saved to results/preset_comparison.txt.
    """
    w = Writer()

    w("=" * 96)
    w("  PRESET COMPARISON — leave-one-season-out cross-validation")
    w(f"  Data: {len(df)} games, {len(SEASONS)} seasons")
    w("  Champ:  seasons where argmax P(champion) == actual champion")
    w("  Series: all playoff series (all rounds); predicted winner = team with P(series win) > 0.5")
    w("=" * 96)

    # ── Compute model-free baselines (one pass over seasons) ──────────────────
    # These don't depend on any trained model so we compute them once up front.
    # All bracket data is cached on disk after the first run, so this is fast.
    #
    # Equivalence note: "always pick home team" and "always pick higher seed"
    # produce IDENTICAL series and championship predictions.  Under the home-
    # team rule, the higher seed (who hosts games 1,2,5,7) wins 4 home games
    # and the lower seed wins 3 (games 3,4,6) — the series always ends 4-3 in
    # favour of the higher seed.  So only one series/champion baseline is needed.
    seed_game_corr  = seed_game_tot  = 0
    home_game_corr  = home_game_tot  = 0
    seed_series_corr = seed_series_tot = 0
    seed_champ_corr  = seed_champ_tot  = 0

    for season in SEASONS:
        test_df = df[df["SEASON"] == season]
        if test_df.empty:
            continue
        try:
            _ts, _   = load_season_data(season, path=DATA_PATH)
            _east, _west, _ = load_playoff_bracket(season, _ts)
            _sl      = build_seed_lookup(_east, _west)

            # Game accuracy — higher seed
            base, n  = higher_seed_baseline(test_df, _sl, _ts)
            seed_game_corr += round(base * n)
            seed_game_tot  += n

            # Game accuracy — home team always wins
            home_game_corr += int((test_df["LABEL"] == 1).sum())
            home_game_tot  += len(test_df)

            # Series accuracy — always pick higher seed (WIN_PCT tiebreak for Finals)
            actual_series = get_actual_series_results(test_df)
            for pair, actual_winner in actual_series.items():
                a, b = list(pair)
                sa, sb = _sl.get(a, 999), _sl.get(b, 999)
                if sa != sb:
                    predicted = a if sa < sb else b
                else:
                    wpa = _ts.get(a, {}).get("WIN_PCT", 0.5)
                    wpb = _ts.get(b, {}).get("WIN_PCT", 0.5)
                    predicted = a if wpa >= wpb else b
                seed_series_corr += int(predicted == actual_winner)
                seed_series_tot  += 1

            # Champion accuracy — always pick each conference's 1-seed;
            # Finals winner = whichever 1-seed has the better regular-season record
            actual_champ = ACTUAL_CHAMPIONS.get(season)
            if actual_champ is not None:
                e_tid = _east[1]["TEAM_ID"]
                w_tid = _west[1]["TEAM_ID"]
                e_wp  = _ts.get(e_tid, {}).get("WIN_PCT", 0.5)
                w_wp  = _ts.get(w_tid, {}).get("WIN_PCT", 0.5)
                pred  = e_tid if e_wp >= w_wp else w_tid
                seed_champ_corr += int(pred == actual_champ)
                seed_champ_tot  += 1

        except Exception:
            pass

    seed_game_acc  = seed_game_corr  / seed_game_tot   if seed_game_tot   > 0 else 0.0
    home_game_acc  = home_game_corr  / home_game_tot   if home_game_tot   > 0 else 0.0

    # ── Main preset comparison table ──────────────────────────────────────────
    w(f"  {'Preset':<30} {'Feat':>4} {'Acc':>7} {'ΔAcc':>7} "
      f"{'LL':>7} {'BS':>7}  {'Champ':>7}  {'Series':>8}")
    w("-" * 96)

    for preset_name, feature_cols in PRESETS.items():
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            w(f"  {preset_name:<30}   [skip — missing columns: {missing}]")
            continue

        accs, losses, briers, deltas = [], [], [], []
        champ_correct = champ_total = 0
        series_correct = series_total = 0

        for season in SEASONS:
            train_df = df[df["SEASON"] != season]
            test_df  = df[df["SEASON"] == season]
            if train_df.empty or test_df.empty:
                continue

            model = LogisticRegression(learning_rate=0.1, epochs=1000, lambda_=0.01)
            model.fit(train_df[feature_cols].values, train_df["LABEL"].values)

            y_proba = model.predict_proba(test_df[feature_cols].values)
            y_true  = test_df["LABEL"].values
            m = compute_metrics(y_true, y_proba)
            accs.append(m["accuracy"])
            losses.append(m["log_loss"])
            briers.append(m["brier"])

            # Bracket-dependent stats: Δ Acc baseline + champion + series.
            # All three share one bracket load so load_playoff_bracket is only
            # called once per season — avoids duplicate API/cache reads and the
            # associated debug printout noise.
            try:
                team_stats, h2h_lookup = load_season_data(season, path=DATA_PATH)
                east_bracket, west_bracket, _ = load_playoff_bracket(season, team_stats)
                seed_lookup = build_seed_lookup(east_bracket, west_bracket)

                # Δ Acc vs seeding baseline
                base, _ = higher_seed_baseline(test_df, seed_lookup, team_stats)
                deltas.append(m["accuracy"] - base)

                # Champion accuracy
                actual_champ = ACTUAL_CHAMPIONS.get(season)
                if actual_champ is not None:
                    champ_probs, _ = simulate_bracket(
                        west_bracket, east_bracket, model, h2h_lookup, feature_cols
                    )
                    predicted_champ = max(champ_probs.items(), key=lambda x: x[1])[0]
                    champ_correct += int(predicted_champ == actual_champ)
                    champ_total += 1

                # Series accuracy — all rounds, derived from game data
                actual_series = get_actual_series_results(test_df)
                for pair, actual_winner in actual_series.items():
                    a, b = list(pair)
                    if a not in team_stats or b not in team_stats:
                        continue
                    seed_a = seed_lookup.get(a, 999)
                    seed_b = seed_lookup.get(b, 999)
                    higher = team_stats[a] if seed_a < seed_b else team_stats[b]
                    lower  = team_stats[b] if seed_a < seed_b else team_stats[a]
                    p = series_win_prob(model, higher, lower, h2h_lookup, feature_cols)
                    predicted_winner = higher["TEAM_ID"] if p > 0.5 else lower["TEAM_ID"]
                    series_correct += int(predicted_winner == actual_winner)
                    series_total += 1

            except Exception:
                deltas.append(0.0)  # fallback if bracket load fails

        if not accs:
            continue

        mean_delta = np.mean(deltas)
        sign = "+" if mean_delta >= 0 else ""
        champ_str  = f"{champ_correct}/{champ_total}"  if champ_total  > 0 else "—"
        series_str = f"{series_correct}/{series_total}" if series_total > 0 else "—"

        w(f"  {preset_name:<30} {len(feature_cols):>4} "
          f"{np.mean(accs):>7.3f} {sign}{mean_delta:>6.3f} "
          f"{np.mean(losses):>7.3f} {np.mean(briers):>7.3f}  "
          f"{champ_str:>7}  {series_str:>8}")

    # ── Baseline reference rows ────────────────────────────────────────────────
    w("=" * 96)
    w()
    w("  MODEL-FREE BASELINES")
    w("  " + "-" * 70)
    w(f"  {'Rule':<40} {'Game Acc':>9}  {'Series':>8}  {'Champ':>7}")
    w("  " + "-" * 70)
    w(f"  {'Always pick higher seed':<40} {seed_game_acc:>9.3f}  "
      f"{seed_series_corr}/{seed_series_tot:>3}  {seed_champ_corr}/{seed_champ_tot}")
    w(f"  {'Always pick home team':<40} {home_game_acc:>9.3f}  "
      f"{'(same)':>8}  {'(same)':>7}")
    w("  " + "-" * 70)
    w("  Series and championship baselines are identical for both rules:")
    w("  'always pick home team' → higher seed wins every series 4-3")
    w("  (higher seed hosts games 1,2,5,7; lower seed hosts 3,4,6)")
    w("  " + "-" * 70)
    w()
    w("  Notes:")
    w("  - ΔAcc: model accuracy minus always-pick-higher-seed baseline per game")
    w("  - Champ: predicted champion = team with highest P(championship) from bracket simulation")
    w("  - Series: all 15 series/season across all rounds; home court = lower seed number")
    w("  - Log-loss and Brier score: lower is better")
    w()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "preset_comparison.txt")
    w.save(out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NBA playoff prediction evaluation")
    parser.add_argument(
        "--preset", default=DEFAULT_PRESET,
        choices=list(PRESETS.keys()),
        help="Feature preset to use (default: %(default)s)"
    )
    parser.add_argument(
        "--compare-all", action="store_true",
        help="Run all presets and print a comparison table (no bracket output)"
    )
    parser.add_argument(
        "--list-presets", action="store_true",
        help="Print available feature presets and exit"
    )
    args = parser.parse_args()

    if args.list_presets:
        list_presets()
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    if args.compare_all:
        run_compare_all(df)
        return

    # ── Single preset full evaluation ─────────────────────────────────────────
    feature_cols = get_preset(args.preset)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"ERROR: columns missing from {DATA_PATH}: {missing}")
        print("Re-run DataScrape.py (with --clutch if needed) to regenerate.")
        return

    w = Writer()

    w("=" * 74)
    w("  NBA PLAYOFF OUTCOME PREDICTION — EVALUATION REPORT")
    w(f"  Preset : {args.preset} ({len(feature_cols)} features)")
    w("  Model  : Logistic Regression (implemented from scratch, NumPy)")
    w("  Method : Leave-one-season-out cross-validation")
    w(f"  Data   : {len(df)} playoff game examples, {df['SEASON'].nunique()} seasons")
    w("=" * 74)

    all_results = []

    for season in SEASONS:
        w()
        w("─" * 74)
        w(f"  SEASON: {season}")
        w("─" * 74)

        result = evaluate_season(season, df, feature_cols, w)
        if result:
            all_results.append(result)

    display_summary(all_results, feature_cols, w)

    report_name = f"evaluation_report_{args.preset}.txt"
    w.save(os.path.join(RESULTS_DIR, report_name))


if __name__ == "__main__":
    main()