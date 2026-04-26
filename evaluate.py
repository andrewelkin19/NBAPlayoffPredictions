"""
NBA Playoff Outcome Prediction - Evaluation Script
CS830 Final Project - Andrew Elkin

Runs leave-one-season-out cross-validation across all seasons.
Outputs:
  - Console: live output as each season is evaluated
  - results/evaluation_report.txt: full readable report
"""

import os
import numpy as np
import pandas as pd
from logistic_regression import LogisticRegression, FEATURE_COLS
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
    """
    Returns {team_id: seed_number} for all 16 playoff teams.
    Lower seed number = higher seed (seed 1 beats seed 8 in R1).
    """
    lookup = {}
    for seed, team in east_bracket.items():
        lookup[int(team["TEAM_ID"])] = seed
    for seed, team in west_bracket.items():
        lookup[int(team["TEAM_ID"])] = seed
    return lookup


def higher_seed_baseline(test_df: pd.DataFrame, seed_lookup: dict) -> tuple[float, int]:
    """
    Fraction of playoff games won by the actual higher seed.
    Joins each game row against verified bracket seedings.

    LABEL=1 means the home team won. Lower seed number = higher seed.
    Returns (accuracy, n_games_matched).
    """
    correct = 0
    total   = 0

    for _, row in test_df.iterrows():
        home_tid  = int(row["TEAM_ID"])
        away_tid  = int(row["OPP_TEAM_ID"])
        home_seed = seed_lookup.get(home_tid)
        away_seed = seed_lookup.get(away_tid)

        if home_seed is None or away_seed is None:
            continue  # unmatched — skip and report

        higher_seed_is_home = home_seed < away_seed
        home_won            = int(row["LABEL"]) == 1
        higher_seed_won     = higher_seed_is_home == home_won

        correct += int(higher_seed_won)
        total   += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, total


# ── Bracket visualization ─────────────────────────────────────────────────────

def predict_series(model, higher_seed, lower_seed, h2h_lookup, name_lookup):
    """
    Returns (winner, prob_winner_wins, winner_name, higher_seed_name, lower_seed_name).
    higher_seed has home court advantage.
    """
    p      = series_win_prob(model, higher_seed, lower_seed, h2h_lookup)
    winner = higher_seed if p >= 0.5 else lower_seed
    prob   = p if p >= 0.5 else 1.0 - p

    wname = name_lookup.get(winner["TEAM_ID"],      str(winner["TEAM_ID"]))
    hname = name_lookup.get(higher_seed["TEAM_ID"], str(higher_seed["TEAM_ID"]))
    lname = name_lookup.get(lower_seed["TEAM_ID"],  str(lower_seed["TEAM_ID"]))
    return winner, prob, wname, hname, lname


def home_court(team_a, team_b, seed_lookup):
    """
    Returns (higher_seed_team, lower_seed_team) using actual bracket seedings.
    Falls back to WIN_PCT if either team is missing from the lookup
    (only possible in Finals where seeds are cross-conference).
    """
    seed_a = seed_lookup.get(int(team_a["TEAM_ID"]))
    seed_b = seed_lookup.get(int(team_b["TEAM_ID"]))

    if seed_a is not None and seed_b is not None:
        return (team_a, team_b) if seed_a < seed_b else (team_b, team_a)

    # Finals fallback: best regular season record gets home court
    return (team_a, team_b) if team_a["WIN_PCT"] >= team_b["WIN_PCT"] else (team_b, team_a)


def display_bracket(west_by_seed, east_by_seed, model, h2h_lookup,
                    name_lookup, seed_lookup, w):
    """Prints a round-by-round predicted bracket to writer w."""

    def run_conference(conf_name, by_seed):
        w(f"  ── {conf_name} " + "─" * (54 - len(conf_name)))

        # Round 1: fixed seeded matchups
        w(f"\n  First Round")
        r1_winners = {}
        for h_s, l_s in [(1, 8), (2, 7), (3, 6), (4, 5)]:
            winner, prob, wname, hname, lname = predict_series(
                model, by_seed[h_s], by_seed[l_s], h2h_lookup, name_lookup
            )
            w(f"    ({h_s}) {hname} vs ({l_s}) {lname:<22}  → {wname} ({prob:.0%})")
            r1_winners[h_s] = winner

        # Semifinals: winner(1/8) vs winner(4/5), winner(2/7) vs winner(3/6)
        w(f"\n  Semifinals")
        semi_pairs   = [(r1_winners[1], r1_winners[4]), (r1_winners[2], r1_winners[3])]
        semi_winners = []
        for team_a, team_b in semi_pairs:
            higher, lower = home_court(team_a, team_b, seed_lookup)
            winner, prob, wname, hname, lname = predict_series(
                model, higher, lower, h2h_lookup, name_lookup
            )
            w(f"    {hname} vs {lname:<26}  → {wname} ({prob:.0%})")
            semi_winners.append(winner)

        # Conference Finals
        w(f"\n  Conference Finals")
        higher, lower = home_court(semi_winners[0], semi_winners[1], seed_lookup)
        winner, prob, wname, hname, lname = predict_series(
            model, higher, lower, h2h_lookup, name_lookup
        )
        w(f"    {hname} vs {lname:<26}  → {wname} ({prob:.0%})")
        w()
        return winner

    w()
    west_champ = run_conference("WESTERN CONFERENCE", west_by_seed)
    east_champ = run_conference("EASTERN CONFERENCE", east_by_seed)

    # NBA Finals
    w(f"  ── NBA FINALS " + "─" * 54)
    # Cross-conference: home court to team with better regular season record
    higher, lower = home_court(west_champ, east_champ, seed_lookup)
    winner, prob, wname, hname, lname = predict_series(
        model, higher, lower, h2h_lookup, name_lookup
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

def evaluate_season(holdout: str, df: pd.DataFrame, w: Writer) -> dict | None:
    train_df = df[df["SEASON"] != holdout]
    test_df  = df[df["SEASON"] == holdout]

    if test_df.empty:
        w(f"  [skip] No test data for {holdout}")
        return None

    # ── Load bracket first — needed for the accurate baseline ─────────────────
    try:
        team_stats, h2h_lookup = load_season_data(holdout, path=DATA_PATH)
        east_bracket, west_bracket, name_lookup = load_playoff_bracket(holdout, team_stats)
        seed_lookup = build_seed_lookup(east_bracket, west_bracket)
    except Exception as e:
        w(f"  [skip] Could not load bracket for {holdout}: {e}")
        return None

    # ── Train model ───────────────────────────────────────────────────────────
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["LABEL"].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df["LABEL"].values

    model = LogisticRegression(learning_rate=0.1, epochs=1000, lambda_=0.01)
    model.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_proba              = model.predict_proba(X_test)
    metrics              = compute_metrics(y_test, y_proba)
    baseline, n_matched  = higher_seed_baseline(test_df, seed_lookup)
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

    # ── Bracket visualization ─────────────────────────────────────────────────
    try:
        w(f"\n  Predicted bracket ({holdout}):")
        display_bracket(
            west_bracket, east_bracket, model, h2h_lookup,
            name_lookup, seed_lookup, w
        )
        champ_probs, conf_probs = simulate_bracket(
            west_bracket, east_bracket, model, h2h_lookup
        )
        w(f"  Championship probabilities:")
        display_champ_table(champ_probs, conf_probs, name_lookup, w)

    except Exception as e:
        w(f"\n  [bracket simulation skipped: {e}]")

    return {**metrics, "baseline": baseline, "n_games": n_games, "season": holdout}


# ── Summary table ─────────────────────────────────────────────────────────────

def display_summary(results: list[dict], w: Writer):
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
    w(f"  Features used in model ({len(FEATURE_COLS)}):")
    for i, col in enumerate(FEATURE_COLS, 1):
        w(f"    {i:2}. {col}")
    w()
    w("  Baseline: always predict the actual higher seed wins, using verified")
    w("  bracket seedings joined per game. Lower seed number = higher seed.")
    w("  Log-loss and Brier score: lower is better.")
    w()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    w = Writer()

    df = pd.read_csv(DATA_PATH)

    w("=" * 74)
    w("  NBA PLAYOFF OUTCOME PREDICTION — EVALUATION REPORT")
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

        result = evaluate_season(season, df, w)
        if result:
            all_results.append(result)

    display_summary(all_results, w)
    w.save(os.path.join(RESULTS_DIR, "evaluation_report.txt"))


if __name__ == "__main__":
    main()