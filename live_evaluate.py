"""
NBA Playoff Outcome Prediction - Live Updating vs Static Comparison
CS830 Final Project - Andrew Elkin

Compares two prediction strategies:

  STATIC:  predict every game using regular season stats only (current approach)
  LIVE:    blend regular season stats with actual playoff box scores from all
           previous games in the same series before making each prediction

The live strategy simulates a real-time system that updates its team
strength estimates as each series progresses. After Game 1, the model
has one game of playoff evidence. After Game 3, it has three.

Blending formula (Bayesian-style weighted average):
  alpha = n_games_played / (n_games_played + weight)
  blended_stat = (1 - alpha) * reg_season_stat + alpha * playoff_running_avg

  weight (default=10) controls how strongly the regular season prior is held.
  Higher weight → slower to update; lower weight → faster to update.

Only Four Factor stats are updated from playoff game logs. Star player PM
and H2H features stay at regular season values throughout (updating star PM
would require player-level playoff game logs beyond what's available here).

Outputs:
  - Console: live output
  - results/live_vs_static_{preset}.txt: full report

Usage:
  python live_evaluate.py
  python live_evaluate.py --preset four_factors_star
  python live_evaluate.py --weight 5
  python live_evaluate.py --weight 20
  python live_evaluate.py --list-presets
"""

import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

from logistic_regression import LogisticRegression
from feature_sets import get_preset, list_presets, DEFAULT_PRESET, PRESETS
from DataScrape import fetch_game_logs, fetch_player_game_logs
from bracket_simulator import build_feature_vector

SEASONS = [
    "2005-06", "2006-07", "2007-08", "2008-09", "2009-10",
    "2010-11",            "2012-13", "2013-14", "2014-15",
    "2015-16", "2016-17", "2017-18", "2018-19",
                          "2020-21", "2021-22", "2022-23",
                          "2023-24", "2024-25"
]

DATA_PATH   = "data/training_data.csv"
RESULTS_DIR = "results"

# Four Factor stat keys updated from team-level playoff game logs.
UPDATABLE_FOUR_FACTORS = [
    "EFG_PCT", "OPP_EFG_PCT",
    "TOV_PCT", "OPP_TOV_PCT",
    "ORB_PCT", "DRB_PCT",
    "FTR",     "OPP_FTR",
]

# Star player keys updated from player-level playoff game logs.
# Only used when --update-stars is passed and the preset includes these features.
UPDATABLE_STAR_KEYS = ["TOP1_PM", "TOP2_AVG_PM", "TOP3_AVG_PM"]

# For backward compatibility
UPDATABLE_KEYS = UPDATABLE_FOUR_FACTORS


# ── Output helper ─────────────────────────────────────────────────────────────

class Writer:
    def __init__(self):
        self.lines = []

    def __call__(self, line=""):
        print(line)
        self.lines.append(line)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))
        print(f"\n✓ Report saved to {path}")


# ── Playoff box score utilities ───────────────────────────────────────────────

def load_playoff_logs(season: str) -> pd.DataFrame:
    """Load cached playoff game logs for a season."""
    logs = fetch_game_logs(season, "Playoffs")
    logs["TEAM_ID"] = logs["TEAM_ID"].astype(int)
    logs["GAME_ID"]  = logs["GAME_ID"].astype(str)
    return logs


def compute_playoff_four_factors(
    playoff_logs: pd.DataFrame,
    team_id: int,
    game_ids: list[str],
) -> dict | None:
    """
    Compute Four Factor stats for a team from a set of playoff game IDs.

    Uses the opponent's rows in those same games to compute defensive stats,
    mirroring the logic in DataScrape.compute_team_features().

    Returns None if no matching games are found.
    """
    game_ids = set(game_ids)

    team_rows = playoff_logs[
        (playoff_logs["TEAM_ID"] == team_id) &
        (playoff_logs["GAME_ID"].isin(game_ids))
    ]
    opp_rows = playoff_logs[
        (playoff_logs["TEAM_ID"] != team_id) &
        (playoff_logs["GAME_ID"].isin(game_ids))
    ]

    if team_rows.empty or opp_rows.empty:
        return None

    def safe_mean(series):
        return float(series.mean()) if len(series) > 0 else 0.0

    fgm      = safe_mean(team_rows["FGM"])
    fga      = safe_mean(team_rows["FGA"])
    fg3m     = safe_mean(team_rows["FG3M"])
    fta      = safe_mean(team_rows["FTA"])
    oreb     = safe_mean(team_rows["OREB"])
    dreb     = safe_mean(team_rows["DREB"])
    tov      = safe_mean(team_rows["TOV"])

    opp_fgm  = safe_mean(opp_rows["FGM"])
    opp_fga  = safe_mean(opp_rows["FGA"])
    opp_fg3m = safe_mean(opp_rows["FG3M"])
    opp_fta  = safe_mean(opp_rows["FTA"])
    opp_oreb = safe_mean(opp_rows["OREB"])
    opp_dreb = safe_mean(opp_rows["DREB"])
    opp_tov  = safe_mean(opp_rows["TOV"])

    efg_pct     = (fgm + 0.5 * fg3m) / fga          if fga  > 0 else 0.5
    tov_pct     = tov / (fga + 0.44 * fta + tov)     if (fga + 0.44*fta + tov) > 0 else 0.1
    orb_pct     = oreb / (oreb + opp_dreb)            if (oreb + opp_dreb) > 0 else 0.3
    drb_pct     = dreb / (dreb + opp_oreb)            if (dreb + opp_oreb) > 0 else 0.7
    ftr         = fta / fga                           if fga  > 0 else 0.25
    opp_efg_pct = (opp_fgm + 0.5*opp_fg3m) / opp_fga if opp_fga > 0 else 0.5
    opp_tov_pct = opp_tov / (opp_fga + 0.44*opp_fta + opp_tov) \
                  if (opp_fga + 0.44*opp_fta + opp_tov) > 0 else 0.1
    opp_ftr     = opp_fta / opp_fga                  if opp_fga > 0 else 0.25

    return {
        "EFG_PCT":     efg_pct,
        "TOV_PCT":     tov_pct,
        "ORB_PCT":     orb_pct,
        "DRB_PCT":     drb_pct,
        "FTR":         ftr,
        "OPP_EFG_PCT": opp_efg_pct,
        "OPP_TOV_PCT": opp_tov_pct,
        "OPP_FTR":     opp_ftr,
    }


def blend_team_stats(reg_stats: dict, playoff_stats: dict,
                     n_games: int, weight: float) -> dict:
    """
    Weighted blend of regular season stats and playoff running averages.

    alpha = n_games / (n_games + weight)
    blended = (1-alpha)*reg + alpha*playoff

    Only updatable Four Factor keys are blended; all other keys
    (TEAM_ID, WIN_PCT, star player features, etc.) stay at reg season values.
    """
    alpha = n_games / (n_games + weight)
    blended = dict(reg_stats)
    for key in UPDATABLE_KEYS:
        if key in playoff_stats and key in reg_stats:
            blended[key] = (1 - alpha) * reg_stats[key] + alpha * playoff_stats[key]
    return blended


def compute_playoff_star_pm(
    player_game_logs: pd.DataFrame,
    team_id: int,
    game_ids: list[str],
    min_mpg: float = 15.0,
) -> dict | None:
    """
    Compute star player plus/minus stats from actual playoff game logs.

    Uses a lower min_mpg threshold than the regular season (15 vs 20)
    since we're averaging across only 1-6 games. min_games is set to 1
    since requiring multiple appearances would exclude valid contributors
    in early games of a series.

    Returns dict with TOP1_PM, TOP2_AVG_PM, TOP3_AVG_PM, or None if
    no qualifying players are found.
    """
    game_ids = set(str(g) for g in game_ids)

    team_players = player_game_logs[
        (player_game_logs["TEAM_ID"] == team_id) &
        (player_game_logs["GAME_ID"].isin(game_ids))
    ]

    if team_players.empty:
        return None

    player_avgs = (
        team_players.groupby("PLAYER_ID")
        .agg(PM=("PLUS_MINUS", "mean"), MIN=("MIN", "mean"))
        .reset_index()
    )

    qualified = player_avgs[player_avgs["MIN"] >= min_mpg].sort_values(
        "PM", ascending=False
    )

    if qualified.empty:
        return None

    top1 = float(qualified.iloc[0]["PM"])
    top2 = float(qualified.head(2)["PM"].mean())
    top3 = float(qualified.head(3)["PM"].mean())

    return {"TOP1_PM": top1, "TOP2_AVG_PM": top2, "TOP3_AVG_PM": top3}


def blend_team_stats(reg_stats: dict, playoff_stats: dict,
                     n_games: int, weight: float,
                     extra_keys: list[str] | None = None) -> dict:
    """
    Weighted blend of regular season stats and playoff running averages.

    alpha = n_games / (n_games + weight)
    blended = (1-alpha)*reg + alpha*playoff

    Four Factor keys in UPDATABLE_FOUR_FACTORS are always blended when
    present in playoff_stats. Pass extra_keys to also blend star player
    features when player game log data is available.
    """
    alpha = n_games / (n_games + weight)
    blended = dict(reg_stats)
    keys_to_update = list(UPDATABLE_FOUR_FACTORS) + (extra_keys or [])
    for key in keys_to_update:
        if key in playoff_stats and key in reg_stats:
            blended[key] = (1 - alpha) * reg_stats[key] + alpha * playoff_stats[key]
    return blended


# ── Series identification ─────────────────────────────────────────────────────

def get_series_key(team_id: int, opp_team_id: int) -> frozenset:
    return frozenset({team_id, opp_team_id})


def group_games_into_series(season_df: pd.DataFrame) -> dict:
    """
    Group games in a season into series by team pair.
    Returns {series_key: [rows sorted by GAME_ID]}.
    """
    series = defaultdict(list)
    for _, row in season_df.iterrows():
        key = get_series_key(int(row["TEAM_ID"]), int(row["OPP_TEAM_ID"]))
        series[key].append(row)

    # Sort each series by GAME_ID (sequential = chronological)
    return {k: sorted(rows, key=lambda r: str(r["GAME_ID"]))
            for k, rows in series.items()}


# ── Per-game prediction ───────────────────────────────────────────────────────

def predict_game(model, home_stats: dict, away_stats: dict,
                 h2h: float, feature_cols: list[str]) -> float:
    """
    Build the feature vector from team stat dicts and return P(home wins).
    Mirrors build_feature_vector() in bracket_simulator.py.
    """
    h2h_lookup = {
        (home_stats["TEAM_ID"], away_stats["TEAM_ID"]): h2h
    }
    features = build_feature_vector(home_stats, away_stats, h2h_lookup, feature_cols)
    return float(model.predict_proba(features)[0])


# ── Per-season evaluation ─────────────────────────────────────────────────────

def evaluate_season_live(
    holdout: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    weight: float,
    update_stars: bool,
    w: Writer,
) -> dict | None:
    """
    For one holdout season, compare static vs live-updating predictions
    game-by-game within each series.

    update_stars: if True and the preset includes star features, also
    update TOP1_PM/TOP2_AVG_PM/TOP3_AVG_PM from player game logs.
    """
    train_df = df[df["SEASON"] != holdout]
    test_df  = df[df["SEASON"] == holdout].copy()

    if test_df.empty:
        return None

    # Train model on all other seasons
    X_train = train_df[feature_cols].values
    y_train = train_df["LABEL"].values
    model   = LogisticRegression(learning_rate=0.1, epochs=1000, lambda_=0.01)
    model.fit(X_train, y_train)

    # Load team-level playoff game logs (always needed for Four Factor updates)
    try:
        playoff_logs = load_playoff_logs(holdout)
    except Exception as e:
        w(f"  [skip] Could not load playoff logs for {holdout}: {e}")
        return None

    # Load player-level game logs if star updating is requested
    player_logs = None
    star_keys_in_preset = [k for k in UPDATABLE_STAR_KEYS
                           if any(k in col for col in feature_cols)]
    if update_stars and star_keys_in_preset:
        try:
            player_logs = fetch_player_game_logs(holdout, "Playoffs")
            player_logs["GAME_ID"] = player_logs["GAME_ID"].astype(str)
        except Exception as e:
            w(f"  [warning] Could not load player game logs for {holdout}: {e}")
            w(f"  [warning] Star features will not be updated this season.")

    # Reconstruct reg-season team stat dicts from HOME_*/AWAY_* columns
    home_stat_cols = [c for c in test_df.columns if c.startswith("HOME_")
                      and not c.endswith("_TEAM_ID")]
    away_stat_cols = [c for c in test_df.columns if c.startswith("AWAY_")
                      and not c.endswith("_TEAM_ID")]

    reg_stats_cache = {}
    for _, row in test_df.iterrows():
        home_tid = int(row["TEAM_ID"])
        away_tid = int(row["OPP_TEAM_ID"])
        if home_tid not in reg_stats_cache:
            entry = {"TEAM_ID": home_tid}
            for col in home_stat_cols:
                entry[col[5:]] = float(row[col])
            reg_stats_cache[home_tid] = entry
        if away_tid not in reg_stats_cache:
            entry = {"TEAM_ID": away_tid}
            for col in away_stat_cols:
                entry[col[5:]] = float(row[col])
            reg_stats_cache[away_tid] = entry

    series_groups = group_games_into_series(test_df)
    results_by_game_num = defaultdict(lambda: {"static": [], "live": [], "labels": []})
    overall = {"static_correct": 0, "live_correct": 0,
               "static_ll": [], "live_ll": [], "n": 0}
    eps = 1e-7

    for series_key, games in series_groups.items():
        games_played_ids = []

        for game_num, row in enumerate(games):
            home_tid = int(row["TEAM_ID"])
            away_tid = int(row["OPP_TEAM_ID"])
            label    = int(row["LABEL"])
            h2h      = float(row["H2H_WIN_PCT"])
            game_id  = str(row["GAME_ID"])

            home_reg = reg_stats_cache.get(home_tid, {})
            away_reg = reg_stats_cache.get(away_tid, {})
            if not home_reg or not away_reg:
                games_played_ids.append(game_id)
                continue

            p_static = predict_game(model, home_reg, away_reg, h2h, feature_cols)

            if games_played_ids:
                n = len(games_played_ids)

                # ── Four Factor updates ────────────────────────────────────
                home_ff = compute_playoff_four_factors(
                    playoff_logs, home_tid, games_played_ids
                )
                away_ff = compute_playoff_four_factors(
                    playoff_logs, away_tid, games_played_ids
                )

                # ── Star player updates (optional) ─────────────────────────
                extra_keys = None
                home_star  = None
                away_star  = None
                if player_logs is not None and star_keys_in_preset:
                    home_star = compute_playoff_star_pm(
                        player_logs, home_tid, games_played_ids
                    )
                    away_star = compute_playoff_star_pm(
                        player_logs, away_tid, games_played_ids
                    )
                    extra_keys = star_keys_in_preset

                # Merge Four Factor and star playoff stats into one dict
                def merge_playoff_stats(ff, star):
                    if ff is None and star is None:
                        return None
                    merged = {}
                    if ff:
                        merged.update(ff)
                    if star:
                        merged.update(star)
                    return merged

                home_playoff = merge_playoff_stats(home_ff, home_star)
                away_playoff = merge_playoff_stats(away_ff, away_star)

                home_live = blend_team_stats(
                    home_reg, home_playoff, n, weight, extra_keys
                ) if home_playoff else home_reg
                away_live = blend_team_stats(
                    away_reg, away_playoff, n, weight, extra_keys
                ) if away_playoff else away_reg
            else:
                home_live = home_reg
                away_live = away_reg

            p_live = predict_game(model, home_live, away_live, h2h, feature_cols)

            # Record
            results_by_game_num[game_num + 1]["static"].append(p_static)
            results_by_game_num[game_num + 1]["live"].append(p_live)
            results_by_game_num[game_num + 1]["labels"].append(label)

            static_correct = int((p_static >= 0.5) == label)
            live_correct   = int((p_live   >= 0.5) == label)

            overall["static_correct"] += static_correct
            overall["live_correct"]   += live_correct
            overall["static_ll"].append(
                -(label * np.log(max(p_static, eps)) + (1-label) * np.log(max(1-p_static, eps)))
            )
            overall["live_ll"].append(
                -(label * np.log(max(p_live, eps)) + (1-label) * np.log(max(1-p_live, eps)))
            )
            overall["n"] += 1

            games_played_ids.append(game_id)

    if overall["n"] == 0:
        return None

    n = overall["n"]
    static_acc = overall["static_correct"] / n
    live_acc   = overall["live_correct"]   / n
    static_ll  = float(np.mean(overall["static_ll"]))
    live_ll    = float(np.mean(overall["live_ll"]))

    sign = "+" if live_acc >= static_acc else ""
    w(f"  Games evaluated : {n}")
    w(f"  Static accuracy : {static_acc:.3f}   log-loss: {static_ll:.3f}")
    w(f"  Live accuracy   : {live_acc:.3f}   log-loss: {live_ll:.3f}  "
      f"({sign}{live_acc - static_acc:.3f} acc, "
      f"{live_ll - static_ll:+.3f} LL)")

    # Per-game-number breakdown
    game_num_results = {}
    for gn in sorted(results_by_game_num):
        s_preds  = results_by_game_num[gn]["static"]
        l_preds  = results_by_game_num[gn]["live"]
        labels   = results_by_game_num[gn]["labels"]
        s_acc = np.mean([(p >= 0.5) == y for p, y in zip(s_preds, labels)])
        l_acc = np.mean([(p >= 0.5) == y for p, y in zip(l_preds, labels)])
        game_num_results[gn] = {
            "n": len(labels),
            "static_acc": float(s_acc),
            "live_acc":   float(l_acc),
        }

    return {
        "season":     holdout,
        "n":          n,
        "static_acc": static_acc,
        "live_acc":   live_acc,
        "static_ll":  static_ll,
        "live_ll":    live_ll,
        "by_game":    game_num_results,
    }


# ── Summary ───────────────────────────────────────────────────────────────────

def display_summary(results: list[dict], weight: float,
                    feature_cols: list[str], w: Writer):
    w()
    w("=" * 74)
    w("  LIVE UPDATING vs STATIC — CROSS-VALIDATION SUMMARY")
    w(f"  Blending weight: {weight}  (alpha = n_games / (n_games + {weight}))")
    w("=" * 74)
    w(f"  {'Season':<10} {'N':>5} {'Static Acc':>12} {'Live Acc':>10} "
      f"{'Δ Acc':>8} {'Static LL':>10} {'Live LL':>9} {'Δ LL':>7}")
    w("-" * 74)

    s_accs, l_accs, s_lls, l_lls = [], [], [], []
    for r in results:
        d_acc = r["live_acc"] - r["static_acc"]
        d_ll  = r["live_ll"]  - r["static_ll"]
        sign_acc = "+" if d_acc >= 0 else ""
        sign_ll  = "+" if d_ll  >= 0 else ""
        w(f"  {r['season']:<10} {r['n']:>5} {r['static_acc']:>12.3f} "
          f"{r['live_acc']:>10.3f} {sign_acc}{d_acc:>7.3f} "
          f"{r['static_ll']:>10.3f} {r['live_ll']:>9.3f} "
          f"{sign_ll}{d_ll:>6.3f}")
        s_accs.append(r["static_acc"])
        l_accs.append(r["live_acc"])
        s_lls.append(r["static_ll"])
        l_lls.append(r["live_ll"])

    w("-" * 74)
    mean_d_acc = np.mean(l_accs) - np.mean(s_accs)
    mean_d_ll  = np.mean(l_lls)  - np.mean(s_lls)
    w(f"  {'MEAN':<10} {'':>5} {np.mean(s_accs):>12.3f} "
      f"{np.mean(l_accs):>10.3f} {mean_d_acc:>+8.3f} "
      f"{np.mean(s_lls):>10.3f} {np.mean(l_lls):>9.3f} "
      f"{mean_d_ll:>+7.3f}")
    w(f"  {'STD':<10} {'':>5} {np.std(s_accs):>12.3f} "
      f"{np.std(l_accs):>10.3f} {'':>8} "
      f"{np.std(s_lls):>10.3f} {np.std(l_lls):>9.3f}")
    w("=" * 74)

    # Per-game-number breakdown across all seasons
    w()
    w("  ACCURACY BY GAME NUMBER WITHIN SERIES (all seasons combined)")
    w(f"  {'Game':>6} {'N':>6} {'Static':>10} {'Live':>10} {'Δ':>8}  "
      f"{'Live better?':>12}")
    w("  " + "-" * 56)

    # Aggregate across all seasons
    by_game = defaultdict(lambda: {"static": [], "live": [], "n": 0})
    for r in results:
        for gn, gdata in r["by_game"].items():
            by_game[gn]["n"]      += gdata["n"]
            by_game[gn]["static"].append(gdata["static_acc"])
            by_game[gn]["live"].append(gdata["live_acc"])

    for gn in sorted(by_game):
        gd = by_game[gn]
        s_mean = np.mean(gd["static"])
        l_mean = np.mean(gd["live"])
        delta  = l_mean - s_mean
        better = "✓ live better" if delta > 0.002 else ("✗ static better" if delta < -0.002 else "≈ tied")
        w(f"  {'Game '+str(gn):>6} {gd['n']:>6} {s_mean:>10.3f} {l_mean:>10.3f} "
          f"{delta:>+8.3f}  {better}")

    w()
    w("  Note: Game 1 is always identical (no prior playoff data to update from).")
    w(f"  Δ LL: negative = live is better (lower loss). Positive = static is better.")
    w()

    # Overall verdict
    seasons_live_better = sum(1 for r in results if r["live_acc"] > r["static_acc"])
    seasons_static_better = sum(1 for r in results if r["static_acc"] > r["live_acc"])
    w(f"  Live beats static: {seasons_live_better}/{len(results)} seasons  "
      f"|  Static beats live: {seasons_static_better}/{len(results)} seasons")
    if mean_d_acc > 0.001:
        w(f"  → VERDICT: Live updating improves accuracy (+{mean_d_acc:.3f} mean)")
    elif mean_d_acc < -0.001:
        w(f"  → VERDICT: Static is better ({mean_d_acc:.3f} mean)")
    else:
        w(f"  → VERDICT: No meaningful difference ({mean_d_acc:+.4f} mean)")
    w()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare live-updating vs static playoff game prediction"
    )
    parser.add_argument(
        "--preset", default=DEFAULT_PRESET,
        choices=list(PRESETS.keys()),
        help="Feature preset to use (default: %(default)s)"
    )
    parser.add_argument(
        "--weight", type=float, default=10.0,
        help="Blending weight for regular season prior (default: 10). "
             "Higher = slower to update. alpha = n_games / (n_games + weight)"
    )
    parser.add_argument(
        "--update-stars", action="store_true",
        help="Also update star player PM features from player-level playoff game logs. "
             "Only applies when the chosen preset includes star player features. "
             "Requires fetching player game logs (cached after first run)."
    )
    parser.add_argument(
        "--list-presets", action="store_true",
        help="Print available feature presets and exit"
    )
    args = parser.parse_args()

    if args.list_presets:
        list_presets()
        return

    feature_cols = get_preset(args.preset)

    # Validate --update-stars makes sense for the chosen preset
    star_in_preset = any(k in " ".join(feature_cols) for k in UPDATABLE_STAR_KEYS)
    update_stars = args.update_stars
    if update_stars and not star_in_preset:
        print(f"Note: --update-stars has no effect with preset '{args.preset}' "
              f"(no star player features). Ignoring.")
        update_stars = False

    os.makedirs(RESULTS_DIR, exist_ok=True)
    w = Writer()

    w("=" * 74)
    w("  NBA PLAYOFF PREDICTION — LIVE UPDATING vs STATIC")
    w(f"  Preset       : {args.preset} ({len(feature_cols)} features)")
    w(f"  Weight       : {args.weight}  "
      f"(after 3 games alpha≈{3/(3+args.weight):.0%}, "
      f"after 6 games alpha≈{6/(6+args.weight):.0%})")
    w(f"  Star updates : {'enabled' if update_stars else 'disabled (pass --update-stars to enable)'}")
    w("  Method       : Leave-one-season-out cross-validation")
    w("=" * 74)

    df = pd.read_csv(DATA_PATH)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"ERROR: columns missing from training data: {missing}")
        print("Re-run DataScrape.py with appropriate flags.")
        return

    all_results = []

    for season in SEASONS:
        w()
        w("─" * 74)
        w(f"  SEASON: {season}")
        w("─" * 74)

        result = evaluate_season_live(
            season, df, feature_cols, args.weight, update_stars, w
        )
        if result:
            all_results.append(result)

    if all_results:
        display_summary(all_results, args.weight, feature_cols, w)

        weight_str = str(int(args.weight)) if args.weight == int(args.weight) \
                     else str(args.weight).replace(".", "p")
        star_suffix = "_stars" if update_stars else ""
        fname = f"live_vs_static_{args.preset}_w{weight_str}{star_suffix}.txt"
        w.save(os.path.join(RESULTS_DIR, fname))


if __name__ == "__main__":
    main()