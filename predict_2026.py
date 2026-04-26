"""
NBA Playoff Outcome Prediction - 2025-26 Season Prediction
CS830 Final Project - Andrew Elkin

Predicts the 2025-26 NBA playoff bracket using:
  - A model trained on all prior seasons (2005-06 through 2024-25)
  - 2025-26 regular season stats fetched live from nba_api

Usage:
  python predict_2026.py                              # default preset (four_factors)
  python predict_2026.py --preset four_factors_star
  python predict_2026.py --preset four_factors_top3
  python predict_2026.py --compare-all                # run every preset, summary table only
  python predict_2026.py --list-presets
"""

import os
import argparse
import numpy as np
import pandas as pd

from DataScrape import fetch_game_logs, compute_team_features, compute_h2h, \
                       compute_clutch_features, fetch_clutch_stats, \
                       compute_star_player_features, fetch_player_stats
from logistic_regression import LogisticRegression
from feature_sets import get_preset, list_presets, DEFAULT_PRESET, PRESETS, \
                         requires_clutch, requires_star
from bracket_simulator import simulate_bracket, series_win_prob

# ── 2025-26 playoff bracket ───────────────────────────────────────────────────

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
DATA_PATH       = "data/training_data.csv"
PREDICTIONS_DIR = "results/predictions_2026"


# ── Output helper ─────────────────────────────────────────────────────────────

class Writer:
    """Prints to console and collects all output for file saving."""
    def __init__(self):
        self.lines = []

    def __call__(self, line=""):
        print(line)
        self.lines.append(line)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))
        print(f"\n✓ Saved to {path}")


# ── Step 1: Train model on all prior seasons ──────────────────────────────────

def train_model(feature_cols: list[str]) -> LogisticRegression:
    print("Loading training data...")
    df = pd.read_csv(DATA_PATH)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns missing from training data: {missing}\n"
            f"Re-run DataScrape.py with appropriate flags to add them."
        )

    print(f"  {len(df)} games across {df['SEASON'].nunique()} seasons.\n")
    X = df[feature_cols].values
    y = df["LABEL"].values

    print("Training logistic regression model...")
    model = LogisticRegression(learning_rate=0.1, epochs=1000, lambda_=0.01)
    model.fit(X, y)
    print("  Done.\n")
    return model


# ── Step 2: Fetch 2025-26 regular season stats ────────────────────────────────

def load_current_season_data(need_clutch: bool, need_star: bool):
    """
    Fetch completed 2025-26 regular season data and compute features.
    Only fetches clutch/player stats if the chosen preset requires them.
    All results are cached so re-runs are instant.
    """
    print(f"Fetching {CURRENT_SEASON} regular season data...")
    reg_logs = fetch_game_logs(CURRENT_SEASON, "Regular Season")
    print(f"  {len(reg_logs)} team-game rows loaded.\n")

    print("Computing team features...")
    team_features = compute_team_features(reg_logs)
    print(f"  {len(team_features)} teams.\n")

    if need_clutch:
        print("Fetching clutch stats...")
        clutch_df     = fetch_clutch_stats(CURRENT_SEASON)
        team_features = compute_clutch_features(team_features, clutch_df)
        print(f"  Clutch features merged.\n")

    if need_star:
        print("Fetching player stats...")
        player_stats  = fetch_player_stats(CURRENT_SEASON)
        team_features = compute_star_player_features(team_features, player_stats)
        print(f"  Star player features merged.\n")

    print("Computing head-to-head records...")
    h2h_df = compute_h2h(reg_logs)
    h2h_lookup = {
        (int(row["TEAM_ID"]), int(row["OPP_TEAM_ID"])): float(row["H2H_WIN_PCT"])
        for _, row in h2h_df.iterrows()
    }
    print(f"  {len(h2h_lookup)} head-to-head records.\n")

    return team_features, h2h_lookup


# ── Step 3: Build bracket dicts ───────────────────────────────────────────────

def build_bracket(seed_map: dict, team_features_df: pd.DataFrame) -> dict:
    """
    Convert {seed: team_id} into the {seed: team_dict} format expected
    by simulate_bracket(). Pulls all available stat columns dynamically
    so any feature preset works without changes here.
    """
    features_by_id = team_features_df.set_index("TEAM_ID")

    # Stat columns to pull into the team dict (everything except metadata)
    skip_cols = {"TEAM_NAME", "TEAM_ABBREVIATION", "GAMES", "WINS",
                 "NET_RATING", "FGM_PG", "FGA_PG", "FG3M_PG", "FTA_PG",
                 "OREB_PG", "DREB_PG", "TOV_PG", "OPP_FGM_PG", "OPP_FGA_PG",
                 "OPP_FG3M_PG", "OPP_FTA_PG", "OPP_OREB_PG", "OPP_DREB_PG",
                 "OPP_TOV_PG"}
    stat_cols = [c for c in team_features_df.columns
                 if c != "TEAM_ID" and c not in skip_cols]

    bracket = {}
    for seed, team_id in seed_map.items():
        if team_id not in features_by_id.index:
            raise ValueError(
                f"Team ID {team_id} ({NAME_LOOKUP.get(team_id, '?')}) "
                f"not found in {CURRENT_SEASON} regular season data."
            )
        row = features_by_id.loc[team_id]
        entry = {"TEAM_ID": int(team_id)}
        for col in stat_cols:
            entry[col] = float(row[col])
        bracket[seed] = entry

    return bracket


# ── Results display ───────────────────────────────────────────────────────────

def print_bracket(west_bracket, east_bracket, model, h2h_lookup,
                  feature_cols, seed_lookup=None, w=print):
    """Print a round-by-round bracket with predicted winners."""

    def home_court(a, b):
        if seed_lookup:
            sa = seed_lookup.get(int(a["TEAM_ID"]))
            sb = seed_lookup.get(int(b["TEAM_ID"]))
            if sa and sb:
                return (a, b) if sa < sb else (b, a)
        return (a, b) if a["WIN_PCT"] >= b["WIN_PCT"] else (b, a)

    def predict(higher, lower):
        p = series_win_prob(model, higher, lower, h2h_lookup, feature_cols)
        winner = higher if p >= 0.5 else lower
        prob   = p if p >= 0.5 else 1 - p
        return winner, prob

    def run_conf(name, by_seed):
        w(f"\n  ── {name} " + "─" * (54 - len(name)))
        w("\n  First Round")
        r1 = {}
        for h, l in [(1,8),(2,7),(3,6),(4,5)]:
            winner, prob = predict(by_seed[h], by_seed[l])
            hn = NAME_LOOKUP.get(by_seed[h]["TEAM_ID"], "?")
            ln = NAME_LOOKUP.get(by_seed[l]["TEAM_ID"], "?")
            wn = NAME_LOOKUP.get(winner["TEAM_ID"], "?")
            w(f"    ({h}) {hn} vs ({l}) {ln:<22}  → {wn} ({prob:.0%})")
            r1[h] = winner

        w("\n  Semifinals")
        sw = []
        for a, b in [(r1[1], r1[4]), (r1[2], r1[3])]:
            hi, lo = home_court(a, b)
            winner, prob = predict(hi, lo)
            hn = NAME_LOOKUP.get(hi["TEAM_ID"], "?")
            ln = NAME_LOOKUP.get(lo["TEAM_ID"], "?")
            wn = NAME_LOOKUP.get(winner["TEAM_ID"], "?")
            w(f"    {hn} vs {ln:<26}  → {wn} ({prob:.0%})")
            sw.append(winner)

        w("\n  Conference Finals")
        hi, lo = home_court(sw[0], sw[1])
        winner, prob = predict(hi, lo)
        hn = NAME_LOOKUP.get(hi["TEAM_ID"], "?")
        ln = NAME_LOOKUP.get(lo["TEAM_ID"], "?")
        wn = NAME_LOOKUP.get(winner["TEAM_ID"], "?")
        w(f"    {hn} vs {ln:<26}  → {wn} ({prob:.0%})")
        w()
        return winner

    wc = run_conf("WESTERN CONFERENCE", west_bracket)
    ec = run_conf("EASTERN CONFERENCE", east_bracket)

    w(f"  ── NBA FINALS " + "─" * 54)
    hi, lo = home_court(wc, ec)
    winner, prob = predict(hi, lo)
    hn = NAME_LOOKUP.get(hi["TEAM_ID"], "?")
    ln = NAME_LOOKUP.get(lo["TEAM_ID"], "?")
    wn = NAME_LOOKUP.get(winner["TEAM_ID"], "?")
    w(f"    {hn} vs {ln:<26}  → {wn} ({prob:.0%})")
    w(f"\n  ★  PREDICTED CHAMPION: {wn} ({prob:.0%} to win Finals)\n")


def print_champ_table(champ_probs, conf_probs, w=print):
    w("  " + "=" * 62)
    w(f"  {'TEAM':<26} {'CONF WIN%':>10} {'CHAMP%':>10}")
    w("  " + "=" * 62)
    for tid, cp in sorted(champ_probs.items(), key=lambda x: -x[1]):
        confp = conf_probs.get(tid, 0.0)
        bar   = "█" * int(cp * 30)
        name  = NAME_LOOKUP.get(tid, str(tid))
        w(f"  {name:<26} {confp:>9.1%} {cp:>9.1%}  {bar}")
    w("  " + "=" * 62)
    w(f"  Sanity check — probs sum to: {sum(champ_probs.values()):.6f}\n")


def print_first_round_probs(west_bracket, east_bracket, model,
                            h2h_lookup, feature_cols, w=print):
    w("First-round series win probabilities (higher seed):")
    w("-" * 56)
    for conf, bracket in [("West", west_bracket), ("East", east_bracket)]:
        for h_seed, l_seed in [(1,8),(4,5),(3,6),(2,7)]:
            h = bracket[h_seed]
            l = bracket[l_seed]
            p  = series_win_prob(model, h, l, h2h_lookup, feature_cols)
            hn = NAME_LOOKUP.get(h["TEAM_ID"], "?")
            ln = NAME_LOOKUP.get(l["TEAM_ID"], "?")
            bar = "█" * int(p * 20)
            w(f"  [{conf}] ({h_seed}) {hn:<20} vs ({l_seed}) {ln:<20}  {p:.1%}  {bar}")
    w()


# ── Compare-all mode ──────────────────────────────────────────────────────────

def run_compare_all(west_bracket_base, east_bracket_base,
                    team_features_full, h2h_lookup, w=print):
    """
    Train each preset and print a single championship probability table.
    Uses team_features_full which has all possible columns pre-computed.
    """
    df = pd.read_csv(DATA_PATH)

    w("\n" + "=" * 72)
    w("  2025-26 CHAMPIONSHIP PROBABILITIES — ALL PRESETS")
    w("=" * 72)

    preset_names = list(PRESETS.keys())
    short = lambda n: (n.replace("four_factors","FF")
                        .replace("_clutch","C").replace("_star_full","SF")
                        .replace("_star","S").replace("_top1","T1")
                        .replace("_top3","T3").replace("_",""))
    shorts = [short(n) for n in preset_names]

    header = f"  {'TEAM':<26}"
    for s in shorts:
        header += f" {s:>8}"
    w(header)
    w("  " + "-" * (26 + 9 * len(preset_names)))

    all_probs = {}
    for name in preset_names:
        feature_cols = get_preset(name)
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            all_probs[name] = None
            continue

        model = LogisticRegression(learning_rate=0.1, epochs=1000, lambda_=0.01)
        model.fit(df[feature_cols].values, df["LABEL"].values)

        west = build_bracket(WEST_SEEDS, team_features_full)
        east = build_bracket(EAST_SEEDS, team_features_full)

        champ_probs, _ = simulate_bracket(west, east, model, h2h_lookup, feature_cols)
        all_probs[name] = champ_probs

    all_tids = list(NAME_LOOKUP.keys())
    avg_probs = {}
    for tid in all_tids:
        vals = [p[tid] for p in all_probs.values() if p and tid in p]
        avg_probs[tid] = np.mean(vals) if vals else 0.0

    for tid in sorted(all_tids, key=lambda t: -avg_probs[t]):
        name = NAME_LOOKUP.get(tid, str(tid))
        row = f"  {name:<26}"
        for pname in preset_names:
            p = all_probs.get(pname)
            if p is None:
                row += f" {'N/A':>8}"
            else:
                row += f" {p.get(tid, 0.0):>7.1%}"
        w(row)

    w("  " + "-" * (26 + 9 * len(preset_names)))
    w(f"\n  Column key: FF=four_factors  FFC=+clutch  FFS=+star(1+2)  FFCS=+clutch+star")
    w(f"              FFT1=+top1  FFT3=+top3  FFSF=+top1+2+3  min=minimal  off=offense_only")
    w("=" * 72)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="2025-26 NBA playoff prediction")
    parser.add_argument(
        "--preset", default=DEFAULT_PRESET,
        choices=list(PRESETS.keys()),
        help="Feature preset to use (default: %(default)s)"
    )
    parser.add_argument(
        "--compare-all", action="store_true",
        help="Run all presets and print a single championship probability table"
    )
    parser.add_argument(
        "--list-presets", action="store_true",
        help="Print available feature presets and exit"
    )
    args = parser.parse_args()

    if args.list_presets:
        list_presets()
        return

    print("=" * 60)
    print("  2025-26 NBA Playoff Prediction")
    print("=" * 60 + "\n")

    # In compare-all mode we need all possible columns, so fetch everything
    need_clutch = args.compare_all or requires_clutch(args.preset)
    need_star   = args.compare_all or requires_star(args.preset)

    team_features, h2h_lookup = load_current_season_data(need_clutch, need_star)

    # Verify all 16 playoff teams are present
    all_playoff_ids = list(WEST_SEEDS.values()) + list(EAST_SEEDS.values())
    found_ids = set(team_features["TEAM_ID"].astype(int))
    missing = [tid for tid in all_playoff_ids if tid not in found_ids]
    if missing:
        print("WARNING: Missing team IDs in fetched data:")
        for tid in missing:
            print(f"  {tid} → {NAME_LOOKUP.get(tid, '?')}")
        print("Check that nba_api has complete 2025-26 data and retry.\n")
        return

    if args.compare_all:
        w = Writer()
        run_compare_all(WEST_SEEDS, EAST_SEEDS, team_features, h2h_lookup, w)
        w.save(os.path.join(PREDICTIONS_DIR, "comparison_all_presets.txt"))
        return

    # ── Single preset full output ─────────────────────────────────────────────
    feature_cols = get_preset(args.preset)
    print(f"Preset: {args.preset} ({len(feature_cols)} features)\n")

    model = train_model(feature_cols)

    west_bracket = build_bracket(WEST_SEEDS, team_features)
    east_bracket = build_bracket(EAST_SEEDS, team_features)

    seed_lookup = {}
    for seed, tid in WEST_SEEDS.items():
        seed_lookup[tid] = seed
    for seed, tid in EAST_SEEDS.items():
        seed_lookup[tid] = seed

    w = Writer()

    w("=" * 60)
    w(f"  2025-26 NBA Playoff Prediction — {args.preset}")
    w("=" * 60)

    print_bracket(west_bracket, east_bracket, model, h2h_lookup,
                  feature_cols, seed_lookup, w)

    w("Simulating full bracket...\n")
    champ_probs, conf_probs = simulate_bracket(
        west_bracket, east_bracket, model, h2h_lookup, feature_cols
    )

    w("\nChampionship probabilities:")
    print_champ_table(champ_probs, conf_probs, w)

    print_first_round_probs(west_bracket, east_bracket, model,
                            h2h_lookup, feature_cols, w)

    w("\nFeature weights (normalized scale):")
    w(f"  {'feature':<28} weight")
    w(f"  {'-'*40}")
    for feat, wt in sorted(zip(feature_cols, model.w), key=lambda x: -abs(x[1])):
        bar  = "█" * int(abs(wt) * 20)
        sign = "+" if wt >= 0 else "-"
        w(f"  {feat:<28} {sign}{abs(wt):.4f}  {bar}")
    w(f"  {'bias':<28} {model.b:+.4f}")

    w.save(os.path.join(PREDICTIONS_DIR, f"prediction_{args.preset}.txt"))


if __name__ == "__main__":
    main()