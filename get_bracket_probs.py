"""
Extracts real model series win probabilities for the 2023-24 Eastern Conference
semis → finals, including both possible East Finals matchups.

Run from your project root:
  python get_east_bracket_probs.py
"""

import pandas as pd
from logistic_regression import LogisticRegression
from feature_sets import get_preset
from bracket_simulator import load_season_data, load_playoff_bracket, series_win_prob

SEASON    = "2023-24"
PRESET    = "four_factors_ts_top1"
DATA_PATH = "data/training_data.csv"

# ── Train on all other seasons ────────────────────────────────────────────────
df           = pd.read_csv(DATA_PATH)
train_df     = df[df["SEASON"] != SEASON]
feature_cols = get_preset(PRESET)

model = LogisticRegression(learning_rate=0.1, epochs=1000, lambda_=0.01)
model.fit(train_df[feature_cols].values, train_df["LABEL"].values)

# ── Load 2023-24 bracket ──────────────────────────────────────────────────────
team_stats, h2h  = load_season_data(SEASON, path=DATA_PATH)
east, west, names = load_playoff_bracket(SEASON, team_stats)

def name(team): return names[team["TEAM_ID"]]
def wp(p):      return f"{p:.1%}"

# ── Semis ─────────────────────────────────────────────────────────────────────
p_semi1 = series_win_prob(model, east[1], east[4], h2h, feature_cols)
p_semi2 = series_win_prob(model, east[2], east[3], h2h, feature_cols)

# ── East Finals — BOTH possible matchups ─────────────────────────────────────
p_vs_knicks = series_win_prob(model, east[1], east[2], h2h, feature_cols)
p_vs_bucks  = series_win_prob(model, east[1], east[3], h2h, feature_cols)

# ── Full path-dependent probability ──────────────────────────────────────────
p_celtics_east = (p_semi1 *
                  (p_semi2 * p_vs_knicks + (1 - p_semi2) * p_vs_bucks))

print(f"\nEASTERN CONFERENCE  {SEASON}  [{PRESET}]")
print(f"{'─'*56}")
print(f"\nSemifinals")
print(f"  {name(east[1])} vs {name(east[4])}")
print(f"  → Celtics win:  {wp(p_semi1)}")
print(f"")
print(f"  {name(east[2])} vs {name(east[3])}")
print(f"  → Knicks win:   {wp(p_semi2)}")
print(f"  → Bucks win:    {wp(1 - p_semi2)}")
print(f"\nConference Finals — two possible matchups")
print(f"  if vs {name(east[2])}: Celtics win {wp(p_vs_knicks)}")
print(f"  if vs {name(east[3])}: Celtics win {wp(p_vs_bucks)}")
print(f"\nPath-dependent probability")
print(f"  P(Celtics win East)")
print(f"  = {wp(p_semi1)} × ({wp(p_semi2)}×{wp(p_vs_knicks)} + {wp(1-p_semi2)}×{wp(p_vs_bucks)})")
print(f"  = {wp(p_celtics_east)}")