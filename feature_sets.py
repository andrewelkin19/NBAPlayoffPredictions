"""
NBA Playoff Outcome Prediction - Feature Set Presets
CS830 Final Project - Andrew Elkin

Single source of truth for feature column lists. Every preset is a named
list of column names that must exist in training_data.csv. Swap presets
by name anywhere in the codebase — no other changes required.

Adding a new feature:
  1. Compute it in DataScrape.py and add a DIFF_ column to training_data.csv
  2. Define or extend a preset below
  3. Re-run DataScrape.py to regenerate training_data.csv
"""

# ── Preset definitions ────────────────────────────────────────────────────────

# Dean Oliver's Four Factors (offense + defense) + head-to-head record.
# Baseline feature set — all features are differentials (home minus away).
FOUR_FACTORS = [
    "DIFF_EFG_PCT",      # offensive effective FG% (3s count 1.5×)
    "DIFF_OPP_EFG_PCT",  # defensive eFG% allowed
    "DIFF_TOV_PCT",      # turnover rate per possession
    "DIFF_OPP_TOV_PCT",  # forced turnover rate
    "DIFF_ORB_PCT",      # offensive rebound rate
    "DIFF_DRB_PCT",      # defensive rebound rate
    "DIFF_FTR",          # free throw attempt rate
    "DIFF_OPP_FTR",      # opponent free throw rate allowed
    "H2H_WIN_PCT",       # regular-season head-to-head win rate (home team's)
]

# Four Factors + clutch performance.
# Clutch = last 5 minutes of a game within 5 points (NBA standard definition).
# Tests whether teams that perform better in high-pressure regular-season moments
# carry that edge into the playoffs.
FOUR_FACTORS_CLUTCH = FOUR_FACTORS + [
    "DIFF_CLUTCH_WIN_PCT",    # win rate in clutch situations
    "DIFF_CLUTCH_NET_RATING", # net scoring margin (pts/game) in clutch situations
]

# Minimal set: only the two shooting efficiency factors + H2H.
# Useful as a lower bound — how much does adding more features actually help?
MINIMAL = [
    "DIFF_EFG_PCT",
    "DIFF_OPP_EFG_PCT",
    "H2H_WIN_PCT",
]

# Offense-only Four Factors (no defensive factors).
# Isolates whether offensive efficiency alone is sufficient.
OFFENSE_ONLY = [
    "DIFF_EFG_PCT",
    "DIFF_TOV_PCT",
    "DIFF_ORB_PCT",
    "DIFF_FTR",
    "H2H_WIN_PCT",
]

# Four Factors + star player impact (top 1 and top 2 average).
FOUR_FACTORS_STAR = FOUR_FACTORS + [
    "DIFF_TOP1_PM",
    "DIFF_TOP2_AVG_PM",
]

# Four Factors + only the single best player's plus/minus.
# Tests whether one transcendent star drives playoff outcomes,
# versus needing a strong second option (compare to four_factors_star).
FOUR_FACTORS_TOP1 = FOUR_FACTORS + [
    "DIFF_TOP1_PM",
]

# Four Factors + average plus/minus of top 3 players.
# Tests whether roster depth (3-man core) predicts better than a 2-man core.
FOUR_FACTORS_TOP3 = FOUR_FACTORS + [
    "DIFF_TOP3_AVG_PM",
]

# Four Factors + all three star depth variants together.
FOUR_FACTORS_STAR_FULL = FOUR_FACTORS + [
    "DIFF_TOP1_PM",
    "DIFF_TOP2_AVG_PM",
    "DIFF_TOP3_AVG_PM",
]

# Kitchen-sink: Four Factors + clutch + star players.
FOUR_FACTORS_CLUTCH_STAR = FOUR_FACTORS + [
    "DIFF_CLUTCH_WIN_PCT",
    "DIFF_CLUTCH_NET_RATING",
    "DIFF_TOP1_PM",
    "DIFF_TOP2_AVG_PM",
]

# Four Factors + clutch + top 1 only.
FOUR_FACTORS_CLUTCH_TOP1 = FOUR_FACTORS + [
    "DIFF_CLUTCH_WIN_PCT",
    "DIFF_CLUTCH_NET_RATING",
    "DIFF_TOP1_PM",
]

# Four Factors + clutch + top 3 average.
FOUR_FACTORS_CLUTCH_TOP3 = FOUR_FACTORS + [
    "DIFF_CLUTCH_WIN_PCT",
    "DIFF_CLUTCH_NET_RATING",
    "DIFF_TOP3_AVG_PM",
]

# Everything: Four Factors + clutch + top 1, top 2, top 3.
FOUR_FACTORS_CLUTCH_STAR_FULL = FOUR_FACTORS + [
    "DIFF_CLUTCH_WIN_PCT",
    "DIFF_CLUTCH_NET_RATING",
    "DIFF_TOP1_PM",
    "DIFF_TOP2_AVG_PM",
    "DIFF_TOP3_AVG_PM",
]

FOUR_FACTORS_TS = [
    "DIFF_TS_PCT",
    "DIFF_OPP_TS_PCT",
    "DIFF_TOV_PCT",
    "DIFF_OPP_TOV_PCT",
    "DIFF_ORB_PCT",
    "DIFF_DRB_PCT",
    "H2H_WIN_PCT",
]

FOUR_FACTORS_TS_STAR_FULL = FOUR_FACTORS_TS + [
    "DIFF_TOP1_PM",
    "DIFF_TOP2_AVG_PM",
    "DIFF_TOP3_AVG_PM",
]

# ── BPM variants ──────────────────────────────────────────────────────────────
# These mirror the PM presets but use Box Plus/Minus from Basketball Reference
# instead of raw per-game plus/minus. BPM adjusts for team context and role,
# making it a cleaner individual quality signal than raw PM.
# Requires running scrape_bpm.py first to populate data/bpm/.

FOUR_FACTORS_BPM_TOP1 = FOUR_FACTORS + [
    "DIFF_TOP1_BPM",
]

FOUR_FACTORS_BPM_TOP3 = FOUR_FACTORS + [
    "DIFF_TOP3_AVG_BPM",
]

FOUR_FACTORS_BPM_FULL = FOUR_FACTORS + [
    "DIFF_TOP1_BPM",
    "DIFF_TOP2_AVG_BPM",
    "DIFF_TOP3_AVG_BPM",
]

PRESETS: dict[str, list[str]] = {
    "four_factors":                  FOUR_FACTORS,
    "four_factors_clutch":           FOUR_FACTORS_CLUTCH,
    "four_factors_star":             FOUR_FACTORS_STAR,
    "four_factors_top1":             FOUR_FACTORS_TOP1,
    "four_factors_top3":             FOUR_FACTORS_TOP3,
    "four_factors_star_full":        FOUR_FACTORS_STAR_FULL,
    "four_factors_clutch_star":      FOUR_FACTORS_CLUTCH_STAR,
    "four_factors_clutch_top1":      FOUR_FACTORS_CLUTCH_TOP1,
    "four_factors_clutch_top3":      FOUR_FACTORS_CLUTCH_TOP3,
    "four_factors_clutch_star_full": FOUR_FACTORS_CLUTCH_STAR_FULL,
    "four_factors_bpm_top1":         FOUR_FACTORS_BPM_TOP1,
    "four_factors_bpm_top3":         FOUR_FACTORS_BPM_TOP3,
    "four_factors_bpm_full":         FOUR_FACTORS_BPM_FULL,
    "minimal":                       MINIMAL,
    "offense_only":                  OFFENSE_ONLY,
    "four_factors_ts":           FOUR_FACTORS_TS,
    "four_factors_ts_star_full": FOUR_FACTORS_TS_STAR_FULL,
}

DESCRIPTIONS: dict[str, str] = {
    "four_factors":                  "Dean Oliver's Four Factors + H2H (9 features) [default]",
    "four_factors_clutch":           "Four Factors + H2H + clutch win% and net rating (11 features)",
    "four_factors_star":             "Four Factors + H2H + top 1 and top 2 avg player PM (11 features)",
    "four_factors_top1":             "Four Factors + H2H + best player PM only (10 features)",
    "four_factors_top3":             "Four Factors + H2H + top 3 avg player PM (10 features)",
    "four_factors_star_full":        "Four Factors + H2H + top 1, top 2 avg, top 3 avg PM (12 features)",
    "four_factors_clutch_star":      "Four Factors + H2H + clutch + top 1 and top 2 PM (13 features)",
    "four_factors_clutch_top1":      "Four Factors + H2H + clutch + top 1 PM only (12 features)",
    "four_factors_clutch_top3":      "Four Factors + H2H + clutch + top 3 avg PM (12 features)",
    "four_factors_clutch_star_full": "Four Factors + H2H + clutch + top 1, top 2, top 3 PM (14 features)",
    "four_factors_bpm_top1":         "Four Factors + H2H + best player BPM (10 features) [requires BPM data]",
    "four_factors_bpm_top3":         "Four Factors + H2H + top 3 avg BPM (10 features) [requires BPM data]",
    "four_factors_bpm_full":         "Four Factors + H2H + top 1, top 2, top 3 avg BPM (12 features) [requires BPM data]",
    "minimal":                       "eFG% differentials + H2H only (3 features)",
    "offense_only":                  "Offensive Four Factors + H2H, no defensive factors (5 features)",
    "four_factors_ts":           "TS% replaces eFG% in Four Factors (9 features)",
    "four_factors_ts_star_full": "TS% Four Factors + top 1/2/3 PM (12 features)",
}

DEFAULT_PRESET = "four_factors"


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_preset(name: str) -> list[str]:
    """Return the feature column list for a named preset."""
    if name not in PRESETS:
        raise ValueError(
            f"Unknown preset {name!r}. Choose from: {list(PRESETS)}"
        )
    return PRESETS[name]


def list_presets() -> None:
    """Print all available presets with descriptions."""
    print("Available feature presets:")
    for name, desc in DESCRIPTIONS.items():
        n_features = len(PRESETS[name])
        print(f"  {name:<25} {desc}  ({n_features} features)")


def requires_clutch(preset_name: str) -> bool:
    """Return True if this preset needs clutch columns in training_data.csv."""
    return any("CLUTCH" in col for col in PRESETS.get(preset_name, []))


def requires_star(preset_name: str) -> bool:
    """Return True if this preset needs star player columns in training_data.csv."""
    return any("TOP1_PM" in col or "TOP2_AVG_PM" in col
               for col in PRESETS.get(preset_name, []))