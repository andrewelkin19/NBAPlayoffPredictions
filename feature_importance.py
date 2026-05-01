"""
Feature Coefficient Analysis — four_factors_ts_top1
CS830 Final Project — Andrew Elkin

Trains the model on each leave-one-season-out fold, collects the learned
weights, standardizes features so coefficients are comparable across features,
then plots mean coefficient ± std across all 18 folds.

Run from project root:
    python feature_importance.py

Outputs:
    feature_importance.png
    feature_importance.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'

from logistic_regression import LogisticRegression
from feature_sets import get_preset

PRESET    = "four_factors_ts_top1"
DATA_PATH = "data/training_data.csv"

SEASONS = [
    "2005-06", "2006-07", "2007-08", "2008-09", "2009-10",
    "2010-11",            "2012-13", "2013-14", "2014-15",
    "2015-16", "2016-17", "2017-18", "2018-19",
                          "2020-21", "2021-22", "2022-23",
                          "2023-24", "2024-25"
]

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN_DARK  = '#007A33'
GREEN_LIGHT = '#E6F5EC'
GREEN_MED   = '#4CAF82'
GRAY        = '#888780'
GRAY_LIGHT  = '#F1EFE8'
GRAY_MED    = '#D3D1C7'
TEXT_DARK   = '#2C2C2A'
TEXT_MED    = '#5F5E5A'

# ── Feature display names ─────────────────────────────────────────────────────
FEATURE_LABELS = {
    "DIFF_TS_PCT":      "True Shooting %",
    "DIFF_OPP_TS_PCT":  "Opp. True Shooting %",
    "DIFF_TOV_PCT":     "Turnover Rate",
    "DIFF_OPP_TOV_PCT": "Opp. Turnover Rate",
    "DIFF_ORB_PCT":     "Off. Rebound Rate",
    "DIFF_DRB_PCT":     "Def. Rebound Rate",
    "DIFF_TOP1_PM":     "Top-1 Player Plus/Minus",
    "H2H_WIN_PCT":      "Head-to-Head Win %",
}

# ── Train on each fold, collect standardized coefficients ─────────────────────
print(f"Training {len(SEASONS)} folds for {PRESET}...")
df           = pd.read_csv(DATA_PATH)
feature_cols = get_preset(PRESET)

all_coefs = []
for season in SEASONS:
    train_df = df[df["SEASON"] != season]
    if train_df.empty:
        continue

    X = train_df[feature_cols].values
    y = train_df["LABEL"].values

    # Standardize features (zero mean, unit variance) so coefficients
    # are comparable across features with different scales
    mu  = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0  # avoid division by zero
    X_std = (X - mu) / std

    model = LogisticRegression(learning_rate=0.1, epochs=1000, lambda_=0.01)
    model.fit(X_std, y)
    all_coefs.append(model.w)
    print(f"  {season} done")

coefs     = np.array(all_coefs)   # shape: (18, 8)
mean_coef = coefs.mean(axis=0)
std_coef  = coefs.std(axis=0)

print("\nMean standardized coefficients:")
for i, f in enumerate(feature_cols):
    print(f"  {FEATURE_LABELS[f]:<30} {mean_coef[i]:+.4f} ± {std_coef[i]:.4f}")

# ── Sort by absolute mean coefficient ─────────────────────────────────────────
order      = np.argsort(np.abs(mean_coef))
sorted_features = [feature_cols[i] for i in order]
sorted_means    = mean_coef[order]
sorted_stds     = std_coef[order]
labels          = [FEATURE_LABELS[f] for f in sorted_features]
colors          = [GREEN_DARK if v > 0 else '#860038' for v in sorted_means]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

y_pos = np.arange(len(labels))

bars = ax.barh(y_pos, sorted_means, xerr=sorted_stds,
               color=colors, alpha=0.85,
               error_kw=dict(ecolor=GRAY, lw=1.2, capsize=3),
               height=0.6, zorder=3)

# Zero line
ax.axvline(0, color=TEXT_DARK, linewidth=0.8, zorder=4)

# Grid
ax.set_axisbelow(True)
ax.xaxis.grid(True, color=GRAY_MED, linewidth=0.5, linestyle='--', alpha=0.7)

# Labels
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10, color=TEXT_DARK)
ax.set_xlabel("Feature Weight",
              fontsize=9, color=TEXT_MED)
ax.set_title("Average Learned Feature Weights Across All Seasons",
             fontsize=11, fontweight='semibold', color=TEXT_DARK, pad=10)

# Value labels on bars
for i, (v, s) in enumerate(zip(sorted_means, sorted_stds)):
    offset = 0.008
    ha = 'left' if v >= 0 else 'right'
    x  = v + offset if v >= 0 else v - offset
    ax.text(x, i, f'{v:+.3f}', ha=ha, va='center',
            fontsize=8.5, color=TEXT_DARK)

# Caption
fig.text(0.5, -0.04,
         "Features standardized via z-score normalization.",
         ha='center', fontsize=8, color=TEXT_MED, style='italic')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(GRAY_MED)
ax.spines['bottom'].set_color(GRAY_MED)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('feature_importance.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("\nSaved: feature_importance.png  feature_importance.pdf")