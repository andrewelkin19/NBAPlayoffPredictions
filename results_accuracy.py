"""
Results Figure — Game Accuracy + Champion Accuracy
CS830 Final Project — Andrew Elkin

Combined bar chart comparing baselines vs best model on two metrics.

Run from project root:
    python results_accuracy.py

Outputs:
    results_accuracy.png
    results_accuracy.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN_DARK  = '#007A33'
GREEN_MED   = '#4CAF82'
GREEN_LIGHT = '#E6F5EC'
GRAY_MED    = '#D3D1C7'
WINE        = '#860038'
TEXT_DARK   = '#2C2C2A'
TEXT_MED    = '#5F5E5A'

# ── Data ──────────────────────────────────────────────────────────────────────
labels      = ['Higher Seed\nBaseline', 'Home Court\nBaseline', 'LR Classifier']
game_acc    = [0.610, 0.628, 0.665]
champ_acc   = [7/18,  7/18,  10/18]   # baseline same for both rules

x      = np.arange(len(labels))
width  = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
fig.patch.set_facecolor('white')

# ── Left: Game Accuracy ────────────────────────────────────────────────────────
colors_game = [GRAY_MED, GRAY_MED, GREEN_DARK]
bars1 = ax1.bar(x, game_acc, color=colors_game, width=0.5,
                edgecolor='white', linewidth=0.5, zorder=3)

ax1.set_ylim(0.55, 0.72)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9, color='black')
ax1.set_ylabel("Accuracy", fontsize=10, color='black')
ax1.set_title("Single Game Prediction Accuracy", fontsize=11,
              fontweight='semibold', color='black', pad=8)
ax1.yaxis.grid(True, color=GRAY_MED, linewidth=0.5,
               linestyle='--', alpha=0.7, zorder=0)
ax1.set_axisbelow(True)
ax1.tick_params(colors='black', labelsize=9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('black')
ax1.spines['bottom'].set_color('black')
ax1.set_facecolor('white')

# Value labels
for bar, val in zip(bars1, game_acc):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{val:.1%}', ha='center', va='bottom',
             fontsize=9.5, fontweight='semibold', color='black')



# ── Right: Champion Accuracy ───────────────────────────────────────────────────
colors_champ = [GRAY_MED, GRAY_MED, GREEN_DARK]
bars2 = ax2.bar(x, champ_acc, color=colors_champ, width=0.5,
                edgecolor='white', linewidth=0.5, zorder=3)

ax2.set_ylim(0, 0.70)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=9, color='black')
ax2.set_ylabel("Accuracy", fontsize=10, color='black')
ax2.set_title("Champion Prediction Accuracy", fontsize=11,
              fontweight='semibold', color='black', pad=8)
ax2.yaxis.grid(True, color=GRAY_MED, linewidth=0.5,
               linestyle='--', alpha=0.7, zorder=0)
ax2.set_axisbelow(True)
ax2.tick_params(colors='black', labelsize=9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('black')
ax2.spines['bottom'].set_color('black')
ax2.set_facecolor('white')

# Fraction + percentage labels
fractions = ['7/18', '7/18', '10/18']
for bar, val, frac in zip(bars2, champ_acc, fractions):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
             f'{frac}\n({val:.0%})',
             ha='center', va='bottom',
             fontsize=9.0, fontweight='semibold', color='black',
             linespacing=1.3)





plt.tight_layout(pad=1.5)
plt.savefig('results_accuracy.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('results_accuracy.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: results_accuracy.png  results_accuracy.pdf")