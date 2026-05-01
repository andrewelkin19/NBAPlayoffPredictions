"""
2025-26 NBA Championship Probability Predictions
CS830 Final Project — Andrew Elkin

Horizontal bar chart showing championship probabilities for all 16 teams.

Run from project root:
    python bracket_2026.py

Outputs:
    bracket_2026.png
    bracket_2026.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'

GREEN_DARK  = '#007A33'
GREEN_LIGHT = '#E6F5EC'
GRAY_MED    = '#D3D1C7'
TEXT_DARK   = '#2C2C2A'
TEXT_MED    = '#5F5E5A'

# Official team primary colors
TEAM_COLORS = {
    "OKC Thunder":       '#007AC1',
    "SA Spurs":          '#000000',
    "DET Pistons":       '#BF0D3E',
    "DEN Nuggets":       '#0E2240',
    "BOS Celtics":       '#007A33',
    "NYK Knicks":        '#F58426',
    "CLE Cavaliers":     '#860038',
    "HOU Rockets":       '#CE1141',
    "LAL Lakers":        '#552583',
    "ATL Hawks":         '#E03A3E',
    "MIN Timberwolves":  '#005083',
    "TOR Raptors":       '#CE1141',
    "ORL Magic":         '#0077C0',
    "PHI 76ers":         '#006BB6',
    "POR Trail Blazers": '#E03A3E',
    "PHX Suns":          '#1D1160',
}

# All 16 teams sorted by championship probability
teams = [
    ("OKC Thunder",        34.8),
    ("SA Spurs",           29.4),
    ("DET Pistons",        11.9),
    ("DEN Nuggets",         9.7),
    ("BOS Celtics",         6.7),
    ("NYK Knicks",          4.5),
    ("CLE Cavaliers",       1.5),
    ("HOU Rockets",         0.7),
    ("LAL Lakers",          0.4),
    ("ATL Hawks",           0.1),
    ("MIN Timberwolves",    0.1),
    ("TOR Raptors",         0.1),
    ("ORL Magic",           0.1),
    ("PHI 76ers",           0.0),
    ("POR Trail Blazers",   0.0),
    ("PHX Suns",            0.0),
]

# Reverse for horizontal bar (highest at top)
names  = [t[0] for t in teams][::-1]
values = [t[1] for t in teams][::-1]
colors = [TEAM_COLORS.get(n, GRAY_MED) for n in names]

fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

y_pos = np.arange(len(names))
bars  = ax.barh(y_pos, values, color=colors, height=0.6,
                zorder=3, edgecolor='white', linewidth=0.4)

ax.set_yticks(y_pos)
labels = [f'★ {n}' if n == 'OKC Thunder' else n for n in names]
ax.set_yticklabels(labels, fontsize=9.5, color='black')
ax.set_xlabel("Championship Probability (%)", fontsize=10, color='black')
ax.set_title("2025-26 Predicted Championship Probabilities",
             fontsize=11, fontweight='semibold', color='black', pad=10)

# Value labels
for bar, val in zip(bars, values):
    if val >= 0.5:
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', ha='left', va='center',
                fontsize=8.5, color='black', fontweight='semibold')

ax.xaxis.grid(True, color=GRAY_MED, linewidth=0.5,
              linestyle='--', alpha=0.7, zorder=0)
ax.set_axisbelow(True)
ax.set_xlim(0, 42)
ax.tick_params(colors='black', labelsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')



# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
           markersize=11, label='Predicted Champion'),
]
ax.legend(handles=legend_elements, fontsize=8.5, loc='lower right',
          framealpha=0.9, edgecolor=GRAY_MED)

fig.text(0.5, -0.01,
         "Trained on 2005-06 through 2024-25",
         ha='center', fontsize=7.5, color=TEXT_MED, style='italic')

plt.tight_layout()
plt.savefig('bracket_2026.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('bracket_2026.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: bracket_2026.png  bracket_2026.pdf")