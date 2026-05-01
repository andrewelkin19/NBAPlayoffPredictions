"""
Model Description Figure — Logistic Regression
CS830 Final Project — Andrew Elkin

Shows the sigmoid function, model equation, and key implementation details.

Run from project root:
    python model_description.py

Outputs:
    model_description.png
    model_description.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN_DARK  = '#007A33'
GREEN_LIGHT = '#E6F5EC'
GREEN_MED   = '#4CAF82'
GRAY_LIGHT  = '#F1EFE8'
GRAY_MED    = '#D3D1C7'
TEXT_DARK   = '#2C2C2A'
TEXT_MED    = '#5F5E5A'

fig = plt.figure(figsize=(8, 3.8))
fig.patch.set_facecolor('white')

# ── Layout: left = sigmoid curve, right = equation + bullets ──────────────────
ax_sig = fig.add_axes([0.04, 0.18, 0.38, 0.72])   # sigmoid plot
ax_txt = fig.add_axes([0.48, 0.05, 0.52, 0.90])   # text panel
ax_txt.axis('off')

# ── Sigmoid curve ─────────────────────────────────────────────────────────────
x   = np.linspace(-6, 6, 300)
sig = 1 / (1 + np.exp(-x))

ax_sig.plot(x, sig, color=GREEN_DARK, linewidth=2.5, zorder=3)
ax_sig.axhline(0.5, color=GRAY_MED, linewidth=0.8, linestyle='--', zorder=2)
ax_sig.axvline(0.0, color=GRAY_MED, linewidth=0.8, linestyle='--', zorder=2)

# Shade above/below 0.5
ax_sig.fill_between(x, sig, 0.5, where=(sig > 0.5),
                    alpha=0.15, color=GREEN_DARK, zorder=2)
ax_sig.fill_between(x, sig, 0.5, where=(sig < 0.5),
                    alpha=0.10, color=GRAY_MED, zorder=2)

# Annotations — moved P=0.5 to left side for visibility
ax_sig.text(-5.8, 0.53, 'P = 0.5', ha='left', va='bottom',
             fontsize=8.0, color='black', fontweight='semibold')

ax_sig.set_xlim(-6.2, 6.2)
ax_sig.set_ylim(-0.05, 1.05)
ax_sig.set_xlabel("w · x  (weighted feature sum)", fontsize=8.5, color='black')
ax_sig.set_ylabel("P(win)", fontsize=8.5, color='black')
ax_sig.set_title("Sigmoid Function", fontsize=9.5,
                 color='black', fontweight='semibold', pad=6)

ax_sig.tick_params(labelsize=7.5, colors='black')
ax_sig.spines['top'].set_visible(False)
ax_sig.spines['right'].set_visible(False)
ax_sig.spines['left'].set_color('black')
ax_sig.spines['bottom'].set_color('black')
ax_sig.set_facecolor('white')

# ── Right panel: equation + bullets ───────────────────────────────────────────
# Equation box
eq_box = FancyBboxPatch((0.04, 0.68), 0.93, 0.28,
                         boxstyle="round,pad=0.02",
                         facecolor=GREEN_LIGHT, edgecolor=GREEN_DARK,
                         linewidth=1.0, transform=ax_txt.transAxes, zorder=3)
ax_txt.add_patch(eq_box)

ax_txt.text(0.50, 0.875, r'$P(\mathrm{win}) = \sigma(\mathbf{w} \cdot \mathbf{x} + b)$',
            transform=ax_txt.transAxes,
            ha='center', va='center', fontsize=14,
            color=GREEN_DARK, fontweight='semibold', zorder=4)
ax_txt.text(0.50, 0.730,
            r'where $\sigma(z) = \frac{1}{1+e^{-z}}$,  '
            r'$\mathbf{x}$ = feature vector,  $\mathbf{w}$ = learned weights',
            transform=ax_txt.transAxes,
            ha='center', va='center', fontsize=8.5,
            color=TEXT_MED, zorder=4)

# Bullet points
bullets = [
    ("Implemented from scratch in Python with Numpy",
     "gradient descent optimization, L2 regularization"),
    ("Leave-one-season-out cross-validation",
     "17 training seasons, 1 test season, repeated for 18 seasons"),
    ("1,000 epochs per run", ""),
]

y_start = 0.57
dy      = 0.195

for i, (header, sub) in enumerate(bullets):
    y = y_start - i * dy

    # Bullet dot
    ax_txt.plot(0.04, y + 0.025, 'o', color=GREEN_DARK,
                markersize=5, transform=ax_txt.transAxes, zorder=4)

    ax_txt.text(0.10, y + 0.025, header,
                transform=ax_txt.transAxes,
                ha='left', va='center', fontsize=9.5,
                color=TEXT_DARK, fontweight='semibold', zorder=4)
    ax_txt.text(0.10, y - 0.055, sub,
                transform=ax_txt.transAxes,
                ha='left', va='center', fontsize=8.0,
                color=TEXT_MED, zorder=4)

# ── Export ────────────────────────────────────────────────────────────────────
plt.savefig('model_description.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('model_description.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: model_description.png  model_description.pdf")