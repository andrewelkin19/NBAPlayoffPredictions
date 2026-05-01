"""
2023-24 Eastern Conference Bracket — Probability Flow Diagram
CS830 Final Project — Andrew Elkin

Shows all possible bracket paths with probability-weighted line thickness.
The model marginalizes over ALL 2^3 = 8 possible East bracket outcomes for
each team, not just the most likely path.

Run from project root:
    python east_bracket_diagram.py

Outputs:
    east_bracket_diagram.png  — 300 DPI for PowerPoint
    east_bracket_diagram.pdf  — vector for LaTeX
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'

from logistic_regression import LogisticRegression
from feature_sets import get_preset
from bracket_simulator import load_season_data, load_playoff_bracket, series_win_prob

# ── Config ────────────────────────────────────────────────────────────────────
SEASON    = "2023-24"
PRESET    = "four_factors_ts_top1"
DATA_PATH = "data/training_data.csv"

# Colours
PURPLE       = '#534AB7'
PURPLE_LIGHT = '#EEEDFE'
PURPLE_TEXT  = '#26215C'
TEAL         = '#0F6E56'
TEAL_LIGHT   = '#E1F5EE'
GRAY         = '#888780'
GRAY_LIGHT   = '#F1EFE8'
GRAY_MED     = '#D3D1C7'
AMBER        = '#BA7517'
AMBER_LIGHT  = '#FAEEDA'
AMBER_TEXT   = '#412402'
TEXT_DARK    = '#2C2C2A'
TEXT_MED     = '#5F5E5A'
TEXT_LIGHT   = '#888780'

# ── Train model on all seasons except holdout ─────────────────────────────────
print("Training model on 2005-2023 seasons (excluding 2023-24)...")
df           = pd.read_csv(DATA_PATH)
train_df     = df[df["SEASON"] != SEASON]
feature_cols = get_preset(PRESET)

model = LogisticRegression(learning_rate=0.1, epochs=1000, lambda_=0.01)
model.fit(train_df[feature_cols].values, train_df["LABEL"].values)
print("  Done.\n")

team_stats, h2h = load_season_data(SEASON, path=DATA_PATH)
east, west, full_names = load_playoff_bracket(SEASON, team_stats)

ABBR = {1:"Celtics", 2:"Knicks", 3:"Bucks", 4:"Cavaliers",
        5:"Magic",   6:"Pacers", 7:"76ers", 8:"Heat"}

# ── Compute all series win probabilities ──────────────────────────────────────
print("Computing probabilities for all possible East bracket matchups...")

def sw(h, l):
    """P(seed h wins) vs seed l.  h must be < l (h is higher seed = home court)."""
    return series_win_prob(model, east[h], east[l], h2h, feature_cols)

def wp(s1, s2):
    """P(s1 wins) vs s2, handling home court automatically."""
    h, l = (s1, s2) if s1 < s2 else (s2, s1)
    p = sw(h, l)
    return p if s1 == h else 1.0 - p

# R1: 1v8, 4v5, 2v7, 3v6
r1 = {}
for h, l in [(1,8), (4,5), (2,7), (3,6)]:
    p = sw(h, l)
    r1[h] = p
    r1[l] = 1.0 - p

# R2: all possible matchups within each half
# Top half: (1 or 8) vs (4 or 5)
# Bottom half: (2 or 7) vs (3 or 6)
r2 = {}
for s1 in [1, 8]:
    for s2 in [4, 5]:
        r2[(s1, s2)] = wp(s1, s2)
for s1 in [2, 7]:
    for s2 in [3, 6]:
        r2[(s1, s2)] = wp(s1, s2)

def r2wp(s1, s2):
    return r2.get((s1, s2), 1.0 - r2.get((s2, s1), 0.5))

# P(reaches CF) for each seed — marginalizes over all possible R2 opponents
p_cf = {}
for s in [1, 8, 4, 5]:
    opps = [4, 5] if s in [1, 8] else [1, 8]
    p_cf[s] = sum(r1[s] * r1[opp] * r2wp(s, opp) for opp in opps)
for s in [2, 7, 3, 6]:
    opps = [3, 6] if s in [2, 7] else [2, 7]
    p_cf[s] = sum(r1[s] * r1[opp] * r2wp(s, opp) for opp in opps)

# CF: all possible matchups (top half team vs bottom half team)
cf = {}
for s1 in [1, 8, 4, 5]:
    for s2 in [2, 7, 3, 6]:
        cf[(s1, s2)] = wp(s1, s2)

# P(wins East) for each seed — marginalizes over all possible CF opponents
p_east = {}
for s in range(1, 9):
    if s in [1, 8, 4, 5]:
        cf_opps = [2, 7, 3, 6]
        def p_beat_opp(seed, opp): return cf[(seed, opp)]
    else:
        cf_opps = [1, 8, 4, 5]
        def p_beat_opp(seed, opp): return 1.0 - cf[(opp, seed)]
    p_east[s] = sum(
        p_cf[s] * p_cf[opp] * p_beat_opp(s, opp)
        for opp in cf_opps
    )

print(f"\n  {'Seed':<5} {'Team':<12} {'P(R2)':>8} {'P(CF)':>8} {'P(Win East)':>12}")
print("  " + "-"*48)
for s in [1, 8, 4, 5, 2, 7, 3, 6]:
    print(f"  {s:<5} {ABBR[s]:<12} {r1[s]:>8.1%} {p_cf[s]:>8.1%} {p_east[s]:>12.1%}")

# ── Layout constants ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 8.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
fig.patch.set_facecolor('white')

BH   = 0.062   # R1 box height
BW1  = 0.150   # R1 box width
BW2  = 0.175   # later-round box width
MAX_LW = 5.5   # line width at 100% probability

# Column x positions (left edge of boxes)
X1 = 0.01    # R1
X2 = 0.27    # Semis
X3 = 0.54    # CF
X4 = 0.80    # East winner

# Bracket junction x values
JX1 = X1 + BW1 + 0.022
JX2 = X2 + BW2 + 0.022
JX3 = X3 + BW2 + 0.022

# Team y-centres in bracket order: 1,8 | 4,5 || 2,7 | 3,6
Y = {1:0.900, 8:0.800, 4:0.678, 5:0.578,
     2:0.433, 7:0.333, 3:0.211, 6:0.111}

# Series midpoints (for bracket stubs)
MID_R1 = {(1,8):(Y[1]+Y[8])/2, (4,5):(Y[4]+Y[5])/2,
           (2,7):(Y[2]+Y[7])/2, (3,6):(Y[3]+Y[6])/2}

# Half midpoints (for R2 → CF connector)
TOP_MID = (MID_R1[(1,8)] + MID_R1[(4,5)]) / 2   # ≈ 0.712
BOT_MID = (MID_R1[(2,7)] + MID_R1[(3,6)]) / 2   # ≈ 0.247
CF_MID  = (TOP_MID + BOT_MID) / 2                # ≈ 0.480

# ── Drawing helpers ───────────────────────────────────────────────────────────
def box(x0, y0, w, h, rows, fc=GRAY_LIGHT, ec=GRAY_MED, alpha=1.0):
    """
    rows: list of (text, fontsize, color, weight, row_alpha)
    Rows are distributed evenly inside the box.
    """
    rect = FancyBboxPatch((x0, y0), w, h,
                          boxstyle="round,pad=0.009",
                          facecolor=fc, edgecolor=ec,
                          linewidth=0.9, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    n = len(rows)
    for i, (txt, fs, tc, fw, ra) in enumerate(rows):
        yc = y0 + h * (n - i) / (n + 1)
        ax.text(x0 + w/2, yc, txt, ha='center', va='center',
                fontsize=fs, color=tc, fontweight=fw, alpha=ra, zorder=4)

def hline(x1, x2, y, lw, color=TEXT_MED, alpha=1.0):
    ax.plot([x1, x2], [y, y], color=color, linewidth=lw,
            solid_capstyle='round', alpha=alpha, zorder=2)

def vline(x, y1, y2, lw, color=TEXT_MED, alpha=0.35):
    ax.plot([x, x], [y1, y2], color=color, linewidth=lw,
            solid_capstyle='round', alpha=alpha, zorder=2)

# ── Team color palette — official NBA primary colors ───────────────────────
# Each tuple is (face_color, edge_color, text_color).
# Face = light tint for readability; edge = official primary; text = dark shade.
TEAM_COLORS = {
    1: ('#C8EBDA', '#007A33', '#003D1A'),   # Celtics — green (richer face)
    2: ('#FDEBD0', '#C45C00', '#622E00'),   # Knicks — orange
    3: ('#8AB89A', '#00471B', '#00200C'),   # Bucks — deeper green face
    4: ('#F2E0D0', '#7B3F00', '#3D1F00'),   # Cavaliers — richer wine face
    5: ('#CCF0FF', '#009BDE', '#004F73'),   # Magic — deeper blue, distinct edge
    6: ('#FEF9E8', '#FDBB30', '#7A5500'),   # Pacers — gold
    7: ('#D0E4F5', '#006BB6', '#003460'),   # 76ers — blue (lighter face than Magic)
    8: ('#F0D0DA', '#98002E', '#4A0016'),   # Heat — richer red face
}
UNIFORM_LW = 2.2

# ── R1 team boxes ─────────────────────────────────────────────────────────────
for s in [1, 8, 4, 5, 2, 7, 3, 6]:
    fc, ec, tc = TEAM_COLORS[s]
    fw = 'semibold'
    box(X1, Y[s]-BH/2, BW1, BH,
        [(f"({s}) {ABBR[s]}", 10, tc, fw, 1.0)],
        fc=fc, ec=ec)

# ── R1 → R2 bracket lines (uniform weight) ────────────────────────────────────
for (h, l) in [(1,8), (4,5), (2,7), (3,6)]:
    y_mid = MID_R1[(h, l)]
    _, ec_h, _ = TEAM_COLORS[h]
    _, ec_l, _ = TEAM_COLORS[l]

    hline(X1+BW1, JX1, Y[h], UNIFORM_LW, color=ec_h, alpha=0.9)
    hline(X1+BW1, JX1, Y[l], UNIFORM_LW, color=ec_l, alpha=0.55)
    vline(JX1, Y[h], y_mid, UNIFORM_LW, color=ec_h, alpha=0.7)  # upper half — higher seed
    vline(JX1, y_mid, Y[l], UNIFORM_LW, color=ec_l, alpha=0.55) # lower half — lower seed
    hline(JX1, X2, y_mid, UNIFORM_LW, color=ec_h, alpha=0.8)

# ── R2 "split boxes" — coloured by most likely occupant ──────────────────────
for (h, l) in [(1,8), (4,5), (2,7), (3,6)]:
    ph, pl = r1[h], r1[l]
    y0    = MID_R1[(h, l)] - BH/2 - 0.008
    h_box = BH + 0.016

    fc, ec, tc_h = TEAM_COLORS[h]
    box(X2, y0, BW2, h_box,
        [(f"{ABBR[h]}  {ph:.0%}", 10, tc_h,      'bold', 1.0),
         (f"{ABBR[l]}  {pl:.0%}", 10, TEXT_MED,   'semibold',  1.0)],
        fc=fc, ec=ec)

# ── R2 → CF bracket lines (uniform weight) ────────────────────────────────────
for half, pairs in [('top', [(1,8),(4,5)]), ('bot', [(2,7),(3,6)])]:
    half_mid = TOP_MID if half == 'top' else BOT_MID
    for (h, l) in pairs:
        _, ec_h, _ = TEAM_COLORS[h]
        hline(X2+BW2, JX2, MID_R1[(h, l)], UNIFORM_LW, color=ec_h, alpha=0.75)

    # Vertical splits at midpoint: each series' higher seed colour runs to centre
    half_junc_mid = (MID_R1[pairs[0]] + MID_R1[pairs[1]]) / 2
    _, ec_p0, _ = TEAM_COLORS[pairs[0][0]]
    _, ec_p1, _ = TEAM_COLORS[pairs[1][0]]
    vline(JX2, MID_R1[pairs[0]], half_junc_mid, UNIFORM_LW, color=ec_p0, alpha=0.7)
    vline(JX2, half_junc_mid, MID_R1[pairs[1]], UNIFORM_LW, color=ec_p1, alpha=0.7)
    # Stub to CF coloured by most likely team to reach CF from this half
    all_seeds = [s for pair in pairs for s in pair]
    top_seed  = max(all_seeds, key=lambda s: p_cf[s])
    _, ec_top, _ = TEAM_COLORS[top_seed]
    hline(JX2, X3, half_mid, UNIFORM_LW, color=ec_top, alpha=0.75)

# ── CF "split boxes" — coloured by most likely occupant ──────────────────────
for half, seeds, y_center in [('top', [1,8,4,5], TOP_MID),
                                ('bot', [2,7,3,6], BOT_MID)]:
    sorted_seeds = sorted(seeds, key=lambda s: p_cf[s], reverse=True)
    h_box = BH * 2.4
    y0    = y_center - h_box / 2

    fc, ec, _ = TEAM_COLORS[sorted_seeds[0]]

    rows = []
    for i, s in enumerate(sorted_seeds):
        tc = TEAM_COLORS[s][2] if i == 0 else TEXT_MED
        fw = 'bold' if i == 0 else 'semibold'
        ra = 1.0 if i == 0 else max(0.65, min(0.85, p_cf[s] * 4))
        rows.append((f"{ABBR[s]}  {p_cf[s]:.0%}", 10, tc, fw, 1.0))

    box(X3, y0, BW2, h_box, rows, fc=fc, ec=ec)

# ── CF → Winner bracket lines (uniform weight) ────────────────────────────────
for seeds, y_mid in [([1,8,4,5], TOP_MID), ([2,7,3,6], BOT_MID)]:
    top_seed = max(seeds, key=lambda s: p_cf[s])
    _, ec_top, _ = TEAM_COLORS[top_seed]
    hline(X3+BW2, JX3, y_mid, UNIFORM_LW, color=ec_top, alpha=0.7)

# CF vertical splits at midpoint: top-half colour above, bottom-half below
top_cf_seed = max([1,8,4,5], key=lambda s: p_cf[s])
bot_cf_seed = max([2,7,3,6], key=lambda s: p_cf[s])
_, ec_top_cf, _ = TEAM_COLORS[top_cf_seed]
_, ec_bot_cf, _ = TEAM_COLORS[bot_cf_seed]
vline(JX3, TOP_MID, CF_MID, UNIFORM_LW, color=ec_top_cf, alpha=0.7)
vline(JX3, CF_MID,  BOT_MID, UNIFORM_LW, color=ec_bot_cf, alpha=0.7)

winner_seed = max(range(1, 9), key=lambda s: p_east[s])
_, ec_win, _ = TEAM_COLORS[winner_seed]
hline(JX3, X4, CF_MID, UNIFORM_LW, color=ec_win, alpha=0.7)

# ── East Winner box — coloured by most likely winner ─────────────────────────
top4   = sorted(range(1, 9), key=lambda s: p_east[s], reverse=True)[:4]
h_win  = BH * 3.6
y0_win = CF_MID - h_win / 2
fc_w, ec_w, _ = TEAM_COLORS[top4[0]]

rows = []
for i, s in enumerate(top4):
    tc = TEAM_COLORS[s][2] if i == 0 else TEXT_MED
    fw = 'bold' if i == 0 else 'semibold'
    ra = 1.0 if i == 0 else max(0.65, min(0.85, p_east[s] * 5))
    rows.append((f"{ABBR[s]}  {p_east[s]:.0%}", 10, tc, fw, 1))

box(X4, y0_win, BW2, h_win, rows, fc=fc_w, ec=ec_w)

# ── Column headers ────────────────────────────────────────────────────────────
for x, label in [(X1+BW1/2, "Round 1"),
                 (X2+BW2/2, "Conf. Semis"),
                 (X3+BW2/2, "Conf. Finals"),
                 (X4+BW2/2, "East Winner")]:
    ax.text(x, 0.97, label, ha='center', va='center',
            fontsize=12, color='BLACK', zorder=4)
    ax.plot([x-0.06, x+0.06], [0.96, 0.96],
            color='BLACK', linewidth=0.7)

# ── Bottom annotation ─────────────────────────────────────────────────────────
ann_box = FancyBboxPatch(
    (0.01, 0.01), 0.98, 0.045,
    boxstyle="round,pad=0.007",
    facecolor=GRAY_LIGHT, edgecolor=GRAY_MED,
    linewidth=0.7, zorder=3
)
ax.add_patch(ann_box)
ax.text(0.50, 0.031,
        "% = LR Model's calculated probability of each team reaching that round, "
        "taking into account the probabilities of all possible previous series outcomes.",
        ha='center', va='center', fontsize=7.5, color=TEXT_DARK, zorder=4)

# ── Export ────────────────────────────────────────────────────────────────────
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('figures/east_bracket_diagram.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('figures/east_bracket_diagram.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("\nSaved: east_bracket_diagram.png  east_bracket_diagram.pdf")