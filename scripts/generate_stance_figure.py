#!/usr/bin/env python3
"""Create figure showing narrative stance shift pattern.

Matches style of fig_narrative_normalization.pdf:
- Positive = increase in rewrites, negative = decrease
- Grayscale shading for categories
- Diverging horizontal bar chart
"""

import matplotlib.pyplot as plt
import numpy as np

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Data: positive = increase in rewrites, negative = decrease
# (flipped from raw d to match paper convention)
markers = [
    'Abstraction density',
    'Causal connectives',
    'Retrospective framing',
    'Eventive clause density',
    'First-person eventive',
]
# Sorted by effect size (largest positive to largest negative)
effect_sizes = [+0.58, -0.47, -0.49, -0.31, -0.30]

# Grayscale shading: darkest = explicit experiential, medium = explicit explanatory, lightest = implicit
# Category order matches markers above
grays = ['0.75', '0.45', '0.45', '0.15', '0.15']  # light gray for abstract, medium for explanatory, dark for experiential

fig, ax = plt.subplots(figsize=(4.5, 2.8))

y_pos = np.arange(len(markers))
bars = ax.barh(y_pos, effect_sizes, color=grays, edgecolor='black', linewidth=0.5, height=0.7)

# Zero line
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

# Threshold lines at +/- 0.5 (medium effect)
ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
ax.axvline(x=-0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Labels
ax.set_yticks(y_pos)
ax.set_yticklabels(markers, fontsize=9)
ax.set_xlabel("Cohen's $d$")
ax.set_xlim(-0.7, 0.8)

# Direction labels
ax.text(-0.65, -0.7, '← decrease', fontsize=8, ha='left', color='gray', style='italic')
ax.text(0.65, -0.7, 'increase →', fontsize=8, ha='right', color='gray', style='italic')

# Legend for grayscale
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='0.15', edgecolor='black', label='Explicit experiential'),
    Patch(facecolor='0.45', edgecolor='black', label='Explicit explanatory'),
    Patch(facecolor='0.75', edgecolor='black', label='Implicit/Abstract'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=7, framealpha=0.9)

# Invert y-axis so largest effect at top
ax.invert_yaxis()

plt.tight_layout()

# Save
from pathlib import Path
out_dir = Path(__file__).parent.parent / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / 'fig_stance_shift.pdf', format='pdf')
plt.savefig(out_dir / 'fig_stance_shift.png', format='png', dpi=300)
print(f"Saved: {out_dir / 'fig_stance_shift.pdf'}")
plt.close()
