#!/usr/bin/env python3
"""
Generate high-quality LSTM Cell Architecture diagram for Chapter 8
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch, PathPatch
from matplotlib.path import Path
import numpy as np

# Style settings for high quality
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.usetex'] = False

# Colors
MAIN_BLUE = '#1A3A6E'
ACCENT_BLUE = '#4A90D9'
IDA_RED = '#DC3545'
FOREST = '#2E7D32'
ORANGE = '#E67E22'
PURPLE = '#8E44AD'
LIGHT_BG = '#F8FAFC'

# Create high-resolution figure
fig, ax = plt.subplots(figsize=(16, 8), dpi=150)
ax.set_xlim(-1.5, 14)
ax.set_ylim(-1, 8)
ax.axis('off')
ax.set_aspect('equal')

# Main cell body (rounded rectangle)
cell_rect = FancyBboxPatch((1.5, 1), 10, 5.5, boxstyle="round,pad=0.15,rounding_size=0.5",
                            facecolor=LIGHT_BG, edgecolor=MAIN_BLUE, linewidth=3.5)
ax.add_patch(cell_rect)

# ============ CELL STATE LINE (top conveyor belt) ============
# Main horizontal line
ax.plot([-1, 13.5], [6, 6], color=MAIN_BLUE, lw=5, solid_capstyle='round', zorder=2)
ax.text(-1.3, 6, '$C_{t-1}$', ha='right', va='center', fontsize=16, fontweight='bold', color=MAIN_BLUE)
ax.text(13.8, 6, '$C_t$', ha='left', va='center', fontsize=16, fontweight='bold', color=MAIN_BLUE)
# Arrow at end
ax.annotate('', xy=(13.5, 6), xytext=(12.8, 6),
            arrowprops=dict(arrowstyle='-|>', color=MAIN_BLUE, lw=4, mutation_scale=25))

# ============ HIDDEN STATE LINE ============
ax.plot([-1, 1.5], [2, 2], color=PURPLE, lw=4)
ax.plot([11.5, 13.5], [2, 2], color=PURPLE, lw=4)
ax.text(-1.3, 2, '$h_{t-1}$', ha='right', va='center', fontsize=16, fontweight='bold', color=PURPLE)
ax.text(13.8, 2, '$h_t$', ha='left', va='center', fontsize=16, fontweight='bold', color=PURPLE)
ax.annotate('', xy=(13.5, 2), xytext=(12.8, 2),
            arrowprops=dict(arrowstyle='-|>', color=PURPLE, lw=4, mutation_scale=25))

# ============ HELPER FUNCTIONS ============
def draw_gate(ax, x, y, label, color, width=1.4, height=1.0):
    """Draw a gate box with label"""
    rect = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.08", facecolor=color,
                          edgecolor='white', linewidth=3, alpha=0.95, zorder=5)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=14, fontweight='bold', color='white', zorder=6)

def draw_op(ax, x, y, symbol, color, radius=0.4):
    """Draw a pointwise operation circle"""
    circle = Circle((x, y), radius, facecolor='white', edgecolor=color, linewidth=3, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, symbol, ha='center', va='center', fontsize=15, fontweight='bold', color=color, zorder=6)

def draw_arrow(ax, start, end, color, lw=2.5):
    """Draw an arrow from start to end"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=18))

# ============ GATES ============
gate_y = 3.5

# Forget Gate (red)
draw_gate(ax, 3, gate_y, '$\\sigma$', IDA_RED)
ax.text(3, 2.2, 'Forget\nGate', ha='center', va='center', fontsize=11, color=IDA_RED, fontweight='bold', linespacing=0.9)

# Input Gate (green)
draw_gate(ax, 5.5, gate_y, '$\\sigma$', FOREST)
ax.text(5.5, 2.2, 'Input\nGate', ha='center', va='center', fontsize=11, color=FOREST, fontweight='bold', linespacing=0.9)

# Candidate (orange)
draw_gate(ax, 7.5, gate_y, 'tanh', ORANGE)
ax.text(7.5, 2.2, 'Candidate', ha='center', va='center', fontsize=11, color=ORANGE, fontweight='bold')

# Output Gate (blue)
draw_gate(ax, 10, gate_y, '$\\sigma$', ACCENT_BLUE)
ax.text(10, 2.2, 'Output\nGate', ha='center', va='center', fontsize=11, color=ACCENT_BLUE, fontweight='bold', linespacing=0.9)

# ============ POINTWISE OPERATIONS ============
# Forget multiplication (on cell state)
draw_op(ax, 3, 6, '×', IDA_RED)

# Add operation (on cell state)
draw_op(ax, 6.5, 6, '+', FOREST)

# Multiply for input*candidate
draw_op(ax, 6.5, 4.8, '×', ORANGE, radius=0.35)

# Tanh before output
draw_op(ax, 10, 5, 'tanh', MAIN_BLUE, radius=0.45)

# Output multiplication
draw_op(ax, 10, 2, '×', ACCENT_BLUE)

# ============ CONNECTIONS ============
# Forget gate → multiply on cell state
ax.plot([3, 3], [4.0, 5.6], color=IDA_RED, lw=2.5, zorder=3)
draw_arrow(ax, (3, 5.0), (3, 5.55), IDA_RED, lw=2.5)

# Input gate → multiply with candidate
ax.plot([5.5, 5.5], [4.0, 4.8], color=FOREST, lw=2.5, zorder=3)
ax.plot([5.5, 6.1], [4.8, 4.8], color=FOREST, lw=2.5, zorder=3)

# Candidate → multiply with input gate
ax.plot([7.5, 7.5], [4.0, 4.8], color=ORANGE, lw=2.5, zorder=3)
ax.plot([7.5, 6.9], [4.8, 4.8], color=ORANGE, lw=2.5, zorder=3)

# Multiply result → add on cell state
ax.plot([6.5, 6.5], [5.15, 5.6], color=FOREST, lw=2.5, zorder=3)
draw_arrow(ax, (6.5, 5.3), (6.5, 5.55), FOREST, lw=2.5)

# Cell state → tanh (branch down)
ax.plot([10, 10], [6, 5.45], color=MAIN_BLUE, lw=2.5, zorder=3)
draw_arrow(ax, (10, 5.7), (10, 5.48), MAIN_BLUE, lw=2.5)

# Tanh → output multiply
ax.plot([10, 10], [4.55, 2.4], color=MAIN_BLUE, lw=2.5, zorder=3)
draw_arrow(ax, (10, 3.0), (10, 2.42), MAIN_BLUE, lw=2.5)

# Output gate → output multiply
ax.plot([10, 10], [3.0, 2.4], color=ACCENT_BLUE, lw=2.5, zorder=3, alpha=0)  # Hidden, same path
ax.plot([9.3, 9.3], [3.5, 2], color=ACCENT_BLUE, lw=2.5, zorder=3)
ax.plot([9.3, 9.6], [3.5, 3.5], color=ACCENT_BLUE, lw=2.5, zorder=3)
ax.plot([9.6, 9.6], [2, 2], color=ACCENT_BLUE, lw=2.5, zorder=3)

# Output multiply → h_t
ax.plot([10.4, 11.5], [2, 2], color=PURPLE, lw=4, zorder=3)

# ============ INPUT x_t ============
ax.annotate('', xy=(6, 1), xytext=(6, -0.5),
            arrowprops=dict(arrowstyle='-|>', color=FOREST, lw=3.5, mutation_scale=22))
ax.text(6, -0.8, '$x_t$', ha='center', va='center', fontsize=16, fontweight='bold', color=FOREST)

# Input distribution to all gates (dashed lines)
input_y = 1.2
ax.plot([3, 10], [input_y, input_y], color='gray', lw=2, ls='--', alpha=0.6, zorder=1)
ax.plot([3, 3], [input_y, 3.0], color='gray', lw=2, ls='--', alpha=0.6, zorder=1)
ax.plot([5.5, 5.5], [input_y, 3.0], color='gray', lw=2, ls='--', alpha=0.6, zorder=1)
ax.plot([7.5, 7.5], [input_y, 3.0], color='gray', lw=2, ls='--', alpha=0.6, zorder=1)
ax.plot([10, 10], [input_y, 3.0], color='gray', lw=2, ls='--', alpha=0.6, zorder=1)
ax.plot([6, 6], [input_y, 1], color='gray', lw=2, ls='--', alpha=0.6, zorder=1)

# h_{t-1} distribution (also feeds to gates)
ax.plot([1.5, 3], [2, 2], color='gray', lw=2, ls=':', alpha=0.5, zorder=1)
ax.plot([3, 3], [2, 3.0], color='gray', lw=2, ls=':', alpha=0.5, zorder=1)

# ============ LABELS ============
# Cell state label
ax.text(0.5, 7.2, 'Cell State $C_t$', ha='center', va='center', fontsize=13,
        color=MAIN_BLUE, fontweight='bold', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=MAIN_BLUE, alpha=0.9))
ax.text(0.5, 6.6, '(Long-term Memory)', ha='center', va='center', fontsize=10,
        color=MAIN_BLUE, style='italic')

# Hidden state label
ax.text(12.5, 0.8, 'Hidden State $h_t$', ha='center', va='center', fontsize=13,
        color=PURPLE, fontweight='bold', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=PURPLE, alpha=0.9))
ax.text(12.5, 0.3, '(Short-term Memory)', ha='center', va='center', fontsize=10,
        color=PURPLE, style='italic')

# ============ LEGEND ============
legend_x = -0.5
legend_y = 4.5
legend_spacing = 0.55

# Legend box
legend_box = FancyBboxPatch((legend_x - 0.6, legend_y - 2.4), 2.4, 2.7,
                             boxstyle="round,pad=0.1", facecolor='white',
                             edgecolor='gray', linewidth=1, alpha=0.95)
ax.add_patch(legend_box)

ax.text(legend_x + 0.5, legend_y + 0.1, 'Gates:', ha='center', va='center', fontsize=11, fontweight='bold')

items = [
    (IDA_RED, 'Forget'),
    (FOREST, 'Input'),
    (ORANGE, 'Candidate'),
    (ACCENT_BLUE, 'Output'),
]

for i, (color, label) in enumerate(items):
    y = legend_y - 0.4 - i * legend_spacing
    rect = Rectangle((legend_x - 0.3, y - 0.15), 0.5, 0.3, facecolor=color, edgecolor='white', lw=1.5)
    ax.add_patch(rect)
    ax.text(legend_x + 0.4, y, label, ha='left', va='center', fontsize=10, color=color, fontweight='bold')

plt.tight_layout()
plt.savefig('ch8_lstm_cell.pdf', bbox_inches='tight', transparent=True, dpi=300)
plt.savefig('ch8_lstm_cell.png', bbox_inches='tight', transparent=True, dpi=300)
plt.close()

print("Generated: ch8_lstm_cell.pdf (high quality)")
