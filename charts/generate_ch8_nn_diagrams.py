#!/usr/bin/env python3
"""
Generate cleaner neural network diagrams for Chapter 8
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch, Polygon
import numpy as np

# Style settings
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'

MAIN_BLUE = '#1A3A6E'
ACCENT_BLUE = '#4A90D9'
IDA_RED = '#DC3545'
FOREST = '#2E7D32'
ORANGE = '#E67E22'
PURPLE = '#8E44AD'

# =============================================================================
# Chart 1: RNN Unfolded - Clean horizontal design
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.set_xlim(-0.5, 14)
ax.set_ylim(-0.5, 4)
ax.axis('off')
ax.set_aspect('equal')

# Cell positions
cell_x = [2, 5, 8, 11]
cell_y = 2
radius = 0.7

# Draw cells and connections
for i, cx in enumerate(cell_x):
    # Main cell circle
    circle = Circle((cx, cell_y), radius, facecolor=ACCENT_BLUE, edgecolor=MAIN_BLUE, linewidth=2.5, zorder=3)
    ax.add_patch(circle)

    # Hidden state label inside
    t_label = ['t-2', 't-1', 't', 't+1'][i]
    ax.text(cx, cell_y, f'$h_{{{t_label}}}$', ha='center', va='center', fontsize=13, fontweight='bold', color='white', zorder=4)

    # Input arrow and label (from bottom)
    ax.annotate('', xy=(cx, cell_y - radius - 0.05), xytext=(cx, 0.3),
                arrowprops=dict(arrowstyle='-|>', color=FOREST, lw=2.5, mutation_scale=15))
    x_label = ['t-2', 't-1', 't', 't+1'][i]
    ax.text(cx, 0, f'$x_{{{x_label}}}$', ha='center', va='center', fontsize=12, fontweight='bold', color=FOREST)

    # Output arrow and label (to top)
    ax.annotate('', xy=(cx, 3.7), xytext=(cx, cell_y + radius + 0.05),
                arrowprops=dict(arrowstyle='-|>', color=IDA_RED, lw=2.5, mutation_scale=15))
    y_label = ['t-2', 't-1', 't', 't+1'][i]
    ax.text(cx, 3.95, f'$y_{{{y_label}}}$', ha='center', va='center', fontsize=12, fontweight='bold', color=IDA_RED)

# Horizontal arrows between cells (memory flow)
for i in range(len(cell_x) - 1):
    ax.annotate('', xy=(cell_x[i+1] - radius - 0.1, cell_y), xytext=(cell_x[i] + radius + 0.1, cell_y),
                arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=3, mutation_scale=18))

# Labels
ax.text(0.3, cell_y, '$h_{t-3}$\n...', ha='center', va='center', fontsize=10, color=ORANGE)
ax.annotate('', xy=(cell_x[0] - radius - 0.1, cell_y), xytext=(0.8, cell_y),
            arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=3, mutation_scale=18))

ax.text(13.2, cell_y, '...', ha='center', va='center', fontsize=14, color=ORANGE)
ax.annotate('', xy=(13, cell_y), xytext=(cell_x[-1] + radius + 0.1, cell_y),
            arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=3, mutation_scale=18))

# Title and legend
ax.text(7, -0.7, 'Time $\\rightarrow$', ha='center', va='center', fontsize=11, style='italic')

# Legend
legend_y = 3.7
ax.plot([0.3, 0.8], [legend_y, legend_y], color=FOREST, lw=2.5)
ax.text(1.0, legend_y, 'Input', ha='left', va='center', fontsize=10, color=FOREST)

ax.plot([0.3, 0.8], [legend_y - 0.35, legend_y - 0.35], color=ORANGE, lw=2.5)
ax.text(1.0, legend_y - 0.35, 'Memory', ha='left', va='center', fontsize=10, color=ORANGE)

ax.plot([0.3, 0.8], [legend_y - 0.7, legend_y - 0.7], color=IDA_RED, lw=2.5)
ax.text(1.0, legend_y - 0.7, 'Output', ha='left', va='center', fontsize=10, color=IDA_RED)

plt.tight_layout()
plt.savefig('ch8_rnn_unfolded.pdf', bbox_inches='tight', transparent=True, dpi=300)
plt.close()
print("Generated: ch8_rnn_unfolded.pdf")

# =============================================================================
# Chart 2: LSTM Cell - Professional schematic
# =============================================================================
fig, ax = plt.subplots(figsize=(13, 6))
ax.set_xlim(-1, 13)
ax.set_ylim(-0.5, 7)
ax.axis('off')

# Main cell body (rectangle)
cell_rect = FancyBboxPatch((2, 1.5), 8, 4, boxstyle="round,pad=0.1",
                            facecolor='#F0F4F8', edgecolor=MAIN_BLUE, linewidth=3)
ax.add_patch(cell_rect)

# Cell state line (top highway)
ax.plot([0, 12], [5.5, 5.5], color=MAIN_BLUE, lw=4, solid_capstyle='round')
ax.text(-0.5, 5.5, '$C_{t-1}$', ha='right', va='center', fontsize=13, fontweight='bold', color=MAIN_BLUE)
ax.text(12.5, 5.5, '$C_t$', ha='left', va='center', fontsize=13, fontweight='bold', color=MAIN_BLUE)
# Arrow at end
ax.annotate('', xy=(12.3, 5.5), xytext=(11.8, 5.5),
            arrowprops=dict(arrowstyle='-|>', color=MAIN_BLUE, lw=3, mutation_scale=20))

# Hidden state output line
ax.plot([0, 2], [2, 2], color=PURPLE, lw=3)
ax.plot([10, 12], [2, 2], color=PURPLE, lw=3)
ax.text(-0.5, 2, '$h_{t-1}$', ha='right', va='center', fontsize=13, fontweight='bold', color=PURPLE)
ax.text(12.5, 2, '$h_t$', ha='left', va='center', fontsize=13, fontweight='bold', color=PURPLE)
ax.annotate('', xy=(12.3, 2), xytext=(11.8, 2),
            arrowprops=dict(arrowstyle='-|>', color=PURPLE, lw=3, mutation_scale=20))

# Gate function - creates a gate box
def draw_gate(ax, x, y, label, color, width=1.2, height=0.9):
    rect = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.05", facecolor=color,
                          edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Pointwise operation circles
def draw_op(ax, x, y, symbol, color):
    circle = Circle((x, y), 0.35, facecolor='white', edgecolor=color, linewidth=2.5)
    ax.add_patch(circle)
    ax.text(x, y, symbol, ha='center', va='center', fontsize=14, fontweight='bold', color=color)

# Gates
draw_gate(ax, 3.5, 3.5, '$\\sigma$', IDA_RED)  # Forget gate
ax.text(3.5, 2.4, 'Forget', ha='center', va='center', fontsize=9, color=IDA_RED, fontweight='bold')

draw_gate(ax, 5.5, 3.5, '$\\sigma$', FOREST)  # Input gate
ax.text(5.5, 2.4, 'Input', ha='center', va='center', fontsize=9, color=FOREST, fontweight='bold')

draw_gate(ax, 6.5, 3.5, 'tanh', ORANGE)  # Candidate
ax.text(6.5, 2.4, 'Candidate', ha='center', va='center', fontsize=9, color=ORANGE, fontweight='bold')

draw_gate(ax, 8.5, 3.5, '$\\sigma$', ACCENT_BLUE)  # Output gate
ax.text(8.5, 2.4, 'Output', ha='center', va='center', fontsize=9, color=ACCENT_BLUE, fontweight='bold')

# Pointwise operations on cell state
draw_op(ax, 3.5, 5.5, '×', IDA_RED)  # Forget multiplication
draw_op(ax, 6, 5.5, '+', FOREST)     # Add new info
draw_op(ax, 9, 4.5, 'tanh', MAIN_BLUE)  # tanh before output

# Output multiplication
draw_op(ax, 9, 2, '×', ACCENT_BLUE)

# Connections
# Forget gate to multiply
ax.plot([3.5, 3.5], [4, 5.15], color=IDA_RED, lw=2)
ax.annotate('', xy=(3.5, 5.15), xytext=(3.5, 4.5),
            arrowprops=dict(arrowstyle='-|>', color=IDA_RED, lw=2, mutation_scale=12))

# Input gate and candidate to add
ax.plot([5.5, 5.5], [4, 4.7], color=FOREST, lw=2)
ax.plot([5.5, 5.8], [4.7, 4.7], color=FOREST, lw=2)
ax.plot([6.5, 6.5], [4, 4.7], color=ORANGE, lw=2)
ax.plot([6.5, 6.2], [4.7, 4.7], color=ORANGE, lw=2)
# Multiply input*candidate then add
ax.plot([6, 6], [4.7, 5.15], color=FOREST, lw=2)
ax.annotate('', xy=(6, 5.15), xytext=(6, 4.8),
            arrowprops=dict(arrowstyle='-|>', color=FOREST, lw=2, mutation_scale=12))

# Cell state to tanh
ax.plot([9, 9], [5.5, 4.85], color=MAIN_BLUE, lw=2)
ax.annotate('', xy=(9, 4.85), xytext=(9, 5.2),
            arrowprops=dict(arrowstyle='-|>', color=MAIN_BLUE, lw=2, mutation_scale=12))

# tanh to output multiply
ax.plot([9, 9], [4.15, 2.35], color=MAIN_BLUE, lw=2)
ax.annotate('', xy=(9, 2.35), xytext=(9, 3),
            arrowprops=dict(arrowstyle='-|>', color=MAIN_BLUE, lw=2, mutation_scale=12))

# Output gate to multiply
ax.plot([8.5, 8.5], [4, 4.3], color=ACCENT_BLUE, lw=2)
ax.plot([8.5, 8.65], [4.3, 4.3], color=ACCENT_BLUE, lw=2)
ax.plot([8.65, 8.65], [4.3, 2], color=ACCENT_BLUE, lw=2)
ax.annotate('', xy=(8.65, 2), xytext=(8.65, 2.5),
            arrowprops=dict(arrowstyle='-|>', color=ACCENT_BLUE, lw=2, mutation_scale=12))

# Input x_t (from bottom)
ax.annotate('', xy=(6, 1.5), xytext=(6, 0),
            arrowprops=dict(arrowstyle='-|>', color=FOREST, lw=2.5, mutation_scale=15))
ax.text(6, -0.3, '$x_t$', ha='center', va='center', fontsize=13, fontweight='bold', color=FOREST)

# Broadcast input to all gates (simplified)
ax.plot([3.5, 3.5], [1.5, 3.05], color='gray', lw=1.5, ls='--', alpha=0.7)
ax.plot([5.5, 5.5], [1.5, 3.05], color='gray', lw=1.5, ls='--', alpha=0.7)
ax.plot([6.5, 6.5], [1.5, 3.05], color='gray', lw=1.5, ls='--', alpha=0.7)
ax.plot([8.5, 8.5], [1.5, 3.05], color='gray', lw=1.5, ls='--', alpha=0.7)
ax.plot([3.5, 8.5], [1.5, 1.5], color='gray', lw=1.5, ls='--', alpha=0.7)

# Labels for cell state
ax.text(1.5, 6.3, 'Cell State (Long-term Memory)', ha='center', va='center', fontsize=10,
        color=MAIN_BLUE, fontweight='bold', style='italic')
ax.text(11, 1.3, 'Hidden State\n(Short-term)', ha='center', va='center', fontsize=9,
        color=PURPLE, fontweight='bold', style='italic')

plt.tight_layout()
plt.savefig('ch8_lstm_cell.pdf', bbox_inches='tight', transparent=True, dpi=300)
plt.close()
print("Generated: ch8_lstm_cell.pdf")

print("\nDone!")
