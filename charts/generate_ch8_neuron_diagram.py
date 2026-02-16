#!/usr/bin/env python3
"""
Generate neuron comparison diagrams for Chapter 8
Biological neuron vs Artificial neuron vs LSTM cell
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Ellipse
import numpy as np

# Style settings - transparent, no grid
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['font.size'] = 10

MAIN_BLUE = '#1A3A6E'
ACCENT_BLUE = '#2A528C'
IDA_RED = '#DC3545'
FOREST = '#2E7D32'
ORANGE = '#E67E22'
PURPLE = '#8E44AD'
LIGHT_BLUE = '#AED6F1'
LIGHT_GREEN = '#A9DFBF'
LIGHT_ORANGE = '#FAD7A0'
LIGHT_RED = '#F5B7B1'

# =============================================================================
# Chart 1: Biological Neuron vs Artificial Neuron
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Biological Neuron (simplified)
ax1 = axes[0]
ax1.set_xlim(-1, 10)
ax1.set_ylim(-2, 6)
ax1.set_aspect('equal')
ax1.axis('off')

# Cell body (soma)
soma = Circle((5, 2), 1.2, color=LIGHT_BLUE, ec=MAIN_BLUE, linewidth=2)
ax1.add_patch(soma)
ax1.text(5, 2, 'Soma\n(Cell Body)', ha='center', va='center', fontsize=9, fontweight='bold')

# Dendrites (inputs)
dendrite_starts = [(0.5, 4), (0.5, 2), (0.5, 0)]
dendrite_labels = ['$x_1$', '$x_2$', '$x_3$']
for i, (dx, dy) in enumerate(dendrite_starts):
    # Wavy dendrite line
    t = np.linspace(0, 1, 50)
    x = dx + t * 3.3
    y = dy + 0.3 * np.sin(t * 6 * np.pi) + (2 - dy) * t * 0.5
    ax1.plot(x, y, color=FOREST, linewidth=2.5)
    ax1.text(dx - 0.3, dy, dendrite_labels[i], ha='right', va='center', fontsize=11, fontweight='bold', color=FOREST)
ax1.text(1.5, 5, 'Dendrites\n(Inputs)', ha='center', va='center', fontsize=9, color=FOREST)

# Axon (output)
ax1.annotate('', xy=(9, 2), xytext=(6.2, 2),
             arrowprops=dict(arrowstyle='->', color=IDA_RED, lw=3))
ax1.text(9.3, 2, '$y$', ha='left', va='center', fontsize=12, fontweight='bold', color=IDA_RED)
ax1.text(7.5, 2.8, 'Axon\n(Output)', ha='center', va='center', fontsize=9, color=IDA_RED)

# Nucleus
nucleus = Circle((5, 2), 0.4, color='white', ec=MAIN_BLUE, linewidth=1.5)
ax1.add_patch(nucleus)

ax1.set_title('Biological Neuron', fontsize=14, fontweight='bold', color=MAIN_BLUE, pad=10)

# Right: Artificial Neuron
ax2 = axes[1]
ax2.set_xlim(-1, 10)
ax2.set_ylim(-2, 6)
ax2.set_aspect('equal')
ax2.axis('off')

# Inputs
inputs_y = [4, 2, 0]
input_labels = ['$x_1$', '$x_2$', '$x_3$']
weight_labels = ['$w_1$', '$w_2$', '$w_3$']

for i, (iy, il, wl) in enumerate(zip(inputs_y, input_labels, weight_labels)):
    # Input node
    inp_circle = Circle((0.5, iy), 0.4, color=LIGHT_GREEN, ec=FOREST, linewidth=2)
    ax2.add_patch(inp_circle)
    ax2.text(0.5, iy, il, ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrow with weight
    ax2.annotate('', xy=(3.6, 2), xytext=(0.9, iy),
                 arrowprops=dict(arrowstyle='->', color=FOREST, lw=2))
    mid_x = 0.9 + (3.6 - 0.9) * 0.5
    mid_y = iy + (2 - iy) * 0.5
    ax2.text(mid_x, mid_y + 0.4, wl, ha='center', va='center', fontsize=9, color=FOREST)

# Sum node
sum_circle = Circle((4, 2), 0.6, color=LIGHT_ORANGE, ec=ORANGE, linewidth=2)
ax2.add_patch(sum_circle)
ax2.text(4, 2, '$\Sigma$', ha='center', va='center', fontsize=14, fontweight='bold')
ax2.text(4, 0.8, '$\sum_i w_i x_i + b$', ha='center', va='center', fontsize=9)

# Arrow to activation
ax2.annotate('', xy=(5.9, 2), xytext=(4.6, 2),
             arrowprops=dict(arrowstyle='->', color=MAIN_BLUE, lw=2))

# Activation function
act_rect = FancyBboxPatch((6, 1.3), 1.5, 1.4, boxstyle="round,pad=0.05",
                           facecolor=LIGHT_BLUE, edgecolor=MAIN_BLUE, linewidth=2)
ax2.add_patch(act_rect)
ax2.text(6.75, 2, '$\sigma$', ha='center', va='center', fontsize=14, fontweight='bold')
ax2.text(6.75, 0.6, 'Activation\n(sigmoid, ReLU)', ha='center', va='center', fontsize=8)

# Output
ax2.annotate('', xy=(9, 2), xytext=(7.6, 2),
             arrowprops=dict(arrowstyle='->', color=IDA_RED, lw=3))
out_circle = Circle((9.3, 2), 0.4, color=LIGHT_RED, ec=IDA_RED, linewidth=2)
ax2.add_patch(out_circle)
ax2.text(9.3, 2, '$y$', ha='center', va='center', fontsize=11, fontweight='bold')

ax2.set_title('Artificial Neuron (Perceptron)', fontsize=14, fontweight='bold', color=MAIN_BLUE, pad=10)

# Add comparison text at bottom
fig.text(0.5, 0.02, 'Dendrites → Inputs with weights | Soma → Weighted sum + activation | Axon → Output',
         ha='center', va='bottom', fontsize=10, style='italic', color=MAIN_BLUE)

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.savefig('ch8_neuron_comparison.pdf', bbox_inches='tight', transparent=True)
plt.close()
print("Generated: ch8_neuron_comparison.pdf")

# =============================================================================
# Chart 2: RNN Unfolded with cleaner design
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(-1, 15)
ax.set_ylim(-1, 5)
ax.axis('off')

# Draw unfolded RNN
positions = [(2, 2), (5, 2), (8, 2), (11, 2)]
labels_t = ['$t-2$', '$t-1$', '$t$', '$t+1$']
labels_x = ['$x_{t-2}$', '$x_{t-1}$', '$x_t$', '$x_{t+1}$']
labels_h = ['$h_{t-2}$', '$h_{t-1}$', '$h_t$', '$h_{t+1}$']
labels_y = ['$y_{t-2}$', '$y_{t-1}$', '$y_t$', '$y_{t+1}$']

for i, (px, py) in enumerate(positions):
    # Hidden state node
    h_circle = Circle((px, py), 0.6, color=LIGHT_BLUE, ec=MAIN_BLUE, linewidth=2)
    ax.add_patch(h_circle)
    ax.text(px, py, labels_h[i], ha='center', va='center', fontsize=11, fontweight='bold')

    # Input
    ax.annotate('', xy=(px, py - 0.6), xytext=(px, -0.2),
                arrowprops=dict(arrowstyle='->', color=FOREST, lw=2))
    ax.text(px, -0.5, labels_x[i], ha='center', va='center', fontsize=11, fontweight='bold', color=FOREST)

    # Output
    ax.annotate('', xy=(px, 4.2), xytext=(px, py + 0.6),
                arrowprops=dict(arrowstyle='->', color=IDA_RED, lw=2))
    ax.text(px, 4.5, labels_y[i], ha='center', va='center', fontsize=11, fontweight='bold', color=IDA_RED)

    # Time label
    ax.text(px, py - 1.5, labels_t[i], ha='center', va='center', fontsize=9, color='gray')

# Horizontal arrows (memory flow)
for i in range(len(positions) - 1):
    px1, _ = positions[i]
    px2, _ = positions[i + 1]
    ax.annotate('', xy=(px2 - 0.7, 2), xytext=(px1 + 0.7, 2),
                arrowprops=dict(arrowstyle='->', color=ORANGE, lw=3))

# Add "Memory flows through time" label
ax.annotate('Memory flows through time', xy=(6.5, 2.8), fontsize=11,
            ha='center', color=ORANGE, fontweight='bold')

ax.set_title('Recurrent Neural Network (Unfolded Through Time)', fontsize=14, fontweight='bold', color=MAIN_BLUE, y=1.02)

plt.tight_layout()
plt.savefig('ch8_rnn_unfolded.pdf', bbox_inches='tight', transparent=True)
plt.close()
print("Generated: ch8_rnn_unfolded.pdf")

# =============================================================================
# Chart 3: LSTM Cell - Cleaner Design
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(-1, 13)
ax.set_ylim(-1, 8)
ax.axis('off')

# Cell state line (top)
ax.annotate('', xy=(11, 6.5), xytext=(1, 6.5),
            arrowprops=dict(arrowstyle='->', color=MAIN_BLUE, lw=4))
ax.text(6, 7.2, 'Cell State $C_t$ (Long-term Memory)', ha='center', va='center',
        fontsize=12, fontweight='bold', color=MAIN_BLUE)

# Hidden state line (bottom)
ax.annotate('', xy=(11, 1.5), xytext=(1, 1.5),
            arrowprops=dict(arrowstyle='->', color=PURPLE, lw=3))
ax.text(6, 0.8, 'Hidden State $h_t$ (Short-term Memory)', ha='center', va='center',
        fontsize=11, fontweight='bold', color=PURPLE)

# Gate boxes
gate_width = 1.8
gate_height = 1.2

# Forget Gate
forget_rect = FancyBboxPatch((2.5, 4), gate_width, gate_height, boxstyle="round,pad=0.1",
                              facecolor=LIGHT_RED, edgecolor=IDA_RED, linewidth=2)
ax.add_patch(forget_rect)
ax.text(3.4, 4.6, 'Forget\n$f_t = \sigma(...)$', ha='center', va='center', fontsize=9, fontweight='bold')

# Input Gate
input_rect = FancyBboxPatch((5, 4), gate_width, gate_height, boxstyle="round,pad=0.1",
                             facecolor=LIGHT_GREEN, edgecolor=FOREST, linewidth=2)
ax.add_patch(input_rect)
ax.text(5.9, 4.6, 'Input\n$i_t = \sigma(...)$', ha='center', va='center', fontsize=9, fontweight='bold')

# Candidate
cand_rect = FancyBboxPatch((5, 2.5), gate_width, gate_height, boxstyle="round,pad=0.1",
                            facecolor=LIGHT_ORANGE, edgecolor=ORANGE, linewidth=2)
ax.add_patch(cand_rect)
ax.text(5.9, 3.1, 'Candidate\n$\\tilde{C}_t = tanh(...)$', ha='center', va='center', fontsize=9, fontweight='bold')

# Output Gate
output_rect = FancyBboxPatch((8, 4), gate_width, gate_height, boxstyle="round,pad=0.1",
                              facecolor=LIGHT_BLUE, edgecolor=ACCENT_BLUE, linewidth=2)
ax.add_patch(output_rect)
ax.text(8.9, 4.6, 'Output\n$o_t = \sigma(...)$', ha='center', va='center', fontsize=9, fontweight='bold')

# Multiplication symbols on cell state
mult1 = Circle((3.4, 6.5), 0.25, color='white', ec=IDA_RED, linewidth=2)
ax.add_patch(mult1)
ax.text(3.4, 6.5, '×', ha='center', va='center', fontsize=12, fontweight='bold', color=IDA_RED)

mult2 = Circle((6.5, 6.5), 0.25, color='white', ec=FOREST, linewidth=2)
ax.add_patch(mult2)
ax.text(6.5, 6.5, '+', ha='center', va='center', fontsize=12, fontweight='bold', color=FOREST)

mult3 = Circle((9.5, 4), 0.25, color='white', ec=ACCENT_BLUE, linewidth=2)
ax.add_patch(mult3)
ax.text(9.5, 4, '×', ha='center', va='center', fontsize=12, fontweight='bold', color=ACCENT_BLUE)

# Connections
# Forget gate to cell state
ax.annotate('', xy=(3.4, 6.25), xytext=(3.4, 5.2),
            arrowprops=dict(arrowstyle='->', color=IDA_RED, lw=2))

# Input gate to cell state
ax.annotate('', xy=(6.5, 6.25), xytext=(5.9, 5.2),
            arrowprops=dict(arrowstyle='->', color=FOREST, lw=2))

# Candidate to input (multiply)
ax.annotate('', xy=(5.9, 4), xytext=(5.9, 3.7),
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=2))

# Output gate to hidden state
ax.annotate('', xy=(9.5, 3.75), xytext=(8.9, 4),
            arrowprops=dict(arrowstyle='->', color=ACCENT_BLUE, lw=2))
ax.annotate('', xy=(9.5, 1.75), xytext=(9.5, 3.5),
            arrowprops=dict(arrowstyle='->', color=ACCENT_BLUE, lw=2))

# tanh on output
tanh_circle = Circle((9.5, 5.5), 0.3, color='white', ec=MAIN_BLUE, linewidth=2)
ax.add_patch(tanh_circle)
ax.text(9.5, 5.5, 'tanh', ha='center', va='center', fontsize=7, fontweight='bold')

# Cell state to tanh
ax.plot([9.5, 9.5], [6.5, 5.8], color=MAIN_BLUE, lw=2)
ax.annotate('', xy=(9.5, 5.2), xytext=(9.5, 5.5),
            arrowprops=dict(arrowstyle='->', color=MAIN_BLUE, lw=2))

# Inputs
ax.text(-0.5, 6.5, '$C_{t-1}$', ha='center', va='center', fontsize=11, fontweight='bold', color=MAIN_BLUE)
ax.text(-0.5, 1.5, '$h_{t-1}$', ha='center', va='center', fontsize=11, fontweight='bold', color=PURPLE)
ax.text(12, 6.5, '$C_t$', ha='center', va='center', fontsize=11, fontweight='bold', color=MAIN_BLUE)
ax.text(12, 1.5, '$h_t$', ha='center', va='center', fontsize=11, fontweight='bold', color=PURPLE)

# Input x_t
ax.text(6, -0.5, '$x_t$ (Input)', ha='center', va='center', fontsize=11, fontweight='bold', color=FOREST)
ax.annotate('', xy=(3.4, 4), xytext=(3.4, 0.5),
            arrowprops=dict(arrowstyle='->', color=FOREST, lw=1.5, ls='--'))
ax.annotate('', xy=(5.9, 2.5), xytext=(5.9, 0.5),
            arrowprops=dict(arrowstyle='->', color=FOREST, lw=1.5, ls='--'))
ax.annotate('', xy=(8.9, 4), xytext=(8.9, 0.5),
            arrowprops=dict(arrowstyle='->', color=FOREST, lw=1.5, ls='--'))

ax.set_title('LSTM Cell Architecture', fontsize=14, fontweight='bold', color=MAIN_BLUE, y=1.02)

plt.tight_layout()
plt.savefig('ch8_lstm_cell.pdf', bbox_inches='tight', transparent=True)
plt.close()
print("Generated: ch8_lstm_cell.pdf")

print("\nAll diagrams generated!")
