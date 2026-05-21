"""
Generate a 2x3 static grid of representative Ising snapshots for Ch13 LPPL.
Row 1 (Bubble): T/Tc = 0.00, 0.65, 1.00
Row 2 (Crash):  T/Tc = 1.10, 1.50, 2.00
Reads existing PNGs from ising_frames/ directory.
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

MainBlue = '#1A3A6E'
Crimson  = '#DC3545'
Forest   = '#2E7D32'
Amber    = '#B5853F'

FRAMES_DIR = '/Users/danielpele/Documents/TSA/charts/ising_frames'
OUTPUT = '/Users/danielpele/Documents/TSA/charts/ch13_lppl_ising_grid.png'

panels = [
    ('ising_bubble_01.png', '$T/T_c = 0.50$', 'Strong herding', MainBlue),
    ('ising_bubble_07.png', '$T/T_c = 0.92$', 'Instability grows', Amber),
    ('ising_bubble_12.png', '$T/T_c = 1.00$', 'Critical point', Crimson),
    ('ising_crash_04.png',  '$T/T_c = 1.10$', 'Herding collapses', Crimson),
    ('ising_crash_08.png',  '$T/T_c = 1.50$', 'Approaching normal', Forest),
    ('ising_crash_12.png',  '$T/T_c = 2.00$', 'Normal market', Forest),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 10))
fig.patch.set_alpha(0)

row_labels = [
    ('Bubble regime: heating toward $T_c$', MainBlue),
    ('Post-regime break: recovery to normal market', Forest),
]

for idx, (fname, ratio_label, desc, color) in enumerate(panels):
    row, col = divmod(idx, 3)
    ax = axes[row][col]
    img = mpimg.imread(os.path.join(FRAMES_DIR, fname))
    # Crop: remove top ~40px (T/Tc label) and bottom ~30px (magnetization label)
    h = img.shape[0]
    img_cropped = img[40:h-30, :, :]
    ax.imshow(img_cropped)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f'{ratio_label}\n{desc}', fontsize=13, fontweight='bold',
                 color=color, pad=8)

fig.text(0.5, 0.98, row_labels[0][0],
         ha='center', fontsize=14, fontstyle='italic', color=row_labels[0][1],
         transform=fig.transFigure)
fig.text(0.5, 0.50, row_labels[1][0],
         ha='center', fontsize=14, fontstyle='italic', color=row_labels[1][1],
         transform=fig.transFigure)

plt.tight_layout(rect=[0.0, 0.02, 1.0, 0.96], h_pad=4.0, w_pad=1.0)
fig.savefig(OUTPUT, dpi=150, bbox_inches='tight', transparent=True, pad_inches=0.1)
plt.close(fig)
print(f'Saved {OUTPUT}')
