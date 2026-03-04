"""
Generate PIB Romania: nivel -> log -> diff(log) = crestere economica.
Uses real Romanian GDP data (World Bank API, constant 2015 USD, annual).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import requests

def fetch_wb_gdp(country='RO', indicator='NY.GDP.MKTP.KD', per_page=50):
    url = (f'https://api.worldbank.org/v2/country/{country}/indicator/{indicator}'
           f'?format=json&per_page={per_page}&mrv={per_page}')
    r = requests.get(url, timeout=15)
    data = r.json()
    vals = [(int(x['date']), x['value']) for x in data[1] if x['value']]
    vals.sort()
    years = np.array([v[0] for v in vals])
    gdp   = np.array([v[1] / 1e9 for v in vals])   # billion USD
    return years, gdp

try:
    years_arr, gdp = fetch_wb_gdp()
    # keep 1992-2023
    mask = (years_arr >= 1992) & (years_arr <= 2023)
    years_arr, gdp = years_arr[mask], gdp[mask]
    source_label = 'Source: World Bank (GDP constant 2015 USD, Romania)'
    print(f"Fetched {len(gdp)} annual obs ({years_arr[0]}-{years_arr[-1]})")
except Exception as e:
    print(f"Download failed ({e}), using hardcoded data")
    years_arr = np.arange(1992, 2024)
    gdp = np.array([
        88.5, 82.1, 79.4, 80.2, 82.7, 87.3, 85.6, 80.1, 78.4, 84.2,
        93.5, 103.1, 113.7, 122.4, 132.8, 143.0, 135.2, 141.8, 149.3,
        157.1, 162.8, 168.5, 175.3, 186.4, 195.1, 200.7, 209.1, 220.8,
        230.0, 235.2, 237.4
    ])
    source_label = 'Source: World Bank (approx., constant 2015 USD)'

log_gdp = np.log(gdp)
growth  = np.diff(log_gdp) * 100
years_g = years_arr[1:]

blue  = '#1565C0'
green = '#2E7D32'
red   = '#C62828'

fig = plt.figure(figsize=(13, 6))
fig.patch.set_alpha(0)
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42)

# ── Panel 1: PIB nivel ───────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.patch.set_alpha(0)
ax1.plot(years_arr, gdp, color=blue, linewidth=2.0, marker='o', markersize=3)
ax1.fill_between(years_arr, 0, gdp, alpha=0.10, color=blue)
ax1.set_title('PIB nivel', fontsize=11, fontweight='bold', color=blue)
ax1.set_xlabel('Year', fontsize=9)
ax1.set_ylabel('Mld. USD (prețuri constante 2015)', fontsize=8)
ax1.tick_params(labelsize=8)
ax1.set_xlim(years_arr[0] - 1, years_arr[-1] + 1)
ax1.annotate('Varianță\ncrescătoare\n$\\Rightarrow I(1)$, nestaționar',
             xy=(2005, gdp[list(years_arr).index(2005)]),
             xytext=(1993, gdp[-5]),
             fontsize=7.5, color=red, ha='center',
             arrowprops=dict(arrowstyle='->', color=red, lw=1.2),
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

fig.text(0.358, 0.56, r'$\ln$', ha='center', va='center', fontsize=14,
         fontweight='bold', color=green,
         bbox=dict(boxstyle='rarrow,pad=0.3', facecolor='#E8F5E9',
                   edgecolor=green, lw=1.5))

# ── Panel 2: log(PIB) ────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.patch.set_alpha(0)
ax2.plot(years_arr, log_gdp, color=green, linewidth=2.0, marker='o', markersize=3)
ax2.fill_between(years_arr, log_gdp.min() - 0.05, log_gdp, alpha=0.10, color=green)
ax2.set_title('log(PIB)', fontsize=11, fontweight='bold', color=green)
ax2.set_xlabel('Year', fontsize=9)
ax2.set_ylabel('log PIB', fontsize=9)
ax2.tick_params(labelsize=8)
ax2.set_xlim(years_arr[0] - 1, years_arr[-1] + 1)
ax2.annotate('Varianță stabilizată\n(dar trend liniar!)',
             xy=(2008, log_gdp[list(years_arr).index(2008)]),
             xytext=(1993, log_gdp[-5]),
             fontsize=7.5, color=blue, ha='center',
             arrowprops=dict(arrowstyle='->', color=blue, lw=1.2),
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

fig.text(0.648, 0.56, r'$\Delta$', ha='center', va='center', fontsize=16,
         fontweight='bold', color=red,
         bbox=dict(boxstyle='rarrow,pad=0.3', facecolor='#FFEBEE',
                   edgecolor=red, lw=1.5))

# ── Panel 3: Δlog(PIB) = growth ──────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.patch.set_alpha(0)
bar_colors = [green if g >= 0 else red for g in growth]
ax3.bar(years_g, growth, color=bar_colors, alpha=0.82, width=0.65)
ax3.axhline(0, color='black', linewidth=0.8)
mean_g = np.mean(growth)
ax3.axhline(mean_g, color=green, linewidth=1.5, linestyle='--',
            label=f'Mean = {mean_g:.1f}%/yr')
ax3.set_title(r'$\Delta\log$(PIB) = Creștere economică', fontsize=10,
              fontweight='bold', color=red)
ax3.set_xlabel('Year', fontsize=9)
ax3.set_ylabel('Rată de creștere anuală (%)', fontsize=8)
ax3.tick_params(labelsize=8)
ax3.set_xlim(years_g[0] - 1, years_g[-1] + 1)
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14),
           ncol=1, frameon=False, fontsize=8)

for yr, label, offset in [(2009, 'Criză\n2009', -3.5), (2020, 'COVID\n2020', -3.5)]:
    if yr in years_g:
        idx = list(years_g).index(yr)
        ax3.annotate(label, xy=(yr, growth[idx]),
                     xytext=(yr, growth[idx] + offset),
                     fontsize=7, color=red, ha='center',
                     arrowprops=dict(arrowstyle='->', color=red, lw=1.0))

fig.suptitle('PIB România: logaritm → diferențiere = creștere economică',
             fontsize=11, fontweight='bold', y=1.01)
fig.text(0.5, -0.04, source_label, ha='center', fontsize=7,
         color='gray', style='italic')

plt.savefig('charts/ch1_log_diff_gdp.pdf', bbox_inches='tight', transparent=True)
print("Saved charts/ch1_log_diff_gdp.pdf")
plt.close()
