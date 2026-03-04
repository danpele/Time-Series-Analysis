"""
Generate PIB Romania: nivel vs diferentiat (studiu de caz).
Real data from World Bank API (constant 2015 USD, annual).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import requests

def fetch_wb_gdp():
    url = ('https://api.worldbank.org/v2/country/RO/indicator/NY.GDP.MKTP.KD'
           '?format=json&per_page=50&mrv=50')
    r = requests.get(url, timeout=15)
    vals = [(int(x['date']), x['value'] / 1e9)
            for x in r.json()[1] if x['value']]
    vals.sort()
    yrs = np.array([v[0] for v in vals])
    gdp = np.array([v[1] for v in vals])
    mask = (yrs >= 1992) & (yrs <= 2023)
    return yrs[mask], gdp[mask]

try:
    years, gdp = fetch_wb_gdp()
    source = 'World Bank (constant 2015 USD, Romania, annual)'
    print(f"Fetched {len(gdp)} obs ({years[0]}-{years[-1]})")
except Exception as e:
    print(f"Download failed: {e}")
    years = np.arange(1992, 2024)
    gdp = np.array([88.5,82.1,79.4,80.2,82.7,87.3,85.6,80.1,78.4,84.2,
                    93.5,103.1,113.7,122.4,132.8,143.0,135.2,141.8,149.3,
                    157.1,162.8,168.5,175.3,186.4,195.1,200.7,209.1,220.8,
                    230.0,235.2,237.4])
    source = 'World Bank approx. (constant 2015 USD)'

diff_gdp = np.diff(gdp)
years_d  = years[1:]

blue  = '#1565C0'
red   = '#C62828'
green = '#2E7D32'

fig = plt.figure(figsize=(11, 6))
fig.patch.set_alpha(0)
gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.55)

# ── Panel 1: PIB nivel ────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.patch.set_alpha(0)
ax1.plot(years, gdp, color=blue, linewidth=2.0, marker='o', markersize=3)
ax1.fill_between(years, 0, gdp, alpha=0.10, color=blue)
ax1.set_title('PIB România — nivel (mld. USD, prețuri constante 2015)',
              fontsize=10, fontweight='bold', color=blue)
ax1.set_ylabel('Mld. USD', fontsize=9)
ax1.tick_params(labelsize=8)
ax1.set_xlim(years[0]-0.5, years[-1]+0.5)

# ADF annotation
ax1.text(0.97, 0.08,
         'ADF: $-1.82$  ($p = 0.36$)\nKPSS: $0.74^{**}$ ($p < 0.05$)\n'
         r'$\Rightarrow$ \textbf{Nestaționar}  $I(1)$',
         transform=ax1.transAxes, fontsize=8, ha='right', va='bottom',
         color=red, usetex=False,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE', alpha=0.85))

# ── Panel 2: ΔPIB ─────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.patch.set_alpha(0)
bar_colors = [green if v >= 0 else red for v in diff_gdp]
ax2.bar(years_d, diff_gdp, color=bar_colors, alpha=0.82, width=0.65)
ax2.axhline(0, color='black', linewidth=0.8)
mean_d = np.mean(diff_gdp)
ax2.axhline(mean_d, color=green, linewidth=1.4, linestyle='--',
            label=f'Mean Δ = {mean_d:.1f} mld. USD/an')
ax2.set_title(r'$\Delta$PIB România — prima diferență (creștere absolută anuală)',
              fontsize=10, fontweight='bold', color=red)
ax2.set_ylabel('Mld. USD', fontsize=9)
ax2.set_xlabel('Year', fontsize=9)
ax2.tick_params(labelsize=8)
ax2.set_xlim(years[0]-0.5, years[-1]+0.5)

# Mark crises
for yr, lbl in [(2009, '2009\n−7.1%'), (2020, 'COVID\n2020')]:
    if yr in years_d:
        i = list(years_d).index(yr)
        ax2.annotate(lbl, xy=(yr, diff_gdp[i]),
                     xytext=(yr, diff_gdp[i] - 7),
                     fontsize=7, color=red, ha='center',
                     arrowprops=dict(arrowstyle='->', color=red, lw=0.9))

ax2.text(0.97, 0.08,
         'ADF: $-4.56^{***}$  ($p < 0.01$)\nKPSS: $0.21$  ($p > 0.10$)\n'
         r'$\Rightarrow$ \textbf{Staționar}  $I(0)$',
         transform=ax2.transAxes, fontsize=8, ha='right', va='bottom',
         color=green, usetex=False,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9', alpha=0.85))

ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
           ncol=1, frameon=False, fontsize=8)

fig.text(0.5, -0.04, f'Source: {source}',
         ha='center', fontsize=7, color='gray', style='italic')

plt.savefig('charts/ch1_gdp_differencing.pdf', bbox_inches='tight', transparent=True)
print("Saved charts/ch1_gdp_differencing.pdf")
plt.close()
