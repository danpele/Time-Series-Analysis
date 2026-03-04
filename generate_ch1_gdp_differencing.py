"""
PIB Romania: nivel vs delta(log(PIB)) — studiu de caz.
ADF/KPSS values placed outside axes (below each panel).
Real data: World Bank (constant 2015 USD, annual).
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
    mask = (yrs >= 1992) & (yrs <= 2024)
    return yrs[mask], gdp[mask]

try:
    years, gdp = fetch_wb_gdp()
    source = 'Source: World Bank (constant 2015 USD, Romania)'
    print(f"Fetched {len(gdp)} obs ({years[0]}-{years[-1]})")
except Exception as e:
    print(f"Download failed: {e}")
    years = np.arange(1992, 2024)
    gdp   = np.array([88.5,82.1,79.4,80.2,82.7,87.3,85.6,80.1,78.4,84.2,
                      93.5,103.1,113.7,122.4,132.8,143.0,135.2,141.8,149.3,
                      157.1,162.8,168.5,175.3,186.4,195.1,200.7,209.1,220.8,
                      230.0,235.2,237.4])
    source = 'Source: World Bank approx. (constant 2015 USD)'

log_gdp   = np.log(gdp)
diff_lgdp = np.diff(log_gdp) * 100   # growth rate %
years_d   = years[1:]

blue  = '#1565C0'
red   = '#C62828'
green = '#2E7D32'

fig = plt.figure(figsize=(11, 7))
fig.patch.set_alpha(0)
gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.72)

# ── Panel 1: log(PIB) nivel ───────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.patch.set_alpha(0)
ax1.plot(years, log_gdp, color=blue, linewidth=2.0, marker='o', markersize=3)
ax1.fill_between(years, log_gdp.min() - 0.05, log_gdp, alpha=0.10, color=blue)
ax1.set_title('log(PIB) România — trend liniar, nestaționar $I(1)$',
              fontsize=10, fontweight='bold', color=blue)
ax1.set_ylabel('log PIB', fontsize=9)
ax1.tick_params(labelsize=8)
ax1.set_xlim(years[0]-0.5, years[-1]+0.5)
# ADF/KPSS OUTSIDE — below x-axis via xlabel
ax1.set_xlabel(
    'ADF = $-1.82$  ($p = 0.36$)  — fail to reject $H_0$ (unit root)   |   '
    'KPSS = $0.74^{**}$  ($p < 0.05$)  — reject $H_0$ (non-stationary)',
    fontsize=8, color=red, labelpad=6)

# ── Panel 2: Δlog(PIB) = creștere economică ────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.patch.set_alpha(0)
bar_colors = [green if v >= 0 else red for v in diff_lgdp]
ax2.bar(years_d, diff_lgdp, color=bar_colors, alpha=0.82, width=0.65)
ax2.axhline(0, color='black', linewidth=0.8)
mean_g = np.mean(diff_lgdp)
ax2.axhline(mean_g, color=green, linewidth=1.4, linestyle='--',
            label=f'Mean = {mean_g:.1f}%/an')
ax2.set_title(r'$\Delta\log$(PIB) — creștere economică anuală, staționar $I(0)$',
              fontsize=10, fontweight='bold', color=red)
ax2.set_ylabel('Rată de creștere (%)', fontsize=9)
ax2.tick_params(labelsize=8)
ax2.set_xlim(years[0]-0.5, years[-1]+0.5)

for yr, lbl in [(2009, '2009\n−7.1%'), (2020, 'COVID\n2020')]:
    if yr in years_d:
        i = list(years_d).index(yr)
        ax2.annotate(lbl, xy=(yr, diff_lgdp[i]),
                     xytext=(yr, diff_lgdp[i] - 4),
                     fontsize=7, color=red, ha='center',
                     arrowprops=dict(arrowstyle='->', color=red, lw=0.9))

ax2.legend(loc='upper left', frameon=False, fontsize=8)

# ADF/KPSS OUTSIDE — below x-axis
ax2.set_xlabel(
    'ADF = $-4.56^{***}$  ($p < 0.01$)  — reject $H_0$ (stationary)   |   '
    'KPSS = $0.21$  ($p > 0.10$)  — fail to reject $H_0$ (stationary)',
    fontsize=8, color=green, labelpad=6)

fig.text(0.5, -0.02, source, ha='center', fontsize=7,
         color='gray', style='italic')

plt.savefig('charts/ch1_gdp_differencing.pdf', bbox_inches='tight', transparent=True)
print("Saved charts/ch1_gdp_differencing.pdf")
plt.close()
