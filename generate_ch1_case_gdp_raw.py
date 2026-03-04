"""
Romania GDP level series — case study chart (slide 55).
Annual data, World Bank (constant 2015 USD), 1990-2023.
"""
import numpy as np
import matplotlib.pyplot as plt
import requests

def fetch_wb_gdp():
    url = ('https://api.worldbank.org/v2/country/RO/indicator/NY.GDP.MKTP.KD'
           '?format=json&per_page=60&mrv=60')
    r = requests.get(url, timeout=15)
    vals = [(int(x['date']), x['value'] / 1e9)
            for x in r.json()[1] if x['value']]
    vals.sort()
    yrs = np.array([v[0] for v in vals])
    gdp = np.array([v[1] for v in vals])
    mask = (yrs >= 1990) & (yrs <= 2023)
    return yrs[mask], gdp[mask]

try:
    years, gdp = fetch_wb_gdp()
    source = 'Source: World Bank (constant 2015 USD, Romania, annual)'
    print(f"Fetched {len(gdp)} obs ({years[0]}-{years[-1]})")
except Exception as e:
    print(f"Download failed: {e}")
    years = np.arange(1990, 2024)
    gdp = np.array([
        116.5, 101.4, 92.5, 93.9, 97.6, 103.7, 107.8, 102.5, 100.5, 100.1,
        101.2, 110.6, 121.5, 131.3, 143.2, 154.3, 160.1, 154.0, 155.0, 158.7,
        163.7, 169.7, 178.1, 186.1, 189.8, 179.5, 189.0, 199.0,
        208.0, 210.2, 215.0, 224.2, 232.8, 235.2
    ])
    source = 'Source: World Bank approx. (constant 2015 USD)'

blue  = '#1565C0'
red   = '#C62828'
green = '#2E7D32'

fig, ax = plt.subplots(figsize=(10, 4.5))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

ax.plot(years, gdp, color=blue, linewidth=2.0, marker='o', markersize=3)
ax.fill_between(years, 0, gdp, alpha=0.10, color=blue)
ax.axhline(np.mean(gdp), color=red, linewidth=1.2, linestyle='--',
           label=f'Mean = {np.mean(gdp):.1f} bn USD')

# Annotate key events
for yr, lbl, yoff in [(1999, 'Crisis\n1999', -25), (2009, 'GFC\n2009', -25),
                      (2020, 'COVID\n2020', -25)]:
    if yr in years:
        i = list(years).index(yr)
        ax.annotate(lbl, xy=(yr, gdp[i]),
                    xytext=(yr, gdp[i] + yoff),
                    fontsize=7.5, color=red, ha='center',
                    arrowprops=dict(arrowstyle='->', color=red, lw=0.9))

ax.set_title('Romania GDP Level — Annual 1990–2023, Non-stationary $I(1)$',
             fontsize=11, fontweight='bold', color=blue)
ax.set_xlabel('Year', fontsize=9)
ax.set_ylabel('GDP (billion USD, constant 2015)', fontsize=9)
ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
ax.tick_params(labelsize=8)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14),
          ncol=1, frameon=False, fontsize=8)

fig.text(0.5, -0.06, source, ha='center', fontsize=7, color='gray', style='italic')

plt.savefig('charts/ch1_case_gdp_raw.pdf', bbox_inches='tight', transparent=True)
print("Saved charts/ch1_case_gdp_raw.pdf")
plt.close()
