import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import combinations

tickers = ['^GSPC', '^GDAXI', '^FTSE', '^N225']
names = ['S&P 500', 'DAX', 'FTSE 100', 'Nikkei 225']

data = yf.download(tickers, start='2004-01-01', end='2024-12-31')['Close']
data.columns = names
returns = data.pct_change().dropna()

window = 60
pairs = list(combinations(names, 2))

rolling_corrs = pd.DataFrame(index=returns.index)
for a, b in pairs:
    rolling_corrs[f'{a}-{b}'] = returns[a].rolling(window).corr(returns[b])

connectivity = rolling_corrs.abs().mean(axis=1).dropna()

fig, ax = plt.subplots(figsize=(12, 4.5))

crises = [
    ('2007-07-01', '2009-06-30', 'GFC'),
    ('2010-04-01', '2012-06-30', 'Euro Crisis'),
    ('2020-02-01', '2020-09-30', 'COVID-19'),
    ('2022-01-01', '2022-10-31', 'Rate Hikes'),
]
for start, end, label in crises:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    ax.axvspan(s, e, alpha=0.12, color='red')
    mid = s + (e - s) / 2
    ax.text(mid, 0.97, label, ha='center', va='top', fontsize=7, color='darkred',
            transform=ax.get_xaxis_transform())

ax.plot(connectivity.index, connectivity.values, color='#2166AC', linewidth=0.8, label='DCC Connectivity $C_t$')

ma = connectivity.rolling(252).mean()
ax.plot(ma.index, ma.values, color='#B2182B', linewidth=1.5, linestyle='--', label='1-year moving average')

ax.axhline(y=0.5, color='gray', linewidth=0.6, linestyle=':')
ax.text(connectivity.index[10], 0.52, 'Diversification threshold', fontsize=7, color='gray')

ax.set_ylabel('Average pairwise $|\\rho_t|$', fontsize=10)
ax.set_xlim(connectivity.index[0], connectivity.index[-1])
ax.set_ylim(0.0, 1.0)
ax.tick_params(labelsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=9, frameon=False)

fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

plt.tight_layout()
plt.savefig('/Users/danielpele/Documents/TSA/charts/ch5b_dcc_connectivity.pdf',
            bbox_inches='tight', transparent=True, dpi=300)
plt.close()
print('Done: ch5b_dcc_connectivity.pdf')
