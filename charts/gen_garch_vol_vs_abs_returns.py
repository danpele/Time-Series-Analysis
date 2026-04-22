"""Generate chart: GARCH conditional volatility vs absolute returns."""
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import yfinance as yf

# Download S&P 500 data
data = yf.download('^GSPC', start='2018-01-01', end='2024-12-31', auto_adjust=True)
returns = 100 * np.log(data['Close'] / data['Close'].shift(1)).dropna()

# Fit GARCH(1,1)
am = arch_model(returns, vol='Garch', p=1, q=1, dist='t')
res = am.fit(disp='off')
cond_vol = res.conditional_volatility

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(returns.index, np.abs(returns.values.flatten()), color='#AAAAAA', alpha=0.6, linewidth=0.5, label='|Returns| (%)')
ax.plot(cond_vol.index, cond_vol.values, color='#CD0000', linewidth=1.2, label='GARCH(1,1) Conditional Volatility $\\hat{\\sigma}_t$')

ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Volatility / |Returns| (%)', fontsize=11)
ax.set_title('S&P 500: GARCH(1,1) Conditional Volatility vs Absolute Returns', fontsize=13, fontweight='bold')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=10, frameon=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig('/Users/danielpele/Documents/TSA/charts/garch_vol_vs_abs_returns.pdf',
            bbox_inches='tight', transparent=True, dpi=150)
plt.close()
print("Done: garch_vol_vs_abs_returns.pdf")
