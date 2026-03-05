"""
Shared module: fetch Romania quarterly real GDP from Eurostat.
Returns pandas Series with DatetimeIndex (quarterly) and GDP volume index (2015=100).
Caches data locally to avoid repeated API calls.
"""

import pandas as pd
import numpy as np
import os

CACHE_FILE = os.path.join(os.path.dirname(__file__), 'romania_gdp_quarterly.csv')

def fetch_gdp(start='1995-Q1', end='2024-Q4', use_cache=True):
    """Fetch Romania quarterly real GDP (chain-linked volumes, 2015=100, SCA)."""

    if use_cache and os.path.exists(CACHE_FILE):
        df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
        df.index.freq = 'QS'
        return df['gdp']

    # Eurostat API: namq_10_gdp, chain-linked volumes index 2015=100, SCA
    url = 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/namq_10_gdp/Q.CLV_I15.SCA.B1GQ.RO?format=TSV'
    raw = pd.read_csv(url, sep='\t')

    # Parse the single row of data
    row = raw.iloc[0]
    data = {}
    for col in raw.columns[1:]:  # skip first col (metadata)
        col_clean = col.strip()
        val_str = str(row[col]).strip().replace(' p', '').replace(' e', '').replace(' b', '')
        if val_str != ':' and val_str != '':
            try:
                val = float(val_str)
                # Convert "1995-Q1" to datetime
                date = pd.Period(col_clean, freq='Q').start_time
                data[date] = val
            except (ValueError, TypeError):
                pass

    series = pd.Series(data).sort_index()
    series.name = 'gdp'
    series.index.name = 'date'

    # Filter range
    mask = (series.index >= pd.Timestamp(start.replace('Q', '-0').replace('-01', '-01-01').replace('-02', '-04-01').replace('-03', '-07-01').replace('-04', '-10-01')))

    # Save cache
    df_cache = pd.DataFrame({'gdp': series})
    df_cache.to_csv(CACHE_FILE)
    print(f"Cached {len(series)} quarters to {CACHE_FILE}")

    return series


def get_gdp_1990_2024():
    """Get GDP for 1990Q1-2024Q4. Eurostat starts ~1995, so we use World Bank annual
    data for 1990-1994 interpolated to quarterly."""

    full = fetch_gdp()

    # Filter to 1995Q1 onwards from Eurostat
    eurostat_data = full[full.index >= '1995-01-01']
    eurostat_data = eurostat_data[eurostat_data.index <= '2024-12-31']

    # For 1990-1994: use World Bank annual data, interpolated
    # World Bank NY.GDP.MKTP.KD (constant 2015 USD) for Romania
    try:
        from pandas_datareader import wb
        wb_data = wb.download(indicator='NY.GDP.MKTP.KD', country='RO', start=1990, end=1994)
        wb_annual = wb_data['NY.GDP.MKTP.KD'].droplevel(0).sort_index()

        # Convert to index (2015=100) using the 2015 value from Eurostat
        gdp_2015 = eurostat_data.loc['2015-01-01':'2015-12-31'].mean()

        # Get 2015 actual GDP from World Bank
        wb_2015 = wb.download(indicator='NY.GDP.MKTP.KD', country='RO', start=2015, end=2015)
        gdp_2015_usd = wb_2015['NY.GDP.MKTP.KD'].values[0]

        # Convert WB annual to index
        wb_index = (wb_annual / gdp_2015_usd) * 100

        # Create quarterly dates and interpolate
        q_dates = pd.date_range('1990-01-01', '1994-10-01', freq='QS')
        annual_dates = [pd.Timestamp(f'{y}-07-01') for y in range(1990, 1995)]

        annual_series = pd.Series(wb_index.values, index=annual_dates)
        combined = annual_series.reindex(annual_dates + list(q_dates)).sort_index()
        quarterly_1990_94 = combined.interpolate(method='cubic').reindex(q_dates)
        quarterly_1990_94.name = 'gdp'

        # Combine
        result = pd.concat([quarterly_1990_94, eurostat_data])
        result = result[~result.index.duplicated(keep='last')]
        result = result.sort_index()
        return result

    except Exception as e:
        print(f"World Bank fetch failed ({e}), starting from 1995")
        return eurostat_data


if __name__ == '__main__':
    gdp = get_gdp_1990_2024()
    print(f"Period: {gdp.index[0]} to {gdp.index[-1]}")
    print(f"Quarters: {len(gdp)}")
    print(f"\nFirst 8 quarters:")
    print(gdp.head(8))
    print(f"\nLast 8 quarters:")
    print(gdp.tail(8))
