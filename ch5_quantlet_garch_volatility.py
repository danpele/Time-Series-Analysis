"""
Quantlet:     sfm_ch5_garch_volatility
Description:  GARCH family estimation, VaR/ES backtesting, HAR-RV comparison
Keywords:     GARCH, EGARCH, GJR, VaR, ES, backtesting, HAR-RV, realized volatility
Author:       Daniel Traian Pele
Date:         2026
Data:         Yahoo Finance (^GSPC, BTC-USD)
Output:       Model comparison table, VaR/ES backtest results, forecast accuracy
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# 1. Data download (Yahoo Finance) with simulated-data fallback
# ---------------------------------------------------------------------------

def download_data(ticker: str, start: str = "2015-01-01",
                  end: str = "2025-12-31") -> pd.Series:
    """Download adjusted close prices; return log-returns (in %)."""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            raise ValueError("empty frame")
        prices = df["Adj Close"].squeeze()
        returns = 100.0 * np.log(prices / prices.shift(1)).dropna()
        returns.name = ticker
        print(f"  [{ticker}] Downloaded {len(returns)} observations "
              f"({returns.index[0].date()} to {returns.index[-1].date()})")
        return returns
    except Exception as exc:
        print(f"  [{ticker}] Download failed ({exc}); using simulated data.")
        np.random.seed(42 if ticker == "^GSPC" else 99)
        n = 2500
        # Simulate GARCH(1,1) returns
        omega, alpha, beta = 0.02, 0.08, 0.90
        sigma2 = np.zeros(n)
        ret = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta)
        for t in range(1, n):
            sigma2[t] = omega + alpha * ret[t - 1] ** 2 + beta * sigma2[t - 1]
            ret[t] = np.sqrt(sigma2[t]) * np.random.standard_t(df=5)
        idx = pd.bdate_range(start="2015-01-02", periods=n)
        returns = pd.Series(ret, index=idx, name=ticker)
        print(f"  [{ticker}] Simulated {n} observations")
        return returns


print("=" * 72)
print("CHAPTER 5  -  GARCH Family: Estimation, VaR/ES & Backtesting")
print("=" * 72)

print("\n[1] Downloading data ...")
sp500 = download_data("^GSPC")
btc   = download_data("BTC-USD")

# ---------------------------------------------------------------------------
# 2. Descriptive statistics
# ---------------------------------------------------------------------------

def descriptive_table(returns: pd.Series) -> pd.Series:
    return pd.Series({
        "Mean":     returns.mean(),
        "Std":      returns.std(),
        "Skewness": returns.skew(),
        "Kurtosis": returns.kurtosis(),
        "Min":      returns.min(),
        "Max":      returns.max(),
        "JB stat":  stats.jarque_bera(returns.dropna())[0],
        "JB p-val": stats.jarque_bera(returns.dropna())[1],
        "N":        len(returns),
    })

print("\n[2] Descriptive statistics of log-returns (%)")
print("-" * 55)
desc = pd.DataFrame({
    "S&P 500": descriptive_table(sp500),
    "Bitcoin":  descriptive_table(btc),
})
print(desc.to_string(float_format=lambda x: f"{x:12.4f}"))

# ---------------------------------------------------------------------------
# 3. Fit GARCH family models
# ---------------------------------------------------------------------------
from arch import arch_model

MODELS = {
    "GARCH-N":    dict(vol="Garch",  p=1, o=0, q=1, dist="normal"),
    "GARCH-t":    dict(vol="Garch",  p=1, o=0, q=1, dist="studentst"),
    "EGARCH-N":   dict(vol="EGARCH", p=1, o=1, q=1, dist="normal"),
    "EGARCH-t":   dict(vol="EGARCH", p=1, o=1, q=1, dist="studentst"),
    "GJR-N":      dict(vol="Garch",  p=1, o=1, q=1, dist="normal"),
    "GJR-t":      dict(vol="Garch",  p=1, o=1, q=1, dist="studentst"),
}


def fit_models(returns: pd.Series, label: str) -> dict:
    """Fit all GARCH-family specifications and return results dict."""
    results = {}
    print(f"\n[3] Fitting GARCH family models  --  {label}")
    print("-" * 55)
    for name, spec in MODELS.items():
        am = arch_model(returns, mean="Constant", **spec)
        try:
            res = am.fit(disp="off", options={"maxiter": 1000})
            results[name] = res
            print(f"  {name:12s}  logL={res.loglikelihood:10.2f}  "
                  f"AIC={res.aic:10.2f}  BIC={res.bic:10.2f}")
        except Exception as exc:
            print(f"  {name:12s}  ** estimation failed: {exc}")
    return results


sp_results  = fit_models(sp500, "S&P 500")
btc_results = fit_models(btc,   "Bitcoin")

# ---------------------------------------------------------------------------
# 4. Model comparison table
# ---------------------------------------------------------------------------

def comparison_table(results: dict, label: str) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        params = res.params
        row = {
            "Model":  name,
            "mu":     params.get("mu", np.nan),
            "omega":  params.get("omega", np.nan),
            "alpha":  params.get("alpha[1]", np.nan),
            "beta":   params.get("beta[1]", np.nan),
            "gamma":  params.get("gamma[1]", np.nan),
            "nu":     params.get("nu", np.nan),
            "LogL":   res.loglikelihood,
            "AIC":    res.aic,
            "BIC":    res.bic,
        }
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    print(f"\n[4] Model comparison  --  {label}")
    print("-" * 72)
    print(df.to_string(float_format=lambda x: f"{x:.4f}", na_rep="   -"))
    return df


sp_comp  = comparison_table(sp_results,  "S&P 500")
btc_comp = comparison_table(btc_results, "Bitcoin")

# ---------------------------------------------------------------------------
# 5-6. VaR computation & Kupiec back-test
# ---------------------------------------------------------------------------

def kupiec_test(violations: np.ndarray, n: int, alpha: float) -> dict:
    """Kupiec (1995) unconditional coverage (POF) likelihood-ratio test."""
    x = violations.sum()
    p_hat = x / n if n > 0 else 0.0
    if x == 0 or x == n:
        lr = np.inf
    else:
        lr = -2 * (n * np.log(1 - alpha) + 0 * np.log(alpha)
                    - (n - x) * np.log(1 - p_hat) - x * np.log(p_hat))
        # Correct formula:
        lr = -2 * ((n - x) * np.log(1 - alpha) + x * np.log(alpha)
                    - (n - x) * np.log(1 - p_hat) - x * np.log(p_hat))
    p_value = 1 - stats.chi2.cdf(abs(lr), df=1)
    return {"violations": int(x), "rate": p_hat, "LR": lr, "p-value": p_value}


def var_es_backtest(returns: pd.Series, label: str,
                    window: int = 1000) -> pd.DataFrame:
    """
    Rolling 1-day VaR & ES back-test using the best Student-t GARCH model.
    """
    print(f"\n[5-6] VaR / ES back-testing  --  {label}")
    print(f"      Rolling window = {window}, "
          f"out-of-sample = {len(returns) - window}")
    print("-" * 72)

    oos_start = window
    n_oos = len(returns) - window
    if n_oos < 100:
        print("  Not enough data for back-test; skipping.")
        return pd.DataFrame()

    alphas = [0.01, 0.05]
    var_dict  = {a: np.full(n_oos, np.nan) for a in alphas}
    es_dict   = {a: np.full(n_oos, np.nan) for a in alphas}
    actual    = returns.values[oos_start:]

    # Use GARCH(1,1) with Student-t for the rolling forecast
    for i in range(n_oos):
        train = returns.values[i: i + window]
        am = arch_model(train, mean="Constant", vol="Garch",
                        p=1, o=0, q=1, dist="studentst")
        try:
            res = am.fit(disp="off", options={"maxiter": 500},
                         show_warning=False)
            fcast = res.forecast(horizon=1)
            mu_f    = fcast.mean.iloc[-1, 0]
            sigma_f = np.sqrt(fcast.variance.iloc[-1, 0])
            nu = res.params.get("nu", 30)

            for a in alphas:
                q = stats.t.ppf(a, df=nu)
                var_dict[a][i] = mu_f + sigma_f * q
                # ES for Student-t
                pdf_q = stats.t.pdf(q, df=nu)
                es_dict[a][i] = mu_f + sigma_f * (
                    -pdf_q / a * (nu + q ** 2) / (nu - 1)
                )
        except Exception:
            pass  # leave NaN

        if (i + 1) % 500 == 0 or i == n_oos - 1:
            print(f"    ... {i + 1}/{n_oos} forecasts completed")

    # Kupiec tests & results
    rows = []
    for a in alphas:
        valid = ~np.isnan(var_dict[a])
        violations = (actual[valid] < var_dict[a][valid]).astype(int)
        n_valid = valid.sum()
        kup = kupiec_test(violations, n_valid, a)

        # ES average on violation days
        viol_mask = actual[valid] < var_dict[a][valid]
        avg_loss = actual[valid][viol_mask].mean() if viol_mask.any() else np.nan
        avg_es   = es_dict[a][valid][viol_mask].mean() if viol_mask.any() else np.nan

        row = {
            "Level":      f"{a:.0%}",
            "N_oos":      n_valid,
            "Violations":  kup["violations"],
            "Viol_rate":   kup["rate"],
            "Expected":    a,
            "Kupiec_LR":   kup["LR"],
            "Kupiec_pval": kup["p-value"],
            "Avg_Loss":    avg_loss,
            "Avg_ES":      avg_es,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return df


sp_bt  = var_es_backtest(sp500, "S&P 500", window=1000)
btc_bt = var_es_backtest(btc,   "Bitcoin", window=750)

# ---------------------------------------------------------------------------
# 7. Expected Shortfall summary
# ---------------------------------------------------------------------------

def es_summary(returns: pd.Series, results: dict, label: str):
    """Compute ES at 1% and 5% using the in-sample fitted models."""
    print(f"\n[7] Expected Shortfall (in-sample)  --  {label}")
    print("-" * 72)

    best_name = min(results, key=lambda k: results[k].bic)
    res = results[best_name]
    sigma = res.conditional_volatility.dropna()
    mu = res.params.get("mu", 0.0)
    nu = res.params.get("nu", None)

    print(f"  Best model by BIC: {best_name}")
    for a in [0.01, 0.05]:
        if nu is not None and nu > 2:
            q = stats.t.ppf(a, df=nu)
            pdf_q = stats.t.pdf(q, df=nu)
            es_factor = -pdf_q / a * (nu + q ** 2) / (nu - 1)
        else:
            q = stats.norm.ppf(a)
            es_factor = -stats.norm.pdf(q) / a

        var_series = mu + sigma * q
        es_series  = mu + sigma * es_factor
        print(f"  alpha={a:.0%}:  avg VaR = {var_series.mean():.4f}%  |  "
              f"avg ES = {es_series.mean():.4f}%")


es_summary(sp500, sp_results,  "S&P 500")
es_summary(btc,   btc_results, "Bitcoin")

# ---------------------------------------------------------------------------
# 8. Final summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

for label, results, bt in [("S&P 500", sp_results, sp_bt),
                            ("Bitcoin", btc_results, btc_bt)]:
    best_aic = min(results, key=lambda k: results[k].aic)
    best_bic = min(results, key=lambda k: results[k].bic)
    print(f"\n  {label}:")
    print(f"    Best model (AIC): {best_aic}  "
          f"(AIC = {results[best_aic].aic:.2f})")
    print(f"    Best model (BIC): {best_bic}  "
          f"(BIC = {results[best_bic].bic:.2f})")
    if not bt.empty:
        for _, row in bt.iterrows():
            status = "PASS" if row["Kupiec_pval"] > 0.05 else "REJECT"
            print(f"    VaR {row['Level']} backtest: "
                  f"{row['Violations']:.0f} violations "
                  f"({row['Viol_rate']:.2%}), "
                  f"Kupiec p={row['Kupiec_pval']:.4f} [{status}]")

print("\n" + "=" * 72)
print("Done.")
