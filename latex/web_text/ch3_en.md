<!--
Course text of the web version, chapter 3 (EN). Rendered into the page by latex/build_web_lecture.py.
Format: # section title (introduction), ## frame title {h=hash} (text), @after (text after the live block), ## @end (synthesis).
Numbered equations: $$ ... $$ {#eq:label}, reference @eq:label. Citations: @Key → Author (year), [@Key] → (Author, year).
Current hashes: python3 build_web_lecture.py 3 --hashes
-->

## @intro

@epigraph Yule1926
> Why do we sometimes get nonsense-correlations between time-series?

@text
Most macroeconomic and financial series rise over time: real gross domestic product (GDP), the price level, the exchange rate, stock indices. Their mean is not constant, so the stationary tools of the previous chapters do not apply to them directly. The question of the chapter is what kind of trend these series have. If the trend is a fixed line, a shock moves the series away from the line only temporarily. If the trend is stochastic, every shock stays in the level of the series for ever. The two cases call for different treatments: in the first the trend is removed, in the second the series is differenced.

The difference has direct practical consequences. A trend-stationary model estimated on US real GDP up to 2007Q4 expected output to return to its pre-crisis line. The return never came: in 2026Q2, observed GDP was about 30 log points below that forecast, while the half-width of the 95% interval was only 8 points. The difference-stationary model also misses, by about 25 points, but its interval, with a half-width of 23, at least signals that the uncertainty is large. The nature of the trend also decides whether a regression between two trending series makes sense: the question of @Yule1926 in the epigraph concerns exactly the correlations found between series that evolve as sums of shocks.

The chapter first treats the difference between the two kinds of trend and spurious regression. It then turns to the tests that decide between them, augmented Dickey–Fuller (ADF), Phillips–Perron (PP) and KPSS, with their limits in short samples and in the presence of structural breaks. The last part builds the ARIMA$(p,d,q)$ models, which add differencing to the ARMA models of [Chapter 2](ch2.html), and uses them to forecast Romanian GDP, inflation and the EUR/RON exchange rate.

@chart tsa_ch3_us_gdp
US real GDP (100 ln) and two forecasts made in 2007Q4: a trend-stationary model (linear trend with AR(2) errors) and a difference-stationary model (ARIMA(1,1,0) with drift), with their 95% intervals; on the right, the outcome minus the forecast, in log points.

@goals
- **Distinguish** trend-stationary from difference-stationary series and explain the effect of a shock in each case.
- **Recognise** a spurious regression from $t$, $R^2$ and the Durbin–Watson statistic and explain why it arises.
- **Apply** the ADF, Phillips–Perron and KPSS tests with suitable deterministic terms and lag lengths and interpret them jointly.
- **Explain** how a structural break can mimic a unit root and how the Zivot–Andrews test searches for it.
- **Identify, estimate and check** an ARIMA$(p,d,q)$ model and derive the width of its forecast intervals.
- **Evaluate** critically the output of an automatic ARIMA selection.

@plan
- Deterministic and stochastic trends | Trend-stationary and difference-stationary series, integrated processes, the unit root and spurious cycles.
- Spurious regression | Yule's nonsense correlations, the Granger–Newbold simulation, Phillips's theory and an example on real data.
- The Dickey–Fuller test | The $\tau$ statistic, the three regressions, critical values, the augmented test, lag choice, power and the order of integration.
- Phillips–Perron, KPSS and their joint use | The long-run variance correction, stationarity as the null and the table of tests on real series.
- Structural breaks | Perron's critique, a simulated level shift and the Zivot–Andrews test on the Nile and on EUR/RON.
- ARIMA(p,d,q) models | The definition, the Box–Jenkins cycle on Romanian GDP, forecast intervals and over-differencing.
- ARIMA in practice | Automatic selection and its checks, Romanian inflation, EUR/RON against the random walk and US GDP after 2007.
- Possible contribution of AI | What an AI assistant can do in this chapter and what must be checked.
- Summary | Key ideas and formulas and the self-assessment.

## Today's question and route {h=977721e9}

The chapter starts from a question about shocks: when a series rises over time, does the effect of today's shock die out or does it last for ever? The answer fixes the treatment of the series, detrending or differencing, and the width of forecast intervals at long horizons. The route has three stages. The first defines the two kinds of trend, integrated processes $I(d)$ and spurious regression. The second presents the unit-root tests, Dickey–Fuller, ADF and Phillips–Perron, the KPSS stationarity test and the effect of structural breaks. The third builds ARIMA$(p,d,q)$ models: identification, estimation, diagnostics and forecasting.

The chapter uses the notions of [Chapter 1](ch1.html), stationarity, the random walk and the difference operator, and the ARMA models of [Chapter 2](ch2.html). Seminar 3 precedes the lecture and introduces the same tests on short examples.

## Reading and tools {h=ec5ba6df}

The core textbook is @HuangPetukhina2022, chapters 4 and 5, on ARIMA models and non-stationarity. The companion textbook, @HyndmanAthanasopoulos2021, free online, covers ARIMA models in chapter 9, with an emphasis on forecasting and automatic selection. For the asymptotic theory of unit roots, the reference is @Hamilton1994, chapters 15–17, and the ARIMA methodology is set out in @BoxJenkinsReinsel2008.

The ADF, KPSS and Zivot–Andrews tests and ARIMA estimation are available in the Python library *statsmodels*; the Phillips–Perron test can be written directly from its formula, given in section 4. The lecture notebook reproduces the charts and tables of the chapter.

# Deterministic and stochastic trends

A trending series can be described in two ways. The first is a fixed line plus stationary deviations from it: a deterministic trend. The second is a cumulative sum of shocks, possibly with a drift: a stochastic trend. On a chart the two look almost the same; they differ in the effect of a shock, in the long-run variance and in the correct treatment. The section defines the two models, integrated processes and the unit root, then shows what happens when a series receives the wrong treatment.

## Four trending series {h=34203332}

The chart brings together four Romanian and international series observed since 2000. Top left is the logarithm of Romania's quarterly real GDP, seasonally adjusted, together with the line fitted by least squares. Top right is the logarithm of the harmonised index of consumer prices (HICP), monthly, with base 2015 = 100. Bottom left is the daily EUR/RON rate, the reference rate of the National Bank of Romania (BNR), from July 2005. Bottom right are the logarithms of the daily prices of the BET and S&P 500 indices, each minus its value in 2000, so that both start at zero.

None of the series fluctuates around a constant mean. In the terms of [Chapter 1](ch1.html), all are non-stationary: their mean depends on time. The shape of the growth differs, however. GDP follows a line from which it departs in long episodes, the HICP rises fast at first and more slowly after 2005, the exchange rate has rare jumps followed by periods of stability, and the BET collapses in 2008–2009 and then recovers slowly.

## Interpreting the four series {h=fa70c04e}

The figures give the order of magnitude of the trends. Real GDP grows on average by 0.80% per quarter, about 3.2% per year, but departs from the fitted line by up to 18%. Consumer prices have risen 7.2-fold since 2000. The EUR/RON rate went from 3.60 to 5.26 lei. The logarithm of the BET rose by 426 log points (that is, $100\Delta\ln$) and that of the S&P 500 by 166 points, up to 18 September 2026.

The 2009 recession illustrates the question of the chapter. After the fall, Romanian GDP grew again, but it is not clear whether it returned to the old line or started on a new one, parallel and lower. The first case corresponds to a temporary shock, the second to a permanent one.

Two models can produce such charts. The first is a fixed line plus stationary noise, a deterministic trend. The second is a random walk with drift, a stochastic trend. The next two frames define them formally and compare their properties.

## Trend-stationary series {h=4d1284f2}

A series is trend-stationary if it can be written as a deterministic function of time plus a stationary process:

$$ y_t = \alpha + \beta t + u_t, \qquad u_t \sim I(0). $$ {#eq:ts}

In @eq:ts, $\alpha$ is the intercept, $\beta$ the slope, that is the average change per period, and $u_t$ a stationary process, for example an AR(1) $u_t = \phi u_{t-1} + \varepsilon_t$ with $|\phi| < 1$ ([Chapter 2](ch2.html)). The mean $E[y_t] = \alpha + \beta t$ depends on time, so $y_t$ is not stationary, but the variance $\mathrm{Var}(y_t) = \mathrm{Var}(u_t)$ is constant and $y_t - \alpha - \beta t$ is stationary.

A shock $\varepsilon_t$ moves the series away from the line only temporarily. With AR(1) errors, its effect after $h$ periods is $\phi^h$, which goes to zero geometrically. Long-run forecasts therefore return to the line $\alpha + \beta(T+h)$, and the forecast-error variance converges to $\mathrm{Var}(u_t)$: forecast intervals have a bounded width. The correct treatment is to estimate the trend by ordinary least squares (OLS) and to model the residuals as an ARMA process.

## Difference-stationary series {h=b4708b5e}

A series is difference-stationary if $y_t = \beta + y_{t-1} + u_t$, that is if its first difference is stationary:

$$ \Delta y_t = \beta + u_t, \qquad u_t \sim I(0). $$ {#eq:ds}

Here $\beta$ is the drift, the average change per period. By repeated substitution, $y_t = y_0 + \beta t + \sum_{i=1}^{t} u_i$, where $y_0$ is the initial value. The series has two trend components: the deterministic term $\beta t$, which comes from the drift, and the cumulative sum of shocks $\sum_{i \le t} u_i$, called the stochastic trend.

The mean is the same as in @eq:ts, $y_0 + \beta t$, but the variance grows with time: for a random walk, with $u_t$ white noise of variance $\sigma^2$, $\mathrm{Var}(y_t) = t\sigma^2$ ([Chapter 1](ch1.html)). Every shock enters the cumulative sum and stays in the level for ever; there is no line to which the series returns. As a consequence, forecast intervals widen without limit as the horizon grows. The correct treatment is to difference the series and to model $\Delta y_t$ as ARMA, that is the ARIMA model of section 6.

## Trend-stationary and difference-stationary series: the effect of a shock {h=78e5aacf}

The simulation compares the two models with the same shocks and the same slope, 0.2 per period. In the left panel, the trend-stationary series has AR(1) errors with $\phi = 0.8$. In the middle panel, the series is a random walk with drift 0.2. The blue lines are the paths without an extra shock, the red ones add a shock of $+6$ at $t = 100$; the dashed line is the line $0.2t$.

The right panel summarises the difference through an impulse response: the effect of a unit shock after $h$ periods. For the trend-stationary series the effect is $\phi^h = 0.8^h$, the green curve falling towards zero. For the random walk the effect is 1 at every horizon, the horizontal purple line.

## Interpreting the two trends {h=c1d4922d}

In the trend-stationary series the shock dies out: its effect is $0.8^{10} = 0.107$ after 10 periods and $0.8^{20} = 0.012$ after 20. At $t = 200$ the red and blue paths coincide. The series has forgotten the shock, and its deviation from the line again depends only on recent shocks.

In the random walk the whole path after $t = 100$ moves up by 6 and stays there. The economic reading is direct. If GDP is trend-stationary, a recession is a temporary fall below trend, followed by recovery. If GDP is difference-stationary, the recession is a permanent loss of output: the future level is lower for ever.

Without the extra shock, the two blue paths look very much alike: both rise around the same line, with long deviations. The naked eye cannot tell them apart, and this difficulty motivates the formal tests of sections 3 and 4.

## Integrated processes {h=421687d4}

The notion that generalises a difference-stationary series is the order of integration. A process $y_t$ is integrated of order $d$, written $y_t \sim I(d)$, if $\Delta^d y_t$ is stationary and $\Delta^{d-1} y_t$ is not. An $I(0)$ process is stationary, with an invertible ARMA representation ([Chapter 2](ch2.html)); an $I(1)$ process becomes stationary after one difference, and an $I(2)$ process after two. The second difference is $\Delta^2 y_t = \Delta y_t - \Delta y_{t-1} = y_t - 2y_{t-1} + y_{t-2}$.

In economics and finance the typical orders are known. Log stock prices, exchange rates and log GDP are usually $I(1)$, while returns and growth rates, their differences, are $I(0)$. The price level in periods of high inflation may be $I(2)$; inflation, the difference of the log price level, is then itself $I(1)$.

Two simple rules help with the classification. The sum of an $I(0)$ and an $I(1)$ process is $I(1)$: the stochastic trend dominates the stationary component. The second rule concerns trend-stationary series: they become $I(0)$ after detrending, not after differencing, and section 6 shows what happens if they are differenced nonetheless.

## The unit root {h=29b86bb0}

The order of integration has an algebraic characterisation through the autoregressive polynomial. An AR($p$) is written with the lag operator $L$, $Ly_t = y_{t-1}$, as $\phi(L)y_t = \varepsilon_t$, where $\phi(z) = 1 - \phi_1 z - \dots - \phi_p z^p$. From [Chapter 2](ch2.html), the process is stationary if all roots of $\phi(z) = 0$ lie outside the unit circle, $|z| > 1$.

A unit root means that $z = 1$ is a root of $\phi$. The polynomial then factors as $\phi(z) = (1 - z)\phi^*(z)$, and if the roots of $\phi^*$ lie outside the circle, $\phi^*(L)(1 - L)y_t = \phi^*(L)\Delta y_t = \varepsilon_t$. The difference $\Delta y_t$ is a stationary AR($p-1$), so $y_t \sim I(1)$. For an AR(1), $\phi(z) = 1 - \phi z$ has the root $1/\phi$, which is a unit root exactly when $\phi = 1$: the random walk.

The example on the slide shows the factorisation. For $y_t = 1.5y_{t-1} - 0.5y_{t-2} + \varepsilon_t$, the polynomial is $1 - 1.5z + 0.5z^2 = (1 - z)(1 - 0.5z)$, with roots 1 and 2. The first is a unit root, the second lies outside the circle, so $\Delta y_t = 0.5\Delta y_{t-1} + \varepsilon_t$: a stationary AR(1) in differences. The coefficients of the levels, 1.5 and $-0.5$, sum to exactly 1, and this is the condition $\phi(1) = 0$ used by the tests of section 3.

## Case study: Nelson and Plosser (1982) {h=e59768f2}

Until the early 1980s, business-cycle models usually assumed that macroeconomic series are stationary around a trend: the cycle was the temporary deviation from a line. @NelsonPlosser1982 put this assumption to the test on 14 long annual US series, some starting in 1860: real gross national product (GNP), employment, prices, wages, interest rates and stock prices.

The method was the Dickey–Fuller test of section 3, in the version with constant and trend, whose alternative is exactly trend stationarity. The unit root was rejected for a single series, the unemployment rate. For the other 13, the data did not allow a stochastic trend to be rejected.

Their conclusion, that shocks to output appear permanent, opened a broad debate on the nature of recessions. It also carried a methodological warning: removing a straight-line trend from a difference-stationary series produces artefacts, illustrated in the next frame. The result was later challenged by @Perron1989, who showed that structural breaks can mimic a unit root (section 5).

## Detrending a random walk {h=997030e8}

The experiment applies the treatment for a deterministic trend to a series that has no such trend. The left panel shows a driftless random walk with $T = 200$ observations and the OLS line $\hat a + \hat b t$ fitted to it. The middle panel shows the residuals, the "detrended" series, and the right panel the autocorrelation function (ACF) of the residuals, with the band $\pm 1.96/\sqrt{T}$.

The random walk has no true slope: the drift is zero. Any fitted line therefore only describes the direction in which this path happened to wander, here downwards. The residuals seem to follow long, regular cycles, although the series contains no cycle at all.

## Interpreting the detrended random walk {h=f0ce7426}

The trend regression looks very successful. The coefficient of determination is $R^2 = 0.61$ and the $t$-statistic of the slope is $-17.7$, although the true slope is 0. The result is not an accident of this path: over 2,000 simulated random walks with $T = 200$, $|t| > 1.96$ occurs in 90% of cases, and the median $R^2$ is 0.44. The usual $t$-test assumes stationary errors, while here the residuals inherit the stochastic trend.

The residuals have a first-order autocorrelation $\hat\rho(1) = 0.96$ and an ACF that decays slowly. They look like a business cycle, but they are an artefact of the method. @NelsonKang1981 described the phenomenon: detrending a difference-stationary series creates spurious cycles, whose length depends on the length of the sample, not on the economy.

The reverse error is differencing a trend-stationary series. It is called over-differencing, was introduced in [Chapter 1](ch1.html) and is treated in section 6 on Romanian GDP.

## Recap: Deterministic and stochastic trends {h=a9ee1212}

The two trend models differ in the memory of shocks. In a trend-stationary series, @eq:ts, shocks die out and forecasts return to the line. In a difference-stationary series, @eq:ds, shocks are permanent and the variance grows with time. The order of integration $I(d)$ counts the differences needed for stationarity, and a unit root of the polynomial $\phi(z)$ means $I(1)$.

The wrong treatment creates artefacts in both directions. Detrending a difference-stationary series produces spurious cycles and misleading $t$-statistics. Differencing a trend-stationary series produces over-differencing. This is why the choice between the two treatments requires a test, not a visual inspection.

## @end

Trending economic series can be stationary around a line or can contain a stochastic trend; in the second case every shock stays in the level and the variance grows with time. The distinction is expressed algebraically by a unit root of the AR polynomial and decides the treatment: detrending or differencing. The simulations show that the two models cannot be told apart by eye and that the wrong treatment produces false cycles and false slopes. [The next section](#sec-2) extends the problem to regressions between two series with stochastic trends.

# Spurious regression

The problem of the previous section, a regression on time that looks significant although there is no slope, also arises when one $I(1)$ series is regressed on another $I(1)$ series independent of it. The phenomenon is called spurious regression. The section follows its history, from Yule (1926) to Granger and Newbold (1974), reproduces it by simulation, explains why the $t$-test fails and illustrates it on real Romanian and US data.

## Yule (1926): nonsense correlations {h=a751362f}

The first to raise the problem systematically was the British statistician George Udny Yule. The paper of @Yule1926 opens with the question in its title: why do we sometimes get nonsense correlations between time series? His example: in England and Wales, between 1866 and 1911, the proportion of marriages celebrated by the Church of England and the mortality rate had a correlation of about 0.95. There is no causal link between them; both series fell over the period.

Yule's explanation anticipates the modern theory. For series that evolve as cumulative sums of random terms, the sample correlation does not concentrate around 0, even when the series are independent. Large values often occur by chance, and the usual standard error of the correlation coefficient, built for independent observations, does not apply.

The problem returned fifty years later, in econometric models estimated on the levels of trending macroeconomic series. The next frame describes its rediscovery.

## Granger and Newbold (1974): spurious regressions {h=189fdbcb}

@GrangerNewbold1974 studied by simulation the regression of one random walk on another, independent random walk: $y_t = a + bx_t + e_t$, with $y_t$ and $x_t$ independent $I(1)$ series. The true slope is $b = 0$. The $t$-test of $b = 0$ nevertheless rejected far too often, the $R^2$ was frequently large and the Durbin–Watson (DW) statistic was close to 0.

The Durbin–Watson statistic measures the first-order autocorrelation of the residuals: $\mathrm{DW} \approx 2(1 - \hat\rho_1)$. A value close to 2 indicates uncorrelated residuals, while a value close to 0 indicates highly persistent, $I(1)$-like residuals. Hence the authors' rule of thumb: a regression should be suspected of being spurious when $R^2 > \mathrm{DW}$, that is when the apparent fit is larger than the signal of autocorrelation in the residuals.

Clive Granger and Paul Newbold were then at the University of Nottingham. Granger received the Nobel Prize in economics in 2003, for methods of analysing series with common trends, cointegration, covered in [Chapter 7](ch7.html).

## Spurious regression by simulation {h=446fde7d}

The simulation reproduces the Granger–Newbold experiment. For each sample size $T$, 2,000 pairs of independent Gaussian random walks are generated, and $y$ is regressed by OLS on a constant and on $x$, once in levels and once in differences. The left panel shows the share of samples with $|t| > 1.96$, that is the frequency with which the true hypothesis $b = 0$ is rejected at the nominal 5% level; the horizontal axis, logarithmic, gives $T$.

The right panel shows the histogram of the $t$-statistic for $T = 200$, in levels (red) and in differences (blue), against the $N(0,1)$ density (green). If the test worked, both histograms should follow the green curve and have 5% of their mass beyond $\pm 1.96$.

## Interpreting the simulation {h=ce400a39}

In levels, the true hypothesis $b = 0$ is rejected in 68% of samples for $T = 50$ and in 92% for $T = 1000$. More data make matters worse: the median of $|t|$ grows from 3.2 for $T = 50$ to 14.2 for $T = 1000$. The red histogram on the right is extremely wide, with values of $t$ between about $-30$ and $30$. The median $R^2$ stays around 0.17 for every $T$, and the median DW statistic is 0.08 at $T = 200$, so the rule $R^2 > \mathrm{DW}$ correctly flags the problem.

In differences, the share of rejections is 4.7% at $T = 200$, close to the nominal 5%, and the blue histogram follows the $N(0,1)$ density. The $t$-test works again, because $\Delta y_t$ and $\Delta x_t$ are independent white noises and the OLS assumptions hold.

## Failure of the t-test in spurious regressions {h=b5c37e6c}

The explanation lies in the error term. If $b = 0$, the regression error is $e_t = y_t - a$, a random walk. The OLS assumptions, stationary and uncorrelated errors, do not hold: the residuals are strongly autocorrelated, and the usual standard error, computed as if they were independent, is far too small. The $t$-ratio is therefore far too large.

@Phillips1986 derived the exact large-sample behaviour, using asymptotic theory for integrated processes. The estimator $\hat b$ does not converge to 0 but to a random variable: each sample produces a different slope, however long it is. The statistic $t_{\hat b}$ grows like $\sqrt{T}$, so the rejection rate goes to 100%. The $R^2$ has a non-degenerate limiting distribution, which explains the stable median of about 0.17 in the simulation, and $\mathrm{DW} \to 0$.

There is an important exception. If $y_t - bx_t$ is stationary for some $b$, the series share a common stochastic trend and are called cointegrated; the regression in levels then makes sense, and $\hat b$ estimates the equilibrium relation [@EngleGranger1987]. Cointegration is the subject of [Chapter 7](ch7.html).

## A spurious regression on real data {h=22c80774}

The real example regresses Romania's HICP (Eurostat) on the S&P 500 index, at the end-of-month close, over 260 months from January 2005. Both series are transformed as $100\ln$, so their monthly differences are approximately percentage changes. The left panel shows the regression in levels, with the OLS line, and the right panel the regression of monthly changes: monthly inflation on the monthly change of $100\ln$ S&P 500.

In levels, the points form elongated bands that correspond to successive periods: both series rose, at different speeds. In differences, the points form a cloud without direction around a practically horizontal line.

## Interpreting the HICP regression {h=9286b036}

In levels, the estimated slope is 0.39, with $t = 34.6$, $R^2 = 0.82$ and $\mathrm{DW} = 0.03$. Read naively, the regression would say that US stocks explain Romanian consumer prices. The Granger–Newbold rule, $R^2 > \mathrm{DW}$, is however satisfied by a wide margin, and the residuals are extremely persistent: this is a typical spurious regression.

In differences, the $t$-statistic of the slope is $-0.21$, with p-value 0.83, and $R^2 = 0.0002$. Monthly Romanian inflation and the monthly return of the S&P 500 have no link detectable by this regression.

The correct conclusion is modest: both levels contain a trend, and the regression in levels only says that both rose over the same period. It says nothing about an economic relation between the two series.

## Recap: Spurious regression {h=e6420a43}

Regressing one $I(1)$ series on another, unrelated $I(1)$ series produces a large $t$, a large $R^2$ and a DW statistic close to 0. More observations do not solve the problem but aggravate it, because $t$ grows like $\sqrt{T}$.

The remedies follow from the diagnosis. First, the order of integration of each series is tested, with the tests of sections 3 and 4. If the series are $I(1)$, the regression is estimated in differences, where the usual inference is valid. Alternatively, cointegration is tested ([Chapter 7](ch7.html)), the case in which the regression in levels has an economic meaning.

## @end

Regression between unrelated integrated series produces $t$-statistics that grow with $\sqrt{T}$, a non-degenerate $R^2$ and a DW statistic that goes to 0 [@Phillips1986]. In the simulation, the true hypothesis $b = 0$ is rejected in 92% of samples of 1,000 observations, and on real data Romania's HICP appears explained by the S&P 500 with $R^2 = 0.82$. In differences, both examples return to the nominal behaviour. Before any regression between trending series, their order of integration must therefore be established, and [the next section](#sec-3) presents the test that does this.

# The Dickey–Fuller test

The Dickey–Fuller test turns the question "deterministic or stochastic trend?" into a hypothesis about one coefficient: is the root of the AR polynomial equal to 1? The difficulty is that, under the null, the test statistic does not have the usual distribution but one of its own, shifted to the left. The section presents the test, its origin, the three versions according to the deterministic terms, the distributions and critical values, a worked example on Romanian GDP, the augmented version for autocorrelated errors, lag choice, the power of the test and the determination of the order of integration.

## Testing φ = 1 {h=b929d6da}

Start from an AR(1), $y_t = \phi y_{t-1} + \varepsilon_t$. Subtracting $y_{t-1}$ from both sides gives the form used by the test:

$$ \Delta y_t = \gamma y_{t-1} + \varepsilon_t, \qquad \gamma = \phi - 1. $$ {#eq:df}

The null hypothesis is $H_0$: $\gamma = 0$, that is $\phi = 1$, a unit root and $y_t \sim I(1)$. The alternative is $H_1$: $\gamma < 0$, that is $|\phi| < 1$ and a stationary series. The test is one-sided, to the left. Its statistic is the usual $t$-ratio from the OLS regression @eq:df, $\tau = \hat\gamma/\mathrm{SE}(\hat\gamma)$, where $\mathrm{SE}$ is the standard error. A strongly negative value of $\tau$ is evidence against the unit root.

The novelty, established by @DickeyFuller1979, is that under $H_0$ the statistic $\tau$ follows neither the Student law nor the Normal distribution, not even asymptotically. The reason is that the regressor $y_{t-1}$ is itself a random walk, not a stationary variable. The sum $\sum y_{t-1}^2$ grows like $T^2$, not $T$, and the estimator $\hat\phi$ converges to 1 at rate $T$, not $\sqrt{T}$: this property is called superconsistency. The limit of $\tau$ is a functional of Brownian motion $W$,

$$ \tau \;\xrightarrow{d}\; \frac{\tfrac12\big(W(1)^2 - 1\big)}{\big(\int_0^1 W(r)^2\,dr\big)^{1/2}}, $$

a skewed distribution, with its mass shifted to the left [@Hamilton1994, ch. 17]. It has no closed form, so the critical values come from simulation. The first tables were published by @Fuller1996; today the response surfaces of @MacKinnon2010 are used, which give the critical value for any $T$.

## Dickey and Fuller at Iowa State University {h=1a19e3c5}

The test comes from the statistics department of Iowa State University, one of the first centres of applied statistics in the United States, housed in Snedecor Hall, shown in the photograph. Wayne A. Fuller, a statistician at Iowa State, is the author of the textbook *Introduction to Statistical Time Series* [@Fuller1996], which contains the tables of critical values. David A. Dickey completed his doctorate with him at Iowa State in 1976 and then moved to North Carolina State University.

Their contributions form the basis of the section. @DickeyFuller1979 derived the distribution of the estimator $\hat\phi$ and of the statistic $\tau$ when the series has a unit root. @DickeyFuller1981 proposed joint tests, of the likelihood-ratio type, for the unit root and for the deterministic terms. @SaidDickey1984 extended the test to ARMA errors, through the augmented version presented below.

## Three Dickey–Fuller regressions {h=b9a68065}

Regression @eq:df has as its alternative a stationary process with mean zero, rarely a realistic description of the data. The test is therefore applied in three versions, according to the deterministic terms included. The version without constant, $\Delta y_t = \gamma y_{t-1} + \varepsilon_t$, has as alternative a stationary process with mean 0. The version with constant, $\Delta y_t = c + \gamma y_{t-1} + \varepsilon_t$, has as null a random walk, with drift if $c \ne 0$, and as alternative a process stationary around a non-zero mean. It suits series without a trend: interest rates, inflation, exchange rates, returns.

The version with constant and trend is

$$ \Delta y_t = c + bt + \gamma y_{t-1} + \varepsilon_t, $$

where $c$ is the constant and $bt$ a linear trend with slope $b$. The null is a random walk with drift and the alternative a trend-stationary series. This is exactly the choice between @eq:ds and @eq:ts in section 1, so this version is used for trending series: log GDP, log prices and log price indices.

Each version has its own distribution of $\tau$ and its own critical values. The deterministic terms change the distribution because the regression has to estimate them, and their estimation absorbs part of the variation of $y_{t-1}$ that would otherwise be attributed to $\gamma$.

## The Dickey–Fuller distributions {h=8c5c5a42}

The chart shows the distributions of the statistic $\tau$ under the null, obtained from 20,000 simulated random walks with $T = 250$ observations. The green curve corresponds to the regression without constant, the blue one to the regression with constant and the red one to the regression with constant and trend; the dotted curve is the $N(0,1)$ density. The vertical dashed lines mark the 5% critical values of @MacKinnon2010 for each version, and the dotted line the value $-1.645$ of the Normal distribution.

Each density should be read together with its vertical line: to the left of the line lies 5% of the mass of the distribution, the probability of rejecting a true unit root.

## Interpreting the three distributions {h=d9258332}

All three distributions lie to the left of $N(0,1)$, and the shift grows with the number of deterministic terms. The simulated 5% quantiles are $-1.95$ without constant, $-2.86$ with constant and $-3.42$ with constant and trend. The tabulated values for the same cases are $-1.94$, $-2.87$ and $-3.43$, so the simulation reproduces the tables to the second decimal.

The difference from the Normal distribution has large consequences. If the one-sided critical value of the Normal distribution, $-1.645$, is used, a true unit root is "rejected" in 46% of samples in the regression with constant and in 77% in the regression with constant and trend, not in 5%. The wrong table turns most random walks into apparently stationary series.

## Critical values of the Dickey–Fuller test {h=969c0835}

The table gives the 1%, 5% and 10% critical values for the three regressions and for four sample sizes: $T = 50$, 100, 250 and the limit $T \to \infty$. The values come from the response surfaces of @MacKinnon2010, the same ones used by the *statsmodels* library, and the p-values reported by software are computed with the approximations of @MacKinnon1994.

The decision rule is one-sided: the unit root is rejected at 5% if $\tau$ is below the 5% critical value, that is more negative than it. For example, in the regression with constant and trend for $T = 100$, the 5% critical value is $-3.46$; a $\tau = -3.0$ does not reject the unit root, although it would reject it with the Normal table.

The values depend little on $T$: in the regression with constant, the 5% value moves from $-2.92$ for $T = 50$ to $-2.86$ asymptotically. They depend strongly on the deterministic terms, and the same critical values apply unchanged to the augmented ADF test.

## Worked example: Romanian real GDP {h=ff90fc5b}

The example applies the test to $y_t = 100\ln$ of Romania's real GDP, quarterly, from 2000Q1 to 2026Q2. The series trends, so the regression with constant and trend is used, here without lags, with $T = 105$ differences. OLS gives $\hat\gamma = -0.0961$ and $\mathrm{SE}(\hat\gamma) = 0.0416$, so the estimated autoregressive coefficient is $\hat\phi = 1 + \hat\gamma = 0.904$.

The test statistic is $\tau = -0.0961/0.0416 = -2.31$. The 5% critical value for the regression with constant and trend and this $T$ is $-3.45$. Since $-2.31 > -3.45$, the unit root is not rejected; the p-value is 0.43. With the critical value of the Normal distribution, $-1.645$, the unit root would have been rejected: the wrong table gives the wrong answer here.

The interpretation requires caution, because "not rejected" does not mean "proven". If the estimated value $\phi = 0.904$ were the true one, a shock would halve in $\ln 0.5/\ln 0.904 \approx 6.9$ quarters, so the series would be trend-stationary, with shocks that last about two years. With 106 observations, the test can barely distinguish this situation from $\phi = 1$, as the power curves below show.

## The augmented Dickey–Fuller test {h=9599f122}

The Dickey–Fuller regression assumes that the error $\varepsilon_t$ is white noise. If $\Delta y_t$ is autocorrelated, the autocorrelation passes into the error and the distribution of $\tau$ changes: the size of the test, the probability of rejecting a true unit root, is no longer the nominal one. The remedy is to add past differences $\Delta y_{t-j}$ as regressors, which absorb the autocorrelation. The result is the augmented Dickey–Fuller (ADF) regression:

$$ \begin{aligned} \Delta y_t ={}& c + bt + \gamma y_{t-1} \\ &+ \sum_{j=1}^{k} \delta_j\,\Delta y_{t-j} + \varepsilon_t. \end{aligned} $$ {#eq:adf}

In @eq:adf, $\delta_j$ are the coefficients of the past differences and $k$ is their number, called the lag length. The statistic remains $\tau = \hat\gamma/\mathrm{SE}(\hat\gamma)$, and the critical values are the same as in the regression without lags. For an AR($p$) in levels, @eq:adf is exact: any AR($p$) can be rewritten as a regression of $\Delta y_t$ on $y_{t-1}$ and $p-1$ past differences, with $k = p - 1$ and $\gamma = -\phi(1)$. The unit root, $\phi(1) = 0$, therefore still corresponds to $\gamma = 0$.

When the errors are ARMA, with an MA component, no finite $k$ is exact. @SaidDickey1984 showed that the test remains valid if $k$ grows slowly with $T$: the past differences approximate the MA part by a long AR. The ADF test is the default unit-root test in statistical software, for example the *adfuller* function in *statsmodels*.

## Choosing the number of lags k {h=84b9df57}

The lag length is chosen in two steps. First a maximum is set; the rule of @Schwert1989 is $k_{\max} = 12\,(T/100)^{1/4}$, which for $T = 105$ gives 12.15, rounded up to 13. Then $k \le k_{\max}$ is chosen by an information criterion, AIC or BIC ([Chapter 2](ch2.html)). All the regressions being compared must be estimated on the same sample, the one left after dropping the first $k_{\max}$ observations; otherwise the values of the criterion are not comparable.

The choice has a cost in both directions. With too few lags, the errors remain autocorrelated and the size of the test is wrong; with a negative MA component, the test rejects a true unit root far too often. With too many lags, the test estimates useless parameters and loses power.

For series with a negative MA component, @NgPerron2001 propose a modified Akaike information criterion (MAIC), which gives a size closer to the nominal one. Combined with the generalised least squares (GLS) detrending of @ElliottRothenbergStock1996, which increases power, it forms the DF-GLS test. Whatever the method, the result is checked with the Ljung–Box test [@LjungBox1978] on the residuals of the ADF regression, and the report gives $k$ together with $\tau$.

## Size and power of the Dickey–Fuller test {h=b767e1e5}

The power of a test is the probability of rejecting the null when it is false. The simulation measures it for the Dickey–Fuller test: for each value of $\phi$, 2,000 AR(1) paths $y_t = \phi y_{t-1} + \varepsilon_t$ are generated, and the curve shows the share of rejections at 5%. At $\phi = 1$ the null is true, and the share of rejections is the size of the test.

The left panel uses the regression with constant and three sample sizes: $T = 100$ (red), 250 (orange) and 500 (green). The right panel fixes $T = 100$ and compares the regression with constant (blue) with the regression with constant and trend (purple), although the simulated data have no trend. The horizontal dotted line marks the 5% level.

## Interpreting the power curves {h=f7460766}

At $\phi = 1$, all curves end around 5%: with a constant, the rejection rates are 5.30%, 5.20% and 4.65% for $T = 100$, 250 and 500 (4.85%, 5.20% and 5.30% with constant and trend), within the simulation error of 2,000 paths, so the test respects its nominal level for all three sample sizes. The problem is power near 1. At $\phi = 0.95$, power is 12% with $T = 100$, 47% with $T = 250$ and 97% with $T = 500$. A quarterly series of 25 years, $T = 100$, almost never distinguishes a stationary process with $\phi = 0.95$ from a unit root.

The relevant length is the span covered, not the number of observations. A quarterly process with $\phi = 0.95$ observed monthly has a monthly coefficient even closer to 1, and switching to daily data over two years adds observations but no information about long-run reversion. Daily data over two years do not help to distinguish a persistence of several years from a unit root.

The right panel shows the cost of an unnecessary deterministic term. At $T = 100$ and $\phi = 0.9$, power is 35% in the regression with constant and only 19% in the regression with constant and trend. The trend term is not needed here, but the distribution with trend is shifted further to the left, and the test rejects less easily.

## Choosing the deterministic terms {h=2507c714}

The rule for choosing the deterministic terms is that the alternative hypothesis should be a plausible description of the data. For trending series, such as log GDP and log prices, the plausible alternative is stationarity around a trend, so the regression includes constant and trend. For series that fluctuate around a level, such as interest rates, inflation, exchange rates and returns, the alternative is stationarity around a mean, so the regression includes only the constant.

The errors have asymmetric costs. If a needed trend is omitted, a trend-stationary series appears to have a unit root: the regression with constant can describe the slope only through a $\hat\phi$ close to 1, and the test almost never rejects. If an unnecessary trend is added, the test remains valid but loses power, as the previous chart shows.

In practice, the chart of the series is examined first and the regression is chosen afterwards. For ambiguous cases, @DickeyFuller1981 propose joint $F$-type tests for the unit root and for the trend coefficient.

## The number of differences: I(1) or I(2) {h=c403d9c1}

When a series may be $I(2)$, the order of the tests matters. @DickeyPantula1987 recommend testing from the highest plausible order downwards. Step 1 applies ADF to $\Delta y_t$; if the test does not reject, $\Delta y_t$ has a unit root and $y_t$ may be $I(2)$. Step 2, only if the first step rejected, applies ADF to $y_t$: a rejection indicates $I(0)$, a non-rejection $I(1)$.

The reason for the downward order is that the ADF test on $y_t$ assumes at most one unit root. If $y_t \sim I(2)$, the null of the test on levels is also wrong, and the distribution of $\tau$ is no longer the tabulated one; the result cannot be interpreted.

The example is Romanian consumer prices after 2005, from the table of section 4. The log HICP, $\ln P$, has a unit root. For 12-month inflation $\pi$, the ADF test has p-value 0.32 and the KPSS statistic is 0.404, so neither test rejects. The change in inflation, $\Delta\pi$, is clearly stationary. Hence $\ln P$ is $I(1)$ or $I(2)$, depending on how persistent inflation is, a question to which section 7 returns.

## Recap: The Dickey–Fuller test {h=fbdb9383}

The ADF test estimates regression @eq:adf and tests $H_0$: $\gamma = 0$, comparing $\tau$ with the Dickey–Fuller critical values, not with those of the Normal distribution. The deterministic terms are chosen from the chart of the series, so that the alternative is plausible, and the lag length by AIC or BIC, up to $k_{\max} = 12(T/100)^{1/4}$.

The test has low power near $\phi = 1$, especially over short spans. Non-rejection is therefore weak evidence in favour of a unit root. The order of integration is established from the top down: first $\Delta y_t$ is tested, then $y_t$.

## @end

The Dickey–Fuller statistic is an ordinary $t$-ratio, but its distribution under a unit root is shifted to the left and depends on the deterministic terms; the 5% critical values are about $-1.94$, $-2.86$ and $-3.41$ asymptotically, not $-1.645$. The augmented version absorbs autocorrelation through past differences. On Romanian GDP, $\tau = -2.31$ does not reject the unit root, but the simulations show that, at this sample length, the test hardly distinguishes $\phi = 0.9$ from $\phi = 1$. [The next section](#sec-4) adds a test with the same null and one with the opposite null.

# Phillips–Perron, KPSS and their joint use

The ADF test handles autocorrelated errors through extra lags and has the unit root as its null. The section presents two alternatives. The Phillips–Perron test keeps the simple regression and corrects the statistic through an estimate of the long-run variance. The KPSS test reverses the hypotheses: stationarity is the null. Used together, tests with opposite nulls allow a more reliable classification, applied then to a table of real series.

## The Phillips–Perron test {h=5d395deb}

@PhillipsPerron1988 handle autocorrelation differently from ADF: they keep the simple Dickey–Fuller regression, without past differences, and correct the statistic $\tau$ after estimation. The corrected statistic is

$$ Z_\tau = \sqrt{\frac{\hat\gamma_0}{\hat\lambda^2}}\;\tau \;-\; \frac{(\hat\lambda^2 - \hat\gamma_0)\,T\,\mathrm{SE}(\hat\gamma)}{2\hat\lambda\,s}, $$

where $\hat\gamma_0$ is the variance of the residuals, $s^2$ their OLS variance (with the degrees-of-freedom correction) and $\hat\lambda^2$ their long-run variance.

The long-run variance, $\lambda^2 = \lim_{T\to\infty}\mathrm{Var}\big(\sum_{t=1}^{T} e_t\big)/T$, includes the autocovariances of the residuals $e_t$ in addition to their variance. It is estimated with Bartlett weights, as in the Newey–West estimator:

$$ \hat\lambda^2 = \hat\gamma_0 + 2\sum_{j=1}^{L}\Big(1 - \frac{j}{L+1}\Big)\hat\gamma_j, $$

where $\hat\gamma_j$ are the autocovariances of the residuals and $L$ is the number of autocovariances included, called the bandwidth. If the residuals are white noise, $\hat\lambda^2 = \hat\gamma_0$, the second term vanishes and $Z_\tau = \tau$. The correction therefore matters only when the residuals are autocorrelated.

The test has the same null and the same critical values as ADF. Its advantages are robustness to heteroskedasticity and the fact that it requires no lag choice, only a bandwidth $L$. Its weak point, documented by @Schwert1989, is the large size distortion when the errors have a negative MA component: the test then rejects a true unit root far too often.

## The KPSS test: stationarity as the null {h=39430728}

The ADF and PP tests start from the unit root and ask for evidence against it. @KwiatkowskiEtAl1992 proposed the KPSS (Kwiatkowski–Phillips–Schmidt–Shin) test, which reverses the hypotheses. The model decomposes the series into a deterministic trend, a random walk and stationary noise:

$$ y_t = \xi t + r_t + \varepsilon_t, \qquad r_t = r_{t-1} + u_t. $$

Here $\xi t$ is the deterministic trend, $r_t$ a random walk, a level that wanders, with shocks $u_t \sim \mathrm{WN}(0, \sigma_u^2)$, and $\varepsilon_t$ stationary noise. The null is $H_0$: $\sigma_u^2 = 0$; the random walk is then a constant, and $y_t$ is stationary around a level or a trend. The alternative $H_1$: $\sigma_u^2 > 0$ means a unit root.

The statistic is built from the residuals $e_t$ of the regression of $y_t$ on a constant, or on a constant and a trend. With the partial sums $S_t = e_1 + \dots + e_t$,

$$ \eta = \frac{1}{T^2\hat\lambda^2}\sum_{t=1}^{T} S_t^2, $$ {#eq:kpss}

where $\hat\lambda^2$ is the long-run variance of $e_t$. Stationarity is rejected for large values of $\eta$. The 5% critical values are 0.463 for the level version and 0.146 for the trend version, and the 1% values are 0.739 and 0.216. KPSS is thus a test of stationarity, not of a unit root: the burden of proof moves from stationarity to the unit root.

## KPSS: the role of partial sums {h=956d22ee}

The chart shows why statistic @eq:kpss uses partial sums. The left panel compares two demeaned series of length $T = 250$: a stationary AR(1) with $\phi = 0.5$ (blue) and a random walk (red). The right panel shows their partial sums $S_t$, the cumulative deviations from the mean.

The KPSS statistic in the level version is 0.16 for the AR(1) series and 0.73 for the random walk. The 5% critical value is 0.463, so the test does not reject stationarity for the first series and rejects it for the second.

## Interpreting the partial sums {h=367c002f}

In the stationary series, deviations from the mean offset each other quickly: a positive value is soon followed by negative ones, and the partial sum $S_t$ stays close to 0. The sum $\sum S_t^2$ grows like $T^2$, like the normalisation in @eq:kpss, so $\eta$ stays bounded as $T$ grows.

In the random walk, the series spends long periods on the same side of the mean, and $S_t$ moves far from 0: on the chart, up to about 300. In this case $\sum S_t^2$ grows like $T^4$, the statistic $\eta$ grows with $T$, and the test rejects with increasing probability.

The idea is the same as in the cumulative sum (CUSUM) charts of quality control: cumulative deviations from a reference value reveal a level that wanders, even when the individual deviations are small.

## ADF and KPSS together {h=fc1bafb2}

The ADF and KPSS tests have opposite nulls, so their combined results give four situations. If ADF rejects and KPSS does not, both point to an $I(0)$ series. If ADF does not reject and KPSS rejects, both point to $I(1)$. In these two cases agreement is stronger evidence than either test alone, because each test has failed to find evidence for the other's model.

The inconclusive situation arises when neither rejects: the data cannot decide. It is frequent in short samples, where both tests have low power. The decision is then made according to the purpose; for forecasting, $d = 1$ is the prudent choice, because it produces wider intervals.

The conflict arises when both reject: the series looks neither stationary nor integrated in the simple sense of the tests. The usual causes are a structural break (section 5), long memory, the fractional integration of [Chapter 8](ch8.html), or outliers.

## Unit-root tests on real series {h=faaa04a0}

The table applies the three tests to Romanian and US series. The column "det." gives the deterministic terms: "c, t" (constant and trend) for trending levels and "c" (constant) for differences, returns, inflation and the exchange rate. The column $T$ gives the number of observations, the ADF column the statistic and its p-value, $k$ the lag length chosen by AIC, and the PP and KPSS columns the statistics of the other two tests. The last column combines ADF and KPSS according to the rules of the previous frame.

The 5% critical values are about $-2.86$ (c) and $-3.41$ (c, t) for ADF and PP, and for KPSS 0.463 (c) and 0.146 (c, t). For the daily series, where $T$ is in the thousands, the asymptotic values are practically exact. Each level appears together with its difference, so the table is read in pairs.

## Interpreting the unit-root table {h=7307a745}

Log stock prices (BET, S&P 500), the EUR/RON rate, log GDP and log HICP are classified as $I(1)$, and their differences as $I(0)$. For Romanian GDP, ADF has p-value 0.43 and KPSS is 0.198, above the 5% critical value for the trend version, 0.146, though below the 1% value, 0.216. The result is a unit root, as @NelsonPlosser1982 found for the United States. Quarterly growth has ADF $-10.77$ and KPSS 0.263, so it is clearly $I(0)$.

Romanian 12-month inflation is the inconclusive case: ADF $-1.94$, with p-value 0.315, does not reject the unit root, and KPSS 0.404, below 0.463, does not reject stationarity. The series is highly persistent and possibly affected by breaks; sections 5 and 7 return to it.

The annual flow of the Nile is the conflicting case: ADF $-4.05$, with p-value 0.001, rejects the unit root, and KPSS 0.869 rejects stationarity. [Chapter 1](ch1.html) showed a level shift in 1898, and the next section uses it to explain this conflict.

For daily returns, PP gives much more negative statistics than ADF, for example $-74.6$ against $-16.5$ for the BET. The difference comes from the different handling of autocorrelation: ADF uses 18 lags, PP corrects the simple regression. Both reject the unit root by a wide margin, however, so the verdict is the same.

## Recap: Phillips–Perron and KPSS {h=cfb266d5}

The Phillips–Perron test uses the simple Dickey–Fuller regression and corrects the statistic through the long-run variance of the residuals. It has the same null and the same critical values as ADF, but is sensitive to negative MA components. The KPSS test has stationarity as its null and rejects it when the partial sums of the residuals are large.

ADF and KPSS are interpreted together. Their agreement classifies the series as $I(0)$ or $I(1)$; if neither rejects, the result is inconclusive; if both reject, the conflict calls for a search for a break or for long memory.

## @end

On the real series of the chapter, the three tests give a coherent picture: the levels of prices, exchange rates and GDP are $I(1)$, and their differences $I(0)$. Two cases do not fit the pattern. Romanian inflation is inconclusive, and the Nile flow produces a conflict, with both tests rejecting. [The next section](#sec-5) shows that a single level shift can produce exactly this kind of result.

# Structural breaks

Unit-root tests compare two models: a series stationary around a fixed mean or trend and a series with a stochastic trend. A third possibility is a series stationary around a mean or a trend that changes once, through a structural break. The section shows that such a break makes the usual tests fail to reject the unit root, presents the Zivot–Andrews test, which searches for the break date, and applies it to the flow of the Nile and to the EUR/RON rate.

## Perron (1989): structural breaks and unit-root tests {h=542dff9a}

@Perron1989 observed that a series stationary around a mean or a trend that changes once looks highly persistent. The Dickey–Fuller regression does not contain the break and can explain it only through a $\hat\phi$ close to 1: after the jump, the series does not return to the old mean, which resembles a permanent shock. The test thus loses power, and the unit root is not rejected.

Perron re-examined the Nelson–Plosser series with two breaks treated as known: the 1929 crash, treated as a change in level, and the 1973 oil shock, treated as a change in slope. With break dummies included in the test regression, the unit root was rejected for most series. His interpretation was that most shocks are temporary and that only a few rare events have permanent effects.

The main criticism concerned the procedure: the break dates were chosen after looking at the series. Choosing the date from the same data makes the critical values computed for a date fixed in advance too lenient, and the test rejects too easily.

## The ADF test under a level shift {h=b149d199}

The simulation builds a stationary series with a known break. The left panel shows AR(1) noise with $\phi = 0.6$ around a mean that jumps from 0 to 5 at $t = 100$; the red dashed line is the true mean. The series is stationary on each side of the break, and a level shift is its only non-stationarity.

The right panel shows the $t$-statistic of the coefficient on $y_{t-1}$ from an ADF regression that includes a level dummy, computed for each possible break date. The horizontal red dashed line is the 5% critical value of the Zivot–Andrews test, $-4.81$, and the vertical dotted line marks the true date, $t = 100$.

## Interpreting the simulated break {h=2734bad4}

The ADF test with constant, which ignores the break, has p-value 0.17: the series, although stationary, appears to have a unit root. The result is not specific to this path. Over 2,000 simulated series with the same break, the Dickey–Fuller test rejects the unit root in 72% of cases, against 100% for the same series without the break. The level shift thus reduces power by 28 percentage points.

When the break is modelled, the picture changes. The statistic of the Zivot–Andrews test, presented in the next frame, is $-7.47$, far below the 5% critical value, $-4.81$, so the unit root is clearly rejected. The curve on the right, the $t$-statistic with a level dummy at each candidate date, reaches its minimum, $-8.37$, at the true date, $t = 100$, while far from this date the statistic stays around $-4$, above the critical value: only a regression that places the break close to its correct location detects stationarity.

## Zivot and Andrews (1992): an unknown break date {h=8d4b1281}

@ZivotAndrews1992 answered the criticism of Perron by letting the data choose the break date $T_B$. For each candidate date in the central 70% of the sample, the ADF regression is estimated with a break dummy:

$$ \begin{aligned} \Delta y_t ={}& c + \theta DU_t + bt + \gamma y_{t-1} \\ &+ \sum_{j}\delta_j\,\Delta y_{t-j} + \varepsilon_t. \end{aligned} $$

The dummy variable $DU_t$ is 0 for $t \le T_B$, before the break, and 1 for $t > T_B$, and $\theta$ measures the size of the level jump. The test statistic is the minimum $t$-ratio of $\gamma$ over all candidate dates, the date most favourable to the stationary alternative. The ends of the sample are excluded so that each regime has enough observations.

The test has three versions: a break in level, as above, a break in slope and a break in both. Because the statistic is a minimum over many dates, its distribution is shifted further to the left than that of $\tau$, and the 5% critical values are about $-4.81$ for the level break and $-5.07$ for the break in both. The null is a unit root without a break, and the alternative a stationary series with one break. For several breaks the method of @BaiPerron1998 is used. A rejection does not prove that the break is the correct explanation: it only shows that the unit-root model without a break is rejected in favour of the alternative tested.

## Zivot–Andrews on two real series {h=fee45e3b}

The chart applies the level-break test to two series. On the left is the annual flow of the Nile at Aswan, 1871–1970, the classic data set included in *statsmodels*. On the right is $100\ln$ EUR/RON, the BNR reference rate at the end of each month. The vertical dotted line marks the break date chosen by the test, and the red dashed lines the means before and after the break.

The two series have different structures. The Nile fluctuates around two distinct levels, without a trend. The EUR/RON rate has a large jump in 2008 but keeps rising after it.

## Interpreting the two breaks {h=ae729e40}

For the Nile, the Zivot–Andrews statistic is $-6.86$, below the 5% critical value, $-4.81$, with the break in 1898. The mean flow falls from 1,098 before the break to 853 after it. The date matches history: the construction of the first Aswan dam, between 1898 and 1902, and the shift to a drier climate. The series is therefore stationary around a mean that changes once, and the ADF–KPSS conflict in the table of section 4 is explained: KPSS rejects stationarity around a constant mean, and ADF rejects the unit root.

For month-end EUR/RON, the ADF test has p-value 0.55 and does not reject the unit root, but the Zivot–Andrews statistic is $-6.89$, with the break in October 2008, at the depreciation of the leu during the financial crisis. The mean rate is 3.50 lei before the break and 4.62 lei after. The result is consistent with a managed-float regime, with long periods of stability and rare adjustments, rather than with a pure random walk.

The interpretation has a limit visible on the chart, however: after 2008 the rate keeps rising, and the values of 2020–2026 lie above the post-break mean. The model "stationary around a break" is itself a simplification; the rejection only shows that it describes the data better than a random walk without a break.

## Recap: Structural breaks {h=f5897681}

A single change in the mean or in the trend makes unit-root tests reject less often: the stationary series with a break looks integrated. The Zivot–Andrews test searches for the break date and uses the more negative critical values that correspond to this search.

The choice between a unit root and a break cannot be made on statistical grounds alone. The estimated break date must be compared with known events, such as the Aswan dam or the 2008 crisis; a break without a corresponding event is a weaker explanation.

## @end

Structural breaks are the usual explanation of a conflict between ADF and KPSS and a source of misleading non-rejections. In a simulated series with a level jump, the power of the Dickey–Fuller test falls from 100% to 72%, and modelling the break brings the statistic to $-7.47$. On real data, the Zivot–Andrews test dates the Nile break in 1898 and the EUR/RON break in October 2008. With the order of integration established, [the next section](#sec-6) builds forecasting models for integrated series.

# ARIMA(p,d,q) models

When a series is integrated, the forecasting model has to contain the differencing. The ARIMA (autoregressive integrated moving average) model combines differencing of order $d$ with an ARMA$(p,q)$ model for the differenced series. The section defines the model and its special cases, adapts the Box–Jenkins cycle to series with unit roots and applies it to Romanian real GDP: identification, choice of orders by AICc, residual checks and forecasting. The second part derives the forecast-error variance and explains why intervals grow with the horizon, and the last part treats over-differencing.

## The ARIMA model {h=1dac7804}

A process $y_t$ follows an ARIMA$(p,d,q)$ model if its difference of order $d$, $\Delta^d y_t$, is a stationary and invertible ARMA$(p,q)$ process:

$$ \phi(L)\,(1 - L)^d\,y_t = c + \theta(L)\,\varepsilon_t. $$ {#eq:arima}

In @eq:arima, $\varepsilon_t \sim \mathrm{WN}(0, \sigma^2)$ is white noise, $p$ is the order of the AR part, $d$ the number of differences, that is the number of unit roots, and $q$ the order of the MA part. The polynomials $\phi(L) = 1 - \phi_1 L - \dots - \phi_p L^p$ and $\theta(L) = 1 + \theta_1 L + \dots + \theta_q L^q$ are those of [Chapter 2](ch2.html), with roots outside the unit circle.

The autoregressive polynomial of the levels is the product $\phi(z)(1 - z)^d$: it has exactly $d$ roots on the unit circle, all equal to 1, and the others outside the circle. The name "integrated" comes from the operation inverse to differencing: $y_t$ is obtained by cumulating an ARMA process $d$ times.

The model was popularised by Box and Jenkins in the 1970s, and their methodology is set out in @BoxJenkinsReinsel2008. In economic and financial applications $d \le 2$, and the most frequent case is $d = 1$.

## Special cases and the constant {h=b90266d1}

Several well-known models are special cases of @eq:arima. ARIMA(0,1,0), $\Delta y_t = c + \varepsilon_t$, is the random walk, with drift $c$. ARIMA(1,1,0), $\Delta y_t = c + \phi\Delta y_{t-1} + \varepsilon_t$, describes growth with momentum: high growth this quarter signals above-average growth in the next. ARIMA(0,1,1), $\Delta y_t = \varepsilon_t + \theta\varepsilon_{t-1}$, generates the same forecasts as simple exponential smoothing from [Chapter 0](ch0.html), with parameter $\alpha = 1 + \theta$. ARIMA(0,2,2) is the model behind Holt's linear method.

The constant $c$ changes the shape of the long-run forecast [@HyndmanAthanasopoulos2021, ch. 9.7]. For $d = 0$, forecasts converge to the mean of the process, $c/\phi(1)$. For $d = 1$, a non-zero constant is a drift: forecasts follow a line whose slope equals the average growth. Without the constant, long-run forecasts are a horizontal line.

For $d = 2$, a non-zero constant means that the second difference has a non-zero mean, so forecasts follow a quadratic trend. Such an extrapolation is rarely reasonable, and in practice $c = 0$ is used when $d = 2$.

## The Box–Jenkins cycle with unit roots {h=6704e4f3}

The Box–Jenkins cycle of [Chapter 2](ch2.html) is extended by a step for choosing $d$. Step 1 is the transformation: if the variance grows with the level of the series, the logarithm or a Box–Cox transformation is applied ([Chapter 1](ch1.html)). Step 2 chooses $d$ from the chart, the ACF and the ADF and KPSS tests; testing starts with $\Delta y_t$, and differencing stops at the first stationary difference.

Step 3 chooses $p$ and $q$ from the ACF and the partial autocorrelation function (PACF) of $\Delta^d y_t$, then compares the candidate models by the AICc criterion, all with the same $d$. Step 4 estimates the model by maximum likelihood and decides whether to include the constant.

Step 5 checks the residuals: the ACF, the Ljung–Box test with $m - p - q$ degrees of freedom and the outliers. Step 6 produces forecasts of the levels, with intervals. If the checks fail, the cycle restarts from step 2 or 3.

## Romanian GDP: identification {h=1d097468}

The application uses Romania's quarterly real GDP, seasonally adjusted (Eurostat), from 2000Q1 to 2026Q2, with $T = 106$ observations. The series is transformed as $y_t = 100\ln Y_t$, so that its difference, $100\Delta\ln Y_t$, is approximately the quarterly growth rate in percent.

The top-left panel shows the level $100\ln Y_t$, and the top-right panel quarterly growth, with its mean as a dashed line. The bottom panels show the ACF and PACF of growth for lags up to 12 quarters, with the band $\pm 1.96/\sqrt{T}$, outside which a sample autocorrelation is significant at 5% for white noise.

## Interpreting the identification plots {h=8617316f}

The order of integration is $d = 1$: the table of section 4 classifies the level as $I(1)$ and growth as $I(0)$. The growth rate fluctuates around a mean of 0.81% per quarter, about 3.2% per year. Two values stand out: $-10.2\%$ in 2009Q1, in the global financial crisis, and $-8.6\%$ in 2020Q2, in the pandemic.

For $p$ and $q$, no autocorrelation and no partial autocorrelation leaves the band $\pm 0.19$. The first two autocorrelations are $\hat\rho(1) = -0.05$ and $\hat\rho(2) = -0.14$. The Ljung–Box test for the first 8 lags gives $Q^*(8) = 7.3$, with p-value 0.51, so the autocorrelations are not separated from noise: the growth rate looks like white noise around its mean.

The plots therefore suggest the ARIMA(0,1,0) model with drift, a random walk with drift for $100\ln Y_t$. The next frame compares this model with the alternatives of higher order.

## Model choice by AICc {h=37337b44}

The table gives the AICc criterion for ARIMA$(p,1,q)$ models with drift, with $p, q \in \{0, 1, 2\}$, estimated by maximum likelihood on $100\ln Y_t$. AICc is the Akaike criterion with the small-sample correction ([Chapter 2](ch2.html)), and a lower AICc indicates a better model. The sign † marks the ARIMA(2,1,2) model, whose optimisation did not converge.

The lowest AICc, 491.9, belongs to ARIMA(0,1,0). The next best model has an AICc higher by 1.7 points. AICc differences below 2 points do not indicate a relevant difference between models, and in such cases the simplest model is kept; here it is also the one with the minimum AICc.

The estimated model is $100\Delta\ln Y_t = 0.81 + \hat\varepsilon_t$. The estimated drift is $\hat c = 0.81$, with standard error 0.30 and $t = 2.72$, so it is significantly different from zero, and the standard deviation of the shocks is $\hat\sigma = 2.47$ percentage points per quarter.

## Romanian GDP: residual checks {h=3a7ecf35}

The check uses the residuals of the ARIMA(0,1,0) model with drift, the deviations of quarterly growth from its mean of 0.81. The top-left panel shows them over time. The top-right panel shows their ACF, with the band $\pm 1.96/\sqrt{T}$. The bottom-left panel gives the p-values of the Ljung–Box test for $m = 2, \dots, 16$ lags, against the 5% line. The bottom-right panel compares the histogram of the residuals with the Normal density with the same mean and variance.

The first three panels check whether autocorrelation remains, that is whether the model has extracted all the linear dynamics. The last checks the normality assumption, on which forecast intervals of the form $\pm 1.96\sigma$ rely.

## Interpreting the residual checks {h=79860b71}

No detectable autocorrelation remains. The Ljung–Box statistic for 8 lags is $Q^*(8) = 7.3$, with p-value 0.51, and for 12 lags the p-value is 0.54. All p-values in the bottom-left panel lie above 5%.

The residuals are not Normal, however. Their kurtosis is 7.9, against 3 for the Normal distribution, and the Jarque–Bera statistic is 145, far above the critical value. The largest residual in absolute value is $-11.0$, in 2009Q1. The fat tails are produced by the two crises, 2009 and 2020, which appear as isolated values on the left of the histogram.

The consequence concerns the forecast intervals. Intervals based on the Normal distribution, used in the following frames, are too narrow in a crisis and slightly too wide in calm periods, because $\hat\sigma$ mixes the two regimes. There are two usual corrections: dummy variables for 2009Q1 and 2020Q2, or a residual bootstrap for building the intervals.

## Forecasting with ARIMA {h=b722ae41}

Point forecasts are obtained by writing the model for the levels and applying it recursively, with the observed values up to $T$ and the future shocks replaced by their mean, $\hat\varepsilon_{T+j} = 0$ for $j \ge 1$. For example, ARIMA(1,1,0) without constant, $\Delta y_t = \phi\Delta y_{t-1} + \varepsilon_t$, is written in levels as $y_t = (1 + \phi)y_{t-1} - \phi y_{t-2} + \varepsilon_t$, an AR(2) with one unit root.

The forecast error at horizon $h$ is written with the weights $\psi_j$ of the MA($\infty$) representation of the levels, defined by $\psi(z) = \theta(z)/[\phi(z)(1 - z)^d]$:

$$ y_{T+h} - \hat y_{T+h} = \sum_{j=0}^{h-1}\psi_j\,\varepsilon_{T+h-j}, $$

and its variance is

$$ \mathrm{Var}(y_{T+h} - \hat y_{T+h}) = \sigma^2\sum_{j=0}^{h-1}\psi_j^2. $$ {#eq:fcvar}

The 95% forecast interval is $\hat y_{T+h} \pm 1.96\,\sigma\sqrt{\sum_{j<h}\psi_j^2}$. For $d = 0$, the weights $\psi_j$ go to zero geometrically, and the variance in @eq:fcvar converges to the variance of the process. For $d \ge 1$, the factor $(1 - z)^{-d}$ prevents the weights from going to zero, and the variance grows without bound with $h$.

Two cases have explicit formulas. For the random walk, $\psi_j = 1$ for every $j$, the variance is $h\sigma^2$, and the interval grows like $\sqrt{h}$. For ARIMA(0,1,1), $\psi_0 = 1$ and $\psi_j = 1 + \theta$ for $j \ge 1$, so the variance is $\sigma^2[1 + (h - 1)(1 + \theta)^2]$.

## Worked example: ARIMA(1,1,0) forecasts {h=bb2081f8}

The example uses the model $\Delta y_t = 0.6\,\Delta y_{t-1} + \varepsilon_t$, with $\sigma = 2$, and the last data $y_T = 108$ and $\Delta y_T = 5$. Point forecasts are obtained in two steps: the change is forecast, then added to the previous level. The first forecast change is $0.6 \cdot 5 = 3$, so $\hat y_{T+1} = 111.00$. The second is $0.6 \cdot 3 = 1.8$, so $\hat y_{T+2} = 112.80$, and the third $0.6 \cdot 1.8 = 1.08$, so $\hat y_{T+3} = 113.88$. Momentum makes the forecast changes decline geometrically towards zero, the mean of $\Delta y_t$ in the absence of a constant.

The $\psi$ weights of the levels are the partial sums of the powers of $\phi$: $\psi_0 = 1$, $\psi_1 = 1 + \phi = 1.60$ and $\psi_2 = 1 + \phi + \phi^2 = 1.96$. From @eq:fcvar, the forecast-error variances are 4 at one step, $4(1 + 1.60^2) = 14.24$ at two steps and $4(1 + 1.60^2 + 1.96^2) = 29.61$ at three steps. The half-widths of the 95% intervals, $1.96$ times the square root of the variance, are 3.92, 7.40 and 10.66.

The interval almost triples in three steps: the ratio is $10.66/3.92 = 2.72$. For a random walk, the same ratio would be $\sqrt{3} = 1.73$. Momentum $\phi = 0.6$ amplifies uncertainty, because today's shock also changes the following growth rates and therefore accumulates in the level with weights larger than 1.

## Growth of the forecast intervals with the horizon {h=0b03e51b}

The chart compares the half-width of the 95% forecast interval, in units of $\sigma$, for horizons 1–20 and five models. All start at $1.96\sigma$ at $h = 1$, because the one-step error is the shock $\varepsilon_{T+1}$ itself. The differences appear from the second step on and come only from the $\psi_j$ weights of each model.

The curves are: a stationary AR(1) with $\phi = 0.8$ ($d = 0$, green), the random walk ARIMA(0,1,0) (blue), ARIMA(1,1,0) with $\phi = 0.5$ (red), ARIMA(0,1,1) with $\theta = -0.5$ (orange) and ARIMA(0,2,0) (purple), which quickly leaves the chart.

## Interpreting the interval widths {h=06ebcb7c}

For $d = 0$, AR(1) with $\phi = 0.8$, the width levels off at $1.96\sigma/\sqrt{1 - \phi^2} = 1.96\sigma/0.6$, that is 3.27$\sigma$ at $h = 20$. The limit is the standard deviation of the stationary process: at long horizons the forecast is the mean, and the uncertainty is the dispersion of the series around it.

For $d = 1$, the width grows like $\sqrt{h}$, with a factor that depends on the ARMA part. At $h = 20$, the half-width is 8.77$\sigma$ for the random walk, 16.78$\sigma$ for ARIMA(1,1,0) with $\phi = 0.5$ and 4.70$\sigma$ for ARIMA(0,1,1) with $\theta = -0.5$. The negative MA component offsets part of each shock in the next period: from $\psi_j = 1 + \theta = 0.5$, only half of each shock remains permanently in the level, and the intervals are narrower.

For $d = 2$, the weights grow linearly, $\psi_j = j + 1$, and the half-width grows like $h^{3/2}$: at $h = 20$ it is 105.00$\sigma$. One extra difference thus makes long-run forecasts practically useless, which makes over-differencing a costly error.

## Romanian GDP: two forecasts {h=ab6d83ca}

The chart compares the forecasts of Romania's real GDP from two models estimated on the same sample, 2000Q1–2026Q2. The difference-stationary model is ARIMA(0,1,0) with drift (red), chosen above. The trend-stationary model is a linear trend with AR(1) errors (blue). The forecasts run to 2029Q2, 12 quarters ahead, and are transformed back from $100\ln$ into billions of euros at 2010 prices.

The shaded bands are the 80% and 95% intervals of each model. The observed series is shown from 2012, so that the 2020 recession and the recent evolution are clearly visible.

## Interpreting the GDP forecasts {h=1b7a0101}

The random walk with drift starts from the last observed value, EUR 48.8 bn at 2010 prices in 2026Q2, and grows by 0.81% per quarter: after 12 quarters it reaches EUR 53.8 bn. The half-width of the 95% interval, expressed in percent of GDP, is 4.8 after one quarter, 9.7 after four and 16.8 after twelve. The ratio of the last to the first, 3.46, is exactly $\sqrt{12}$, as the formula $h\sigma^2$ of the random walk requires.

The trend-stationary model pulls GDP back towards the old line, with an estimated AR(1) coefficient $\hat\phi = 0.92$. Because recent GDP lies below the line, the forecast grows faster: EUR 56.3 bn after 12 quarters, with a 95% half-width of 11.2%. The model is more optimistic and has narrower intervals, because it assumes that the recent slowdown is temporary.

The difference between the two forecasts, EUR 2.5 bn after three years, does not come from new data but from the assumption about $d$. The tests of section 4 lean towards $d = 1$, but the KPSS statistic rejects stationarity at 5%, not at 1%, and the power of the ADF test is low for 106 quarters, so a model with $\hat\phi = 0.92$ cannot be ruled out. The choice of $d$ is thus a forecasting decision with consequences for the level and for the uncertainty reported.

## Over-differencing {h=829e9050}

Over-differencing means differencing a series that is already stationary or trend-stationary. The simplest case shows the effect: if $\varepsilon_t$ is white noise, the difference $\Delta\varepsilon_t = \varepsilon_t - \varepsilon_{t-1}$ is an MA(1) with $\theta = -1$, a process with a unit root in its MA part ([Chapter 1](ch1.html)). The resulting process is not invertible: it has no AR($\infty$) representation, and its maximum-likelihood estimation is difficult, with estimates piling up at the boundary $\theta = -1$ [@PlosserSchwert1977].

The symptoms follow from the same example. The first-order autocorrelation of the differenced series is $\rho(1) = \theta/(1 + \theta^2) = -0.5$, so a $\hat\rho(1)$ close to $-0.5$ is a signal. The variance increases after differencing instead of falling: $\mathrm{Var}(\Delta\varepsilon_t) = 2\sigma^2$. An estimated MA coefficient close to $-1$ shows that the extra difference is cancelled by the MA part.

The practical rule is simple: difference only as long as the variance of the series falls and the tests call for differencing. The first difference that makes the variance rise is, as a rule, one too many.

## Over-differencing Romanian GDP {h=b83271ba}

The chart applies the diagnosis to Romanian GDP. The left panel shows the ACF of quarterly growth $\Delta\,100\ln Y_t$, the series with $d = 1$. The right panel shows the ACF of the second difference, $\Delta^2\,100\ln Y_t$, the series with $d = 2$. The band $\pm 1.96/\sqrt{T}$ is the same in both panels.

On the left, all autocorrelations lie inside the band. On the right, the first autocorrelation leaves the band by a wide margin, downwards, while the others remain small: the typical shape of an MA(1) with a negative coefficient.

## Interpreting over-differencing {h=5dcc37e8}

The figures confirm all three symptoms. For $d = 1$, $\hat\rho(1) = -0.05$ and the variance of the series is 6.16. For $d = 2$, $\hat\rho(1) = -0.43$, close to $-0.5$, and the variance rises to 12.80, roughly double that at $d = 1$, as $\mathrm{Var}(\Delta\varepsilon_t) = 2\sigma^2$ predicts.

The third symptom appears in estimation. An MA(1) estimated on $\Delta^2\,100\ln Y_t$ gives $\hat\theta = -0.987$, with standard error 0.080, practically $-1$. The MA part cancels the extra difference: the factor $(1 - L)$ from differencing and $(1 + \hat\theta L) \approx (1 - L)$ from the MA part cancel out. The correct choice for Romanian GDP is therefore $d = 1$.

## Recap: ARIMA models {h=db20a653}

An ARIMA$(p,d,q)$ model is an ARMA$(p,q)$ for $\Delta^d y_t$, and the constant decides the slope of the long-run forecast. For Romanian real GDP since 2000, the Box–Jenkins cycle leads to ARIMA(0,1,0) with a drift of 0.81% per quarter, with uncorrelated residuals but fat tails produced by the crises of 2009 and 2020.

The forecast-error variance is $\sigma^2\sum_{j<h}\psi_j^2$: bounded for $d = 0$, growing for $d \ge 1$, at rate $\sqrt{h}$ for $d = 1$ and $h^{3/2}$ for $d = 2$. Over-differencing is recognised by three symptoms: $\hat\rho(1) \approx -0.5$, a larger variance after differencing and an estimated MA coefficient close to $-1$.

## @end

The ARIMA model links the test results to forecasting: the $d$ established by the tests determines whether forecast uncertainty is bounded or grows with the horizon. On Romanian GDP, a random walk with drift describes the linear dynamics well, and its 95% interval after 12 quarters has a half-width of 16.8%, against 11.2% for the trend-stationary model. A second difference of the same series doubles the variance and produces an MA coefficient of $-0.987$. [The next section](#sec-7) applies the methodology to other series and discusses automatic model selection.

# ARIMA in practice

In applications, ARIMA models are often chosen automatically, for many series at once. The section describes the standard automatic selection algorithm and the points at which it must be checked. It then applies the methodology to three problems: the choice of $d$ for Romanian inflation, the comparison between an ARIMA model and the random walk for the EUR/RON rate, in the tradition of the Meese–Rogoff study, and the forecast of US GDP after the 2008 crisis.

## Automatic ARIMA {h=f0b6a518}

The algorithm of @HyndmanKhandakar2008, implemented in the *auto.arima* function in R and in the Python libraries *pmdarima* and *statsforecast*, automates the Box–Jenkins cycle. The order $d$ is chosen by repeated KPSS tests at 5%: the series is differenced as long as KPSS rejects stationarity. The orders $p$ and $q$ are chosen by a stepwise search by AICc, starting from a few initial models and changing $p$ or $q$ by one unit at each step, with $d$ fixed. For $d \le 1$, the algorithm also decides whether to include a constant or drift.

The algorithm discards models whose roots are too close to the unit circle and those whose estimation fails. It is useful when many series must be modelled at once and as a starting point for an individual analysis. Evaluated by its authors on the 3,003 series of the M3 forecasting competition, the method was competitive with the best methods in the competition.

Each step of the algorithm contains a choice, however: the test used for $d$, the criterion, the search space. Whoever uses the result must be able to justify each choice, and the next frame shows where problems can arise.

## Automatic ARIMA: checks {h=a26edd66}

The first check concerns the comparability of the criterion. AICc cannot be compared across different values of $d$: with $d = 1$, the likelihood is computed for $\Delta y_t$, a different series from $y_t$, and the values of the criterion are on different scales. The order $d$ is therefore chosen by tests and judgement, and AICc compares only models with the same $d$.

The second check concerns roots that nearly cancel. On Romanian GDP with the sample extended back to 1995Q1, the lowest AICc, 607.5, belongs to ARIMA(2,1,2), against 611.1 for ARIMA(0,1,0), and its estimation did not converge. The MA roots have modulus 1.0002 and the AR roots modulus 1.014: the AR and MA polynomials have almost equal roots, which cancel. The second MA coefficient is estimated at 1.000, with standard error 6.96. The model describes the noise of the 1990s data, not the dynamics of the economy.

The general rule is to check the output of automatic selection every time. The data are plotted, the convergence of the estimation and the position of the roots are checked, the residual tests are applied, and the forecasts are compared with a simple benchmark, for example the random walk ([Chapter 0](ch0.html)).

## Romanian inflation: the choice between d = 0 and d = 1 {h=04b06af4}

Romania's 12-month HICP inflation, from January 2005 to August 2026, is the inconclusive series of the table in section 4. The chart compares two models, each the best by AICc in its class, with $p \le 3$ and $q \le 2$: one with $d = 0$ and a mean, the other with $d = 1$. Both produce forecasts for 24 months, with 95% intervals.

The observed series is shown from 2018. It shows the period of low inflation in 2019–2021, the rise to a peak of more than 13% at the end of 2022, the subsequent fall and a new rise in 2025–2026. The horizontal dotted line is the estimated mean of the model with $d = 0$.

## Interpreting the two inflation models {h=6bd12224}

The model with $d = 0$ is ARIMA(2,0,0), with $\hat\phi_1 = 1.32$ and $\hat\phi_2 = -0.34$. The sum of the coefficients is 0.976, so $\phi(1) = 0.024$: the model is stationary, but only just, with a mean of 5.2%. The model with $d = 1$ is ARIMA(1,1,0), with $\hat\phi = 0.33$. In the short run the two have almost the same dynamics: an AR(2) with a root very close to 1 behaves almost like an AR(1) in differences.

After 24 months, the model with $d = 0$ forecasts 5.1%, with a 95% interval from $-0.6\%$ to 10.9%, and the model with $d = 1$ forecasts 5.2%, with an interval from $-2.9\%$ to 13.3%. The point forecasts are almost equal, but the intervals differ: $d$ decides how much uncertainty is reported.

The tests do not settle the question. KPSS, with statistic 0.404, does not reject stationarity and keeps $d = 0$, as does automatic selection, which relies on KPSS. ADF, with p-value 0.32, does not reject the unit root and suggests $d = 1$. Both models leave autocorrelated residuals, because 12-month rates overlap: two successive 12-month rates share 11 months, and the series also inherits the seasonality of prices. [Chapter 4](ch4.html) models monthly inflation instead, with its seasonal component.

## Case study: Meese and Rogoff (1983) {h=041d31cd}

The study of @MeeseRogoff1983 asked a simple question: can exchange-rate models forecast better than a random walk? The authors compared the monetary exchange-rate models of the 1970s, time-series models, ARIMA and vector autoregressions (VAR), and the random walk, on US dollar exchange rates, at horizons of 1–12 months.

The result was negative for all models. Out of sample, no model had a lower root mean squared error (RMSE) than the random walk. The result held even when the structural models used the actual future values of the fundamental variables, information that a real forecaster does not have.

The study left two legacies. The first is the "Meese–Rogoff puzzle": economic models of the exchange rate do not forecast it better than a model without economics. The second is the use of the random walk as a mandatory benchmark for any exchange-rate forecast. The next frame makes the same comparison for EUR/RON.

## EUR/RON: ARIMA against the random walk {h=ac09eb47}

The comparison uses the daily BNR EUR/RON reference rate. The model parameters are estimated on 4,673 daily changes up to December 2023, and the test period, from January 2024, contains 679 one-day-ahead forecasts. The models compared are the random walk, the random walk with drift and ARIMA(1,1,0).

The left panel shows the rate since 2022, with a dotted line at the start of the test period. The right panel shows the cumulative difference of one-day squared errors between each model and the random walk. A curve that falls below zero indicates a model more accurate than the random walk; a jump of the curve indicates a day on which the difference in accuracy was large.

## Interpreting the EUR/RON comparison {h=822e29cd}

The RMSE of the daily change, measured as $100\Delta\ln$, is 0.1330 for the random walk, 0.1327 for the random walk with drift and 0.1300 for ARIMA(1,1,0), whose estimated coefficient is $\hat\phi = 0.169$. The gain of the ARIMA model over the random walk is 2.2%. At the 20-day horizon, with 33 non-overlapping windows, the RMSE is 0.667 for the random walk and 0.638 for ARIMA(1,1,0), a gain of 4.3%.

The gain comes from a few days. Over the test period the rate rose from 4.9746 to 5.2644 lei, mainly through two jumps, and both continued on the following day, which the positive AR coefficient anticipates. On the right-hand chart, the red curve drops sharply on the days after the jumps and stays almost flat otherwise: on calm days, the random walk is just as good.

A gain concentrated in two episodes is fragile evidence. A formal comparison of two forecasts requires the Diebold–Mariano test ([Chapter 4](ch4.html)). The Meese–Rogoff lesson remains valid: the random walk is reported alongside any exchange-rate forecast.

## A classic series: US real GDP after 2007 {h=e0643e99}

The last application uses US real GDP, the GDPC1 series from the FRED (Federal Reserve Economic Data) database, available since 1947. On data up to 2007Q4, two models are estimated: a trend-stationary model, with a linear trend and AR(2) errors, and a difference-stationary model, ARIMA(1,1,0) with drift. Both then forecast the series up to 2026Q2, 74 quarters ahead, with 95% intervals.

The left panel shows the series $100\ln$ GDP from 1995 and the two forecasts, with their intervals: the narrow one belongs to the trend-stationary model, the wide one to the ARIMA model. The right panel shows the outcome minus the forecast, in log points, for each model.

## Interpreting the US GDP forecasts {h=00165aaa}

The trend-stationary model expected GDP to return to the pre-2008 line. The return never came: the gap widened almost continuously and is $-30$ log points at the end, when the half-width of the 95% interval is only 8. The output lost in 2008–2009 was never recovered, which corresponds to a permanent shock, as the interpretation of @NelsonPlosser1982 implies.

The difference-stationary model also misses, by $-25$ points at the end, just outside its 95% interval, whose half-width is 23. The difference is that its interval warns of large uncertainty, while the narrow interval of the trend model gives a false sense of precision. Both models assume that the pre-2008 drift continues, but growth after 2008 was slower: the data suggest a break in the drift, of the kind discussed in section 5.

The unit-root tests on the whole series are clear: ADF has p-value 0.82 and KPSS is 0.58, above the 5% critical value for the trend version. Both classify US real GDP as $I(1)$.

## Recap: ARIMA in practice {h=3cb4a0ba}

Automatic ARIMA selection chooses $d$ by KPSS tests and the orders $p$ and $q$ by AICc. Its output is checked: convergence of the estimation, roots close to the unit circle and the residuals. AICc compares only models with the same $d$.

For exchange rates, the random walk is a hard benchmark to beat [@MeeseRogoff1983]: on EUR/RON, the gain of an ARIMA(1,1,0) is small and concentrated in a few days. The choice between trend stationarity and difference stationarity changes the long-run forecast and its interval, as the forecasts of US GDP after 2008 show.

## @end

In practice, the choice of $d$ is the decision with the largest effect on the forecast. For Romanian inflation, $d = 0$ and $d = 1$ give the same point forecast but 95% intervals of different widths. For US GDP, the trend-stationary model missed a permanent loss of output by 30 log points. Automatic selection is a useful starting point, and its output must be checked and compared with the random walk.

# Possible contribution of AI

An assistant based on artificial intelligence (AI) can speed up the coding and exploration in this chapter, but unit-root tests involve many choices that change the conclusion: the deterministic terms, the lags, the critical values, the direction of the null. The section shows what the assistant can do and what must be checked in its output.

## Possible contribution of AI {h=8bba117f}

An AI assistant can write a first version of a script that applies ADF, PP and KPSS to many series and tabulates the verdicts, on the model of the table in section 4. It can offer a second explanation of a theoretical result, for example why the Dickey–Fuller distribution is not Normal, or of a Zivot–Andrews result. It can also support large-scale exploration: unit roots and breaks in the inflation rates of all European Union (EU) countries, or benchmark ARIMA models for many Romanian series.

The example prompt asks for complete code for the analysis of section 6: loading Romania's quarterly real GDP from Eurostat, the ADF tests with constant and trend and KPSS on the level and on the difference, estimation of ARIMA$(p,1,q)$ models with drift for $p, q \le 2$, and a chart of the 8-quarter forecasts with 95% intervals. A precise prompt, with the series, the transformation, the tests and the grid of models specified, reduces the number of choices the assistant makes implicitly.

## Necessary checks {h=f4af6bcf}

The code and text produced by the assistant are checked at the points where errors are frequent. The first is the direction of each test: ADF and PP have the unit root as their null, KPSS has stationarity, and a reversal changes the conclusion. The second is the deterministic terms, the lag length and the critical values actually used: the Dickey–Fuller values, not those of the Normal distribution.

The wording of the conclusions needs the same attention. A "not rejected" result is not reported as "proven", and regressions between trending series are not presented as evidence of causality. For ARIMA models, the convergence warnings, roots close to the unit circle, residual tests and the fact that AICc is compared only for the same $d$ are checked. Every reference cited by the assistant must exist; the DOI is checked directly.

## @end

The AI assistant shortens the writing of code and the exploration of many series, but the choices that decide the outcome of a unit-root test remain the analyst's. The check covers the same points that the chapter showed to be sensitive: the direction of the null, the critical values, the deterministic terms, convergence and the comparability of criteria.

# Summary

The summary brings together the main ideas of the chapter, the formulas to remember and three self-assessment questions with their answers.

## Key takeaways {h=f204e075}

Trending series are either trend-stationary, with shocks that die out, or difference-stationary, with permanent shocks. The treatment differs: in the first case the trend is removed, in the second the series is differenced, and the wrong treatment produces spurious cycles or over-differencing. Regressions between unrelated $I(1)$ series are spurious: they have large $t$ and $R^2$ and a DW statistic close to 0.

ADF and PP test the null of a unit root, with Dickey–Fuller critical values, while KPSS tests the null of stationarity; the two kinds of test are used together. The tests have low power near $\phi = 1$ and are misled by structural breaks; the Zivot–Andrews test allows for one break at an unknown date.

An ARIMA$(p,d,q)$ model is an ARMA for $\Delta^d y_t$, and its forecast intervals grow with the horizon when $d \ge 1$. Automatic selection is a starting point: its output is checked for $d$, convergence, roots, residuals and the comparison with the random walk.

## Key formulas {h=7a59b0a1}

The table brings together the formulas of the chapter. The first three rows give the two trend models, @eq:ts and @eq:ds, the ADF regression @eq:adf, with null $\gamma = 0$ and statistic $\tau$, and Schwert's rule for the maximum lag length. Next comes the KPSS statistic @eq:kpss, with the 5% critical values, 0.463 with constant and 0.146 with constant and trend.

The last three rows concern ARIMA models: the definition @eq:arima, the forecast-error variance @eq:fcvar, with the special case $h\sigma^2$ of the random walk, and the variance for ARIMA(0,1,1), $\sigma^2[1 + (h - 1)(1 + \theta)^2]$, together with the link $\alpha_{\mathrm{SES}} = 1 + \theta$ to simple exponential smoothing.

## Self-assessment {h=12edfc6f}

The first question asks for the decision rule: an ADF test with constant and trend gives $\tau = -2.9$ for $T = 200$. The 5% critical value for this case is about $-3.43$, and $-2.9 > -3.43$, so the unit root is not rejected. With the value $-1.645$ of the Normal distribution, it would have been wrongly rejected.

The second question concerns the ADF–KPSS combination: if neither rejects, the result is inconclusive, and the sample is too short for a decision. Both models are reported, or $d$ is chosen according to the purpose. The third applies the ARIMA(0,1,1) formula: with $\theta = -0.6$, $\sigma = 1$ and $h = 5$, the forecast-error variance is $1 + 4 \cdot 0.4^2 = 1.64$.

The next chapter, [Chapter 4](ch4.html), treats seasonality: seasonal differences, seasonal ARIMA (SARIMA) models, TBATS and Prophet.
