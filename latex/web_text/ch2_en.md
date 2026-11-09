<!--
Course text of the web version, Chapter 2 (EN). Inserted into the page by latex/build_web_lecture.py.
Format: # section title (introduction), ## slide title {h=hash} (text), @after (text after the live block), ## @end (summary).
Numbered equations: $$ ... $$ {#eq:label}, reference @eq:label. Citations: @Key -> Author (year), [@Key] -> (Author, year).
Current hashes: python3 build_web_lecture.py 2 --hashes
-->

## @intro

@epigraph Yule1927 | On a method of investigating periodicities in disturbed series, with special reference to Wolfer's sunspot numbers
> If we take a curve representing a simple harmonic function of the time, and superpose on the ordinates small random errors, the only effect is to make the graph somewhat irregular, leaving the suggestion of periodicity still quite clear to the eye.

@text
@Yule1927 contrasted this picture with a different mechanism: a pendulum whose motion is disturbed by random shocks. In the first picture the cycle is a fixed function of time and the errors only blur it. In the second the shocks enter the dynamics, every shock changes the future path, and the series is correlated with its own past. Yule fitted this second mechanism to the yearly sunspot numbers with two coefficients, and the result was the first autoregression. The chapter starts from the same question: a stationary series is correlated with its own past, and a small model is needed that captures this memory and turns it into forecasts with honest uncertainty.

The Wold decomposition from Chapter 1 shows that every stationary, purely nondeterministic process is an infinite weighted sum of past shocks. It does not say how to estimate infinitely many weights from $T$ observations. The ARMA($p,q$) models of @BoxJenkinsReinsel2008 write these weights as the ratio of two short polynomials, with $p + q + 2$ parameters. The chapter derives their properties (stationarity, invertibility, autocorrelations, impulse responses), the estimators, the information criteria, the residual tests and the forecasts, and then applies the whole method to real series.

The main application is the annual growth of Romanian real GDP, 106 quarters from 2000Q1 to 2026Q2. Its autocorrelation function cuts off after lag 3, the Bayesian information criterion (BIC) chooses an MA(3) model, and the MA(3) leaves residuals without autocorrelation, but with fat tails due to 2020. The same tools are then applied to Romanian inflation, to the daily returns of the BET index and of the EUR/RON exchange rate, and to Yule's sunspots.

@chart tsa_ch2_gdp_ident
Annual growth of Romanian real GDP, 2000Q1–2026Q2, in % (top), with its sample autocorrelation function (ACF) and partial autocorrelation function (PACF) for lags of 1 to 16 quarters (bottom); the red dashed lines are the band $\pm 1.96/\sqrt{T} = \pm 0.19$.

@goals
- **Check** the stationarity of an AR($p$) and the invertibility of an MA($q$) from the roots of their polynomials.
- **Derive** the mean, the autocovariances and the $\psi$ weights of AR(1), AR(2), MA($q$) and ARMA(1,1).
- **Identify** candidate orders from the ACF and the PACF, and choose among them with AIC and BIC.
- **Estimate** ARMA models by Yule–Walker, conditional least squares and maximum likelihood, and interpret the output.
- **Test** the residuals with Ljung–Box on $m - p - q$ degrees of freedom and with Jarque–Bera.
- **Forecast** with ARMA models, compute the intervals and explain mean reversion.

@plan
- From Chapter 1 to ARMA models | Stationarity, white noise, the lag operator, the Wold decomposition and the problem of infinitely many weights.
- Autoregressive models | AR(1) and AR($p$): stationarity from the roots, moments, the AR(2) triangle and the cut-off of the PACF.
- Moving-average models | MA($q$): the cut-off of the ACF, the bound on $\rho(1)$ and invertibility.
- ARMA(p,q) models | The $\psi$ weights, impulse responses, common factors and the ACF–PACF identification table.
- Estimation | Yule–Walker, conditional least squares and Gaussian maximum likelihood, and their small-sample behaviour.
- Model selection and diagnostics | AIC and BIC, the residual tests and the degrees of freedom of the Ljung–Box test.
- Forecasting with ARMA models | The optimal forecast, the error variance, the intervals and mean reversion.
- The Box–Jenkins method | The loop of identification, estimation, checking and forecasting.
- Case study: Romanian GDP growth | The whole method on 106 quarters of annual GDP growth.
- More real series | Inflation, daily returns of BET and EUR/RON, Yule's sunspots and the forecasting competitions.
- Possible contribution of AI | What AI can draft and the checks the analyst must run.
- Summary | Key takeaways, key formulas and self-assessment.

# From Chapter 1 to ARMA models

This section collects the tools from Chapter 1 that the chapter uses and states the problem that ARMA models solve: the Wold decomposition describes every stationary process, but with infinitely many weights.

## Tools from Chapter 1 {h=4a771584}

A process $X_t$ is weakly stationary when its mean $\mu$ is constant and its autocovariance $\gamma(h) = \mathrm{Cov}(X_t, X_{t+h})$ depends only on the lag $h$, not on the date $t$. The autocorrelation function (ACF), $\rho(h) = \gamma(h)/\gamma(0)$, rescales the autocovariances to the interval $[-1, 1]$. The partial autocorrelation function (PACF), $\phi_{hh}$, measures the correlation between $X_t$ and $X_{t+h}$ after the linear effect of the intermediate values $X_{t+1}, \dots, X_{t+h-1}$ has been removed. These two functions are the identification tools of the chapter: their shapes indicate which model to fit.

White noise, $\varepsilon_t \sim \mathrm{WN}(0, \sigma^2)$, has mean 0, variance $\sigma^2$ and no autocorrelation at any nonzero lag. It is the raw material of every model in the chapter: each model is a linear filter applied to white noise. The lag operator shifts a series back in time, $LX_t = X_{t-1}$ and $L^kX_t = X_{t-k}$. A polynomial in $L$, such as $1 - 0.5L$, therefore acts on a series as a weighted sum of its lags, and the algebra of polynomials (products, ratios, roots) becomes the algebra of models.

The Wold decomposition [@Wold1938] connects these pieces. Every stationary, purely nondeterministic process can be written as $X_t = \mu + \sum_{j \ge 0}\psi_j\varepsilon_{t-j}$, with $\psi_0 = 1$ and $\sum_j\psi_j^2 < \infty$. "Purely nondeterministic" excludes components that the infinite past predicts exactly, such as a fixed sine wave. The theorem says that such a process is a moving average of infinite order of its past shocks; the rest of the chapter writes this sum with few parameters.

## The problem: infinitely many weights {h=22327161}

The Wold decomposition says what a stationary process looks like, not how to estimate it. The weights $\psi_1, \psi_2, \dots$ are infinitely many unknowns, and the sample has only $T$ observations. Truncating the sum at a long lag would still leave many parameters, each estimated with a large error.

The idea is to write the lag polynomial $\psi(L) = \sum_j\psi_jL^j$ as a ratio of two short polynomials, $\psi(L) = \theta(L)/\phi(L)$. Multiplying both sides of the Wold form by $\phi(L)$ gives the ARMA($p,q$) model

$$\begin{gathered}\phi(L)(X_t - \mu) = \theta(L)\varepsilon_t, \\ \phi(L) = 1 - \phi_1L - \dots - \phi_pL^p, \\ \theta(L) = 1 + \theta_1L + \dots + \theta_qL^q.\end{gathered} $$ {#eq:arma}

The model has $p + q + 2$ parameters: the $p$ autoregressive coefficients, the $q$ moving-average coefficients, the mean $\mu$ and the noise variance $\sigma^2$. One parameter can already generate infinitely many weights: $1/(1 - 0.8L) = 1 + 0.8L + 0.64L^2 + \dots$.

A ratio of two low-order polynomials approximates almost any smooth $\psi(L)$ well. It reproduces geometric decays, damped oscillations and weights that stop after a finite lag, the shapes met in practice. The principle that guides the choice of $p$ and $q$ is parsimony: the smallest model whose residuals are white noise. Every additional parameter adds estimation error, and beyond the point where the residuals are white noise it no longer improves the forecasts.

## Three building blocks {h=7d1cb143}

The autoregressive model AR($p$) makes today's value depend on the last $p$ values of the series and on a new shock: $X_t = c + \phi_1X_{t-1} + \dots + \phi_pX_{t-p} + \varepsilon_t$. It goes back to @Yule1927 and his analysis of the sunspots. The moving-average model MA($q$) makes today's value depend on the last $q$ shocks: $X_t = \mu + \varepsilon_t + \theta_1\varepsilon_{t-1} + \dots + \theta_q\varepsilon_{t-q}$. @Slutzky1937 showed that such sums of random causes produce series that look cyclical. The ARMA($p,q$) model contains both parts; @BoxJenkinsReinsel2008, whose first edition appeared in 1970, turned it into a complete method of identification, estimation, checking and forecasting.

In the autoregressive form, $c$ is the intercept, not the mean. Taking expectations on both sides and using stationarity ($E[X_t] = E[X_{t-k}] = \mu$) gives $\mu = c + (\phi_1 + \dots + \phi_p)\mu$, so $\mu = c/(1 - \phi_1 - \dots - \phi_p)$. The distinction matters in practice: for $X_t = 2 + 0.8X_{t-1} + \varepsilon_t$ the intercept is 2 and the mean is 10, and software output may report either.

Throughout the chapter, $\varepsilon_t$ is white noise; this is enough for the means, the autocovariances and the point forecasts. The likelihood and the forecast intervals require more: $\varepsilon_t$ independent and identically distributed (i.i.d.) $N(0, \sigma^2)$. The difference between the two assumptions appears in the case study, where the residuals of the GDP model are uncorrelated but not Normal: the point estimates remain valid, the Normal intervals do not.

## @end

The Wold decomposition guarantees that every stationary, purely nondeterministic process is a filter of white noise, and the ARMA model @eq:arma writes this filter as $\theta(L)/\phi(L)$, with $p + q + 2$ parameters. The [next section](#sec-2) studies the autoregressive part alone, the [section after it](#sec-3) the moving-average part, and the [fourth section](#sec-4) the two together.

# Autoregressive models

An autoregressive model regresses the series on its own past. The section derives the moments of the AR(1) model, extends the stationarity condition to AR($p$) through the roots of the characteristic polynomial, maps the stationary AR(2) models in the plane of their coefficients and shows the property that identifies an AR($p$) in data: its PACF cuts off after lag $p$.

## The AR(1) model {h=77f7004f}

The AR(1) model is $X_t = c + \phi X_{t-1} + \varepsilon_t$, or, with the lag operator, $(1 - \phi L)X_t = c + \varepsilon_t$. Written in deviations from the mean, $X_t - \mu = \phi(X_{t-1} - \mu) + \varepsilon_t$: the coefficient $\phi$ is the share of today's deviation from the mean that survives until tomorrow. With $\phi = 0.8$, a deviation of 2 today is expected to be 1.6 tomorrow.

Substituting the equation into itself $k$ times expresses $X_t$ through a distant value and the shocks in between (first line); for $|\phi| < 1$ and $k \to \infty$ this gives the second line:

$$\begin{aligned} X_t &= c(1 + \phi + \dots + \phi^{k-1}) + \phi^kX_{t-k} \\ &\quad + \sum_{j=0}^{k-1}\phi^j\varepsilon_{t-j}, \\ X_t &= \frac{c}{1 - \phi} + \sum_{j=0}^{\infty}\phi^j\varepsilon_{t-j}. \end{aligned} $$ {#eq:ar1-wold}

When $|\phi| < 1$, the term $\phi^kX_{t-k}$ vanishes: the process forgets its starting point. The limit is the Wold form with $\psi_j = \phi^j$, called the causal or MA($\infty$) representation, because $X_t$ depends only on present and past shocks.

The condition $|\phi| < 1$ is essential. With $\phi = 1$ the model is the random walk of Chapter 1: shocks never fade, the variance grows with time, and there is no stationary solution. With $|\phi| > 1$ the backward substitution explodes; a stationary solution exists only if $X_t$ is written as a function of future shocks, which is useless for forecasting from the past, so this case is excluded.

## Moments of a stationary AR(1) {h=b2a9dba1}

The mean follows from taking expectations of the model. Stationarity is used here: $E[X_t] = E[X_{t-1}] = \mu$, so $\mu = c + \phi\mu$ and $\mu = c/(1 - \phi)$. The variance follows in the same way. Since $\varepsilon_t$ is uncorrelated with $X_{t-1}$, which by @eq:ar1-wold depends only on shocks up to $t - 1$, the variance of the sum is the sum of the variances: $\gamma(0) = \phi^2\gamma(0) + \sigma^2$.

$$\begin{gathered}\mu = \frac{c}{1 - \phi}, \qquad \gamma(0) = \frac{\sigma^2}{1 - \phi^2}, \\ \gamma(h) = \phi\,\gamma(h - 1), \qquad \rho(h) = \phi^h.\end{gathered} $$ {#eq:ar1-mom}

The variance grows without bound as $|\phi| \to 1$: a persistent series wanders far from its mean, although each shock has the same variance $\sigma^2$. The autocovariances come from multiplying $X_t - \mu = \phi(X_{t-1} - \mu) + \varepsilon_t$ by $X_{t-h} - \mu$, $h \ge 1$, and taking expectations; the shock term drops out because $\varepsilon_t$ is uncorrelated with the past. The result is the recursion $\gamma(h) = \phi\gamma(h - 1)$, so $\gamma(h) = \phi^h\gamma(0)$ and $\rho(h) = \phi^h$: the ACF decays geometrically, with alternating signs when $\phi < 0$.

A useful summary of persistence is the half-life of a shock, the horizon $h$ at which half of its effect is gone: $\phi^h = 1/2$, so $h = \ln 0.5/\ln\phi$ for $0 < \phi < 1$. It translates a coefficient into a time span that can be compared across series and frequencies.

## Worked example: an AR(1) {h=58a756e0}

Take $X_t = 2 + 0.8X_{t-1} + \varepsilon_t$ with $\sigma^2 = 1$. The model is stationary because $|\phi| = 0.8 < 1$. Its mean is $\mu = 2/(1 - 0.8) = 10$: the intercept is 2 and the mean is 10, and confusing the two is a common error when reading software output.

The variance is $\gamma(0) = 1/(1 - 0.64) = 2.78$, so the series has a standard deviation of about 1.67, larger than the standard deviation 1 of a single shock, because past shocks accumulate. The autocorrelation at lag 3 is $\rho(3) = 0.8^3 = 0.512$: observations three periods apart have a correlation of 0.512.

The half-life is $\ln 0.5/\ln 0.8 = 3.11$ periods: after about three periods, half of a shock is gone. The $\psi$ weights are $1, 0.8, 0.64, 0.512, \dots$, so a unit shock today raises $X_{t+2}$ by 0.64. For an AR(1) the impulse response $\psi_j$ and the autocorrelation $\rho(j)$ are both equal to $\phi^j$; this coincidence is specific to the AR(1) and does not hold for the other models of the chapter.

## AR(1) for three values of φ {h=969d8d08}

The chart shows three simulated AR(1) series of 200 observations, with $\phi = 0.9$, $0.3$ and $-0.7$. All three are driven by the same 200 shocks, so every difference between the paths in the top row comes from $\phi$ alone. The vertical scale is the same in the three panels, which makes the difference in variance visible.

The bottom row shows, for lags 1 to 15, the sample ACF of each path (bars) and the theoretical ACF $\rho(h) = \phi^h$ (orange dots). The red dashed lines are the band $\pm 1.96/\sqrt{T}$, the approximate 95% band for the sample autocorrelations of white noise. A bar outside the band is an autocorrelation that this check separates from noise; the gap between a bar and its dot is the sampling error of one series of length 200.

## Interpreting the AR(1) paths {h=afc800eb}

With $\phi = 0.9$ the series makes long swings away from 0. Its sample first-order autocorrelation is $\hat\rho(1) = 0.88$, and its sample variance is 4.98, against the theoretical value $1/(1 - 0.81) = 5.26$. Over a window of a few dozen observations such a series looks like a trend: strong persistence and a deterministic trend are hard to tell apart by eye, which is why Chapter 3 introduces formal tests.

With $\phi = 0.3$ the series is close to white noise: only $\hat\rho(1) = 0.35$ is clearly outside the band, and the sample variance is 1.18, against a theoretical 1.10. With $\phi = -0.7$ the series zig-zags: $\hat\rho(1) = -0.58$ and $\hat\rho(2) = 0.34$, with alternating signs. In economic data, negative autocorrelation appears with overshooting followed by correction, or with a measurement error that is corrected in the next period.

In the panel for $\phi = 0.9$, the sample ACF lies below the theoretical one at long lags. With $T = 200$, sample autocorrelations of a persistent series are biased towards zero. The same small-sample bias affects the estimators of $\phi$, as the [estimation section](#sec-5) shows.

## The AR(p) model and its characteristic polynomial {h=811a9a1c}

The AR($p$) model is $\phi(L)X_t = c + \varepsilon_t$, with the characteristic polynomial $\phi(z) = 1 - \phi_1z - \dots - \phi_pz^p$ in a complex variable $z$. By the fundamental theorem of algebra the polynomial factorises into $p$ linear factors:

$$\phi(z) = (1 - \lambda_1z)(1 - \lambda_2z)\cdots(1 - \lambda_pz), \qquad z_i = 1/\lambda_i. $$ {#eq:charpoly}

The $z_i$ are the roots of $\phi(z) = 0$, and the $\lambda_i$ are the inverse roots. Each factor $1 - \lambda_iL$ is the operator of an AR(1) with coefficient $\lambda_i$, so an AR($p$) is a product of $p$ AR(1) filters, some of which may have complex coefficients.

The stationarity (causality) condition is that all roots lie outside the unit circle, $|z_i| > 1$, or equivalently that all inverse roots lie inside it, $|\lambda_i| < 1$; software usually plots the inverse roots. The reason is the AR(1) result applied to each factor: $1/(1 - \lambda L) = \sum_j\lambda^jL^j$ converges only if $|\lambda| < 1$, and the product of the $p$ convergent series is the causal MA($\infty$) form.

A root on the unit circle, $|z| = 1$, for example $z = 1$, is a unit root: the process is not stationary, and Chapter 3 treats this case. A quick necessary check is $\phi_1 + \dots + \phi_p < 1$, that is $\phi(1) > 0$. Since $\phi(0) = 1 > 0$, a value $\phi(1) \le 0$ would place a real root between 0 and 1, inside the circle. The check is not sufficient: a model can pass it and still have a root inside the circle, so the roots themselves must be computed.

## AR(2): the stationarity triangle {h=3e4abcb0}

For the AR(2) model $X_t = \phi_1X_{t-1} + \phi_2X_{t-2} + \varepsilon_t$, the root condition reduces to three linear inequalities:

$$\phi_1 + \phi_2 < 1, \qquad \phi_2 - \phi_1 < 1, \qquad |\phi_2| < 1. $$ {#eq:ar2-tri}

They define a triangle in the plane $(\phi_1, \phi_2)$ with vertices $(-2, -1)$, $(2, -1)$ and $(0, 1)$. Every point inside the triangle is a stationary AR(2), and every point outside has at least one root on or inside the unit circle.

The roots are complex when $\phi_1^2 + 4\phi_2 < 0$, below a parabola inside the triangle. The ACF is then a damped cosine: it produces pseudo-cycles of average length $2\pi/\arccos\big(\phi_1/(2\sqrt{-\phi_2})\big)$ periods, and their amplitude shrinks by the factor $\sqrt{-\phi_2}$ per period, the modulus of the inverse roots. The cycles are "pseudo" because the shocks keep changing their amplitude and phase; the series has no fixed period, only a typical one.

The ACF of an AR(2) comes from the Yule–Walker equations: multiply the model by $X_{t-h}$, take expectations and divide by $\gamma(0)$.

$$\begin{gathered}\rho(1) = \frac{\phi_1}{1 - \phi_2}, \\ \rho(h) = \phi_1\rho(h - 1) + \phi_2\rho(h - 2), \quad h \ge 2.\end{gathered} $$ {#eq:ar2-yw}

The first equation comes from $\rho(1) = \phi_1 + \phi_2\rho(1)$, using $\rho(-1) = \rho(1)$. The second shows that the ACF obeys the same recursion as the process without its shocks, so its shape (geometric decay or damped wave) is set by the roots of $\phi(z)$.

## Three AR(2) processes and the sunspots {h=b49d7b95}

The left panel draws the stationarity triangle @eq:ar2-tri, shaded, and the dashed parabola $\phi_1^2 + 4\phi_2 = 0$ that separates real roots (above it) from complex roots (below it). Four models are marked: A with $\phi = (0.5, 0.3)$, B with $(1.0, -0.6)$, C with $(0.6, 0.5)$, and the AR(2) that Yule fitted to the sunspots, estimated in [Section 10](#sec-10). A lies inside the triangle in the region of real roots, B and the sunspots inside in the region of complex roots, and C just outside the right edge.

The middle panel shows the inverse roots $\lambda_i$ in the complex plane, with the unit circle. A model is stationary when all its points lie inside the circle. A has two real inverse roots inside; B and the sunspots each have a pair of complex conjugate inverse roots inside; C has one inverse root on the real axis outside the circle.

The right panel shows the theoretical ACF of A and B up to lag 30. A decays smoothly and slowly; B oscillates with decreasing amplitude, the damped cosine of complex roots.

## Interpreting the AR(2) examples {h=c4d89ffa}

Model A has the real roots $-2.84$ and $1.17$, both outside the unit circle, so it is stationary. Its autocorrelations, from @eq:ar2-yw, are $\rho(1) = 0.5/0.7 = 0.714$ and $\rho(2) = 0.5\rho(1) + 0.3 = 0.657$: a slow, smooth decay, dominated by the root 1.17, the one closest to the circle.

Model B has the complex roots $0.833 \pm 0.986\,i$, with modulus $1.291 > 1$, so it is stationary. Its ACF is a wave with a period of 7.2 lags, damped by the factor 0.775 per lag. Such a model generates a stochastic cycle: oscillations with a typical length but random amplitude and timing.

Model C has $\phi_1 + \phi_2 = 1.1 > 1$, so it fails the necessary condition; one root is $0.936 < 1$, inside the circle, and the process is explosive. Each coefficient is below 1, yet the process explodes: stationarity is a property of the roots, not of the coefficients taken one at a time. Complex roots are the mechanism by which linear models produce business cycles and, in [Section 10](#sec-10), the 11-year solar cycle.

## The PACF of an AR(p) cuts off after lag p {h=149406a5}

The partial autocorrelation $\phi_{hh}$ is the last coefficient in the best linear prediction of $X_t$ from $X_{t-1}, \dots, X_{t-h}$ (Chapter 1). It does not require a regression for every $h$: the Durbin–Levinson recursion [@Durbin1960] computes all the $\phi_{hh}$ from the ACF.

For an AR($p$) and $h > p$, the best linear predictor of $X_t$ from $h$ lags is the model itself, $\phi_1X_{t-1} + \dots + \phi_pX_{t-p}$: the error $\varepsilon_t$ is uncorrelated with all past values, so no additional lag can improve the prediction. The coefficients of lags $p + 1, \dots, h$ are therefore zero, which gives $\phi_{hh} = 0$ for $h > p$, and $\phi_{pp} = \phi_p$.

This is the identification rule for autoregressions: when the ACF decays gradually and the PACF has $p$ significant values and then stays inside the band, the candidate is an AR($p$). For the sample PACF of an AR($p$) at lags $h > p$ the band is $\pm 1.96/\sqrt{T}$. With many lags examined, about one value in twenty falls outside the band by chance, so an isolated value just outside it at a long lag is weak evidence.

## Recap: Autoregressive models {h=d3b34593}

An AR($p$), $\phi(L)X_t = c + \varepsilon_t$, is stationary if and only if all roots of $\phi(z)$ lie outside the unit circle; checking the coefficients one by one is not enough. Its mean is $\mu = c/\phi(1)$, and its ACF follows the Yule–Walker recursion @eq:ar2-yw, which produces a mix of geometric decays (real roots) and damped waves (complex roots).

The AR(1) is the building block: $\rho(h) = \psi_h = \phi^h$ and the half-life of a shock is $\ln 0.5/\ln\phi$. In data, an AR($p$) is recognised by its PACF, which cuts off after lag $p$.

## @end

Autoregressive models describe memory through the past values of the series. Stationarity requires the roots of $\phi(z)$ outside the unit circle @eq:charpoly; for an AR(2) this is the triangle @eq:ar2-tri. Real roots produce geometric decays of the ACF, complex roots damped waves and stochastic cycles. The signature of an AR($p$) in data is a PACF that cuts off after lag $p$. The [next section](#sec-3) studies the opposite case: memory through past shocks.

# Moving-average models

A moving-average model describes the series as a weighted sum of the current and the last $q$ shocks. Such a model is always stationary, and its ACF cuts off after lag $q$. The new issue is invertibility: two different MA models can have the same autocorrelations, and a convention is needed to choose between them.

## The MA(q) model {h=14bb2e6e}

The MA($q$) model is $X_t = \mu + \theta(L)\varepsilon_t = \mu + \varepsilon_t + \theta_1\varepsilon_{t-1} + \dots + \theta_q\varepsilon_{t-q}$. It is a Wold sum with finitely many nonzero weights, $\psi_j = \theta_j$ for $j \le q$ and 0 beyond. A finite sum of white-noise terms always has a constant mean and finite variance, so an MA($q$) is stationary for any value of the coefficients.

The moments follow from the fact that the shocks are uncorrelated. With $\theta_0 = 1$, $E[X_t] = \mu$ and

$$\begin{gathered}\gamma(h) = \sigma^2\sum_{j=0}^{q-h}\theta_j\theta_{j+h}, \quad 0 \le h \le q, \\ \gamma(h) = 0, \quad h > q.\end{gathered} $$ {#eq:ma-acov}

Two values more than $q$ periods apart contain no common shock, so their covariance is zero.

The identification rule is the mirror image of the AR rule: the ACF cuts off after lag $q$, and the PACF decays gradually. For the sample ACF at lags $h > q$, Bartlett's formula gives the band $\pm 1.96\sqrt{(1 + 2\sum_{j=1}^{q}\hat\rho(j)^2)/T}$. It is wider than $\pm 1.96/\sqrt{T}$, because the sampling variance of an autocorrelation beyond lag $q$ depends on the autocorrelations up to lag $q$; with the narrow band, a sample ACF beyond the true order would look significant too often.

## MA(1): θ = 0.8 and θ = -0.8 {h=1386f80d}

The chart shows two simulated MA(1) series of $T = 300$ observations, with $\theta = 0.8$ (top row) and $\theta = -0.8$ (bottom row). For each, the left panel is the path, the middle panel the ACF and the right panel the PACF, for lags 1 to 12; the bars are sample values, the orange dots the theoretical values and the red dashed lines the band $\pm 1.96/\sqrt{T}$.

In both ACF panels there is a single clear value, at lag 1: positive for $\theta = 0.8$ and negative for $\theta = -0.8$. The PACF does not cut off: it decays, with alternating signs for $\theta = 0.8$ and with all values negative for $\theta = -0.8$. The path with $\theta = 0.8$ is smoother, because neighbouring values share a shock with the same sign; the path with $\theta = -0.8$ is more jagged.

## Interpreting the MA(1) correlograms {h=2c098173}

For an MA(1), @eq:ma-acov gives $\rho(1) = \theta/(1 + \theta^2)$, which equals $\pm 0.488$ for $\theta = \pm 0.8$, and $\rho(h) = 0$ for $h \ge 2$. The sample values are $\hat\rho(1) = 0.56$ and $-0.51$; the differences from $\pm 0.488$ are sampling error with 300 observations.

The PACF does not cut off: for $\theta = 0.8$ the sample value at lag 2 is $\hat\phi_{22} = -0.23$. The reason, developed in the next frames, is that an invertible MA(1) can be written as an AR($\infty$): every past value of the series carries a little information about the current shock, so every lag enters the best linear prediction with a nonzero coefficient.

An MA(1) remembers exactly one period: a shock today affects tomorrow's value and is then forgotten. This structure appears in economic data in several ways: overlapping observations, as in the GDP growth rates of [Section 9](#sec-9); a measurement error that enters one observation and is corrected in the next; and the bid–ask bounce, in which transaction prices alternate between the bid and the ask, so that consecutive price changes are negatively correlated.

## The largest autocorrelation of an MA(1) {h=3eb38ad5}

The chart plots $\rho(1) = \theta/(1 + \theta^2)$ for $\theta$ between $-4$ and $4$. The shaded band marks the invertible region $|\theta| < 1$, and the red dashed lines mark $\pm 0.5$. Two points are highlighted: $\theta = 0.5$ and $\theta = 2$, both with $\rho(1) = 0.4$.

The curve is odd in $\theta$, reaches its maximum 0.5 at $\theta = 1$ and its minimum $-0.5$ at $\theta = -1$, and tends to 0 as $|\theta|$ grows. Every value of $\rho(1)$ strictly between $-0.5$ and $0.5$, other than 0, is reached twice: once inside the shaded region and once outside it.

## Interpreting the curve ρ(1) {h=d393da3c}

The largest possible value is $|\rho(1)| = 0.5$, reached at $\theta = \pm 1$. It follows from $(1 - |\theta|)^2 \ge 0$, which gives $1 + \theta^2 \ge 2|\theta|$. An MA(1) can therefore never produce $|\rho(1)| > 0.5$. If a sample ACF shows $\hat\rho(1) = 0.7$ and nothing significant afterwards, the series is not an MA(1); the candidates are an AR model or an MA($q$) with more terms.

The values $\theta = 0.5$ and $\theta = 2$ give the same $\rho(1) = 0.4$. In general $\theta$ and $1/\theta$ give the same ACF: replacing $\theta$ by $1/\theta$ leaves $\theta/(1 + \theta^2)$ unchanged, and the autocovariances also coincide if $\sigma^2$ is multiplied by $\theta^2$. The data cannot tell the two models apart; this is an identification problem.

The convention that removes the ambiguity is invertibility, $|\theta| < 1$, the subject of the next frame.

## Invertibility {h=3395a745}

For an MA(1) with $\mu = 0$, the model can be solved for the shock: $\varepsilon_t = X_t - \theta\varepsilon_{t-1}$. Substituting this expression into itself repeatedly gives

$$\begin{gathered}\varepsilon_t = X_t - \theta X_{t-1} + \theta^2X_{t-2} - \dots \\ = \sum_{j \ge 0}(-\theta)^jX_{t-j}, \qquad |\theta| < 1.\end{gathered} $$ {#eq:ma1-inv}

Rearranged, this is an AR($\infty$) representation, $X_t = \theta X_{t-1} - \theta^2X_{t-2} + \dots + \varepsilon_t$, which explains why the PACF of an MA(1) never cuts off.

An MA model is invertible when its shocks can be recovered from the past values of $X$. The condition is that all roots of $\theta(z) = 0$ lie outside the unit circle; for the MA(1) this is $|\theta| < 1$. It is the same algebra as the stationarity of an AR, applied to $\theta(z)$ instead of $\phi(z)$.

Invertibility is required for three reasons. It makes the model unique: of $\theta$ and $1/\theta$, only one is invertible. It makes forecasting possible: the shocks $\varepsilon_t$ are computed from the data by @eq:ma1-inv, and the forecasts use them. Software therefore reports invertible estimates, and an estimated root close to the unit circle is a warning: it is the typical sign of over-differencing (Chapter 1), since differencing a series that was already stationary creates an MA root equal to 1.

## Worked example: θ = 2 or θ = 0.5? {h=74b4189c}

Consider $X_t = \varepsilon_t + 2\varepsilon_{t-1}$ with $\sigma^2 = 1$, and $Y_t = u_t + 0.5u_{t-1}$ with $\mathrm{Var}(u_t) = 4$. By @eq:ma-acov, $X$ has $\gamma(0) = 1 \cdot (1 + 4) = 5$ and $\gamma(1) = 2 \cdot 1 = 2$, while $Y$ has $\gamma(0) = 4 \cdot (1 + 0.25) = 5$ and $\gamma(1) = 0.5 \cdot 4 = 2$. Both have zero mean.

The two processes have the same mean and the same autocovariances. With Gaussian shocks, a stationary process is fully determined by its mean and autocovariances, so $X$ and $Y$ have the same distribution: no amount of data separates them.

Only $Y$ is invertible: $u_t = Y_t - 0.5Y_{t-1} + 0.25Y_{t-2} - \dots$, a convergent sum. For $X$ the same sum has weights $(-2)^j$ and explodes. The rule is to report the invertible version, $\theta = 0.5$ with $\sigma^2 = 4$. Its shocks $u_t$ are the errors of the best one-step prediction from the past of the series, which is why the noise variance of the invertible version (4) is larger than that of the non-invertible one (1).

## Recap: Moving-average models {h=6716403a}

An MA($q$) is always stationary; its ACF cuts off after lag $q$ and its PACF decays. For an MA(1), $\rho(1) = \theta/(1 + \theta^2)$, which is at most 0.5 in absolute value; a larger first autocorrelation rules out the MA(1).

An MA is invertible if and only if the roots of $\theta(z)$ lie outside the unit circle; it can then be written as an AR($\infty$). The parameters $\theta$ and $1/\theta$ are observationally equivalent, and the invertible one is reported.

## @end

The MA($q$) model has finite memory: by @eq:ma-acov its autocorrelations are exactly zero after lag $q$, and this cut-off identifies it in data. Stationarity is automatic; the condition that matters is invertibility, which makes the model unique and lets the shocks be computed from the data @eq:ma1-inv. The [next section](#sec-4) combines the autoregressive and the moving-average parts.

# ARMA(p,q) models

The ARMA($p,q$) model has both an autoregressive and a moving-average part. The section states its three conditions (causality, invertibility, no common factors), computes its impulse response through the $\psi$ weights, derives the moments of the ARMA(1,1) and collects the identification rules of the whole chapter in one table.

## The ARMA(p,q) model {h=105eab34}

The ARMA($p,q$) model is $\phi(L)(X_t - \mu) = \theta(L)\varepsilon_t$, as in @eq:arma, or, written out,

$$\begin{aligned} X_t &= c + \phi_1X_{t-1} + \dots + \phi_pX_{t-p} \\ &\quad + \varepsilon_t + \theta_1\varepsilon_{t-1} + \dots + \theta_q\varepsilon_{t-q}, \\ c &= \mu\,\phi(1). \end{aligned} $$ {#eq:arma-full}

Three conditions make the model well defined; @BrockwellDavis2016 treat them in their Chapter 3. It is stationary and causal when the roots of $\phi(z)$ lie outside the unit circle; then $X_t - \mu = \psi(L)\varepsilon_t$ with $\psi(L) = \theta(L)/\phi(L)$. It is invertible when the roots of $\theta(z)$ lie outside the unit circle; then $\pi(L)(X_t - \mu) = \varepsilon_t$ with $\pi(L) = \phi(L)/\theta(L)$. Finally, $\phi(z)$ and $\theta(z)$ must have no common root, otherwise the model has more parameters than the process it describes.

The value of mixing the two parts shows in the ARMA(1,1). Its ACF decays geometrically at the rate $\phi$, like an AR(1), but its first value differs from $\phi$: it lies below $\phi$ when $\theta < 0$ and above it when $\theta > 0$. A pure AR or a pure MA would need many terms to reproduce this shape; the ARMA(1,1) does it with two coefficients.

## The ψ weights and the impulse response {h=2cecb0f6}

The causal form requires the coefficients of $\psi(L) = \theta(L)/\phi(L)$. Writing $\phi(L)\psi(L) = \theta(L)$ and matching the coefficients of each power $L^j$ gives a recursion:

$$\begin{gathered}\psi_0 = 1, \qquad \psi_j = \theta_j + \sum_{i=1}^{\min(j,p)}\phi_i\psi_{j-i}, \\ \theta_j = 0 \text{ for } j > q.\end{gathered} $$ {#eq:psi}

For the ARMA(1,1), $\psi_1 = \phi + \theta$ and $\psi_j = \phi^{j-1}(\phi + \theta)$ for $j \ge 1$. With $\phi = 0.7$ and $\theta = 0.4$: $\psi_1 = 1.10$ and $\psi_2 = 0.77$.

The weights have a direct interpretation: $\psi_j = \partial X_{t+j}/\partial\varepsilon_t$, the effect of a unit shock at time $t$ on the value $j$ periods later, with all other shocks held fixed. The sequence $(\psi_j)$ is the impulse response of the model. The same object determines the variance of the forecast errors ([Section 7](#sec-7)) and, in a multivariate form, the impulse responses of VAR models in Chapter 6.

## Impulse responses of four models {h=db2f74f0}

The chart shows $\psi_j$ for $j = 0, \dots, 15$, computed with the recursion @eq:psi, for four models: AR(1) with $\phi = 0.8$, AR(2) with $\phi = (1.0, -0.6)$, MA(2) with $\theta = (0.6, 0.3)$ and ARMA(1,1) with $\phi = 0.7$, $\theta = 0.4$. Every response starts at $\psi_0 = 1$, the immediate effect of the shock on the current value.

The height of bar $j$ is the change in $X_{t+j}$ after a unit shock at time $t$. The sum of the squared heights, multiplied by $\sigma^2$, is the variance $\gamma(0)$ of the process, so a response that dies out slowly means both long memory and a large variance.

## Interpreting the impulse responses {h=fb8fe1c6}

For the AR(1) the response decays geometrically, $\psi_j = 0.8^j$. The shock never vanishes completely, but it fades fast enough for $\sum_j\psi_j^2$ to be finite, which is the condition of the Wold decomposition. For the AR(2) with complex roots, $\psi_2 = 0.4$ and the response then changes sign: $\psi_3 = -0.20$, $\psi_4 = -0.44$. A single shock creates a cycle: the series first rises, then falls below its mean, then returns.

For the MA(2) the response is exactly zero after two periods: the memory of the model is finite. For the ARMA(1,1) the moving-average term lifts the first response to $\psi_1 = 1.10 > 1$, above the immediate effect; after that the autoregressive part takes over, and each response is 0.7 times the previous one.

These shapes reappear in forecasting: the speed at which the $\psi_j$ decay is the speed at which forecasts return to the mean and at which the forecast intervals widen to their long-run width.

## Moments of the ARMA(1,1) {h=8f7bfbdc}

Consider $X_t = \phi X_{t-1} + \varepsilon_t + \theta\varepsilon_{t-1}$ with $|\phi| < 1$ (causal), $|\theta| < 1$ (invertible) and $\phi + \theta \ne 0$ (no common factor). The variance follows from the causal form, $\gamma(0) = \sigma^2\sum_j\psi_j^2$, with the weights $\psi_j = \phi^{j-1}(\phi + \theta)$: summing the geometric series gives $\gamma(0) = \sigma^2(1 + 2\phi\theta + \theta^2)/(1 - \phi^2)$. For $\phi = 0.7$, $\theta = 0.4$ and $\sigma^2 = 1$ this is 3.37.

The autocorrelations are

$$\begin{gathered}\rho(1) = \frac{(1 + \phi\theta)(\phi + \theta)}{1 + 2\phi\theta + \theta^2}, \\ \rho(h) = \phi\,\rho(h - 1), \quad h \ge 2.\end{gathered} $$ {#eq:arma11-rho}

In the example $\rho(1) = 0.819$ and $\rho(2) = 0.7 \cdot 0.819 = 0.573$.

From lag 1 onwards the ACF decays like that of an AR(1) with coefficient $\phi$, but its starting value is $\rho(1) \ne \phi$: the moving-average part moves only the starting point. This is the pattern listed in the identification table for ARMA models: the ACF decays after lag $q - p$, here after lag 0, with a first value that the AR part alone would not give.

## Common factors: a model that is too large {h=ff9c2d4f}

Consider $X_t = 0.5X_{t-1} + \varepsilon_t - 0.5\varepsilon_{t-1}$. In lag-operator form it is $(1 - 0.5L)X_t = (1 - 0.5L)\varepsilon_t$; cancelling the common factor gives $X_t = \varepsilon_t$, white noise. The ARMA(1,1) is an over-parameterised description of a process with no memory at all.

More generally, any ARMA(1,1) with $\phi = -\theta$ is white noise, whatever the common value. The likelihood is therefore flat along the line $\phi = -\theta$, and the data cannot locate the parameters on it: they are not identified.

In software output, a common factor shows up as large estimates of opposite sign, for example $\hat\phi = 0.9$ and $\hat\theta = -0.88$, with very large standard errors, and as a log-likelihood that is no better than that of the smaller model. The remedy is to reduce $p$ and $q$ together and to prefer the smallest model whose residuals are white noise.

## Identification: the ACF–PACF table {h=d4a26395}

The table summarises the theoretical patterns. White noise has no significant autocorrelation in either function. An AR($p$) has an ACF that decays (geometrically or as a damped wave) and a PACF that cuts off after lag $p$. An MA($q$) has the opposite: an ACF that cuts off after lag $q$ and a PACF that decays. An ARMA($p,q$) has both functions decaying: the ACF after lag $q - p$ and the PACF after lag $p - q$, meaning that the first values are set freely by the MA part and the later ones follow the AR recursion.

The last row anticipates Chapter 3: for a series with a unit root, the sample ACF decays very slowly, almost linearly, and the first partial autocorrelation $\hat\phi_{11}$ is close to 1. Such a series is not modelled as an ARMA in levels; it is first differenced.

The table proposes candidates; it does not choose the model. The information criteria and the residual diagnostics of [Sections 5](#sec-5) and [6](#sec-6) decide among them. Mixed ARMA models are the hardest to recognise by eye, because both functions simply decay.

## Theoretical and sample ACF and PACF {h=e65b5cb2}

The chart applies the table to three simulated series of $T = 500$ observations each: an AR(2) with $\phi = (1.0, -0.6)$, an MA(2) with $\theta = (0.6, 0.3)$ and an ARMA(1,1) with $\phi = 0.7$, $\theta = 0.4$. The top row shows the ACF, the bottom row the PACF, for lags 1 to 15.

In every panel the bars are the sample values from the single simulated path, the orange dots the theoretical values and the red dashed lines the band $\pm 1.96/\sqrt{T}$. Comparing bars with dots shows how much noise a sample correlogram contains even with 500 observations; counting the bars outside the band, function by function, is the identification step.

## Interpreting the three patterns {h=381911cd}

For the AR(2), the PACF has two significant values, the second being $\phi_{22} = \phi_2 = -0.60$, and then stays at zero; the ACF is a damped wave. Identification is easy: the order $p$ is read from the PACF.

For the MA(2), the theoretical ACF is $\rho(1) = 0.538$, $\rho(2) = 0.207$ and zero afterwards, and the PACF decays slowly. The sample ACF at lag 2 is only just outside the band: with real data, an MA(1) and an MA(2) can look alike, and the choice between them is left to the information criteria.

For the ARMA(1,1), both functions decay. The ACF looks like that of an AR(1), and the PACF has a second, smaller value of opposite sign. An analyst who reads only the PACF would choose an AR(2) or an AR(3): a close approximation, but with more parameters. Sample correlograms are noisy even with $T = 500$; with 100 quarterly observations the band is wider and the patterns are much less clear.

## Recap: ARMA models {h=244978fb}

An ARMA($p,q$), $\phi(L)(X_t - \mu) = \theta(L)\varepsilon_t$, is causal when the roots of $\phi(z)$ lie outside the unit circle and invertible when those of $\theta(z)$ do. Its impulse response, the $\psi$ weights, follows from the recursion @eq:psi.

The two polynomials must have no common factor; a flat likelihood and large estimates of opposite sign are its symptoms. Identification uses the table: an ACF that cuts off points to an MA, a PACF that cuts off to an AR, two decaying functions to an ARMA.

## @end

The ARMA($p,q$) model combines a short AR polynomial, which sets the long-run decay of the memory, with a short MA polynomial, which shapes the first lags. Its impulse response @eq:psi summarises the dynamics, and for the ARMA(1,1) the autocorrelations @eq:arma11-rho decay like an AR(1) from a different starting point. Correlograms only suggest candidate orders. The [next section](#sec-5) estimates the parameters of a candidate, and the [one after it](#sec-6) chooses among candidates and checks the chosen model.

# Estimation

Once candidate orders are chosen, the parameters must be estimated. The section presents three estimators: Yule–Walker, a method of moments; conditional least squares; and exact Gaussian maximum likelihood. It then compares their behaviour in samples of 100 observations.

## Three estimators {h=77a43347}

The Yule–Walker estimator [@Yule1927; @Walker1931] is a method of moments: it replaces the theoretical autocorrelations in the Yule–Walker equations by sample autocorrelations and solves for the coefficients. It applies only to pure AR models. It is fast and always gives a stationary model, but it is less precise when a root is close to the unit circle.

Conditional least squares minimises the sum of squared shocks $\sum_t\varepsilon_t^2$, treating the first observations as fixed. For an AR($p$) this is ordinary least squares (OLS) of $X_t$ on $1, X_{t-1}, \dots, X_{t-p}$. For MA and ARMA models the shocks are not observed: they are computed recursively, $\varepsilon_t = X_t - \mu - \theta\varepsilon_{t-1} - \dots$, starting from $\varepsilon_0 = 0$. The sum of squares is then a nonlinear function of $\theta$ and is minimised numerically.

Exact Gaussian maximum likelihood (MLE, maximum likelihood estimation) maximises the joint density of $(X_1, \dots, X_T)$ and therefore also uses the information in the first observations. It is the default of the ARIMA class of statsmodels, which evaluates the likelihood with the Kalman filter (Chapter 10). In large samples the three estimators agree for pure AR models; for MA and ARMA models, MLE is the reference.

## Yule–Walker for an AR(p) {h=19a5df30}

For an AR($p$), multiplying the model by $X_{t-h}$, taking expectations and dividing by $\gamma(0)$ gives, for $h = 1, \dots, p$, the equations $\rho(h) = \phi_1\rho(h - 1) + \dots + \phi_p\rho(h - p)$. In matrix form they read $R\,\phi = \rho$, where $R$ is the $p \times p$ matrix with elements $R_{ij} = \rho(|i - j|)$, $\phi = (\phi_1, \dots, \phi_p)^\top$ and $\rho = (\rho(1), \dots, \rho(p))^\top$. The estimator replaces each $\rho(h)$ by its sample value, marked by a hat:

$$\hat\phi = \hat R^{-1}\hat\rho, \qquad \hat\sigma^2 = \hat\gamma(0)\,(1 - \hat\phi^\top\hat\rho). $$ {#eq:yw}

The second formula splits the variance of the series into the part explained by the past, $\hat\gamma(0)\hat\phi^\top\hat\rho$, and the variance of the shock.

For an AR(2), Cramer's rule gives explicit formulas: $\hat\phi_1 = \hat\rho_1(1 - \hat\rho_2)/(1 - \hat\rho_1^2)$ and $\hat\phi_2 = (\hat\rho_2 - \hat\rho_1^2)/(1 - \hat\rho_1^2)$. The second is the sample partial autocorrelation at lag 2 from Chapter 1, in agreement with the result $\phi_{pp} = \phi_p$. Gilbert Walker (1868–1958), whose portrait accompanies the slide, extended Yule's method to autoregressions of any order in 1931.

## Worked example: Yule–Walker on Romanian GDP growth {h=e7b71dcb}

The annual growth of Romanian real GDP, 2000Q1–2026Q2, studied in [Section 9](#sec-9), has sample autocorrelations $\hat\rho(1) = 0.62$ and $\hat\rho(2) = 0.40$, and sample variance $\hat\gamma(0) = 18.77$, dividing by $T$ as in the definition of Chapter 1. The determinant of $\hat R$ is $1 - \hat\rho_1^2 = 0.620$, computed with the unrounded value of $\hat\rho(1)$.

The Cramer formulas give $\hat\phi_1 = 0.62\,(1 - 0.40)/0.620 = 0.600$ and $\hat\phi_2 = (0.40 - 0.62^2)/0.620 = 0.027$. The noise variance, from @eq:yw, is $\hat\sigma^2 = 18.77\,(1 - \hat\phi_1\hat\rho_1 - \hat\phi_2\hat\rho_2) = 11.62$, against a variance of 18.77 for the series itself.

The second coefficient is practically zero: within the AR family, the second lag adds nothing, and an AR(1) would be enough. Yule–Walker does not ask whether an AR model is the right family at all. [Section 9](#sec-9) shows that for this series an MA(3) fits better.

## Maximum likelihood {h=de3ccae9}

The Gaussian likelihood of an ARMA model is computed by the prediction-error decomposition. The joint density factorises into conditional densities, $f(x_1, \dots, x_T) = \prod_t f(x_t \mid x_{t-1}, \dots, x_1)$, and with Gaussian shocks each factor is a Normal density with mean $\hat x_{t|t-1}$, the one-step prediction, and variance $v_t$, its error variance:

$$\ln L = -\frac12\sum_{t=1}^{T}\Big[\ln(2\pi v_t) + \frac{(x_t - \hat x_{t|t-1})^2}{v_t}\Big]. $$ {#eq:loglik}

The variances $v_t$ are larger for the first observations, which are predicted from little or no past.

For an AR(1) the decomposition can be written by hand. The first observation has the stationary distribution, $X_1 \sim N(\mu, \sigma^2/(1 - \phi^2))$, and each later one is $X_t \mid X_{t-1} \sim N(c + \phi X_{t-1}, \sigma^2)$. Dropping the first factor leaves the conditional likelihood, whose maximisation is conditional least squares. The difference between the two estimators therefore comes from the first observation, and it matters most in small samples and when $\phi$ is close to 1.

Standard errors (se) come from the curvature of $\ln L$ at its maximum. Asymptotically, $\mathrm{se}(\hat\phi) \approx \sqrt{(1 - \phi^2)/T}$ for an AR(1) and $\mathrm{se}(\hat\theta) \approx \sqrt{(1 - \theta^2)/T}$ for an MA(1). The $t$-ratios $\hat\phi/\mathrm{se}$ are compared with $\pm 1.96$, as in regression. The formula for the AR(1) shrinks to zero as $|\phi| \to 1$; the next frames show that near the unit circle this approximation is too optimistic. By default, statsmodels reports a different estimate, the OPG (outer product of gradients) standard error, built from the squared scores of each observation. If the model is right, both estimates have the same limit; when the errors have heavy tails the OPG standard errors are smaller than those from the curvature. All standard errors in this chapter are OPG.

## Sampling distributions of the estimators {h=4d81e961}

The chart shows the distributions of the estimates over 400 simulated samples of $T = 100$ observations each. The left panel is an AR(1) with $\phi = 0.9$, estimated by Yule–Walker (orange), conditional least squares (blue) and Gaussian maximum likelihood (red); the right panel is an MA(1) with $\theta = 0.5$, estimated by conditional least squares and maximum likelihood. The dashed vertical line marks the true value.

The position of each histogram relative to the dashed line shows the bias of the estimator, its width shows the standard deviation, and its shape shows whether intervals based on the Normal approximation are adequate. A symmetric histogram centred on the line is the ideal; the left panel departs from it, the right panel much less.

## Interpreting the estimator comparison {h=68a248b0}

For the AR(1) with $\phi = 0.9$, the means of the estimates are 0.853 for Yule–Walker, 0.864 for conditional least squares and 0.866 for maximum likelihood. All three are biased downwards. The textbook approximation of the least squares bias, $-(1 + 3\phi)/T = -0.037$, is close to the bias of the simulated least squares estimates. Yule–Walker has the largest bias near the unit circle.

The standard deviations of the estimates are 0.056, 0.055 and 0.054, all above the asymptotic value 0.044. The distribution is skewed to the left: the estimates cannot go far above 1 without leaving the stationary region, but they can fall well below 0.9. Asymptotic intervals are therefore too optimistic near $\phi = 1$.

For the MA(1) with $\theta = 0.5$, the means are 0.504 for conditional least squares, with a standard deviation of 0.097, and 0.506 for maximum likelihood, with a standard deviation of 0.098; the asymptotic standard deviation is 0.087. Both estimators are practically unbiased here, and both vary more than the asymptotic formula says. The practical rule is to use maximum likelihood, and Yule–Walker only for quick starting values or for teaching.

## Recap: Estimation {h=bb02bb2a}

Yule–Walker is a method of moments for pure AR models, $\hat\phi = \hat R^{-1}\hat\rho$. Conditional least squares is OLS for an AR and a numerical minimisation over recursively computed shocks for an MA. Exact Gaussian maximum likelihood, computed through the prediction-error decomposition @eq:loglik, is the standard in software.

In small samples, persistent AR coefficients are biased downwards, and their sampling distribution is skewed, so the asymptotic standard errors understate the uncertainty near the unit circle.

## @end

The three estimators coincide asymptotically for pure AR models and differ in small samples: with $T = 100$ and $\phi = 0.9$ all are biased downwards, Yule–Walker the most. Maximum likelihood @eq:loglik handles MA and ARMA models and is the default. Estimating a model does not say whether it is the right one: the [next section](#sec-6) compares candidate models and tests their residuals.

# Model selection and diagnostics

The correlograms propose several candidate orders. This section chooses among them with information criteria, which trade fit against the number of parameters, and checks the chosen model by testing whether its residuals behave like white noise. Two simulation experiments show how often AIC and BIC find the true order, and why the Ljung–Box test on residuals needs $m - p - q$ degrees of freedom.

## Information criteria {h=7154b645}

The fit of a model, measured by its maximised log-likelihood $\ln L$, improves with every added parameter. Information criteria therefore charge a penalty per parameter. Let $k$ be the number of estimated parameters (the $\phi$ and $\theta$ coefficients, $\mu$ and $\sigma^2$):

$$\begin{gathered}\mathrm{AIC} = -2\ln L + 2k, \\ \mathrm{AICc} = \mathrm{AIC} + \frac{2k(k + 1)}{T - k - 1}, \\ \mathrm{BIC} = -2\ln L + k\ln T.\end{gathered} $$ {#eq:ic}

The Akaike information criterion (AIC) comes from @Akaike1974; its small-sample correction, AICc, from @HurvichTsai1989; the Bayesian information criterion (BIC) from @Schwarz1978. Hirotugu Akaike, pictured on the slide, worked at the Institute of Statistical Mathematics in Tokyo.

The two criteria differ only in the penalty per parameter: 2 for AIC and $\ln T$ for BIC. Since $\ln T > 2$ once $T \ge 8$, BIC penalises more heavily in every realistic sample and prefers smaller models. The model with the smallest value is chosen. Only models fitted to the same observations can be compared: if, for example, AR models of different orders are fitted by conditional least squares, each drops a different number of initial observations, and their likelihoods are no longer comparable.

## AIC or BIC? {h=3daa0ea4}

BIC is consistent: if the true model is among the candidates, the probability that BIC selects it tends to 1 as $T$ grows. The price is that, in small samples, its heavy penalty may choose a model that is too small, missing a coefficient that is real but small.

AIC is designed for forecasting: it estimates the expected out-of-sample fit of a model, measured by the Kullback–Leibler distance between the model and the true distribution. It is not consistent: even with large $T$ it overfits with a positive probability, because the penalty 2 does not grow with the sample.

In practice, as described by @HyndmanAthanasopoulos2021 and @HyndmanKhandakar2008, automatic searches such as auto.arima in R or pmdarima in Python minimise AICc over a grid of orders $(p, q)$. When AIC and BIC choose different models, both candidates are kept, and the residual diagnostics and the out-of-sample forecasts decide between them.

## How often AIC and BIC find the true order {h=43ee12f1}

The experiment simulates a true AR(2) with $\phi = (0.5, 0.25)$, fits AR($p$) models for $p = 0, \dots, 6$ by OLS on the same sample, and records the order chosen by each criterion. Each panel uses 500 samples: on the left of length $T = 100$, on the right of length $T = 1000$.

The bars give the percentage of samples in which AIC (blue) or BIC (red) chose each order; the green dashed line marks the true order $p = 2$. Mass at $p = 2$ is a correct choice, mass to the left is underfitting, and mass to the right is overfitting.

## Interpreting the selection experiment {h=fdc94b61}

With $T = 100$, AIC finds $p = 2$ in 54.4% of the samples and BIC in 47.4%. BIC chooses a model that is too small in 50.4% of the samples: the second coefficient, $\phi_2 = 0.25$, is small relative to its sampling error with 100 observations, and BIC's penalty often outweighs its contribution. AIC chooses a model that is too large in 24.8% of the samples.

With $T = 1000$, BIC finds the true order in 98.6% of the samples and AIC in 74.4%. This is consistency at work: BIC converges to the true model, while AIC still overfits in 25.6% of the samples.

Neither criterion proves the order. With quarterly macroeconomic data, $T$ is around 100, and the order remains uncertain. The two errors have different costs: overfitting adds a little estimation noise to the forecasts, while underfitting leaves autocorrelation in the residuals, which the diagnostics of the next frame are designed to catch.

## Residual diagnostics {h=7bf61992}

If the model is correctly specified, its residuals $\hat\varepsilon_t$ behave like white noise. The first check is visual: the residual plot should show no pattern, no change in variance and no isolated extreme values. The residual ACF should stay inside $\pm 1.96/\sqrt{T}$, except for about one lag in twenty.

The portmanteau test of @LjungBox1978 combines the first $m$ residual autocorrelations:

$$\begin{gathered}Q^*(m) = T(T + 2)\sum_{h=1}^{m}\frac{\hat\rho_{\hat\varepsilon}(h)^2}{T - h}, \\ Q^*(m) \approx \chi^2(m - p - q) \quad \text{under } H_0.\end{gathered} $$ {#eq:lb}

The null hypothesis is that the model is correct. Estimating $p + q$ coefficients uses up $p + q$ degrees of freedom [@BoxPierce1970]. The number of lags $m$ should be well above $p + q$: $m = 10$ for annual or quarterly data and $m = 2s$ for seasonal data with period $s$ [@HyndmanAthanasopoulos2021].

Normality is tested with the Jarque–Bera statistic [@JarqueBera1987], $\mathrm{JB} = \frac{T}{6}\big(S^2 + (K - 3)^2/4\big) \approx \chi^2(2)$, where $S$ is the skewness and $K$ the kurtosis of the residuals (0 and 3 for the Normal distribution), together with a quantile–quantile (QQ) plot. Normality is rejected at 5% if $\mathrm{JB} > 5.99$, the 95% quantile of $\chi^2(2)$. Non-Normal residuals do not bias $\hat\phi$, but they make the Normal forecast intervals wrong. Finally, a Ljung–Box test on the squared residuals $\hat\varepsilon_t^2$ detects volatility clustering, the subject of Chapter 5.

## Why m - p - q degrees of freedom {h=b103dff0}

The experiment applies the Ljung–Box test at the 5% level to the residuals of the true model, with $T = 200$ and $m = 10$: 1,000 samples of an AR(1) and 500 samples of an ARMA(1,1). For each model the chart shows the rejection rate with $m = 10$ degrees of freedom (orange) and with $m - p - q$ degrees of freedom (blue); the red dashed line marks the nominal level of 5%.

Since the fitted model is the true one, the null hypothesis holds in every sample. A test with the correct size rejects in 5% of the samples, so a good choice of degrees of freedom gives a bar that reaches the dashed line.

## Interpreting the degrees-of-freedom experiment {h=be828b61}

With $m = 10$ degrees of freedom the test rejects too rarely: in 3.2% of the samples for the AR(1) and in 1.4% for the ARMA(1,1), instead of 5%. The residuals are fitted: estimation chooses the coefficients so that the residuals look uncorrelated at the first lags. Their autocorrelations are therefore smaller than those of true white noise, and $Q^*$ is smaller than its $\chi^2(m)$ reference suggests.

With $m - p - q$ degrees of freedom the rejection rates are 4.8% and 3.4%, close to the nominal 5%. The correction matters most when $p + q$ is large relative to $m$: it is larger for the ARMA(1,1), with two coefficients, than for the AR(1).

The function acorr_ljungbox of statsmodels uses $m$ degrees of freedom by default; the correction requires the argument model_df = p + q. With the wrong degrees of freedom the p-values are too large, and a bad model looks acceptable.

## Recap: Model selection and diagnostics {h=8fb3102e}

The criteria @eq:ic are $\mathrm{AIC} = -2\ln L + 2k$ and $\mathrm{BIC} = -2\ln L + k\ln T$; the smallest value wins, and BIC picks smaller models. BIC is consistent; AIC aims at forecasting and sometimes overfits.

The residuals are checked with a plot, their ACF, the Ljung–Box test @eq:lb with $m - p - q$ degrees of freedom, the Jarque–Bera test and the Ljung–Box test on squared residuals. A model passes when its residuals are white noise; normality matters for the intervals.

## @end

Selection and checking are separate steps. The criteria @eq:ic rank the candidates, with BIC consistent and AIC oriented towards forecasting; with about 100 observations both often miss the true order. The diagnostics decide whether the chosen model is adequate, and the Ljung–Box test @eq:lb must use $m - p - q$ degrees of freedom to have the right size. The [next section](#sec-7) uses an adequate model to forecast.

# Forecasting with ARMA models

The purpose of most ARMA models is forecasting. This section derives the optimal point forecast, the variance of its error and the forecast intervals, and shows the property that all stationary models share: their forecasts return to the mean.

## The optimal forecast {h=78da9485}

The forecast of $X_{T+h}$ made at time $T$ is the conditional expectation given the past, $\hat X_{T+h|T} = E[X_{T+h} \mid X_T, X_{T-1}, \dots]$. Among all functions of the past, it minimises the mean squared error (MSE) $E[(X_{T+h} - \hat X_{T+h|T})^2]$.

For an ARMA model the conditional expectation follows from a simple recipe. Write the model at time $T + h$ and replace the future shocks $\varepsilon_{T+j}$, $j \ge 1$, by 0, their conditional mean, and the future values $X_{T+j}$ by their forecasts; past values and past residuals stay as they are. The forecasts are computed recursively for $h = 1, 2, \dots$.

For an AR(1) the recipe gives

$$\hat X_{T+h|T} = \mu + \phi^h(X_T - \mu). $$ {#eq:ar1-fc}

The deviation from the mean shrinks by the factor $\phi$ per period: this is mean reversion at speed $\phi$. For an MA($q$), every term of the model at $T + h$ with $h > q$ is a future shock, so $\hat X_{T+h|T} = \mu$: after $q$ steps the model knows nothing beyond the mean.

## Forecast errors and intervals {h=0a628884}

The forecast error follows from the causal form $X_{T+h} = \mu + \sum_j\psi_j\varepsilon_{T+h-j}$. The forecast keeps the terms with shocks up to time $T$, so the error is the sum of the future shocks weighted by the first $\psi$ weights:

$$\begin{gathered}e_{T+h} = \varepsilon_{T+h} + \psi_1\varepsilon_{T+h-1} + \dots + \psi_{h-1}\varepsilon_{T+1}, \\ \sigma_h^2 = \sigma^2(1 + \psi_1^2 + \dots + \psi_{h-1}^2).\end{gathered} $$ {#eq:fvar}

The error variance $\sigma_h^2$ increases with the horizon $h$.

With Gaussian shocks, the 95% forecast interval is $\hat X_{T+h|T} \pm 1.96\,\sigma_h$. At $h = 1$ it is $\pm 1.96\,\sigma$. As $h \to \infty$, $\sigma_h^2 \to \sigma^2\sum_j\psi_j^2 = \gamma(0)$, the unconditional variance: in the long run the forecast interval is the band of the unconditional distribution.

The interval leaves out three sources of uncertainty: the parameters are estimated ($\hat\phi$ instead of $\phi$), the model itself may be wrong, and the shocks may have fatter tails than the Normal distribution. The actual coverage of nominal 95% intervals is therefore usually below 95%. Moreover, the errors of forecasts made at the same time for horizons $h$ and $h + 1$ are correlated, because they share the shocks $\varepsilon_{T+1}, \dots, \varepsilon_{T+h}$: a forecast path that is too high at one horizon tends to be too high at the next.

## Worked example: forecasting an AR(1) {h=7491f865}

Return to $X_t = 2 + 0.8X_{t-1} + \varepsilon_t$, with $\sigma^2 = 1$ and mean $\mu = 10$, and suppose the last observed value is $X_T = 12$, two units above the mean. By @eq:ar1-fc the point forecasts are $\hat X_{T+1} = 10 + 0.8 \cdot 2 = 11.60$, $\hat X_{T+2} = 10 + 0.64 \cdot 2 = 11.28$ and $\hat X_{T+10} = 10.21$.

The error variances follow from @eq:fvar with $\psi_1 = 0.8$: $\sigma_1^2 = 1$ and $\sigma_2^2 = 1 + 0.8^2 = 1.64$. The 95% intervals are $[9.64, 13.56]$ at $h = 1$ and $[8.77, 13.79]$ at $h = 2$: the interval widens, and its centre moves towards the mean.

In the long run the forecast tends to the mean 10, and the standard deviation of the forecast error tends to $\sqrt{\gamma(0)} = 1.67$, the unconditional standard deviation computed in the first worked example.

## Mean reversion in forecasts {h=32bd3dbe}

The chart compares the forecasts of three models over horizons 0 to 20: an AR(1) with $\phi = 0.9$, an AR(1) with $\phi = 0.5$ and an MA(1) with $\theta = 0.6$ and last shock $\varepsilon_T = 2$. In all three the last value is $X_T = 3$, the mean is $\mu = 0$ and the shock standard deviation is $\sigma = 1$.

The solid line is the point forecast, the shaded area the 95% interval, and the red dashed lines the unconditional band $\mu \pm 1.96\,\sigma_X$, where $\sigma_X$ is the standard deviation of the process. The distance of the forecast from 0 measures the information the present still carries about the future; the gap between the shaded interval and the dashed band measures how much the forecast reduces uncertainty compared with knowing only the unconditional distribution.

## Interpreting the forecast paths {h=7c91e771}

With $\phi = 0.9$ the reversion is slow: after 5 periods the forecast is still 1.77, with a standard error of 1.85. The interval approaches its long-run width, $\pm 1.96 \times 2.29$, only after many periods.

With $\phi = 0.5$ the forecast after 5 periods is 0.094, practically the mean. The MA(1) has a single informative step: the forecast at $h = 1$ is $\theta\varepsilon_T = 1.2$, and from $h = 2$ on it is the mean 0; the interval jumps to its long-run width at $h = 2$.

A stationary model always forecasts a return to the mean. This is useful for series that do revert, such as spreads and growth rates, and misleading when the mean itself shifts: the model then pulls the forecasts towards a level that no longer applies.

## Recap: Forecasting {h=b63b5f17}

The optimal forecast $\hat X_{T+h|T}$ replaces future shocks by 0 and future values by their forecasts. The error variance is $\sigma^2\sum_{j<h}\psi_j^2$, so the intervals widen with the horizon up to $\pm 1.96\sqrt{\gamma(0)}$.

Forecasts revert to $\mu$; for an MA($q$) they equal $\mu$ beyond $q$ steps. The Normal intervals ignore parameter and model uncertainty and are therefore too narrow in practice.

## @end

The forecast of a stationary ARMA model is a return to the mean at the speed set by the $\psi$ weights, @eq:ar1-fc for the AR(1); the error variance @eq:fvar accumulates the same weights and tends to $\gamma(0)$. The [next section](#sec-8) assembles identification, estimation, checking and forecasting into one method.

# The Box–Jenkins method

The previous sections supply the parts: identification from correlograms, estimation, selection, diagnostics and forecasting. Box and Jenkins assembled them into an iterative method, which the case study of [Section 9](#sec-9) then applies step by step.

## George Box and Gwilym Jenkins {h=c870f5db}

George E. P. Box (1919–2013) was a statistician at the University of Wisconsin–Madison. He is known for the sentence "All models are wrong, but some are useful". Gwilym M. Jenkins (1932–1982) was a statistician at Lancaster University. Their book *Time Series Analysis: Forecasting and Control* appeared in 1970; the fourth edition, with Gregory Reinsel, is @BoxJenkinsReinsel2008.

Their contribution was not new models: autoregressions go back to @Yule1927 and moving averages to @Slutzky1937. It was a method: a loop of identification, estimation and checking that any analyst can apply to a series. The residual tests of @BoxPierce1970 and @LjungBox1978 come from this programme.

The sentence about wrong models summarises the attitude of the method. A model is not judged by whether it is true, which no finite sample can establish, but by whether its residuals are white noise and its forecasts are useful.

## The Box–Jenkins loop {h=fd1bdcd5}

The diagram shows the method as a loop. Step 0 prepares the series: plot it, transform it (for example, take logs) and make it stationary. Step 1 identifies candidate orders $(p, q)$ from the ACF and the PACF. Step 2 estimates the candidates by maximum likelihood and compares them with AIC and BIC. Step 3 checks the chosen model: residual plots, the Ljung–Box test and the roots of both polynomials. If the model passes, Step 4 produces point forecasts and intervals; if it fails, the analyst returns to identification.

Step 0 belongs to Chapters 1 and 3: logarithms, differences and unit-root tests. ARMA models require a stationary series, and fitting them to a series with a unit root produces estimates near the boundary of the stationary region and misleading forecasts.

The loop ends with the smallest model whose residuals are white noise and whose roots lie well inside the admissible region. Out-of-sample checks, with the training and test samples of Chapter 0, complete the method: a model that passes every in-sample check must still forecast better than simple benchmarks.

## @end

The Box–Jenkins method turns the results of the chapter into a procedure: prepare, identify, estimate, check, and return to identification until the residuals are white noise, then forecast and compare with simple benchmarks out of sample. The [next section](#sec-9) applies it to Romanian GDP growth.

# Case study: Romanian GDP growth

The case study applies the Box–Jenkins loop to the annual growth of Romanian real GDP. Each step of the method has its frame: the data, identification from the correlograms, estimation and the information criteria, diagnostics of the chosen model and forecasts. The data structure itself explains the model that the criteria select.

## The data {h=5f487534}

The series is Romanian real GDP in chain-linked volumes (2010 prices), quarterly and not seasonally adjusted, published by Eurostat; the national source is the National Institute of Statistics (INS). Chain-linked volumes remove the effect of price changes, so growth rates measure changes in real output.

The analysis uses the annual growth rate $y_t = 100\,(\ln Y_t - \ln Y_{t-4})$, in %. Comparing each quarter with the same quarter of the previous year removes the seasonal pattern, as Chapter 1 showed. The sample covers 2000Q1–2026Q2, $T = 106$ quarters, with a mean of 3.29% and a standard deviation of 4.35%. The extremes are $-9.9\%$ in 2020Q2 and $+12.5\%$ in 2004Q4, and the last value is $-2.0\%$.

The growth rate shows no trend, which supports treating it as stationary; the formal test is in Chapter 3. The photograph on the slide shows the building of the National Institute of Statistics in Bucharest.

## Step 1: identification {h=7571bbe3}

The top panel shows the annual growth rate with its mean of 3.29% (dashed). The large movements of the sample stand out: the recession of 2009–2010, the fall to $-9.9\%$ in 2020Q2 during the pandemic, the rebound of 2021, and the slowdown at the end of the sample, with negative values in its last quarters.

The bottom panels show the sample ACF and PACF for lags 1 to 16 quarters, with the band $\pm 1.96/\sqrt{T} = \pm 0.19$. Values outside the band are those this check separates from noise; they are read against the identification table of [Section 4](#sec-4).

## Interpreting the correlograms of GDP growth {h=7452213b}

The ACF takes the values 0.62, 0.40 and 0.21 at lags 1 to 3 and $-0.01$ at lag 4: it cuts off after lag 3. By the identification table, the candidate is an MA(3). The PACF is 0.62 at lag 1 and small afterwards, except for 0.32 at lag 5. Read alone, it points to an AR(1); the value at lag 5 may be chance or a remnant of the seasonal pattern.

The MA(3) has a structural explanation. The annual growth rate is approximately the sum of four quarterly growth rates, $y_t \approx g_t + g_{t-1} + g_{t-2} + g_{t-3}$, where $g_t = 100\,(\ln Y_t - \ln Y_{t-1})$. If the quarterly rates $g_t$ are close to white noise (Seminar 2, task B2), two annual rates one quarter apart share three of their four quarterly terms, two quarters apart share two, three quarters apart share one, and four quarters apart share none. The result is an MA(3) by construction.

Before any model is fitted, the Ljung–Box statistic on the series itself is $Q^*(8) = 64.9$. The series is strongly autocorrelated, so a model is needed: the mean alone would leave predictable structure in the errors.

## Step 2: estimation and information criteria {h=cbbacc6d}

The table compares seven models, all estimated by exact Gaussian maximum likelihood on the same 106 quarters. For each it gives the number of parameters $k$, the maximised log-likelihood $\ln L$, AIC, BIC, the residual standard deviation $\hat\sigma$ and the p-value of the Ljung–Box test $Q^*(8)$ on the residuals, with $8 - p - q$ degrees of freedom. The full search covers all $p, q \le 3$.

The mean-only model, ARMA(0,0), has $\ln L = -305.83$ and a Ljung–Box p-value below 0.001. The AR(1) raises $\ln L$ to $-279.67$ and lowers $\hat\sigma$ from 4.33 to 3.38, but its residuals remain autocorrelated, with a p-value of 0.023. Adding a second parameter, in the AR(2) or the ARMA(1,1), leaves $\ln L$ at $-279.66$: the extra parameter is useless, and both residual tests still reject at 5%.

The MA(3) reaches $\ln L = -270.02$, AIC 550.04, BIC 563.36, $\hat\sigma = 3.07$ and a Ljung–Box p-value of 0.375. The ARMA(1,3) has the lowest AIC, 548.97, lower by only 1.07, but a higher BIC, 564.95. BIC therefore chooses the MA(3) and AIC the ARMA(1,3); the ARMA(2,3) is not better on either criterion. The AR(1), the candidate suggested by the PACF, is rejected by the diagnostics.

## Interpreting the estimated MA(3) {h=35fb7422}

The estimated model is $\hat y_t = 3.17 + \hat\varepsilon_t + 0.834\,\hat\varepsilon_{t-1} + 0.675\,\hat\varepsilon_{t-2} + 0.591\,\hat\varepsilon_{t-3}$, with $\hat\sigma = 3.07$. The OPG standard errors are 1.03 for the mean and 0.083, 0.102 and 0.082 for the three MA coefficients, so all three MA terms are significant.

The coefficients decline from 0.834 to 0.591. They are close to, but below, the pattern $(1, 1, 1)$ that the overlap of four quarters would give if quarterly growth were exactly white noise. That pattern would put the roots of $\theta(z) = 1 + z + z^2 + z^3$ on the unit circle; for the estimated model the smallest root modulus is 1.17, above 1, so the model is invertible.

In the ARMA(1,3), the first MA coefficient is $\hat\theta_1 = 1.019 > 1$, yet all roots of $\theta(z)$ have modulus at least 1.10, so this model is still invertible: for $q > 1$ invertibility is read from the roots, not from single coefficients. Its AR coefficient is $\hat\phi = -0.284$, with a standard error of 0.143: one extra parameter for a small gain, which BIC drops. The mean growth is 3.17% per year, but with a standard error of 1.03: overlapping observations carry less information than the 106 quarters suggest.

## Step 3: diagnostics of the MA(3) {h=60f8a88c}

The chart has four panels. The top left shows the residuals of the MA(3) over time, and the top right their ACF with the band $\pm 1.96/\sqrt{T}$. The bottom left gives the p-values of the Ljung–Box test for $m = 4, \dots, 16$, each with $m - 3$ degrees of freedom, against the 5% level (dashed line). The bottom right is a Normal QQ plot of the standardised residuals.

In a QQ plot, the sample quantiles of the residuals are plotted against the quantiles of the Normal distribution; under normality the points lie on the red line. Points below the line in the lower left corner are negative residuals more extreme than the Normal distribution allows, a heavy left tail.

## Interpreting the MA(3) diagnostics {h=fb25cdc0}

The residuals show no autocorrelation. The Ljung–Box statistic is $Q^*(8) = 5.35$ on 5 degrees of freedom, with a p-value of 0.375, and the test at $m = 16$ has a p-value of 0.525. Every p-value in the chart is above 0.12. With 8 instead of 5 degrees of freedom the p-value of $Q^*(8)$ would be 0.720, a result that looks more reassuring than the data justify.

The residuals are not Normal: the Jarque–Bera statistic is 39.0, with a p-value below 0.001, the skewness is $-0.75$ and the kurtosis 5.61. The largest residual is $-11.4$ in 2020Q2, $-3.89$ standard deviations: the pandemic lockdown, a shock that no model of past GDP could anticipate.

The Ljung–Box test on the squared residuals has a p-value of 0.779, so there is no volatility clustering at this frequency. The verdict is that the MA(3) captures the dependence in the series, but the Normal intervals understate the risk of recessions.

## Step 4: forecasts {h=1c015c6e}

The chart shows the observed growth rate from 2016 and the forecasts for 2026Q3–2028Q2 from two models: the MA(3), chosen by BIC (red), and the AR(1) (blue), each with its 95% interval.

Both forecasts start just below zero, close to the last observed values, and rise towards the mean. The MA(3) reaches its mean after three quarters and then stays flat; the AR(1) approaches its mean gradually. The two shaded intervals are wide compared with the movements of the series in recent years and overlap almost entirely.

## Interpreting the GDP forecasts {h=04ba6c73}

The MA(3) forecasts $-0.14\%$, $1.81\%$ and $2.77\%$ for the next three quarters and then the mean, 3.17%, from the fourth quarter on. This is exactly the MA($q$) rule: after $q = 3$ steps the model knows only the mean.

The AR(1) returns to its mean smoothly, reaching 2.99% after 8 quarters, against a mean of 3.12%. The two models agree in the short run and differ in how fast they forget the recent slowdown.

The 95% intervals of the MA(3) are $[-6.1, 5.9]$ one quarter ahead and $[-6.3, 12.7]$ from four quarters on, that is $\pm 6.0$ and $\pm 9.5$ percentage points. Because the residuals have fat tails, these intervals understate the probability of a deep recession. A univariate model cannot anticipate a turning point; it shows how unusual the next value would be given the past of the series.

## Recap: Romanian GDP growth {h=56c9a771}

For the annual growth of quarterly GDP, the ACF cuts off after lag 3; BIC chooses the MA(3) and AIC the ARMA(1,3). The MA(3) reflects the overlap of four quarters in the annual growth rate: the structure of the data explains the model.

The residuals are white noise but have fat tails because of 2020, so the Normal intervals are too narrow in crises. The forecasts reach the mean after three quarters.

## @end

The Box–Jenkins loop on GDP growth ends with an MA(3): the correlograms suggest it, BIC selects it, the Ljung–Box tests with $m - 3$ degrees of freedom accept it, and the construction of annual growth rates explains it. The diagnostics also show its limit, residuals with fat tails, which make the Normal intervals too narrow in recessions. The [next section](#sec-10) applies the same tools to series of other kinds.

# More real series

The method is applied here to three series with different characters: monthly inflation, which is very persistent; daily returns of the BET index and of the EUR/RON exchange rate, which are nearly unpredictable in the mean; and the historical sunspot numbers of Yule. A last case study summarises what forecasting competitions say about ARMA models.

## Romanian inflation: an AR(2) forecast {h=a00dd9e8}

The series is the 12-month rate of the harmonised index of consumer prices (HICP) for Romania, published by Eurostat, in %: 260 months up to August 2026. The AR order is chosen by BIC among $p \le 6$.

The chart shows the inflation rate (red), the AR(2) forecast with 80% and 95% intervals (shaded), the estimated mean of 5.16% (orange dashed) and the inflation target of the National Bank of Romania (BNR), 2.5% ± 1 percentage point (green band). The sample includes periods of negative inflation around 2015–2016 and the peak of 2022–2023.

## Interpreting the inflation model {h=dfc09f58}

BIC is 495.7 for the AR(1), 469.7 for the AR(2) and 473.4 for the AR(3), so the AR(2) is chosen. Its coefficients are $\hat\phi_1 = 1.320$ (standard error 0.057) and $\hat\phi_2 = -0.344$ (standard error 0.055). Their sum is 0.976, and the largest inverse root has modulus 0.963.

The model is very close to a unit root, which raises the question whether inflation is stationary at all; the test is in Chapter 3. The estimated mean, 5.16% with a standard error of 1.35, is the average of 2005–2026 and lies above the BNR target of 2.5% ± 1 percentage point. The forecast goes from the last value, 6.09%, to 5.44% next month and to 5.13% in August 2028, with a 95% interval of $[-0.6, 10.9]$ at that horizon. The model reverts to its historical mean, not to the target: it contains no information about monetary policy.

The diagnostics fail: the Ljung–Box statistic is $Q^*(12) = 71.6$, with a p-value below 0.001. The residuals are correlated at lag 12, a consequence of base effects: a price shock enters the 12-month rate when it occurs and leaves it twelve months later. Modelling this seasonal structure is a task for Chapter 4.

## Daily returns: BET and EUR/RON {h=c0ab7238}

The left panels show the ACF of daily log returns of the BET index of the Bucharest Stock Exchange and of the EUR/RON exchange rate, for lags of 1 to 20 days. The right panels show the ACF of the squared residuals of an AR(1) fitted to each series. The band is $\pm 1.96/\sqrt{T}$, very narrow because both samples have thousands of observations.

In the returns, lag 1 is clearly positive and the other lags are small, although several cross the narrow band. In the squared residuals every lag is positive and far outside the band: the returns themselves are hardly predictable, but their magnitude is.

## Interpreting the AR(1) for daily returns {h=90d5942b}

The table reports the AR(1) for both series. For BET (6,670 observations), $\hat\phi = 0.109$ with a standard error of 0.005, a $t$-ratio of 20.9 and an $R^2$ of 1.2%. For EUR/RON (5,352 observations), $\hat\phi = 0.170$ with a standard error of 0.006, a $t$-ratio of 30.6 and an $R^2$ of 2.9%. These are OPG standard errors. With residual kurtosis of 14 and 17 they are too small: standard errors from the curvature of the likelihood, or robust ones, are several times larger, yet both coefficients remain significant. The coefficients are highly significant, yet they explain only 1–3% of the variance: the effect is statistically real and economically small once trading costs are taken into account.

The Ljung–Box statistics $Q^*(10)$ on the residuals, with 9 degrees of freedom, are 29.7 and 61.5, both with p-values below 0.001, so the AR(1) does not capture all the dependence in the mean; BIC over $p, q \le 2$ chooses an ARMA(1,0) for BET and an ARMA(2,1) for EUR/RON. The positive $\hat\phi$ has economic explanations: slow price adjustment for BET, a managed exchange rate for EUR/RON.

The Ljung–Box statistics on the squared residuals are 2496 for BET and 2351 for EUR/RON, and the kurtosis of the residuals is 14.0 and 17.0. The mean is modelled, the variance is not: large moves cluster in time, and the combination of ARMA with GARCH models in Chapter 5 addresses this.

## Case study: Yule (1927) and the sunspots {h=8f8297da}

@Yule1927 fitted the first autoregression, to Wolfer's yearly sunspot numbers for 1749–1924. Until then, cycles in data were modelled as sums of sine waves, analysed with the periodogram. Yule proposed a different mechanism, a "pendulum bombarded by peas": a damped oscillator kicked by random shocks. The cycle comes from the dynamics of the system, and its irregularity from the shocks.

A least squares AR(2) fitted to the same years, 176 observations, gives $\hat\phi_1 = 1.336$ and $\hat\phi_2 = -0.650$. The roots are complex, and the modulus of the inverse roots is 0.806. The implied period, $2\pi/\arccos\big(\hat\phi_1/(2\sqrt{-\hat\phi_2})\big)$, is 10.57 years: two coefficients recover the solar cycle.

@Walker1931 generalised the equations to AR($p$), and they have since been called the Yule–Walker equations. These two papers mark the start of parametric time series analysis.

## The sunspots: AR(2) and the information criteria {h=2132e49b}

The left panel shows the yearly sunspot numbers for 1700–2008, 309 years, together with the one-step predictions of an AR(2) estimated by maximum likelihood on the full sample; the shaded area marks Yule's sample, 1749–1924. The right panel shows AIC and BIC for AR($p$) models with $p = 1, \dots, 10$, estimated on the full sample.

The predictions follow the observed cycles closely. In the right panel both criteria fall sharply from $p = 1$ to $p = 2$, stay roughly flat for a few orders and reach their minimum at $p = 9$.

## Interpreting the sunspot models {h=67ab7749}

The full-sample AR(2) has $\hat\phi = (1.391, -0.689)$, close to Yule's estimates on the shorter sample. It explains 83% of the variance of the sunspot numbers one year ahead. The one-step predictions follow the cycle but lag behind at the peaks: the model reacts to a peak only after observing it.

AIC and BIC both choose $p = 9$ among the orders up to 10. The additional lags capture a feature that the AR(2) cannot: the solar cycle is asymmetric, with a fast rise and a slow decline, while a linear AR(2) generates symmetric oscillations.

The lesson extends beyond sunspots. Two parameters capture the essence of the series, its cycle; more lags capture details; and nonlinear features such as the asymmetry of the cycle are beyond ARMA models.

## Case study: forecasting competitions {h=509a9484}

@MakridakisHibon2000 report the M3 competition: 3,003 series (yearly, quarterly, monthly and other), forecast by 24 methods. The forecasts were compared out of sample, on several accuracy measures and at several horizons.

The competition confirmed the conclusions of the earlier M-competitions. Statistically sophisticated methods do not necessarily forecast better than simple ones. The ranking of methods depends on the accuracy measure and on the horizon, and combinations of methods perform well.

For ARMA models the consequence is that the Box–Jenkins loop must end with an out-of-sample comparison against simple benchmarks: the mean, the last value and the exponential smoothing of Chapter 0. A model that passes all in-sample diagnostics has not yet shown that it forecasts better. Seminar 2, task C1, runs this comparison for Romanian inflation.

## Recap: More real series {h=819e5132}

Inflation is fitted by an AR(2) near a unit root; the model reverts to the historical mean and leaves residual correlation at lag 12. Daily returns have significant but tiny AR effects, and their variance needs GARCH models.

Yule's AR(2) recovers the 11-year solar cycle from complex roots. The forecasting competitions show that every model must be compared with simple benchmarks.

## @end

The same tools give different verdicts on different series. For inflation the AR(2) is close to a unit root and fails the diagnostics at lag 12; for daily returns the AR coefficients are significant but explain little, and the dependence is in the variance; for the sunspots two coefficients produce the 11-year cycle, and nine lags describe its asymmetry better. The forecasting competitions add the last check: comparison with simple benchmarks out of sample.

# Possible contribution of AI

Tools based on artificial intelligence (AI) can draft code, give a second explanation of a derivation and repeat the Box–Jenkins loop on many series. The two frames of this section describe what they can contribute and what the analyst must check before using their output.

## Possible contribution of AI {h=e93e3335}

An AI assistant can produce a first draft of code that fits ARMA($p,q$) models on a grid of orders, tabulates AIC and BIC and runs the residual tests. It can also give a second derivation of a result, for example the ACF of an ARMA(1,1) or the reason why the PACF of an MA decays, which the reader can compare line by line with the derivation in the course.

It can extend an analysis to many series: the same Box–Jenkins loop applied to the GDP growth of all EU (European Union) countries, for example, to compare the persistence of their business cycles.

The example prompt on the slide shows the level of detail a useful request needs: the data source, the grid of orders, the criteria, the degrees of freedom of the Ljung–Box test (model_df = p + q) and the forecast horizon. A vague request leaves these choices to the tool, and the defaults are often the ones the next frame warns against.

## Checks you must run {h=1e95475c}

The first checks concern conventions. Software writes the MA polynomial as $\theta(L) = 1 + \theta L$ (as statsmodels does) or as $1 - \theta L$ (as some textbooks do), so the sign of every $\theta$ must be checked. The constant reported by the ARIMA class of statsmodels is the mean $\mu$, not the intercept $c$.

The next checks concern inference. The Ljung–Box test on residuals must use $m - p - q$ degrees of freedom, and all models compared by information criteria must be fitted to the same sample. Stationarity and invertibility are read from the roots of both polynomials, not from the size of single coefficients.

The last checks concern the method. The series must be stationary before an ARMA model is fitted (Chapter 3), and the forecasts must be compared with simple benchmarks. Every reference cited by an AI tool must exist; its DOI must be checked.

## @end

AI tools shorten the coding and can repeat the analysis on many series, but the decisions that make an ARMA analysis correct remain with the analyst: sign conventions, the meaning of the constant, the degrees of freedom of the residual test, the roots, the stationarity of the series and the comparison with benchmarks.

# Summary

The summary collects the main results, the formulas to remember and three questions for self-assessment.

## Key takeaways {h=11647bc9}

The ARMA($p,q$) model replaces the infinite Wold sum by a ratio of two short polynomials, $\theta(L)/\phi(L)$. It is stationary when the roots of $\phi(z)$ lie outside the unit circle, invertible when the roots of $\theta(z)$ do, and well identified when the two polynomials have no common factor. In data, an ACF that cuts off points to an MA, a PACF that cuts off to an AR, and two decaying functions to an ARMA.

Models are estimated by maximum likelihood, chosen with AIC or BIC and checked with the Ljung–Box test on $m - p - q$ degrees of freedom. Their forecasts revert to the mean, and the intervals widen to $\pm 1.96\sqrt{\gamma(0)}$ while ignoring parameter uncertainty.

On real data, Romanian GDP growth is an MA(3) by construction, inflation is close to a unit root, and daily returns need GARCH models for their variance.

## Key formulas {h=24563724}

The table collects the formulas of the chapter. The first rows describe the models: the ARMA equation and its causal form $\psi(L) = \theta(L)/\phi(L)$; the mean, variance and ACF of the AR(1); the Yule–Walker recursion of the AR(2); the first autocorrelation and the invertibility condition of the MA(1); the $\psi$ weights and the ACF recursion of the ARMA(1,1).

The last rows describe the method: the Yule–Walker estimator, the information criteria, the reference distribution of the Ljung–Box statistic and the forecast interval $\hat X_{T+h|T} \pm 1.96\,\sigma\sqrt{\sum_{j=0}^{h-1}\psi_j^2}$, which follows from the error variance @eq:fvar.

## Self-assessment {h=005b2bd6}

The first question asks whether $X_t = 1.2X_{t-1} - 0.32X_{t-2} + \varepsilon_t$ is stationary. The characteristic polynomial factorises as $1 - 1.2z + 0.32z^2 = (1 - 0.8z)(1 - 0.4z)$, so the roots are 1.25 and 2.5, both outside the unit circle, and the process is stationary. The inverse roots, 0.8 and 0.4, give the same answer read from inside the circle.

The second question describes a series whose ACF has a single significant value, $\hat\rho(1) = -0.45$, and whose PACF decays. The pattern is that of an MA(1) with $\theta < 0$. The value of $\theta$ solves $\theta/(1 + \theta^2) = -0.45$, a quadratic equation with two real roots whose product is 1; the invertible one, with $|\theta| < 1$, is kept.

The third question gives $Q^*(10) = 15.2$ for the residuals of an ARMA(2,1). The degrees of freedom are $10 - 3 = 7$, and the 95% quantile of $\chi^2(7)$ is 14.07, below 15.2, so the test rejects at 5%. With 10 degrees of freedom the critical value would be 18.31, and the model would be wrongly accepted. Chapter 3 continues with unit roots and ARIMA models, for the case in which a root lies on the unit circle.

## @end

The chapter has built the stationary ARMA model from the Wold decomposition, derived its properties, estimated it, checked it and used it to forecast. Every application assumed a stationary series. Chapter 3 treats the case excluded here, a root on the unit circle, and the ARIMA models that handle it by differencing.
