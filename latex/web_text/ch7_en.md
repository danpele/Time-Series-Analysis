<!--
Course text of the web version, Chapter 7 (EN). Inserted into the page by latex/build_web_lecture.py.
Format: # section title (introduction), ## slide title {h=hash} (text), @after (text after the live block), ## @end (summary).
Numbered equations: $$ ... $$ {#eq:label}, reference @eq:label. Citations: @Key -> Author (year), [@Key] -> (Author, year).
Current hashes: python3 build_web_lecture.py 7 --hashes
-->

## @intro

@epigraph EngleGranger1987 | Co-integration and error correction: Representation, estimation, and testing
> The relationship between co-integration and error correction models, first suggested in Granger (1981), is here extended and used to develop estimation procedures, tests, and empirical examples. If each element of a vector of time series x first achieves stationarity after differencing, but a linear combination a'x is already stationary, the time series x are said to be co-integrated with co-integrating vector a.

@portrait photos/ch3_clive_granger_2008.jpg | Clive W. J. Granger | Photo: Olaf Storbeck (2008), CC BY-SA 2.0, Wikimedia Commons | https://commons.wikimedia.org/wiki/File:Clive_Granger_by_Olaf_Storbeck_(3x4_cropped).jpg

@text
[Chapter 3](../EN/Courses/chapter3_unit_roots_arima_models.pdf) established two facts that seem to close the discussion of series with a stochastic trend. Many economic and financial series are integrated of order 1, $I(1)$: they become stationary only after one difference. And the regression of one $I(1)$ series on another, unrelated $I(1)$ series often produces a spurious result, with large $t$-statistics and a large $R^2$. The cautious conclusion would be to always work in differences. This chapter shows why that conclusion is too hasty.

There is an important case in which two or more $I(1)$ series are tied together in the long run: some linear combination of them is stationary. The series can drift apart for a while, but the distance between them keeps returning towards an equilibrium level. This case is called cointegration. @Granger1981 formulated the idea, and @EngleGranger1987 gave the definition, the two-step estimation method and the link to error correction models. Differencing the data would erase exactly this information about the long run.

The chapter builds three tools: the residual-based Engle–Granger test, the single-equation error correction model, and the vector error correction model (VECM), with the Johansen tests for the number of equilibrium relations. It applies them to US Treasury yields, to US consumption and income, to purchasing power parity for the leu, to ROBOR and Euribor, to three Central European currencies and to a pairs-trading strategy on the Bucharest Stock Exchange. The results are mixed, and learning to read a negative result is one of the lessons of the chapter.

@chart tsa_ch7_examples
Four pairs of trending series: US Treasury yields at 3 months and 10 years (monthly, since 1960), US real GDP and consumption (quarterly, 1959–2009), adjusted share prices of Banca Transilvania and BRD (daily, since 2014) and the EUR/RON and EUR/HUF exchange rates (end of month, since July 2005).

@goals
- **Define** cointegration and explain it as a stochastic trend shared by several series.
- **Test** for cointegration with the Engle–Granger and Phillips–Ouliaris tests, using the right critical values.
- **Estimate** an error correction model and interpret the speed of adjustment and the half-life.
- **Derive** the VECM form of a VAR and choose the cointegrating rank with the Johansen tests.
- **Interpret** the cointegrating vectors, the adjustment coefficients and the impulse responses of a VECM.
- **Evaluate** the forecasts of a VECM and a pairs-trading strategy out of sample and after costs.

@plan
- The idea of cointegration | Spurious regression, the drunk and her dog, the definition and common stochastic trends.
- The Engle–Granger two-step method | The regression in levels, the residual test and its special critical values.
- Error correction models | Speed of adjustment, half-life and the DHSY consumption function.
- The VECM and the Granger representation theorem | Rewriting a VAR, the rank of $\Pi$, deterministic terms and weak exogeneity.
- The Johansen tests | Reduced-rank regression, the trace and maximum-eigenvalue statistics, three real systems.
- Estimating and interpreting β and α | The US term structure of yields: equilibria, adjustment and impulse responses.
- Forecasting with a VECM and with a VAR in differences | An out-of-sample forecasting experiment for levels and for a spread.
- Economic examples: parity and interest rates | PPP for the leu, ROBOR and Euribor, three Central European currencies.
- An application: pairs trading | A trading rule without look-ahead bias on the Bucharest Stock Exchange and on US banks.
- Possible contribution of AI | What an assistant can draft and what must be checked.
- Summary | Key ideas, formulas and self-assessment questions.

## Today's question and route {h=15f1e6b8}

The question of the chapter starts from two prices that both move like random walks. Each of them, taken alone, cannot be forecast over long horizons. A combination of them, for example their difference or a ratio, may nonetheless be stable over time. If so, the series share a stochastic trend and a long-run equilibrium, and a regression in levels estimates a real relation, not a spurious one.

The route has three parts. The first defines cointegration and tests it with the Engle–Granger method, using the critical values that fit it, then builds the error correction model. The second moves to systems: the VECM, the Granger representation theorem, the Johansen tests, the interpretation of the coefficients and forecasting. The third applies the tools: purchasing power parity, interest rates, three Central European currencies and pairs trading on the Bucharest Stock Exchange. The chapter builds on [Chapter 3](../EN/Courses/chapter3_unit_roots_arima_models.pdf), for unit roots and spurious regression, and on [Chapter 6](../EN/Courses/chapter6_var_models_granger_causality.pdf), for VAR models.

## Learning outcomes {h=ad0ee8ee}

The first outcomes concern the concept and its single-equation test. The reader should be able to define cointegration and explain it as a common stochastic trend, to apply the Engle–Granger and Phillips–Ouliaris tests, and to compare the statistics with the critical values of cointegration tests, not with the Dickey–Fuller ones. The reader should then be able to estimate an error correction model and translate the speed of adjustment into a half-life.

The next outcomes concern systems: writing a VAR as a VECM, stating the Granger representation theorem and choosing the cointegrating rank with the Johansen tests, followed by the interpretation of the cointegrating vectors and the adjustment coefficients. The last outcomes concern use: when a VECM forecasts better than a VAR in differences, and how to evaluate an application, from parity conditions to pairs trading, out of sample and after costs.

## Reading and tools {h=bb258724}

The course textbook is @HuangPetukhina2022, whose Chapter 9 covers non-stationarity and cointegration. The theory is treated at length in @Hamilton1994, Chapters 19–20, and in @Lutkepohl2005, Chapters 6–8. The maximum-likelihood approach developed by Søren Johansen and Katarina Juselius is set out in @Johansen1995 and, with an emphasis on applications, in @Juselius2006.

The reference papers are @EngleGranger1987, which defined cointegration, @Johansen1988 and @Johansen1991, which gave the tests for systems, and Granger's Nobel lecture, @Granger2004. The computations are done in Python: *coint*, *coint_johansen*, *VECM* and *VAR* from *statsmodels*, and *phillips_ouliaris* from *arch*. The chapter's Quantlets reproduce every chart and table, and the lecture notebook runs in Google Colab.

# The idea of cointegration

The section starts from the problem of spurious regression and shows the case in which a regression between $I(1)$ series makes sense. The intuition comes from a well-known story, a drunk walking her dog, checked afterwards by simulation. Then come the formal definition of cointegration, its representation through common stochastic trends, the examples from economic theory and the first tests on four pairs of real series.

## From spurious regression to cointegration {h=9c39bc4f}

@GrangerNewbold1974 showed, by simulation, that regressing one random walk on another, independent random walk mostly gives a large $t$-statistic, a large $R^2$ and a Durbin–Watson statistic close to 0. The result is a spurious regression, discussed in [Chapter 3](../EN/Courses/chapter3_unit_roots_arima_models.pdf). The cause is the regression error: whatever the slope $b$, the combination $y_t - a - bx_t$ still contains the stochastic trend of $y_t$ and that of $x_t$, so it is itself $I(1)$. Nothing ties the two series together, and the usual tests, which assume stationary errors, are no longer valid.

The exception is the case in which, for one particular value of $b$, the combination $u_t = y_t - a - bx_t$ is stationary. The stochastic trends of the two series then cancel exactly in this combination, and the series are called cointegrated [@Granger1981; @EngleGranger1987]. The regression in levels is no longer spurious: it estimates a real long-run relation. The two series can drift apart for a while, but $u_t$, being stationary, returns towards its mean and brings them back towards each other. In this sense, $u_t = 0$ describes an equilibrium.

The rest of the chapter answers three questions about this case: how to recognise it in the data, how to test formally whether a combination is stationary, and how to build a model that uses the information in the equilibrium.

## A drunk and her dog {h=3c331315}

@Murray1994 gave an illustration that has become a classic. A drunk leaves the pub and wanders aimlessly: her position $x_t$ is a random walk, each step being unpredictable. Her dog wanders too, so its position $y_t$ is also a random walk. If the two were independent passers-by, the distance between them would be the difference of two random walks, hence again a random walk, and its standard deviation would grow without bound, in proportion to $\sqrt{t}$.

The drunk and the dog are not independent, however. When they drift apart, the dog hears her call and runs back, and she hears the dog and turns towards it. Each closes part of the distance at every step: $\Delta x_t = c\,(y_{t-1} - x_{t-1}) + u_t$ and $\Delta y_t = d\,(x_{t-1} - y_{t-1}) + w_t$, where $c, d > 0$ are the fractions of the distance closed at each step and $u_t$ and $w_t$ are independent shocks. The terms in $c$ and $d$ are error correction terms.

The result combines two properties that seem contradictory. Each path remains unpredictable, because the shocks accumulate, but the distance $y_t - x_t$ stays bounded, because it is corrected at every step. This is, in a single picture, the idea of the chapter: cointegration and error correction are two sides of the same phenomenon.

## The drunk and her dog by simulation {h=747fcad4}

The simulation runs the two equations for $T = 400$ steps, with Gaussian shocks and $c = 0.15$ and $d = 0.25$: at each step the drunk closes 15% of the distance and the dog 25%. For comparison, a stray dog $z_t$ also appears, an independent random walk with the same shock variance.

The left panel shows the three paths. The drunk ($x_t$) and her dog ($y_t$) wander together: their common path rises and falls without any predictable tendency, but the two lines stay almost on top of each other. The stray dog heads off in another direction and ends, after 400 steps, about 50 units below the other two. The right panel shows the distances: $y_t - x_t$ oscillates tightly around zero, while $z_t - x_t$ drifts far away, down to about $-50$.

## Interpreting the simulation: the distances {h=1408d10d}

Unit root tests confirm what the chart shows. The augmented Dickey–Fuller (ADF) test applied to the drunk's path gives a p-value of 0.21, so the unit root is not rejected, as expected for a random walk. The distance to her own dog, however, is an AR(1) process: subtracting the two equations, $y_t - x_t = (1 - c - d)(y_{t-1} - x_{t-1}) + (w_t - u_t)$, with coefficient $\rho = 1 - 0.15 - 0.25 = 0.6$. The ADF test on this distance gives a p-value below 0.001, and its standard deviation is 1.9. The distance to the stray dog has an ADF p-value of 0.81 and a standard deviation of 14.1, so it behaves like a random walk.

The coefficient $\rho = 0.6$ also tells how fast a distance closes. After $h$ steps without new shocks a fraction $0.6^h$ of it remains, so half of it has gone after $h = \ln 0.5/\ln 0.6 = 1.4$ steps. This quantity is the half-life, which returns in [Section 3](#sec-3) for the estimated models.

The practical consequence matters. Neither the drunk's path nor the dog's can be forecast over long horizons, but knowing where the drunk is tells us where to look for the dog. Cointegration does not make the series predictable; it makes the relation between them predictable.

## Definition of cointegration {h=4e3e45f7}

The definition of @EngleGranger1987 applies to a vector of $n$ series, $\mathbf y_t = (y_{1t}, \dots, y_{nt})^\top$. Its components are cointegrated of order (1,1), written $\mathbf y_t \sim CI(1,1)$, if two conditions hold. First, each component is $I(1)$, that is, it becomes stationary after one difference, as in [Chapter 3](../EN/Courses/chapter3_unit_roots_arima_models.pdf). Second, there is a vector $\beta \neq 0$ for which the linear combination $\beta^\top\mathbf y_t$ is $I(0)$, that is, stationary. The first 1 in the notation gives the order of integration of the series, the second by how much this order falls in the combination.

The vector $\beta$ is called the cointegrating vector, and $u_t = \beta^\top\mathbf y_t - \mu$, where $\mu$ is the mean of the combination, is called the equilibrium error. The vector is not unique: if $\beta^\top\mathbf y_t$ is stationary, so is $2\beta^\top\mathbf y_t$. One coefficient is therefore normalised to 1; with two variables one usually writes $\beta = (1, -b)^\top$, which corresponds to the relation $y_t = a + bx_t + u_t$.

With $n$ variables there can be several linearly independent cointegrating vectors, but at most $n - 1$; their number $r$ is called the cointegrating rank. Cointegration is a property of the levels: if all series are differenced before modelling, the combination $\beta^\top\mathbf y_t$ disappears from the data and, with it, the information about the equilibrium.

## Common stochastic trends {h=391df8e8}

A simple example shows the mechanism. Let $x_t = x_{t-1} + \varepsilon_t$ be a random walk and $y_t = bx_t + u_t$, with $u_t$ stationary. Both series contain the same stochastic trend, the sum of past shocks $\sum_{s \le t}\varepsilon_s$: $x_t$ contains it with weight 1, $y_t$ with weight $b$. The combination $y_t - bx_t = u_t$ removes it completely and stays stationary. Cointegration therefore means that the series are driven by shared trends, not each by its own.

@StockWatson1988 turned this observation into a general common trends representation: $n$ variables with $r$ cointegrating vectors are driven by $n - r$ common stochastic trends. The two extremes are instructive. With $r = 0$ there are $n$ separate trends and no cointegration. With $r = n - 1$ there is a single trend, shared by all the series. The Treasury yields at 1, 5 and 10 years, with $r = 2$, have a single common trend, the general level of interest rates, as [Section 5](#sec-5) shows.

The worked example uses three series: $x_t$ a random walk, $y_t = 2x_t + u_t$ and $z_t = -x_t + v_t$, with $u_t$ and $v_t$ stationary. All three are driven by the same trend, so $n = 3$, $n - r = 1$ and $r = 2$. In the order $(x_t, y_t, z_t)$, two cointegrating vectors are $\beta_1 = (-2, 1, 0)^\top$, because $y_t - 2x_t = u_t$, and $\beta_2 = (0, 1, 2)^\top$, because $y_t + 2z_t = 2x_t + u_t - 2x_t + 2v_t = u_t + 2v_t$. Any linear combination of the two vectors is again a cointegrating vector; the rank says how many independent directions exist.

## Cointegration in economic theory {h=6d54f776}

Many economic theories state that some combination of non-stationary variables must be stable. Purchasing power parity (PPP) writes the log exchange rate as $s_t = p_t - p_t^* + q_t$, where $s_t$ is the log exchange rate (lei per euro), $p_t$ and $p_t^*$ are the logs of the domestic and foreign price levels, and $q_t$ is the real exchange rate. The theory requires $q_t$ to be stationary, which means cointegration of $s_t$, $p_t$ and $p_t^*$; [Section 8](#sec-8) tests this requirement for the leu.

The term structure of interest rates gives the second example. Under the expectations hypothesis, a long-term yield is the average of expected short-term yields plus a premium; the difference (the spread) between the long and the short yield is then stationary [@CampbellShiller1987]. The third example is the "great ratios": if consumption follows permanent income, the ratio of consumption to income, $c_t - y_t$ in logs, is stable.

The last examples come from financial markets. Arbitrage keeps small the difference between the spot price and the futures price of the same asset, or between an index and a fund that tracks it. The shares of similar firms are exposed to the same sector shocks and may share a trend; this is the basis of the pairs-trading strategy in [Section 9](#sec-9). In every case, theory proposes a relation and the data must confirm it.

## Four pairs of trending series {h=197c42b6}

The chart shows four pairs of series used throughout the chapter. The first panel contains the US Treasury yields at 3 months and 10 years, in percent per year, monthly since 1960, from the FRED database (Federal Reserve Economic Data). The second contains US real GDP and consumption, quarterly from 1959 to 2009, as 100 × the natural log, set to 0 in the first quarter of 1959; on this scale a difference of 1 unit means approximately 1%.

The third panel shows the adjusted share prices of Banca Transilvania (TLV) and BRD since 2014, and the fourth the EUR/RON and EUR/HUF reference rates published by the National Bank of Romania (BNR), at the end of the month, since July 2005, both as 100 × log, set to 0 at the start. In every panel the two series seem to move together. The chart alone cannot say, however, whether this impression reflects a common trend or just two separate trends that happened to point the same way in the sample.

## Interpreting the four pairs {h=5b5bbd37}

Each of the eight series is $I(1)$ according to the tests of [Chapter 3](../EN/Courses/chapter3_unit_roots_arima_models.pdf). The question is whether the two series of each pair move together in the long run, and the answer comes from the Engle–Granger test of [Section 2](#sec-2). Its statistic, $\tau$, is compared with a 5% critical value of about $-3.34$; a more negative value rejects the null hypothesis of no cointegration.

For the 10-year yield on the 3-month yield, $\tau = -3.35$ with p-value 0.048, a borderline result. For consumption and income, $\tau = -3.54$ with p-value 0.029, and for the two banks $\tau = -4.33$ with p-value 0.002: these are cointegrated in the sample since 2014. For EUR/RON and EUR/HUF, $\tau = -2.47$ with p-value 0.294: the test does not reject the null of no cointegration, so the data give no evidence of a common trend of the two exchange rates.

The last pair is the most instructive. The two exchange rates have risen by similar amounts since 2005, and the regression in levels has $R^2 = 0.79$. A common direction in the sample is not a common stochastic trend, however: the paths of the two series have parted several times, for example after 2020, and no force brings them back together.

## Question for the room {h=e5b60e03}

First question: two $I(1)$ series have a correlation of 0.95 in levels; are they cointegrated? The correlation does not answer the question. Unrelated random walks often have large correlations in levels; this is the very phenomenon of spurious regression in [Chapter 3](../EN/Courses/chapter3_unit_roots_arima_models.pdf). Only a test applied to the equilibrium error, that is, the question whether a combination of the series is stationary, can decide.

Second question: two cointegrated series have a correlation of 0.2 between their daily changes; is this a contradiction? No. Cointegration concerns the long run: the series do not drift apart indefinitely. In the short run their changes can be almost uncorrelated, and the correction of the distance can work slowly, over many days. The correlation of the changes and the cointegration of the levels are different properties.

## Recap: The idea of cointegration {h=54cdf90f}

$I(1)$ series are cointegrated if a linear combination $\beta^\top\mathbf y_t$ is stationary; the combination describes a long-run equilibrium, and deviations from it are corrected. An equivalent statement is that the series share stochastic trends: $n$ variables with $r$ cointegrating vectors are driven by $n - r$ trends. Economic theory suggests cointegration in many places, from parity conditions to spreads and "great ratios", but the data must confirm it through a test.

## @end

Cointegration is the exception to the rule of spurious regression: series that, taken alone, are unpredictable random walks can be tied by a stationary combination. The simulation of the drunk and her dog shows that a distance corrected at every step stays bounded although both paths wander. On real data, the Engle–Granger test finds cointegration for consumption and income and for two Romanian banks, a borderline result for two US yields and no evidence for EUR/RON and EUR/HUF. The next section explains the test and its critical values.

# The Engle–Granger two-step method

The method of @EngleGranger1987 is the simplest way to test two series for cointegration: a regression in levels, followed by a unit root test on the residuals. The simplicity hides a trap: the critical values are no longer the Dickey–Fuller ones. The section explains why, obtains them by simulation, presents the alternative Phillips–Ouliaris test, applies both tests to real data and closes with the limits of the method.

## The two steps of the method {h=fbb4001b}

Step 1 estimates the long-run relation by ordinary least squares (OLS) in levels, $y_t = a + bx_t + u_t$, and keeps the residuals $\hat u_t = y_t - \hat a - \hat bx_t$. If the series are cointegrated, the estimator $\hat b$ has an unusual property: it is super-consistent, meaning that its error shrinks like $1/T$, faster than the usual rate $1/\sqrt{T}$. The intuition is that any wrong value of $b$ leaves a stochastic trend in the residuals, whose variance then grows with the sample; OLS, which minimises this variance, finds the right value very quickly.

Step 2 applies an ADF test, as in [Chapter 3](../EN/Courses/chapter3_unit_roots_arima_models.pdf), to the step-1 residuals, without a constant, because OLS residuals already have mean zero:

$$ \begin{gathered} \Delta\hat u_t = \gamma\,\hat u_{t-1} + \sum_{j=1}^{k}\delta_j\,\Delta\hat u_{t-j} \\ + e_t, \\ H_0: \gamma = 0 \ \text{(no cointegration)}, \\ H_1: \gamma < 0 \ \text{(cointegration)}. \end{gathered} $$ {#eq:eg}

Here $\delta_j$ are the coefficients of the $k$ lagged differences, which absorb the autocorrelation of $\hat u_t$, and $e_t$ is white noise. The null hypothesis $\gamma = 0$ states that the residuals have a unit root, so there is no cointegration; the alternative $\gamma < 0$ states that the residuals revert towards zero. The test statistic is $\tau = \hat\gamma/\mathrm{SE}(\hat\gamma)$, where SE is the standard error, and the null is rejected when $\tau$ is below the critical value.

The test has a precondition: each series must be $I(1)$. An $I(0)$ series and an $I(1)$ series cannot be cointegrated, because any combination with a non-zero coefficient on the $I(1)$ series remains $I(1)$. That is why every cointegration analysis starts with unit root tests on each series.

## Engle–Granger residuals and the Dickey–Fuller critical values {h=c5252494}

The step-1 residuals are not an observed series but the result of an optimisation. OLS chooses $\hat b$ so that the variance of the residuals is as small as possible, that is, so that $\hat u_t$ looks as "calm" as possible. Even when the series are not cointegrated, $\hat u_t$ therefore looks more stationary than the true error $u_t$. Under the null, the statistic $\tau$ is pushed towards negative values, and its distribution lies to the left of the Dickey–Fuller distribution.

The right critical values depend on the number of variables $n$ in the regression and on the deterministic terms in step 1 (a constant, or a constant and a trend). @EngleGranger1987 obtained them by simulation; @MacKinnon1994 and @MacKinnon2010 then gave accurate approximations, for any sample size, which the *coint* function of *statsmodels* uses.

There is one case in which the Dickey–Fuller table is the right one: when the cointegrating vector is known from theory. If theory says that the spread between two rates is stationary, that is $\beta = (1, -1)^\top$, the spread is computed directly, without any estimation, and the ordinary ADF test, with its own table, is applied to this observed series.

## The null distribution by simulation {h=d8a5ee6b}

The simulation builds the distribution of the statistic $\tau$ under the null. It generates 10,000 samples of $T = 200$ observations, each made of independent Gaussian random walks, hence without cointegration. For each sample, step 1 is run with a constant and step 2 without lags, and $\tau$ is recorded. The smoothed histograms of the 10,000 values give the distributions in the chart.

The chart shows four curves. The dashed line is the standard Normal distribution, the usual distribution of a $t$-statistic. The blue curve is the Dickey–Fuller distribution for one series with a constant, already far to the left of zero. The red and green curves are the Engle–Granger distributions for 2 and 3 variables, shifted even further left. The dotted vertical lines mark the 5% quantiles of the three non-standard distributions: the more variables, the more negative the critical value.

## Interpreting the simulation: the critical values {h=7cb4a4d8}

The 5% quantiles from the simulation are $-2.86$ for a single series (the Dickey–Fuller case), $-3.33$ for 2 variables and $-3.75$ for 3 variables. The MacKinnon values for $T = 200$ are $-2.88$, $-3.37$ and $-3.78$, so the simulation reproduces them up to simulation error.

The difference between the tables has large consequences. If the residual statistic were compared with the Dickey–Fuller value $-2.88$, a true null would be rejected in 14.5% of the samples with 2 variables and in 30.3% of those with 3 variables, instead of 5%. In other words, cointegration between independent random walks would be "found" three to six times more often than the level of the test promises.

The reason why the critical values grow in absolute value with the number of variables is the one from the previous frame: each additional variable gives OLS one more coefficient with which to reduce the variance of the residuals, so the residuals look even more stationary.

## Critical values of the Engle–Granger test {h=0d725bc1}

The table gives the asymptotic critical values, for $T \to \infty$, from @MacKinnon2010, at the 1%, 5% and 10% levels, for 1–4 variables and for two specifications of step 1. The first row, with a single variable, is the ordinary Dickey–Fuller test and serves as a benchmark: the 5% value with a constant is $-2.86$. For 2 variables it becomes $-3.34$, for 3 variables $-3.74$, and for 4 variables $-4.10$.

The row is chosen by the total number of variables in the regression, including $y_t$; a regression of $y_t$ on $x_t$ uses the row $n = 2$. In finite samples the critical values are slightly more negative than the asymptotic ones, for example $-3.37$ instead of $-3.34$ for 2 variables and $T = 200$; software computes the value appropriate for $T$.

The right-hand columns are used when step 1 also includes a deterministic trend. This specification fits when the series have drifts (average growth) that the cointegrating relation does not cancel; the critical values are then even more negative, for example $-3.78$ at 5% for 2 variables.

## The Phillips–Ouliaris test {h=e57aa753}

@PhillipsOuliaris1990 proposed a test with the same step 1 and the same null hypothesis, no cointegration. The difference is in step 2: the Dickey–Fuller regression on the residuals is estimated without lagged differences, and the autocorrelation of the residuals is handled by correcting the $t$-statistic with the long-run variance of the residuals, as in the Phillips–Perron test of [Chapter 3](../EN/Courses/chapter3_unit_roots_arima_models.pdf). The test has two versions: $Z_t$, the corrected $t$-ratio, and $Z_\alpha$, the corrected statistic $T\hat\gamma$. The asymptotic critical values of $Z_t$ are the same as for the Engle–Granger test.

Using both tests has a practical reason. The Engle–Granger test depends on the number of lags $k$, usually chosen by the Akaike information criterion (AIC), while the Phillips–Ouliaris test depends on the bandwidth used for the long-run variance. The two choices handle the same autocorrelation in different ways. When the tests agree, the verdict is robust; when they disagree, the residual plot and the sample must be examined.

In Python, the Engle–Granger test is *coint(y, x)* from *statsmodels*, and the Phillips–Ouliaris test is *phillips_ouliaris(y, x, test_type="Zt")* from *arch*.

## Worked example: US consumption and income {h=039474f1}

The data are US real consumption and real GDP, quarterly, from the first quarter of 1959 to the third quarter of 2009, that is, 203 quarters. Both are transformed into $c_t$ and $y_t$, 100 × the natural log, so that one unit means approximately 1%. The preliminary ADF tests with trend give p-values of 0.31 for $c_t$ and 0.39 for $y_t$, so both series are treated as $I(1)$.

Step 1 gives $\hat c_t = -107.6 + 1.075\,y_t$, with $R^2 = 0.9992$ and a Durbin–Watson (DW) statistic of the residuals of 0.24. The DW statistic is about 2 when the residuals are not autocorrelated and approaches 0 when they are highly persistent. The slope close to 1 suggests a stable share of consumption in income. Neither an $R^2$ near 1 nor a small DW proves anything, however: the same combination appears in a spurious regression.

Step 2 gives $\tau = -3.54$, with 0 lags chosen by AIC. The 5% critical value for $n = 2$ and $T = 203$ is $-3.37$. Since $-3.54 < -3.37$, the hypothesis "no cointegration" is rejected at 5%, with p-value 0.029; the Phillips–Ouliaris test gives $Z_t = -3.52$, with p-value 0.031. Comparing with the Dickey–Fuller value, $-2.88$, would have led to the same decision here, but for the wrong reason: with a statistic between $-3.37$ and $-2.88$, it would have led to a wrong conclusion.

## The step-1 residuals {h=9da912dc}

The chart shows the step-1 residuals $\hat u_t$ for two regressions: consumption on income (left, quarterly) and the 10-year yield on the 3-month yield (centre, monthly). Both oscillate around zero, but in long waves: the consumption residuals stay below zero for most of the 1960s and above zero after 2000, and the yield residuals have cycles of several years, with the largest deviations around 1980.

The right panel compares the sample autocorrelation functions (ACF) of the two residual series with the ACF of a simulated random walk (dashed line). The ACF of the consumption residuals falls quickly and crosses below zero after about 14 lags. The ACF of the yield residuals falls much more slowly. The ACF of the random walk stays close to 1 and falls only to about 0.7 after 24 lags.

## Interpreting the residuals {h=8e30c9e4}

For consumption, the residual autocorrelation is 0.86 at lag 1 but only 0.31 at lag 8. Deviations from equilibrium, of the order of 1–3%, persist from one quarter to the next but shrink considerably within about two years. For the yields, the autocorrelation is 0.97 at lag 1 and still 0.58 at lag 12: deviations last for years, and the test is borderline ($\tau = -3.35$, p-value 0.048).

High persistence is the usual situation for equilibrium errors, not an exception. An AR(1) process with $\rho$ close to 1 is hard to tell apart from a random walk in a finite sample; the power of unit root tests against such alternatives is low, as in [Chapter 3](../EN/Courses/chapter3_unit_roots_arima_models.pdf). A borderline result for very persistent series must therefore be read with caution in both directions.

The comparison with the random walk also shows the limit of visual inspection. The ACF of a random walk decays much more slowly than that of the residuals, but the difference between slow and very slow decay cannot be settled by eye. The test turns this difference into a decision with a known significance level.

## Engle–Granger and Phillips–Ouliaris on real data {h=5b58ebb6}

The table applies both tests to seven regressions. For each, the columns give the data frequency, the number of observations $T$, the estimated slope $\hat b$, the Engle–Granger statistic $\tau$ with the number of lags $k$ chosen by AIC, its p-value, the Phillips–Ouliaris statistic $Z_t$ and its p-value. The p-values come from the approximations of @MacKinnon2010 for 2 variables.

The first rows repeat the examples already discussed: US consumption on income ($\tau = -3.54$, p-value 0.029) and the 10-year yield on the 3-month yield ($\tau = -3.35$ with 20 lags, p-value 0.048, while $Z_t = -4.04$, p-value 0.006). Then comes the bank pair in three versions: TLV on BRD since 2014 (p-value 0.002), the reverse regression BRD on TLV since 2014 (p-value 0.004) and TLV on BRD since 2010 (p-value 0.622). The EUR/RON and EUR/HUF exchange rates have a p-value of 0.294.

The last row uses Romanian household consumption and GDP, from Eurostat, in chain-linked volumes, seasonally adjusted, quarterly since 1995, 126 observations. The slope is 1.46, and the two tests give opposite verdicts: p-value 0.363 for Engle–Granger, with 8 lags, and below 0.001 for Phillips–Ouliaris.

## Interpreting the table {h=f22cff51}

The first lesson concerns normalisation. The regression of TLV on BRD gives $\tau = -4.33$, and the regression of BRD on TLV gives $\tau = -4.16$; the slopes, 1.36 and 0.71, are not exact inverses of each other, since $1/1.36 = 0.74$. OLS minimises errors in the direction of the left-hand variable, so the result depends on that choice. Here the verdict is the same in both directions, but this is not guaranteed in general.

The second lesson concerns the sample. Since 2010, the same banks have a p-value of 0.622, and the test does not detect cointegration; since 2014, the p-value is 0.002. Choosing the start date after examining the data and seeing which one gives the desired result is a form of data snooping, that is, searching the data until a result appears; a p-value obtained this way no longer has its nominal meaning.

The third lesson concerns disagreement between the tests. For Romania, Engle–Granger with 8 lags does not reject no cointegration, while Phillips–Ouliaris rejects it at any usual level. The slope of 1.46 is far from 1: the share of consumption in GDP rose strongly over the period, and such a relation can be hard to tell apart from an unstable one. Seminar 7 (B2) discusses this case.

## Limits of the Engle–Granger method {h=6fcea17a}

The method has four limits. First, it can find at most one cointegrating vector. With two variables this is not a problem, but with $n \ge 3$ variables there may be several vectors, and a single regression estimates only one combination of them ([Sections 4](#sec-4)–[5](#sec-5)). Second, the results depend on the variable placed on the left, as the bank pair showed.

The third limit concerns inference on $\hat b$. The estimator is super-consistent but biased in small samples, and the $t$-statistics reported by OLS in step 1 are not valid, because the errors are autocorrelated and correlated with the changes in $x_t$. One remedy is the dynamic OLS (DOLS) estimator of @StockWatson1993, which adds leads and lags of $\Delta x_t$ to the regression and gives valid standard errors.

The fourth limit comes from the two-step structure: estimation errors in step 1 carry over into step 2, and the power of the test is low when adjustment to equilibrium is slow. The Johansen method of [Section 5](#sec-5) avoids these problems by estimating everything at once, by maximum likelihood, in a system.

## Case study: Engle and Granger (1987) {h=84851d25}

The question of the paper was how to model $I(1)$ series tied in the long run without falling into the trap of spurious regression and without removing, by differencing, the information about the long run. @EngleGranger1987, written at the University of California, San Diego, became one of the most cited papers in econometrics.

The contribution has three parts. The first is the definition of cointegration and the representation theorem of [Section 4](#sec-4), which shows that cointegrated series always have an error correction representation and conversely. The second is the two-step estimator and the residual-based tests, with critical values obtained by simulation, that is, the method of this section. The third is the applications to US series: consumption and income, prices and wages, short-term and long-term interest rates.

Robert Engle received the 2003 Nobel Prize in economics jointly with Clive Granger; Engle's share was for the ARCH models of [Chapter 5](../EN/Courses/chapter5_conditional_volatility_garch.pdf), and Granger's for cointegration.

## Recap: The Engle–Granger method {h=548a5c36}

The method has two steps: OLS in levels, with a super-consistent estimator $\hat b$, and an ADF test on the residuals @eq:eg, with the null hypothesis of no cointegration. The critical values are the Engle–Granger values computed by MacKinnon, more negative than the Dickey–Fuller ones and dependent on the number of variables. A verdict is checked with the Phillips–Ouliaris test, with the reverse normalisation and on another sample.

## @end

The Engle–Granger test is simple, but it needs the right table: because OLS makes the residuals look stationary, the Dickey–Fuller critical values would find cointegration between independent series three to six times more often than they promise. On real data, US consumption and income and the TLV–BRD pair since 2014 are cointegrated according to both tests; the same pair since 2010, the EUR/RON and EUR/HUF rates and, according to Engle–Granger, Romanian consumption and GDP give no evidence of cointegration. Dependence on normalisation, on the sample and on the lag choice are the limits that motivate methods for systems.

# Error correction models

If two series are cointegrated, yesterday's equilibrium error contains information about today's changes. The error correction model (ECM) puts this information into a regression in differences. The section defines the model, translates the adjustment coefficient into a half-life, estimates ECMs for US consumption and for the 10-year yield and presents the DHSY consumption function, the first famous application of this form.

## The error correction model {h=e0fff57a}

Let $u_{t-1} = y_{t-1} - a - bx_{t-1}$ be the equilibrium error of the previous period. The error correction model explains the change in $y_t$ by the current and past changes of the variables and by this error:

$$ \begin{aligned} \Delta y_t = {} & c + \gamma\,u_{t-1} + \delta_0\,\Delta x_t \\ & + \sum_{j \ge 1}\left(\phi_j\,\Delta y_{t-j} + \delta_j\,\Delta x_{t-j}\right) \\ & + \varepsilon_t. \end{aligned} $$ {#eq:ecm}

Here $c$ is the intercept, $\phi_j$ and $\delta_j$ are the short-run coefficients of the lagged differences, and $\varepsilon_t$ is white noise. If the series are cointegrated, all the terms of the equation are stationary: the differences because the series are $I(1)$, and $u_{t-1}$ by the definition of cointegration. That is why OLS and the usual $t$-tests are valid, and $\hat u_{t-1}$ from step 1 of the Engle–Granger method can be used in place of $u_{t-1}$, thanks to the super-consistency of $\hat b$.

The coefficients have distinct interpretations. The slope $b$ of the equilibrium relation is the long-run effect of $x$ on $y$; $\delta_0$ is the short-run effect, within the same period. The coefficient $\gamma$ is the speed of adjustment. If $y_{t-1}$ was above equilibrium, $u_{t-1} > 0$, and $\gamma < 0$, then $\gamma u_{t-1}$ is negative and $y_t$ tends to fall: a fraction $|\gamma|$ of the distance is closed in each period. Error correction requires $-2 < \gamma < 0$; for $\gamma$ between $-1$ and 0 the adjustment is monotone.

## Speed of adjustment and half-life {h=0a38bf96}

The speed $\gamma$ is hard to read directly; a time is easier to communicate. In a two-variable system in which only $y$ adjusts, without new shocks, the equilibrium error evolves as $u_t = (1 + \gamma)\,u_{t-1}$, so after $h$ periods $u_h = (1 + \gamma)^h u_0$ remains. The half-life is the number of periods after which half of a deviation has gone:

$$ (1 + \gamma)^{h} = 0.5 \quad\Longrightarrow\quad h_{1/2} = \frac{\ln 0.5}{\ln(1 + \gamma)}. $$ {#eq:halflife}

The worked example uses $\gamma = -0.2$: $h_{1/2} = \ln 0.5/\ln 0.8 = 3.1$ periods. On monthly data, half of a deviation disappears in about three months; on quarterly data, in about three quarters, that is, nine months. The same value of $\gamma$ therefore means very different real speeds.

That is why half-lives are compared only in the same time unit. A $\gamma$ estimated on monthly data and one estimated on quarterly data are not directly comparable; they must first be turned into half-lives and expressed in the same unit.

## Half-lives: theory and estimates {h=40f9d404}

The left panel shows the function $(1 + \gamma)^h$, the fraction of a unit deviation left after $h$ periods, for three speeds. With $\gamma = -0.05$ the curve falls slowly, and half of the deviation has gone after 13.5 periods. With $\gamma = -0.2$, after 3.1 periods. With $\gamma = -0.5$, after a single period. The dotted horizontal line marks the level 0.5; the point where each curve crosses it is the half-life.

The right panel shows the same decay for the two ECMs estimated in the next frame: US consumption, in quarters, with $\hat\gamma = -0.047$, and the 10-year yield, in months, with $\hat\gamma = -0.033$. On the horizontal axis the same numbers mean quarters for the first curve and months for the second; the two curves look alike but describe different real speeds.

## Interpreting the estimated ECMs {h=b4aa9b09}

For US consumption, the estimated speed of adjustment is $\hat\gamma = -0.047$, with $t = -1.75$, which gives a half-life of 14.3 quarters, more than three and a half years. The short-run effect of income is $\hat\delta_0 = 0.54$: roughly half of an increase in income reaches consumption in the same quarter. The rest comes slowly, through error correction, and the adjustment term is only marginally significant, at the 10% level.

For the 10-year yield, regressed on the 3-month yield, $\hat\gamma = -0.033$, with $t = -4.94$, and the half-life is 20.8 months; the short-run effect is $\hat\delta_0 = 0.38$. The 3-month rate also reacts to the same equilibrium error, with $\hat\gamma = 0.024$ and $t = 2.08$. The opposite sign makes sense: when the 10-year yield is too high relative to the 3-month yield, the former falls and the latter rises, so the two rates move towards each other.

The last observation shows the limit of a single-equation ECM. If both variables adjust, the equation for $\Delta y_t$ describes only part of the correction, and the other part stays outside the model. The model that describes the adjustment of all variables at once is the VECM of [Section 4](#sec-4).

## Worked example: one quarter of consumption {h=f45277dc}

The ECM estimated for US consumption has, as its main terms, $\Delta c_t = 0.418 + 0.54\,\Delta y_t - 0.047\,\hat u_{t-1} + \dots$, where $\hat u_{t-1} = c_{t-1} - 1.075\,y_{t-1} - \hat a$ is the equilibrium error of the previous quarter. Since the variables are 100 × log, $\hat u_{t-1} = 2$ means consumption about 2% above its equilibrium level.

The scenario assumes exactly this situation and a 1% rise in income in the current quarter, with the lagged differences equal to 0. The short-run effect of income is $0.537 \times 1 = 0.537$ (the coefficient $0.54$ before rounding). Error correction contributes $-0.047 \times 2 = -0.095$. Together with the intercept, the forecast growth of consumption is $0.418 + 0.537 - 0.095 \approx 0.86\%$.

Without the correction term, the model would forecast higher growth and would ignore the fact that consumption is already too high relative to income. Error correction is the information that a model in differences alone does not have.

## Case study: the consumption function of DHSY (1978) {h=3108bc5b}

The question arose in the United Kingdom of the 1970s: consumption functions estimated in levels or in differences forecast consumers' expenditure poorly. @DavidsonEtAl1978, the paper by Davidson, Hendry, Srba and Yeo, known by its initials as DHSY, analysed quarterly British data on consumers' expenditure and income.

The method started from a large dynamic model and reduced it step by step, through a general-to-specific search. The final model is written in differences but also contains the lagged log ratio of expenditure to income, that is, an error correction term. This term pulls consumption back towards a stable share of income and explains why the models in differences alone went wrong.

The legacy of the paper is that the ECM form appeared before the theory that justifies it. @Granger1981 and @EngleGranger1987 then showed why it works: expenditure and income are cointegrated, and the representation theorem guarantees an error correction form. The $t$-statistic of $\gamma$ in an ECM can even serve as a cointegration test, with its own critical values [@BanerjeeDoladoMestre1998].

## Recap: Error correction models {h=d60dfde7}

An ECM regresses $\Delta y_t$ on $\Delta x_t$, on lagged differences and on the equilibrium error $\hat u_{t-1}$; all terms are stationary, so the usual inference is valid. The coefficient $\gamma < 0$ is the speed of adjustment, and the half-life @eq:halflife is expressed in the units of the data. A single equation contains the short-run effect ($\delta_0$) and the long-run relation ($b$); when several variables adjust, a system is needed.

## @end

The error correction model turns cointegration into a forecasting equation: today's change responds to yesterday's deviation from equilibrium. For US consumption, half of an increase in income passes into consumption in the same quarter, and a deviation from equilibrium halves in about 14 quarters; for the 10-year yield, in about 21 months, and the 3-month rate adjusts too. This mutual adjustment requires a model of the system, the subject of the next section.

# The VECM and the Granger representation theorem

The section moves from one equation to a system. It starts from the VAR model of [Chapter 6](../EN/Courses/chapter6_var_models_granger_causality.pdf) and rewrites it as a vector error correction model (VECM). The rank of the long-run matrix $\Pi$ says whether and how many cointegrating relations exist, and the Granger representation theorem links cointegration to error correction. Then come a two-variable example, the five cases of deterministic terms and the notion of weak exogeneity.

## From a VAR to a VECM {h=e5d3795b}

A VAR($p$) in levels, as in [Chapter 6](../EN/Courses/chapter6_var_models_granger_causality.pdf), is $\mathbf y_t = \mathbf c + A_1\mathbf y_{t-1} + \dots + A_p\mathbf y_{t-p} + \mathbf u_t$. Subtracting $\mathbf y_{t-1}$ from both sides and regrouping the lags, the same model is written in vector error correction form:

$$ \begin{aligned} \Delta\mathbf y_t &= \mathbf c + \Pi\,\mathbf y_{t-1} + \sum_{i=1}^{p-1}\Gamma_i\,\Delta\mathbf y_{t-i} \\ &\quad + \mathbf u_t, \\ \Pi &= A_1 + \dots + A_p - I, \\ \Gamma_i &= -(A_{i+1} + \dots + A_p). \end{aligned} $$ {#eq:vecm}

The matrix $\Pi$ is the long-run matrix: it multiplies the lagged levels. The matrices $\Gamma_i$ are the short-run matrices, which multiply the lagged differences, and $I$ is the identity matrix.

The derivation for $p = 2$ shows how the regrouping works. Start from $\mathbf y_t - \mathbf y_{t-1} = (A_1 - I)\mathbf y_{t-1} + A_2\mathbf y_{t-2} + \mathbf u_t$, add and subtract $A_2\mathbf y_{t-1}$, and obtain $\Delta\mathbf y_t = (A_1 + A_2 - I)\mathbf y_{t-1} - A_2\Delta\mathbf y_{t-1} + \mathbf u_t$, so $\Pi = A_1 + A_2 - I$ and $\Gamma_1 = -A_2$.

The VECM form is a rewriting, not a new model: it has the same parameters and the same likelihood as the VAR in levels. Its advantage is that it separates the short-run dynamics, in $\Gamma_i$, from the long-run structure, which is concentrated entirely in the matrix $\Pi$.

## The rank of {h=f8ccae52}

In @eq:vecm the left-hand side $\Delta\mathbf y_t$ is stationary, because the series are $I(1)$. The right-hand side must then be stationary too, hence also $\Pi\mathbf y_{t-1}$, although $\mathbf y_{t-1}$ is not. This condition restricts the matrix $\Pi$, and the restriction is expressed through its rank $r = \mathrm{rank}(\Pi)$, with three cases.

If $r = 0$, then $\Pi = 0$, there is no cointegration, and the right model is a VAR in differences. If $r = n$, the matrix $\Pi$ has full rank and is invertible; then $\mathbf y_{t-1} = \Pi^{-1}(\Pi\mathbf y_{t-1})$ is itself stationary, and the right model is a VAR in levels, as in [Chapter 6](../EN/Courses/chapter6_var_models_granger_causality.pdf). The interesting case is $0 < r < n$: the series are cointegrated with $r$ vectors, and the matrix factors as $\Pi = \alpha\beta^\top$, with $\alpha$ and $\beta$ of dimension $n \times r$.

In this factorisation, $\beta^\top\mathbf y_{t-1}$ are the $r$ equilibrium errors, and $\alpha$ contains the adjustment coefficients (loadings): the element $\alpha_{ij}$ says how strongly variable $i$ reacts to equilibrium error $j$. Only the product is unique: for any invertible $r \times r$ matrix $Q$, $(\alpha Q)(\beta Q^{-\top})^\top = \alpha QQ^{-1}\beta^\top = \alpha\beta^\top$. That is why $\beta$ is normalised, as in the single-equation case. Testing for cointegration thus becomes testing the rank of $\Pi$, the subject of [Section 5](#sec-5).

## The Granger representation theorem {h=ae1d9f28}

The Granger representation theorem, proved in @EngleGranger1987, states that if $\mathbf y_t$ is $I(1)$ and cointegrated with rank $r$, then it has a VECM representation with $\Pi = \alpha\beta^\top$, and conversely. Cointegration and error correction are therefore equivalent: if an equilibrium exists, the deviation from it must influence the changes of at least one variable; otherwise nothing would pull them back towards equilibrium.

The theorem has three practical consequences. First, a VAR in differences omits the term $\alpha\beta^\top\mathbf y_{t-1}$, so it is misspecified when the series are cointegrated. Second, if $\alpha \neq 0$, at least one variable adjusts to the equilibrium error, which depends on the other variables, so there is Granger causality, in the sense of [Chapter 6](../EN/Courses/chapter6_var_models_granger_causality.pdf), in at least one direction. Third, the levels are driven by $n - r$ common stochastic trends, and the effects of shocks on the levels do not die out.

A VAR in levels is not misspecified either: its estimators are consistent, since the VECM is only a rewriting of it. It ignores, however, the restriction $\mathrm{rank}(\Pi) = r$, so it estimates more parameters than necessary, and some of its tests have non-standard distributions.

## Clive Granger and the 2003 Nobel Prize {h=2a166cec}

The 2003 Nobel Prize in economics was shared between Clive Granger, for cointegration, and Robert Engle, for the ARCH models of [Chapter 5](../EN/Courses/chapter5_conditional_volatility_garch.pdf). Both contributions concern economic time series, and both changed empirical practice: one for series with common trends, the other for series with time-varying volatility.

Granger studied and began his career at the University of Nottingham, where a building now bears his name. His Nobel lecture, @Granger2004, titled "Time series analysis, cointegration, and applications", tells how the idea of cointegration arose and how it was received.

## Worked example: a bivariate VECM {h=5f441cfa}

The example uses the system $\Delta y_{1t} = -0.10\,(y_{1,t-1} - y_{2,t-1}) + u_{1t}$ and $\Delta y_{2t} = 0.10\,(y_{1,t-1} - y_{2,t-1}) + u_{2t}$. The equilibrium error is the distance $y_1 - y_2$, so $\beta = (1, -1)^\top$, and the adjustment coefficients are $\alpha = (-0.10;\ 0.10)^\top$. The long-run matrix is

$$ \Pi = \alpha\beta^\top = \begin{pmatrix} -0.10 & 0.10 \\ 0.10 & -0.10 \end{pmatrix}, $$

with $\det\Pi = 0.01 - 0.01 = 0$; its columns are proportional, so the rank is 1, as in the case $0 < r < n$.

The signs of $\alpha$ say who moves. If $y_1$ is above $y_2$, then $y_1$ falls, because $\alpha_1 < 0$, and $y_2$ rises, because $\alpha_2 > 0$: both variables close the distance, like the drunk and her dog. Subtracting the two equations, the distance $z_t = y_{1t} - y_{2t}$ follows $z_t = (1 + \beta^\top\alpha)\,z_{t-1} + (u_{1t} - u_{2t})$, with $\beta^\top\alpha = -0.10 - 0.10 = -0.2$, so $z_t = 0.8\,z_{t-1} + (u_{1t} - u_{2t})$. This is a stationary AR(1), with half-life $\ln 0.5/\ln 0.8 = 3.1$ periods; the stationarity condition is $-2 < \beta^\top\alpha < 0$.

Adding the two equations, the correction terms cancel, so the average $(y_{1t} + y_{2t})/2$ changes only by $(u_{1t} + u_{2t})/2$: it is a random walk. This is the common trend, and $n - r = 2 - 1 = 1$ confirms that there is only one.

## Deterministic terms: five cases {h=8370fe87}

A VECM can contain a constant and a linear trend, and their place matters: inside the cointegrating relations, $\beta^\top\mathbf y_{t-1}$, or outside them, unrestricted. The table gives the five standard cases, the implication of each for the data and the corresponding code of the *VECM* class in *statsmodels*.

Case 1, with no constant and no trend, means series without drift and relations through the origin; it is rarely realistic. Case 2 puts a constant only inside the cointegrating relations: the series have no drift, but the equilibrium errors have a non-zero mean, the right situation for yields and spreads. Case 3, with an unrestricted constant, allows linear trends in the levels but not in the relations, the typical situation for GDP and prices. Case 4 adds a linear trend inside the cointegrating relations, hence trends both in the levels and in the relations. Case 5, with an unrestricted constant and trend, implies quadratic trends in the levels and is rarely plausible.

The critical values of the Johansen tests change from one case to another [@Johansen1995], so the case must be chosen before testing, from the plot of the series and from theory. The *coint_johansen* function of *statsmodels* offers case 1 (*det_order=-1*), case 3 (*det_order=0*), the case used for the tests in this chapter, and case 5 (*det_order=1*, a constant and a linear trend).

## Weak exogeneity {h=c8a2db4f}

If a row of $\alpha$ is zero, $\alpha_i = 0$, variable $y_i$ does not react to any equilibrium error. We say that $y_i$ is weakly exogenous for $\beta$: the other variables do all the adjusting. Such a variable drives the common trend, and the others follow it. An example is Euribor for Romanian interest rates, analysed in Seminar 7 (B4): it is plausible that Romanian rates adjust to euro-area rates, not the other way round.

Weak exogeneity can be tested. With a single cointegrating vector, a $t$-test on $\alpha_i$ suffices; in general a likelihood ratio (LR) test of the restriction $\alpha_i = 0$ is used [@Johansen1991]. Once the rank $r$ has been fixed, the statistic is asymptotically $\chi^2$ with $r$ degrees of freedom, one for each element of the row $\alpha_i$.

The notion also has a practical use. If $x_t$ is weakly exogenous, a single-equation ECM for $y_t$, as in [Section 3](#sec-3), loses no information about $\beta$: all the error correction takes place in the equation of $y_t$, and the equation of $x_t$ adds nothing about the equilibrium.

## Recap: The VECM {h=da6476bd}

A VECM, $\Delta\mathbf y_t = \alpha\beta^\top\mathbf y_{t-1} + \sum\Gamma_i\Delta\mathbf y_{t-i} + \mathbf u_t$, is a rewritten VAR, and the rank $r$ of the matrix $\Pi$ gives the number of cointegrating relations. The Granger representation theorem says that cointegration and error correction are equivalent, so a VAR in differences omits the long-run term. The matrix $\beta$ contains the equilibria, $\alpha$ shows who adjusts, the deterministic terms have five cases, and a zero row of $\alpha$ means weak exogeneity.

## @end

The VECM form separates the short-run dynamics of a VAR from its long-run structure: all the information about the equilibria lies in $\Pi = \alpha\beta^\top$, and its rank counts the cointegrating relations. The Granger representation theorem explains why a VAR in differences is misspecified for cointegrated series and why shocks have permanent effects on the levels. What remains is to estimate the rank from the data; this is the role of the Johansen tests.

# The Johansen tests

@Johansen1988 and @Johansen1991 gave the method for estimating a VECM by maximum likelihood and, with it, tests for the cointegrating rank. Unlike the Engle–Granger method, the approach works directly with the system, can find several cointegrating vectors and does not depend on the choice of the left-hand variable. The section explains the idea of reduced-rank regression, defines the two test statistics, computes them on an example and applies them to three real systems.

## Reduced-rank regression {h=1315be61}

The method of @Johansen1988 and @Johansen1991 estimates a VECM by maximum likelihood, assuming Gaussian errors. The idea is a reduced-rank regression. First the short-run dynamics are removed: $\Delta\mathbf y_t$ and $\mathbf y_{t-1}$ are each regressed on the lagged differences, and the residuals are denoted $R_{0t}$ and $R_{1t}$. What remains is the relation between current changes and lagged levels, cleaned of short-run effects.

Then one looks for the linear combinations of $R_{1t}$ that are most strongly correlated with $R_{0t}$; these are the canonical correlations. Their squares are the estimated eigenvalues $1 > \hat\lambda_1 \ge \dots \ge \hat\lambda_n \ge 0$. A large eigenvalue corresponds to a combination of the levels that predicts the changes well, that is, to a stationary combination acting as an equilibrium error. An eigenvalue close to 0 corresponds to a combination with no predictive power that remains $I(1)$.

The eigenvectors associated with the $r$ largest eigenvalues give the estimator $\hat\beta$; then $\hat\alpha$ follows by OLS, regressing $R_{0t}$ on $\hat\beta^\top R_{1t}$. The method was developed at the University of Copenhagen by Søren Johansen and Katarina Juselius.

## Trace and maximum-eigenvalue statistics {h=3cc81013}

The two Johansen tests are likelihood ratio tests built from the eigenvalues, where $T$ is the number of observations:

$$ \begin{gathered} \lambda_{\mathrm{trace}}(r) = -T\sum_{i=r+1}^{n}\ln(1 - \hat\lambda_i), \\ \lambda_{\max}(r) = -T\ln(1 - \hat\lambda_{r+1}). \end{gathered} $$ {#eq:johansen}

The trace test has the null hypothesis "the rank is at most $r$" against the alternative "the rank is $n$". It sums the information in the smallest $n - r$ eigenvalues: if all of them are close to 0, each $-\ln(1 - \hat\lambda_i)$ is close to 0, the statistic is small and the null is not rejected. The maximum-eigenvalue test has the null hypothesis "the rank is $r$" against the alternative "the rank is $r + 1$" and uses only the next eigenvalue, $\hat\lambda_{r+1}$.

The procedure is sequential. First $r = 0$ is tested; if it is rejected, $r \le 1$ is tested, and so on. The estimated rank is the first $r$ whose null hypothesis is not rejected. The null distributions of the statistics are non-standard, functionals of Brownian motions, and the critical values depend on $n - r$ and on the case of deterministic terms; they are tabulated by @OsterwaldLenum1992 and @MacKinnonHaugMichelis1999. The two tests usually agree; when they do not, most applied work prefers the trace test.

## Worked example: from eigenvalues to statistics {h=1757d821}

The example uses the US Treasury yields at 1, 5 and 10 years, monthly: 801 months, of which the first 3 are used up by the two lagged differences, so the effective sample is $T = 801 - 2 - 1 = 798$; the estimated eigenvalues are $\hat\lambda = (0.0405;\ 0.0229;\ 0.0039)$. The first step is the transformation $-\ln(1 - \hat\lambda_i)$, which gives 0.0414, 0.0231 and 0.0039; for small eigenvalues the transformation barely changes the number.

For $r = 0$, the trace statistic is $798 \times (0.0414 + 0.0231 + 0.0039) = 54.6$, above the 5% critical value of 29.80, so the hypothesis "rank 0" is rejected. The maximum-eigenvalue statistic is $798 \times 0.0414 = 33.0$, above 21.13, so this test rejects too. For $r \le 1$, the trace statistic is $798 \times (0.0231 + 0.0039) = 21.6$, above 15.49, so it is rejected. For $r \le 2$, the trace statistic is $798 \times 0.0039 = 3.1$, below 3.84, so the hypothesis is not rejected. The estimated rank is 2: two cointegrating vectors and a single common trend.

The example shows why even small eigenvalues matter: they are multiplied by $T$. A squared canonical correlation of 0.0229 looks negligible, but over 798 observations it gives a statistic of about 18.5, enough to reject. The values in the next table agree with these; its column $T$ gives the full 801 months, before the observations lost to the lags.

## Johansen tests on three systems {h=9482ceda}

The table applies the tests to three systems of three variables each, all in case 3, with an unrestricted constant. For each system the columns give the number of observations $T$, the number of lagged differences $k$, chosen by the Bayesian information criterion (BIC) for the VAR in levels, the trace statistics for $r = 0, 1, 2$, the maximum-eigenvalue statistics for $r = 0, 1$ and the chosen rank. The last row gives the 5% critical values.

The first system contains the US yields at 1, 5 and 10 years, from FRED, between 1960 and 2026. The second contains US GDP, consumption and investment, as 100 × log, between 1959 and 2009. The third contains the EUR/RON, EUR/HUF and EUR/PLN exchange rates, from the BNR reference rates at the end of the month, between 2005 and 2026. The ranks are chosen by the trace test; the maximum-eigenvalue test gives the same ranks here.

For the yields, the trace statistics are 54.6, 21.6 and 3.1, hence rank 2. For the macroeconomic aggregates, the first trace statistic, 28.9, is below 29.80, so the rank is 0. For the currencies, the first trace statistic, 24.2, is below 29.80, so the rank is also 0.

## Interpreting the three systems {h=b18f9144}

For the yields, rank 2 means a single common trend, $n - r = 3 - 2 = 1$. This trend is the level of interest rates, which moves all maturities together, and the two cointegrating relations are, approximately, the two spreads, which are stationary. The result agrees with the expectations hypothesis of the term structure [@CampbellShiller1987] in its weak form, which requires only that the spreads be stationary.

For GDP, consumption and investment, the trace statistic for $r = 0$ is 28.9 against the critical value of 29.80: the rank is 0 at 5%, but the result is very close to rejection. Balanced-growth theory predicts rank 2, because the shares of consumption and investment in GDP should be stable. Moreover, the Engle–Granger test found consumption and income cointegrated, with p-value 0.029. The non-rejection in the Johansen test reflects low power over 203 quarters, not proof that the relations are absent.

For the three currencies, the trace statistic of 24.2 is clearly below the critical value. The test finds no cointegrating relation, and the data are compatible with three separate stochastic trends; [Section 8](#sec-8) returns to this case.

## Sensitivity of the Johansen tests to the deterministic terms {h=30bc29a0}

The table repeats the trace test for the US yields at 1, 5 and 10 years with three specifications of the deterministic terms. Without deterministic terms (*det_order=-1*, case 1), the statistics are 45.9, 13.3 and 0.4, with 5% critical values of 24.28, 12.32 and 4.13: rank 2. With a constant (*det_order=0*, case 3), the statistics are 54.6, 21.6 and 3.1, against 29.80, 15.49 and 3.84: rank 2 again. With a constant and a linear trend (*det_order=1*, case 5), the statistics are 60.0, 23.7 and 4.8, against 35.01, 18.40 and 3.84: rank 3.

Rank 3 in a system of three variables means that the matrix $\Pi$ has full rank, so each yield would be stationary around a deterministic trend. For interest rates over 66 years, which rose until 1981 and then fell for four decades, this conclusion is not plausible. The lesson is that the deterministic terms are chosen from theory and from the plot, before testing, not by the result that suits us.

There are other practical issues too. The number of lags $k$ affects the statistics. In small samples the tests reject too often; @ReinselAhn1992 propose multiplying the statistic by the correction factor $(T - nk)/T$. Structural breaks, such as changes of monetary regime, can hide or mimic cointegration.

## Case study: Johansen and Juselius (1990) {h=c5d31277}

The question of the paper was whether there is a stable money demand in Denmark and Finland. @JohansenJuselius1990 analysed quarterly data on real money, real income, interest rates and, for one of the systems, inflation. A stable money demand means exactly a cointegrating relation among these variables.

The method had two stages. The first estimated a VECM by maximum likelihood and chose the rank with the trace and maximum-eigenvalue tests. The second tested economic restrictions on $\beta$ and $\alpha$ as likelihood ratio tests: for example a unit income elasticity of money demand, $\beta = (1, -1, \cdot)$, or the weak exogeneity of income and interest rates.

The legacy of the paper is the "cointegrated VAR" methodology, set out at length by @Juselius2006: first test the rank, then test the economic hypotheses, formulated as restrictions on $\beta$ and $\alpha$. The paper is one of the most cited in applied econometrics.

## Recap: The Johansen tests {h=1d24c120}

The Johansen tests use the eigenvalues of a reduced-rank regression; the trace and maximum-eigenvalue statistics @eq:johansen are applied sequentially, starting from $r = 0$. The critical values depend on $n - r$ and on the case of deterministic terms. For the US yields the rank is 2, for the macroeconomic aggregates the result is borderline, and for the three currencies the rank is 0.

## @end

The Johansen tests estimate the cointegrating rank directly in the system and do not depend on normalisation. For the US yields at 1, 5 and 10 years they find two cointegrating relations and a single common trend; for GDP, consumption and investment the trace statistic stays just below the critical value, a case of low power; for three Central European currencies they find no relation. The same yield system gets rank 3 if an implausible linear trend is added, so the choice of deterministic terms is part of the model specification.

# Estimating and interpreting β and α

Once the rank is chosen, the estimated VECM yields two matrices with an economic interpretation: $\hat\beta$, the equilibrium relations, and $\hat\alpha$, the speed with which each variable reacts to deviations from them. The section estimates them for the term structure of US yields, interprets them and closes with the impulse response functions, which show the permanent effects of shocks in a cointegrated system.

## The term-structure VECM {h=2812b0d2}

The model uses the US Treasury yields at 1, 5 and 10 years, the FRED series GS1, GS5 and GS10, monthly, from January 1960 to September 2026. The specification has rank 2, chosen by the Johansen tests, 2 lagged differences and a constant restricted to the cointegrating relations, that is, case 2. Case 2 suits interest rates: they have no long-run drift, but the spreads have non-zero means, the term premia.

The left panel shows the three yields: a rise to a peak around 1981, above 15%, a long decline to values close to zero after 2008 and in 2020, then a rise after 2022. The three lines move together, with the 1-year yield usually the lowest. The right panel shows the two estimated equilibrium errors, $\hat\beta^\top\mathbf y_t$: they oscillate around zero without a trend, with larger deviations for the combination that includes the 1-year yield.

## Interpreting the cointegrating vectors {h=064a7591}

With the normalisation on the 1-year and 5-year yields, the two estimated relations are $\hat\beta_1^\top\mathbf y_t = i^{(1)}_t - 1.045\,i^{(10)}_t + 1.22$ and $\hat\beta_2^\top\mathbf y_t = i^{(5)}_t - 1.044\,i^{(10)}_t + 0.59$, where $i^{(m)}_t$ is the $m$-year yield in percent per year. Both slopes are close to 1, so the equilibria are almost the spreads $i^{(1)} - i^{(10)}$ and $i^{(5)} - i^{(10)}$. The constants, 1.22 and 0.59 percentage points, reflect the average term premia: long-term yields are on average higher than short-term ones.

The equilibrium errors are stationary: the ADF test gives a p-value below 0.001 for the first and 0.008 for the second. They are large in the 1980s, when monetary policy pushed short-term rates up sharply, and in periods of near-zero rates, when the short end of the curve stayed pinned at zero while long-term yields kept moving.

A formal test of the restriction $\beta = (1, -1)$ for each spread is a likelihood ratio test [@Johansen1991]. Even without it, values so close to 1 are informative: because $\hat\beta$ is super-consistent, its estimation error is small over a sample of more than 800 months.

## The adjustment coefficients {h=65b79755}

The table gives the estimated adjustment coefficients, with their $t$-statistics, for the three equations of the VECM. The column $\hat\alpha_{\cdot 1}$ shows the reaction of each yield to the first equilibrium error (the 1-year – 10-year spread), and the column $\hat\alpha_{\cdot 2}$ the reaction to the second (the 5-year – 10-year spread).

The 1-year yield does not react significantly to either error: the coefficients are $-0.021$ and $0.024$, with $|t| < 1$. It behaves as a weakly exogenous variable, which makes economic sense, because the short end of the curve follows monetary policy. The adjusting is done by the 5-year and 10-year yields. The 5-year yield has coefficients 0.045 ($t = 2.21$) and $-0.122$ ($t = -2.00$), and the 10-year yield has 0.040 ($t = 2.22$) and $-0.077$ ($t = -1.43$); the reaction of the 10-year yield to the second error is thus less precisely determined than the others.

The overall speed of correction is read from the dynamics of the equilibrium errors, $\mathbf z_t = \hat\beta^\top\mathbf y_t$, which follow $\mathbf z_t = (I + \hat\beta^\top\hat\alpha)\,\mathbf z_{t-1} + \dots$. The eigenvalues of the matrix $I + \hat\beta^\top\hat\alpha$ are 0.969 and 0.926, which give half-lives of $\ln 0.5/\ln 0.969 \approx 22.1$ and $\ln 0.5/\ln 0.926 \approx 9.0$ months. Error correction is slow, which is typical of interest rates.

## Impulse responses of the VECM {h=1fad15c5}

The chart shows the impulse response functions of the three yields to a one-standard-deviation shock to the 1-year yield, over a horizon of 60 months, in percentage points. The shocks are orthogonalised by the Cholesky decomposition, with the ordering 1 year, 5 years, 10 years, as in [Chapter 6](../EN/Courses/chapter6_var_models_granger_causality.pdf): a shock to the 1-year yield can affect the other yields in the same month, but not the other way round.

The three curves rise in the first month, the 1-year yield the most, and then level off. After the first year, the 1-year yield declines slowly while the 5-year and 10-year yields rise slowly, so the curves draw closer to each other. None of them returns towards zero within the 60 months.

## Interpreting the impulse responses {h=7a2c4feb}

On impact, the yields rise by 0.36, 0.24 and 0.19 percentage points for the 1-, 5- and 10-year maturities. After 60 months, the responses are 0.37, 0.36 and 0.33 percentage points: the effects do not die out and become almost equal. The shock has permanently shifted all yields by about the same amount.

This is the signature of a single common trend, $n - r = 1$. The shock permanently moves the common trend, the level of interest rates, while the spreads, being stationary, return to their means; that is why the long-run responses of the three yields become equal. The initial differences between the responses are temporary deviations from equilibrium, corrected by the terms $\alpha\beta^\top\mathbf y_{t-1}$.

The contrast with a stationary VAR, from [Chapter 6](../EN/Courses/chapter6_var_models_granger_causality.pdf), matters. In a stationary VAR every impulse response returns to zero, because shocks die out. In a VECM the long-run responses of the levels are in general different from zero, while the responses of the equilibrium errors return to zero.

## Recap: β and α {h=fc549b38}

The vectors $\hat\beta$ are normalised and interpreted as equilibria, for example spreads or ratios, and economic restrictions on them are tested. The coefficients $\hat\alpha$ show who adjusts and how fast; a zero row means weak exogeneity. Shocks have permanent effects on the cointegrated levels, but not on the equilibrium errors.

## @end

For the term structure of US yields, the rank-2 VECM gives two equilibrium relations close to the spreads against the 10-year yield, with positive term premia. The 1-year yield does not adjust significantly and drives the system; the 5-year and 10-year yields do the correcting, with half-lives of about 9 and 22 months for the two equilibrium errors. A shock to the 1-year yield shifts the whole curve permanently, while the spreads revert.

# Forecasting with a VECM and with a VAR in differences

The representation theorem says that a VAR in differences is misspecified for cointegrated series. The practical question is whether the correctly specified VECM also forecasts better. The section presents the theoretical arguments, then an out-of-sample forecasting experiment on US yields, for the levels and for a spread.

## Theoretical arguments for the VECM {h=0e487433}

VECM forecasts preserve the equilibria. As the horizon $h$ grows, the forecast of the equilibrium error, $\hat\beta^\top\hat{\mathbf y}_{T+h}$, returns to its mean, because the correction term pushes the forecasts towards equilibrium. A VAR in differences does not use the information in the levels: its forecasts of the spreads stay, over long horizons, close to the current value, however far from the mean it is.

@EngleYoo1987 showed, in simulated systems, that imposing cointegration improves long-horizon forecasts. @ChristoffersenDiebold1998 specified where these gains arise: in the cointegrating combinations. For the individual levels, the long-horizon forecast error is dominated by the common trend, which no model can predict, so a VECM and a VAR in differences have similar errors.

There is also an argument the other way. A VECM has more parameters to estimate: $\alpha$, $\beta$ and the constants. Their estimation error can make a VECM forecast worse in practice, especially at short horizons. The answer can only come from an out-of-sample check, with the models re-estimated at every forecast origin.

## The forecasting experiment {h=6b381c43}

The data are the US Treasury yields at 1, 5 and 10 years, monthly, since 1960. The experiment uses 135 forecast origins, one every three months, from January 1990 to July 2023. At each origin the models are re-estimated using only the data available up to then, and forecasts are made for horizons from 1 to 36 months.

Three models are compared. The first is a VECM with rank 2 and a restricted constant, the specification of [Section 6](#sec-6). The second is a VAR in differences without a constant, with the same number of lags. The third is the random walk, which forecasts that each yield stays at its last observed value (the "no-change" forecast).

The accuracy measure is the root mean squared error (RMSE) of each model divided by the RMSE of the random walk. A ratio below 1 means a model more accurate than the random walk, a ratio above 1 a less accurate one. The comparison is made for each of the three yields and for the spread between the 10-year and the 1-year yield.

## VECM against a VAR in differences, out of sample {h=01d38efd}

The chart has four panels: the first three for the yields at 1, 5 and 10 years, the last for the spread 10 years minus 1 year. The horizontal axis is the forecast horizon in months, and the vertical axis the ratio of the model's RMSE to the random walk's RMSE. The dashed line at 1 is the random walk; the red line is the VECM and the blue line the VAR in differences.

In the first three panels the blue line stays close to 1 at all horizons. The red line usually lies above it: it rises to about 1.19 for the 1-year yield in the first months and drops below 1 only after about two years; for the 5-year and 10-year yields it stays above 1 at almost all horizons. In the spread panel the picture reverses after the first year: the red line falls steadily, to about 0.74 at 36 months, while the blue line stays at 1.

## Interpreting the forecast comparison {h=e4118fd8}

For the levels of the yields, no model is clearly more accurate than the random walk. The VECM is even worse at short horizons: at 6 months the RMSE ratio is 1.14 for the 1-year yield and 1.06 for the 5-year yield. The VAR in differences stays close to 1 at every horizon; for the 1-year yield at 12 months its ratio is 1.00.

For the spread, the VECM is more accurate at long horizons: the ratio is 0.85 at 24 months and 0.74 at 36 months, that is, an error 15% and 26% smaller than that of the random walk, while the VAR in differences stays at 1.00. This is exactly the pattern described by @ChristoffersenDiebold1998: cointegration helps to forecast the equilibrium errors, not the common trend.

The practical lesson is clear. A VECM is used when the interest lies in the relation between series: a spread, a ratio, a pair of prices. For the levels taken alone, the random walk is hard to beat, and a richer model can lose through estimation error.

## Recap: Forecasting {h=b618ec03}

A VECM brings the forecasts of the equilibrium errors back to their means; a VAR in differences does not. Out of sample, on US yields, the VECM gains for the spread at long horizons and does not gain for the levels. The comparison is always made against the random walk, with re-estimation at every origin and without using future data.

## @end

Cointegration makes the relations between series predictable, not the series themselves. Over 135 origins between 1990 and 2023, the VECM of US yields reduces the forecast error of the 10-year – 1-year spread by 26% at 36 months relative to the random walk, but does not forecast the levels of the yields better, and at short horizons it is even worse. This result confirms the theoretical argument of @ChristoffersenDiebold1998.

# Economic examples: parity and interest rates

The section applies the tools of the chapter to three questions about the Romanian economy and the region: whether the leu exchange rate obeys purchasing power parity, whether ROBOR and Euribor are tied in the long run, and whether three Central European currencies share a trend. All three are cases in which theory suggests a cointegrating relation, and the data give a weak or negative answer.

## Purchasing power parity {h=36133ce0}

Relative purchasing power parity states that the change in the exchange rate offsets the inflation difference between two countries. With $s_t = 100\ln(\mathrm{EUR/RON})$ and with $p_t$ and $p_t^*$ equal to $100\ln$ of the harmonised index of consumer prices (HICP) of Romania and of the euro area, the real exchange rate is $q_t = s_t - p_t + p_t^*$. A rise in $q_t$ means a real depreciation of the leu, a fall a real appreciation. PPP holds in the long run if $q_t$ is stationary, that is, if $(s_t, p_t, p_t^*)$ are cointegrated with the vector $\beta = (1, -1, 1)$.

There are two reasons why PPP may fail. The first is the Balassa–Samuelson effect [@Balassa1964]: in a converging economy, productivity grows quickly in the tradable-goods sector, wages rise across the economy, and the prices of non-tradable services go up. The result is a trend real appreciation, hence a non-stationary $q_t$.

The second reason is speed. Even where PPP holds, deviations from it die out slowly; the survey by @TaylorTaylor2004 reports half-lives of several years. In a short sample, a relation with such slow correction is hard to tell apart from no relation.

## PPP for the leu {h=4f39ca38}

The data are monthly, from July 2005 to August 2026: the EUR/RON exchange rate as the monthly average of the BNR reference rate, and the HICP of Romania and of the euro area, from Eurostat, with base 2015 = 100. All series are expressed as 100 × log, set to 0 in July 2005.

The left panel compares the exchange rate $s_t$ (green) with relative prices $p_t - p_t^*$ (red). Until around 2020 the two lines move close together, with temporary deviations, the largest in 2007, when the leu appreciated strongly, and the rebound during the 2008–2009 crisis. After 2021, relative prices rise much faster than the exchange rate. The right panel shows the real exchange rate $q_t$: it oscillates between about $-18$ and $+3$ until 2021, then falls steadily, to about $-14$ at the end of the sample.

## Interpreting PPP for the leu {h=18659118}

Since 2005 the leu has lost 38.7 log points against the euro; a log point is 100 × the difference of the logs, approximately a 1% change for small changes. Over the same period, Romanian prices rose 52.4 log points more than euro-area prices. The difference, $38.7 - 52.4 = -13.7$ log points, is the change in the real exchange rate: a real appreciation of the leu, concentrated after 2021.

All three tests give the same answer. The ADF test on the real exchange rate $q_t$ has a p-value of 0.24, so the unit root is not rejected. The Engle–Granger test for $s_t$ on $p_t - p_t^*$ gives a slope of 0.89 and a p-value of 0.34. The Johansen test on $(s_t, p_t, p_t^*)$ chooses rank 0.

The data offer no evidence of PPP over the 21 years analysed. The result agrees with the Balassa–Samuelson convergence of an economy approaching the euro-area price level. It does not prove, however, that PPP is absent: 21 years may be too few to detect a relation whose deviations have half-lives of several years.

## ROBOR and Euribor {h=025df613}

The data are the three-month money-market interest rates, from Eurostat, monthly from August 1995 to August 2026: ROBOR 3M (Romanian Interbank Offer Rate) for Romania and Euribor 3M for the euro area. The left panel uses a log scale, because ROBOR exceeded 100% per year in the late 1990s; Euribor values below 0.05%, including the negative ones between 2015 and 2022, are shown at 0.05.

The right panel shows the period since 2010, in percent per year, together with the ROBOR minus Euribor spread (dashed green line). ROBOR falls from about 8% in 2010 to below 1% in 2016, climbs to almost 8% in 2022 and then stays around 6%. Euribor sits near or below zero between 2015 and 2022, then rises to about 4% in 2023–2024. The spread mostly follows ROBOR, which already suggests a weak link.

## Interpreting ROBOR and Euribor {h=42360b99}

Over the full sample the tests disagree: Engle–Granger has a p-value of 0.52, Phillips–Ouliaris 0.010. The disinflation of the 1990s, when ROBOR fell from over 100% to a few percent, dominates the sample and makes the result depend on how each test handles the autocorrelation.

Since 2010, both tests reject no cointegration at 5%: p-value 0.047 for Engle–Granger and 0.029 for Phillips–Ouliaris. The slope of the relation is 1.19, and the average spread is 3.11 percentage points. An ECM for ROBOR, however, gives a speed of adjustment $\hat\gamma = -0.020$, with $t = -1.18$, hence not significant, and a half-life of about 34 months: weak and slow adjustment.

The result has an economic explanation. The National Bank of Romania pursues its own inflation target, and ROBOR reflects primarily its policy, not that of the European Central Bank. The link with Euribor exists but is weak; Seminar 7 (B4) extends the analysis with Romania's 10-year yield.

## Three Central European currencies {h=0ab2e648}

The chart shows the EUR/RON, EUR/HUF and EUR/PLN exchange rates, computed as cross rates from the BNR reference rates, at the end of the month, as 100 × log, set to 0 in July 2005. A rise means a depreciation of the national currency against the euro.

The leu (green) depreciates in jumps until 2012 and very slowly after that. The forint (orange) follows a path close to the leu's until 2019, then depreciates further, with a peak in 2022, and comes back. The zloty (purple) appreciates until 2008, depreciates sharply in the crisis and then oscillates without a clear tendency, ending close to its 2005 level. The three currencies share the same anchor, the euro, and reacted to the same crises, but their long-run paths differ.

## Interpreting the three currencies {h=21b7d1bd}

Since July 2005, EUR/RON has risen by 40.1 log points, EUR/HUF by 39.5 and EUR/PLN by only 6.7. Each exchange rate is treated as $I(1)$: the ADF tests give p-values of 0.55, 0.75 and 0.08; for the zloty the unit root is not rejected at 5% but would be rejected at 10%, so the $I(1)$ conclusion is more fragile.

The Johansen trace statistic for $r = 0$ is 24.2, below the critical value of 29.80, so the chosen rank is 0. Even in the sample since 2012 the test finds no evidence of cointegration: the trace statistic is 12.7. A common anchor and similar shocks do not create a common stochastic trend: each central bank and each inflation path adds its own trend.

Without a detected cointegrating relation, the appropriate model for these exchange rates is a VAR in differences, as in [Chapter 6](../EN/Courses/chapter6_var_models_granger_causality.pdf). Seminar 7 (B3) returns to the analysis.

## Recap: Economic examples {h=7345d7c3}

For the leu, between 2005 and 2026, the tests find no evidence of PPP: the real exchange rate has appreciated, as happens in a converging economy. ROBOR and Euribor have a weak and slow link since 2010, and the chosen sample decides the verdict. In every case, theory proposes the vector $\beta$, and the tests, the sample and the deterministic terms decide whether the data agree with it.

## @end

The three examples show what a negative or weak result looks like. Purchasing power parity is not supported by the leu data for 2005–2026, in line with the Balassa–Samuelson effect, although the sample may be too short for a correction lasting several years. ROBOR and Euribor are cointegrated only since 2010, with insignificant and slow adjustment. The three Central European currencies have no detectable common trend. An economic theory is a hypothesis to test, not a guaranteed result.

# An application: pairs trading

Pairs trading is the most direct financial application of cointegration: if the spread of two shares reverts to its mean, its large deviations are trading opportunities. The section describes the idea, builds a rule that uses no future information, applies it to the Banca Transilvania – BRD pair and to two groups of shares, in Bucharest and in the US, and closes with the pitfalls of the method.

## The idea of pairs trading {h=e2f033fc}

Two shares exposed to the same shocks, for example two banks or two oil companies, may be cointegrated: their spread, the difference of the log prices adjusted by a coefficient, reverts to its mean. When the spread is unusually large, the strategy sells the relatively expensive share and buys the relatively cheap one; the position is closed when the spread has reverted.

It is a relative-value trade: because the position is long one share and short the other, the direction of the market largely cancels out, and the profit comes from error correction. @GatevGoetzmannRouwenhorst2006 studied the distance method, a variant without a cointegration test, on US shares and found profits over 1962–2002. @DoFaff2010 showed that profits declined in the following years.

Pairs trading is a good test for any time-series idea, because it requires three things that an in-sample analysis can hide: the rule must work out of sample, after costs and on pairs chosen without looking ahead.

## A trading rule without look-ahead bias {h=ed31c59d}

The rule has two windows that follow each other. In the formation window, the previous 252 trading days, about one year, each pair of shares is tested with Engle–Granger, and the pairs with a p-value below 0.05 are kept. For a kept pair, with prices $P_{A,t}$ and $P_{B,t}$, the hedge ratio $\hat b$ is estimated by OLS of $\ln P_A$ on $\ln P_B$. The spread is $s_t = \ln P_{A,t} - \hat a - \hat b\ln P_{B,t}$, and its mean $\bar s$ and standard deviation $\hat\sigma_s$ are also computed in the formation window.

In the trading window, the next 126 days, about six months, the formation parameters stay fixed, and the rule tracks the standardised score $z_t = (s_t - \bar s)/\hat\sigma_s$. If $z_t > 2$, A is relatively expensive: the strategy short-sells A and buys $\hat b$ units of B. If $z_t < -2$, it does the reverse. The position is closed when $z_t$ returns to 0 or at the end of the window. The daily return per unit of gross exposure is $\pm(r_A - \hat b\,r_B)/(1 + |\hat b|)$, where $r_A$ and $r_B$ are the daily log returns of the two shares, and each selected pair receives the same capital.

Trading costs are 0.20% of the gross exposure traded in Bucharest and 0.05% in the US, at every opening and every closing of a position. The key point is that every decision uses only data available at the time; a rule that estimated $\hat b$ or $\hat\sigma_s$ on the whole sample would have an advantage impossible in practice, called look-ahead bias.

## Banca Transilvania and BRD {h=f4cf5bb8}

The left panel shows the daily adjusted share prices of Banca Transilvania (TLV) and BRD since 2014, as 100 × log, set to 0 at the start. Both rise over the long run, TLV faster, and large falls, for example at the start of 2020, occur at the same time in both.

The right panel shows the spread $\mathrm{TLV} - 1.36\,\mathrm{BRD}$, from the full-sample regression, standardised, together with the entry bands $\pm 2$. The spread starts below $-3$ in 2014, rises above $+2$ in 2016 and falls again towards $-2$ in 2026; between these episodes it oscillates inside the bands, in cycles that last months or years.

## Interpreting the bank pair {h=8cf35d54}

Over the full sample the hedge ratio is $\hat b = 1.36$, and the spread behaves like an AR(1) with $\hat\rho = 0.990$. The half-life is $\ln 0.5/\ln 0.990 \approx 69$ trading days, more than three months. Only 5.0% of the days lie outside the $\pm 2$ band, so trading opportunities are few, and each lasts for months.

Applying the rule to this chart, with parameters estimated on the full sample, gives a return of 3.32% per year after costs, with a Sharpe ratio of 0.44. The Sharpe ratio is the average annual return divided by the annual standard deviation of returns. The result is an illusion, however: $\hat b$, $\bar s$ and $\hat\sigma_s$ use future data, and the pair was chosen precisely because it looks cointegrated since 2014.

The correct rule, with rolling windows, exposes the problem. The pair does not pass the formation test in any of the 24 windows: with a half-life of about 69 days, one year of data contains too few complete cycles for the test to detect the correction. A relation visible over 12 years can be undetectable over one year.

## Pairs trading out of sample {h=6b523382}

The chart shows the cumulative growth, in log percent, of the rule without look-ahead on two groups of shares, gross of costs (blue) and net of costs (red). The left panel uses 8 liquid shares on the Bucharest Stock Exchange (TLV, BRD, SNP, TGN, TEL, EL, SNG, SNN), hence 28 pairs, with data since 2015 and a cost of 0.20%. The right panel uses 6 US banks (JPM, BAC, C, WFC, GS, MS), hence 15 pairs, with data since 2005 and a cost of 0.05%.

The flat stretches are periods without open positions: either no pair passed the formation test or no spread left the band. In Bucharest the curve oscillates between about $-20\%$ and $+10\%$ and ends close to zero after costs. In the US the curve has a fall of more than 50% in the autumn of 2008, recovers and ends around $+20\%$.

## Interpreting the backtest {h=b14268e7}

In Bucharest the backtest has 22 trading windows; pairs were found in 14 of them, and the rule made 73 trades. Of these, 62% were profitable, but the average return per trade is $-0.26\%$. The annual return is 1.17% gross of costs, with a Sharpe ratio of 0.15, and 0.41% net of costs, with a Sharpe ratio of 0.05. The cost at which the net profit becomes zero is 0.31%, very close to the assumed cost of 0.20%.

For the US banks, the annual return is 2.77% gross of costs, with a Sharpe ratio of 0.16, and 2.57% net of costs, with a Sharpe ratio of 0.14, from 80 trades. The result includes a large loss in the autumn of 2008: the cointegrating relation between banks can vanish exactly when it matters most, in a crisis that hits banks very differently.

Both results show the typical profile of a mean-reversion strategy: many small gains, when the spread reverts, and a few large losses, when it does not. That is why the share of profitable trades says little without the size of the losses.

## Limitations of pairs trading {h=0831591f}

The first pitfall is multiple testing. With 8 shares there are 28 pairs; if none of them is cointegrated, a test at the 5% level still lets through, on average, $28 \times 0.05 = 1.4$ pairs in each window, purely by chance. Some of the selected pairs are therefore false from the start.

The second pitfall is look-ahead: choosing the pairs, the start dates or the thresholds after seeing the whole sample, as in the TLV–BRD example. The third is breaks: mergers, recapitalisations, regulatory changes and crises change the vector $\beta$, and the spread may no longer return to its old mean.

The fourth pitfall is market frictions: short selling is limited on the Bucharest Stock Exchange, borrowing shares costs a fee, and bid-ask spreads are wide for small companies. The fifth is slow error correction: a half-life of several months means capital tied up for months and, sometimes, a long wait in a losing position.

## Recap: Pairs trading {h=aaa6031a}

Pairs trading bets on error correction in the spread of two cointegrated shares. In sample, with parameters estimated on all the data, the strategy looks attractive; out of sample and after costs, the profits on the Bucharest Stock Exchange are close to zero. A proper evaluation reports the selection method, the costs, the number of trades and the worst episode, not just the Sharpe ratio.

## @end

The TLV–BRD pair shows the difference between a relation visible in hindsight and an exploitable one: with full-sample parameters the rule earns 3.32% per year, but with one-year windows the pair never passes the formation test. On 8 Bucharest shares the correct rule earns 0.41% per year after costs; on 6 US banks, 2.57%, with a large loss in 2008. Cointegration is a statistical property, and turning it into profit requires selection without look-ahead, realistic costs and stable relations.

# Possible contribution of AI

An AI assistant can shorten the routine part of a cointegration analysis: writing code, explaining results and exploring many series. The section lists these uses, gives an example prompt and states the checks that remain the analyst's job, because the typical errors of this chapter are silent: the code runs and the verdicts look plausible.

## Possible contribution of AI {h=d138950d}

Three uses are realistic. The first is code: a first draft of a script that applies unit root, Engle–Granger, Phillips–Ouliaris and Johansen tests to many series and tabulates the verdicts. The second is explanation: a second account of the Granger representation theorem or of the columns of a *VECM* results table. The third is exploration at scale: candidate pairs on the Bucharest Stock Exchange or parity conditions for several Central European countries.

The example prompt states every choice that matters: the data (the monthly yields GS1, GS5 and GS10 from FRED), the test (the Johansen trace test with a constant and BIC lags), the model (a VECM with the chosen rank, in *statsmodels*) and the required output ($\beta$ normalised on the first variables, $\alpha$ with $t$-statistics and the half-lives of the equilibrium errors). A precise prompt makes the result checkable against the figures in [Section 6](#sec-6).

## Necessary checks {h=5932d8da}

The first check concerns the critical values: for residuals, the Engle–Granger values computed by MacKinnon, not the Dickey–Fuller ones, and for the Johansen tests, the values for the chosen case of deterministic terms. The second concerns the direction of the null hypothesis: "no cointegration" in the Engle–Granger test and "rank at most $r$" in the trace test. The third: each series must be $I(1)$ before testing, and a high correlation between levels is not reported as cointegration.

The adjustment coefficients must have the sign that corrects the error, and half-lives must be expressed in the time unit of the data. Backtests may not use future information when choosing pairs, must include costs and must report the number of pairs tested. Finally, every reference cited by an assistant must exist: its DOI must lead to the paper with the stated title.

## @end

An assistant can draft code, explanations and large-scale explorations, but its output is accepted only after checking the critical values, the direction of the null hypotheses, the order of integration of the series, the signs and units of the adjustment, the absence of look-ahead and the references.

# Summary

The summary gathers the main ideas of the chapter, the formulas to remember and three self-assessment questions.

## Key takeaways {h=6358d707}

Cointegrated $I(1)$ series share stochastic trends, and a combination $\beta^\top\mathbf y_t$ of them is stationary. The Engle–Granger method estimates the relation by OLS in levels and applies an ADF test to the residuals, with the MacKinnon critical values, not the Dickey–Fuller ones; the Phillips–Ouliaris test serves as a check. The error correction model explains the changes by the lagged equilibrium error, and the coefficient $\gamma$ gives the speed of adjustment and the half-life.

For systems, the VECM writes the long-run matrix as $\Pi = \alpha\beta^\top$, and the Granger representation theorem says that cointegration and error correction are equivalent; the Johansen tests choose the rank $r$. Cointegration helps to forecast the relations between series, such as spreads, not the levels, and applications, from parity to pairs trading, are evaluated out of sample and after costs.

## Key formulas {h=e8adb9a3}

The table gathers the formulas of the chapter in the order in which they appeared: the definition of cointegration; the two steps of the Engle–Granger method @eq:eg; the error correction model @eq:ecm and the half-life @eq:halflife; the VECM form of a VAR @eq:vecm, with the matrices $\Pi$ and $\Gamma_i$; the Johansen statistics @eq:johansen; the dynamics of the equilibrium errors, $\mathbf z_t = (I + \beta^\top\alpha)\mathbf z_{t-1} + \dots$, from which the half-lives of a VECM are computed.

## Self-assessment {h=2c2ece6c}

The first question applies the Engle–Granger critical values. With two variables and $T = 300$, the statistic $\tau = -3.1$ is above the 5% critical value of $-3.36$, so the hypothesis "no cointegration" is not rejected. With the Dickey–Fuller value, $-2.87$, it would have been wrongly rejected.

The second question applies @eq:halflife: with $\hat\gamma = -0.1$ on monthly data, the half-life is $\ln 0.5/\ln 0.9 \approx 6.6$ months. The third applies the common trends representation: four $I(1)$ variables with cointegrating rank 3 are driven by $n - r = 4 - 3 = 1$ common trend.

The next chapter, [Chapter 8](../EN/Courses/chapter8_long_memory_arfima.pdf), covers long memory: series between $I(0)$ and $I(1)$, fractional differencing and ARFIMA models.

## @end

The chapter has shown that $I(1)$ series should not be differenced automatically: when they are cointegrated, their levels contain an equilibrium that is tested with Engle–Granger or Johansen and modelled with an ECM or a VECM. On real data, US yields have two equilibrium relations and a single common trend, and the VECM forecasts the spread better at long horizons. PPP for the leu, the three Central European currencies and pairs trading in Bucharest show, just as importantly, how to read a weak or negative result.
